from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnose_v8a_paired_clean_null_behavior import (
    NULL_MODES,
    PRIMARY_MODE,
    apply_pair_orientations,
    magnetite_probability,
    model_specs,
    orientation_map_for_mode,
    pair_table,
    selected_threshold,
    threshold_metrics,
)
from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only threshold-free paired-clean null protocol audit for v8A H/M sidecar features; "
    "not training evidence, product accuracy, hardware validation, shadow/final validation, full ten-material "
    "matrix, or manuscript-grade powder XRD"
)

THRESHOLDS = {
    "shuffle_seed_count_min": 60,
    "effective_shuffle_fraction_min": 0.45,
    "effective_shuffle_fraction_max": 0.55,
    "oriented_auc_p95_max": 0.58,
    "oriented_auc_single_seed_max": 0.68,
    "positive_threshold_inflation_p95_max": 0.05,
    "rank_overlap_p05_min": 0.84,
    "paired_recall_p95_ceiling_reference": 0.55,
}


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise SystemExit("Missing scikit-learn in the active environment.") from exc
    return {"roc_auc_score": roc_auc_score}


def score_metrics(y_true: np.ndarray, probabilities: np.ndarray, roc_auc_score: Any) -> dict[str, float]:
    y_true = y_true.astype(str)
    y_binary = (y_true == "Magnetite").astype(int)
    if len(set(y_binary.tolist())) < 2:
        auc = 0.5
    else:
        try:
            auc = float(roc_auc_score(y_binary, probabilities))
        except ValueError:
            auc = 0.5
    hematite = probabilities[y_true == "Hematite"]
    magnetite = probabilities[y_true == "Magnetite"]
    h_mean = float(np.mean(hematite)) if len(hematite) else 0.0
    m_mean = float(np.mean(magnetite)) if len(magnetite) else 0.0
    pooled_std = float(np.sqrt(0.5 * (np.var(hematite) + np.var(magnetite)) + 1e-12)) if len(hematite) and len(magnetite) else 1.0
    signed_gap = m_mean - h_mean
    rank_separation_abs = abs(2.0 * auc - 1.0)
    return {
        "auc_magnetite_positive": auc,
        "oriented_auc": float(max(auc, 1.0 - auc)),
        "rank_separation_abs": float(rank_separation_abs),
        "rank_overlap_index": float(1.0 - rank_separation_abs),
        "score_mean_hematite": h_mean,
        "score_mean_magnetite": m_mean,
        "signed_score_gap_m_minus_h": float(signed_gap),
        "abs_score_gap": float(abs(signed_gap)),
        "signed_score_gap_z": float(signed_gap / pooled_std) if pooled_std else 0.0,
        "abs_score_gap_z": float(abs(signed_gap) / pooled_std) if pooled_std else 0.0,
        "score_std_pooled": pooled_std,
    }


def evaluate_threshold_free_null(frame: pd.DataFrame, main_cols: list[str], seeds: list[int], roc_auc_score: Any) -> pd.DataFrame:
    train = frame[frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    validation = frame[frame["split"].astype(str).eq("validation") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    holdout = frame[frame["split"].astype(str).eq("stress_holdout") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    if train.empty or validation.empty or holdout.empty:
        raise RuntimeError("Threshold-free null audit requires source-on train/validation/stress_holdout rows.")
    pairs = pair_table(train)
    x_train = train[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    eval_frames = {"validation": validation, "stress_holdout": holdout}
    rows: list[dict[str, Any]] = []
    for mode in NULL_MODES:
        for seed in seeds:
            orientations = orientation_map_for_mode(pairs, seed, mode)
            y_train, effective_shuffle_fraction, orientation_diag = apply_pair_orientations(train, orientations)
            for model_name, estimator in model_specs(
                {
                    "make_pipeline": __import__("sklearn.pipeline", fromlist=["make_pipeline"]).make_pipeline,
                    "StandardScaler": __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler,
                    "LogisticRegression": __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression,
                    "ExtraTreesClassifier": __import__("sklearn.ensemble", fromlist=["ExtraTreesClassifier"]).ExtraTreesClassifier,
                },
                seed,
            ):
                fitted = deepcopy(estimator)
                fitted.fit(x_train, y_train)
                validation_prob = magnetite_probability(fitted, validation[main_cols].fillna(0.0).to_numpy(dtype=np.float64))
                validation_true = validation["material"].astype(str).to_numpy()
                selected_validation_threshold, selected_validation_metrics = selected_threshold(validation_true, validation_prob)
                fixed_validation_metrics = threshold_metrics(validation_true, validation_prob, 0.5)
                for eval_split, eval_frame in eval_frames.items():
                    x_eval = eval_frame[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
                    y_true = eval_frame["material"].astype(str).to_numpy()
                    prob = magnetite_probability(fitted, x_eval)
                    fixed = threshold_metrics(y_true, prob, 0.5)
                    if eval_split == "validation":
                        selected = selected_validation_metrics
                        selected_threshold_value = selected_validation_threshold
                    else:
                        selected = threshold_metrics(y_true, prob, selected_validation_threshold)
                        selected_threshold_value = selected_validation_threshold
                    threshold_inflation = float(selected["hm_min_recall"] - fixed["hm_min_recall"])
                    rows.append(
                        {
                            "shuffle_mode": mode,
                            "shuffle_seed": seed,
                            "model": model_name,
                            "eval_split": eval_split,
                            "effective_shuffle_fraction": effective_shuffle_fraction,
                            **orientation_diag,
                            **score_metrics(y_true, prob, roc_auc_score),
                            "fixed_threshold": 0.5,
                            "selected_threshold": float(selected_threshold_value),
                            "fixed_hm_min_recall": float(fixed["hm_min_recall"]),
                            "selected_hm_min_recall": float(selected["hm_min_recall"]),
                            "positive_threshold_inflation": float(max(threshold_inflation, 0.0)),
                            "selected_minus_fixed_hm_min_recall": threshold_inflation,
                            "fixed_accuracy": float(fixed["accuracy"]),
                            "selected_accuracy": float(selected["accuracy"]),
                        }
                    )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["shuffle_mode", "model", "eval_split"], sort=True)
    out = []
    for keys, group in grouped:
        mode, model, split = keys
        out.append(
            {
                "shuffle_mode": mode,
                "model": model,
                "eval_split": split,
                "seed_count": int(group["shuffle_seed"].nunique()),
                "effective_shuffle_fraction_min": float(group["effective_shuffle_fraction"].min()),
                "effective_shuffle_fraction_max": float(group["effective_shuffle_fraction"].max()),
                "oriented_auc_mean": float(group["oriented_auc"].mean()),
                "oriented_auc_p95": float(group["oriented_auc"].quantile(0.95)),
                "oriented_auc_max": float(group["oriented_auc"].max()),
                "rank_overlap_p05": float(group["rank_overlap_index"].quantile(0.05)),
                "rank_overlap_min": float(group["rank_overlap_index"].min()),
                "abs_score_gap_mean": float(group["abs_score_gap"].mean()),
                "abs_score_gap_p95": float(group["abs_score_gap"].quantile(0.95)),
                "abs_score_gap_z_p95": float(group["abs_score_gap_z"].quantile(0.95)),
                "fixed_hm_min_recall_p95": float(group["fixed_hm_min_recall"].quantile(0.95)),
                "selected_hm_min_recall_p95": float(group["selected_hm_min_recall"].quantile(0.95)),
                "fixed_hm_min_recall_max": float(group["fixed_hm_min_recall"].max()),
                "selected_hm_min_recall_max": float(group["selected_hm_min_recall"].max()),
                "positive_threshold_inflation_p95": float(group["positive_threshold_inflation"].quantile(0.95)),
                "positive_threshold_inflation_max": float(group["positive_threshold_inflation"].max()),
            }
        )
    return pd.DataFrame(out)


def write_report(output_dir: Path, gate: dict[str, Any], summary: pd.DataFrame) -> None:
    cols = [
        "shuffle_mode",
        "model",
        "eval_split",
        "oriented_auc_p95",
        "oriented_auc_max",
        "rank_overlap_p05",
        "fixed_hm_min_recall_p95",
        "selected_hm_min_recall_p95",
        "positive_threshold_inflation_p95",
    ]
    lines = [
        "# v8A threshold-free paired-clean null protocol audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Threshold-free gate passed: `{str(gate['threshold_free_gate_passed']).lower()}`",
        f"- Training unlocked: `{str(gate['training_unlocked']).lower()}`",
        f"- Primary oriented AUC p95: `{gate['primary_oriented_auc_p95']:.4f}`",
        f"- All-mode oriented AUC p95: `{gate['all_modes_oriented_auc_p95']:.4f}`",
        f"- Primary positive threshold inflation p95: `{gate['primary_positive_threshold_inflation_p95']:.4f}`",
        f"- Primary fixed recall p95: `{gate['primary_fixed_hm_min_recall_p95']:.4f}`",
        "",
        "## Summary",
        "",
        "```csv",
        summary[cols].to_csv(index=False, lineterminator="\n").rstrip(),
        "```",
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This audit can diagnose whether the null failure is threshold/protocol-driven or representation-driven. It does not unlock training by itself.",
            "",
        ]
    )
    (output_dir / "v8a_threshold_free_null_protocol_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run threshold-free paired-clean null protocol audit for a v8A H/M feature view.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-seeds", default=",".join(str(seed) for seed in range(11001, 11061)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    for name, payload in {"schema_gate": schema_gate, "manifest": manifest}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing threshold-free null audit because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing threshold-free null audit because {name} reports existing XRT cube reads.")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features available.")
    seeds = [int(item.strip()) for item in args.shuffle_seeds.split(",") if item.strip()]
    sk = require_sklearn()
    rows = evaluate_threshold_free_null(frame, main_cols, seeds, sk["roc_auc_score"])
    summary = summarize(rows)

    primary = summary[summary["shuffle_mode"].astype(str).eq(PRIMARY_MODE)]
    all_modes = summary.copy()
    primary_oriented_auc_p95 = float(primary["oriented_auc_p95"].max()) if not primary.empty else 1.0
    all_modes_oriented_auc_p95 = float(all_modes["oriented_auc_p95"].max()) if not all_modes.empty else 1.0
    primary_oriented_auc_max = float(primary["oriented_auc_max"].max()) if not primary.empty else 1.0
    all_modes_oriented_auc_max = float(all_modes["oriented_auc_max"].max()) if not all_modes.empty else 1.0
    primary_rank_overlap_p05 = float(primary["rank_overlap_p05"].min()) if not primary.empty else 0.0
    all_modes_rank_overlap_p05 = float(all_modes["rank_overlap_p05"].min()) if not all_modes.empty else 0.0
    primary_inflation_p95 = float(primary["positive_threshold_inflation_p95"].max()) if not primary.empty else 1.0
    all_modes_inflation_p95 = float(all_modes["positive_threshold_inflation_p95"].max()) if not all_modes.empty else 1.0
    primary_fixed_recall_p95 = float(primary["fixed_hm_min_recall_p95"].max()) if not primary.empty else 1.0
    primary_selected_recall_p95 = float(primary["selected_hm_min_recall_p95"].max()) if not primary.empty else 1.0
    effective_min = float(all_modes["effective_shuffle_fraction_min"].min()) if not all_modes.empty else 0.0
    effective_max = float(all_modes["effective_shuffle_fraction_max"].max()) if not all_modes.empty else 1.0
    seed_count_min = int(all_modes["seed_count"].min()) if not all_modes.empty else 0

    pass_items = {
        "shuffle_seed_count": seed_count_min >= THRESHOLDS["shuffle_seed_count_min"],
        "effective_shuffle_fraction_min_ok": effective_min >= THRESHOLDS["effective_shuffle_fraction_min"],
        "effective_shuffle_fraction_max_ok": effective_max <= THRESHOLDS["effective_shuffle_fraction_max"],
        "primary_oriented_auc_p95_under_ceiling": primary_oriented_auc_p95 <= THRESHOLDS["oriented_auc_p95_max"],
        "all_modes_oriented_auc_p95_under_ceiling": all_modes_oriented_auc_p95 <= THRESHOLDS["oriented_auc_p95_max"],
        "primary_oriented_auc_max_under_ceiling": primary_oriented_auc_max <= THRESHOLDS["oriented_auc_single_seed_max"],
        "all_modes_oriented_auc_max_under_ceiling": all_modes_oriented_auc_max <= THRESHOLDS["oriented_auc_single_seed_max"],
        "primary_rank_overlap_p05_ok": primary_rank_overlap_p05 >= THRESHOLDS["rank_overlap_p05_min"],
        "all_modes_rank_overlap_p05_ok": all_modes_rank_overlap_p05 >= THRESHOLDS["rank_overlap_p05_min"],
        "primary_threshold_inflation_p95_ok": primary_inflation_p95 <= THRESHOLDS["positive_threshold_inflation_p95_max"],
        "all_modes_threshold_inflation_p95_ok": all_modes_inflation_p95 <= THRESHOLDS["positive_threshold_inflation_p95_max"],
    }
    failure_labels = {
        "shuffle_seed_count": "shuffle_seed_count_below_minimum",
        "effective_shuffle_fraction_min_ok": "effective_shuffle_fraction_below_minimum",
        "effective_shuffle_fraction_max_ok": "effective_shuffle_fraction_above_maximum",
        "primary_oriented_auc_p95_under_ceiling": "primary_threshold_free_auc_p95_exceeded_ceiling",
        "all_modes_oriented_auc_p95_under_ceiling": "all_modes_threshold_free_auc_p95_exceeded_ceiling",
        "primary_oriented_auc_max_under_ceiling": "primary_threshold_free_auc_single_seed_max_exceeded_ceiling",
        "all_modes_oriented_auc_max_under_ceiling": "all_modes_threshold_free_auc_single_seed_max_exceeded_ceiling",
        "primary_rank_overlap_p05_ok": "primary_rank_overlap_below_minimum",
        "all_modes_rank_overlap_p05_ok": "all_modes_rank_overlap_below_minimum",
        "primary_threshold_inflation_p95_ok": "primary_threshold_selection_inflation_detected",
        "all_modes_threshold_inflation_p95_ok": "all_modes_threshold_selection_inflation_detected",
    }
    stop_reasons = [failure_labels[name] for name, passed in pass_items.items() if not passed]
    threshold_free_gate_passed = not stop_reasons
    paired_recall_still_high = max(primary_fixed_recall_p95, primary_selected_recall_p95) > THRESHOLDS["paired_recall_p95_ceiling_reference"]
    if threshold_free_gate_passed and paired_recall_still_high:
        decision = "threshold_protocol_artifact_suspected"
    elif not threshold_free_gate_passed and (
        primary_oriented_auc_p95 > THRESHOLDS["oriented_auc_p95_max"]
        or all_modes_oriented_auc_p95 > THRESHOLDS["oriented_auc_p95_max"]
    ):
        decision = "threshold_free_null_direction_found"
    elif not threshold_free_gate_passed:
        decision = "threshold_free_null_protocol_needs_review"
    else:
        decision = "threshold_free_null_clean"

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_threshold_free_null_protocol.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_threshold_free_paired_clean_null_protocol_audit",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "main_feature_count": int(len(main_cols)),
        "shuffle_seed_count": int(len(seeds)),
        "threshold_free_gate_passed": threshold_free_gate_passed,
        "gate_passed": threshold_free_gate_passed,
        "decision": decision,
        "primary_oriented_auc_p95": primary_oriented_auc_p95,
        "all_modes_oriented_auc_p95": all_modes_oriented_auc_p95,
        "primary_oriented_auc_max": primary_oriented_auc_max,
        "all_modes_oriented_auc_max": all_modes_oriented_auc_max,
        "primary_rank_overlap_p05": primary_rank_overlap_p05,
        "all_modes_rank_overlap_p05": all_modes_rank_overlap_p05,
        "primary_positive_threshold_inflation_p95": primary_inflation_p95,
        "all_modes_positive_threshold_inflation_p95": all_modes_inflation_p95,
        "primary_fixed_hm_min_recall_p95": primary_fixed_recall_p95,
        "primary_selected_hm_min_recall_p95": primary_selected_recall_p95,
        "paired_recall_still_high": paired_recall_still_high,
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    rows.to_csv(output_dir / "v8a_threshold_free_null_rows.csv", index=False, lineterminator="\n")
    summary.to_csv(output_dir / "v8a_threshold_free_null_summary.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_threshold_free_null_protocol_gate.json", json_clean(gate))
    write_report(output_dir, gate, summary)
    print(
        "decision={decision} threshold_free_gate_passed={passed} primary_auc_p95={auc:.4f} recall_p95={recall:.4f}".format(
            decision=decision,
            passed=str(threshold_free_gate_passed).lower(),
            auc=primary_oriented_auc_p95,
            recall=primary_fixed_recall_p95,
        )
    )


if __name__ == "__main__":
    main()
