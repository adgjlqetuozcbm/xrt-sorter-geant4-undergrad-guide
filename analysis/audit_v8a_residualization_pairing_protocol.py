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
)
from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only residualization and paired-orientation protocol audit for v8A H/M sidecar features; "
    "not training evidence, product accuracy, hardware validation, shadow/final validation, full ten-material "
    "matrix, or manuscript-grade powder XRD"
)

THRESHOLDS = {
    "oriented_auc_p95_increase_max": 0.03,
    "score_std_ratio_max": 1.75,
    "score_std_ratio_min": 0.57,
    "higher_order_orientation_abs_sum_max": 0.0,
    "residualized_oriented_auc_p95_reference_max": 0.58,
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


def source_feature_map(clean_manifest: dict[str, Any], clean_cols: list[str]) -> dict[str, str]:
    features = clean_manifest.get("residualization", {}).get("features", {})
    mapping = {}
    for clean_col in clean_cols:
        source_col = features.get(clean_col, {}).get("source_column")
        if source_col:
            mapping[clean_col] = str(source_col)
    if len(mapping) != len(clean_cols):
        missing = sorted(set(clean_cols) - set(mapping))
        raise RuntimeError(f"Residualization manifest is missing source columns for: {missing[:5]}")
    return mapping


def zscore_train_apply(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    train_mask = frame["split"].astype(str).eq("train")
    result = pd.DataFrame(index=frame.index)
    for col in columns:
        train_values = frame.loc[train_mask, col].fillna(0.0).to_numpy(dtype=np.float64)
        center = float(np.mean(train_values)) if len(train_values) else 0.0
        scale = float(np.std(train_values)) if len(train_values) else 1.0
        scale = scale if scale > 1e-12 else 1.0
        result[col] = (frame[col].fillna(0.0).to_numpy(dtype=np.float64) - center) / scale
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def aligned_source_frame(clean: pd.DataFrame, source: pd.DataFrame, clean_cols: list[str], mapping: dict[str, str]) -> pd.DataFrame:
    source_indexed = source.set_index("sample_id", drop=False)
    rows = []
    for sample_id in clean["sample_id"].astype(str):
        if sample_id not in source_indexed.index:
            raise RuntimeError(f"Source feature frame is missing sample_id={sample_id}")
        rows.append(source_indexed.loc[sample_id])
    aligned = pd.DataFrame(rows).reset_index(drop=True)
    for col in clean.columns:
        if col not in aligned.columns and not col.startswith("diffraction_"):
            aligned[col] = clean[col].to_numpy()
    source_cols = [mapping[col] for col in clean_cols]
    scaled_source = zscore_train_apply(aligned, source_cols)
    renamed = scaled_source.rename(columns={mapping[col]: col for col in clean_cols})
    output = pd.concat([clean[[col for col in clean.columns if not col.startswith("diffraction_")]].copy(), renamed[clean_cols]], axis=1)
    return output


def score_metrics(y_true: np.ndarray, probabilities: np.ndarray, roc_auc_score: Any) -> dict[str, float]:
    y_true = y_true.astype(str)
    y_binary = (y_true == "Magnetite").astype(int)
    try:
        auc = float(roc_auc_score(y_binary, probabilities)) if len(set(y_binary.tolist())) >= 2 else 0.5
    except ValueError:
        auc = 0.5
    return {
        "auc_magnetite_positive": auc,
        "oriented_auc": float(max(auc, 1.0 - auc)),
        "rank_overlap_index": float(1.0 - abs(2.0 * auc - 1.0)),
        "score_std": float(np.std(probabilities)),
        "score_mean": float(np.mean(probabilities)),
    }


def evaluate_view_scores(frame: pd.DataFrame, main_cols: list[str], seeds: list[int], roc_auc_score: Any, view_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    validation = frame[frame["split"].astype(str).eq("validation") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    holdout = frame[frame["split"].astype(str).eq("stress_holdout") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    pairs = pair_table(train)
    x_train = train[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    eval_frames = {"validation": validation, "stress_holdout": holdout}
    rows = []
    score_rows = []
    sk = {
        "make_pipeline": __import__("sklearn.pipeline", fromlist=["make_pipeline"]).make_pipeline,
        "StandardScaler": __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler,
        "LogisticRegression": __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression,
        "ExtraTreesClassifier": __import__("sklearn.ensemble", fromlist=["ExtraTreesClassifier"]).ExtraTreesClassifier,
    }
    for mode in NULL_MODES:
        for seed in seeds:
            orientations = orientation_map_for_mode(pairs, seed, mode)
            y_train, effective_shuffle_fraction, orientation_diag = apply_pair_orientations(train, orientations)
            for model_name, estimator in model_specs(sk, seed):
                fitted = deepcopy(estimator)
                fitted.fit(x_train, y_train)
                for eval_split, eval_frame in eval_frames.items():
                    prob = magnetite_probability(fitted, eval_frame[main_cols].fillna(0.0).to_numpy(dtype=np.float64))
                    y_true = eval_frame["material"].astype(str).to_numpy()
                    metrics = score_metrics(y_true, prob, roc_auc_score)
                    rows.append(
                        {
                            "view": view_name,
                            "shuffle_mode": mode,
                            "shuffle_seed": seed,
                            "model": model_name,
                            "eval_split": eval_split,
                            "effective_shuffle_fraction": effective_shuffle_fraction,
                            **orientation_diag,
                            **metrics,
                        }
                    )
                    for sample_id, split, material, seed_block, thickness, pose, count_bin, score in zip(
                        eval_frame["sample_id"].astype(str),
                        eval_frame["split"].astype(str),
                        eval_frame["material"].astype(str),
                        eval_frame["seed_block"].astype(str),
                        eval_frame["thickness_mm"],
                        eval_frame["pose_index"],
                        eval_frame["count_target_bin"].astype(str),
                        prob,
                    ):
                        score_rows.append(
                            {
                                "view": view_name,
                                "shuffle_mode": mode,
                                "shuffle_seed": seed,
                                "model": model_name,
                                "eval_split": split,
                                "sample_id": sample_id,
                                "material": material,
                                "seed_block": seed_block,
                                "thickness_mm": float(thickness),
                                "pose_index": int(pose),
                                "count_target_bin": count_bin,
                                "score": float(score),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["view", "shuffle_mode", "model", "eval_split"], sort=True)
        .agg(
            seed_count=("shuffle_seed", "nunique"),
            oriented_auc_p95=("oriented_auc", lambda values: float(pd.Series(values).quantile(0.95))),
            oriented_auc_max=("oriented_auc", "max"),
            rank_overlap_p05=("rank_overlap_index", lambda values: float(pd.Series(values).quantile(0.05))),
            score_std_mean=("score_std", "mean"),
            score_std_p95=("score_std", lambda values: float(pd.Series(values).quantile(0.95))),
            effective_shuffle_fraction_min=("effective_shuffle_fraction", "min"),
            effective_shuffle_fraction_max=("effective_shuffle_fraction", "max"),
            higher_order_orientation_abs_sum_max=("seed_block_thickness_pose_max_abs_orientation_sum", "max"),
        )
        .reset_index()
    )


def compare_summary(summary: pd.DataFrame) -> pd.DataFrame:
    clean = summary[summary["view"].eq("residualized_clean")].copy()
    source = summary[summary["view"].eq("pre_residual_source_scaled")].copy()
    merged = clean.merge(
        source,
        on=["shuffle_mode", "model", "eval_split"],
        suffixes=("_residualized", "_source"),
        how="inner",
    )
    merged["oriented_auc_p95_delta_residual_minus_source"] = merged["oriented_auc_p95_residualized"] - merged["oriented_auc_p95_source"]
    merged["score_std_p95_ratio_residual_over_source"] = merged["score_std_p95_residualized"] / merged["score_std_p95_source"].replace(0.0, np.nan)
    merged["score_std_p95_ratio_residual_over_source"] = merged["score_std_p95_ratio_residual_over_source"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged


def score_variance_by_group(score_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        score_rows.groupby(["view", "model", "eval_split", "seed_block", "thickness_mm", "pose_index", "count_target_bin"], sort=True)
        .agg(score_mean=("score", "mean"), score_std=("score", "std"), sample_count=("score", "size"))
        .reset_index()
    )
    grouped["score_std"] = grouped["score_std"].fillna(0.0)
    return grouped


def write_report(output_dir: Path, gate: dict[str, Any], comparison: pd.DataFrame) -> None:
    cols = [
        "shuffle_mode",
        "model",
        "eval_split",
        "oriented_auc_p95_source",
        "oriented_auc_p95_residualized",
        "oriented_auc_p95_delta_residual_minus_source",
        "score_std_p95_ratio_residual_over_source",
    ]
    lines = [
        "# v8A residualization and pairing protocol audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Max residualized AUC p95: `{gate['max_residualized_oriented_auc_p95']:.4f}`",
        f"- Max source AUC p95: `{gate['max_source_oriented_auc_p95']:.4f}`",
        f"- Max AUC delta residual-source: `{gate['max_oriented_auc_p95_delta_residual_minus_source']:.4f}`",
        f"- Score std ratio max/min: `{gate['score_std_ratio_max']:.4f}` / `{gate['score_std_ratio_min']:.4f}`",
        f"- Higher-order orientation max abs sum: `{gate['higher_order_orientation_abs_sum_max']:.4f}`",
        "",
        "## Residualized vs Source",
        "",
        "```csv",
        comparison[cols].to_csv(index=False, lineterminator="\n").rstrip(),
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
            "This audit diagnoses residualization/pairing behavior only. It does not unlock training.",
            "",
        ]
    )
    (output_dir / "v8a_residualization_pairing_protocol_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit residualization and paired-orientation protocol behavior for v8A clean H/M views.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-seeds", default=",".join(str(seed) for seed in range(11001, 11061)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    clean_dir = project_root / args.clean_dir
    source_dir = project_root / args.source_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    clean_manifest = load_json(clean_dir / "v8a_event_feature_manifest.json")
    source_manifest = load_json(source_dir / "v8a_event_feature_manifest.json")
    for name, payload in {"clean_manifest": clean_manifest, "source_manifest": source_manifest}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing residualization audit because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing residualization audit because {name} reports existing XRT cube reads.")
    clean = pd.read_csv(clean_dir / "v8a_event_sidecar_features.csv")
    source = pd.read_csv(source_dir / "v8a_event_sidecar_features.csv")
    clean_cols, _, _, _, _ = feature_sets(clean)
    mapping = source_feature_map(clean_manifest, clean_cols)
    source_aligned = aligned_source_frame(clean, source, clean_cols, mapping)
    seeds = [int(item.strip()) for item in args.shuffle_seeds.split(",") if item.strip()]
    roc_auc_score = require_sklearn()["roc_auc_score"]

    residual_rows, residual_scores = evaluate_view_scores(clean, clean_cols, seeds, roc_auc_score, "residualized_clean")
    source_rows, source_scores = evaluate_view_scores(source_aligned, clean_cols, seeds, roc_auc_score, "pre_residual_source_scaled")
    rows = pd.concat([residual_rows, source_rows], ignore_index=True)
    score_rows = pd.concat([residual_scores, source_scores], ignore_index=True)
    summary = summarize_rows(rows)
    comparison = compare_summary(summary)
    variance = score_variance_by_group(score_rows)

    max_residual_auc = float(summary[summary["view"].eq("residualized_clean")]["oriented_auc_p95"].max())
    max_source_auc = float(summary[summary["view"].eq("pre_residual_source_scaled")]["oriented_auc_p95"].max())
    max_delta = float(comparison["oriented_auc_p95_delta_residual_minus_source"].max()) if not comparison.empty else 0.0
    ratio_values = comparison["score_std_p95_ratio_residual_over_source"].replace([np.inf, -np.inf], np.nan).dropna()
    ratio_max = float(ratio_values.max()) if not ratio_values.empty else 0.0
    ratio_min = float(ratio_values.min()) if not ratio_values.empty else 0.0
    higher_order = float(summary["higher_order_orientation_abs_sum_max"].max()) if not summary.empty else 0.0
    pass_items = {
        "residualization_did_not_increase_auc_p95_too_much": max_delta <= THRESHOLDS["oriented_auc_p95_increase_max"],
        "residualized_auc_p95_below_reference": max_residual_auc <= THRESHOLDS["residualized_oriented_auc_p95_reference_max"],
        "score_std_ratio_not_too_high": ratio_max <= THRESHOLDS["score_std_ratio_max"],
        "score_std_ratio_not_too_low": ratio_min >= THRESHOLDS["score_std_ratio_min"],
        "higher_order_orientation_balanced": higher_order <= THRESHOLDS["higher_order_orientation_abs_sum_max"],
    }
    failure_labels = {
        "residualization_did_not_increase_auc_p95_too_much": "residualization_increases_null_auc_p95",
        "residualized_auc_p95_below_reference": "residualized_threshold_free_null_direction_remains_high",
        "score_std_ratio_not_too_high": "residualized_score_variance_too_high",
        "score_std_ratio_not_too_low": "residualized_score_variance_too_low",
        "higher_order_orientation_balanced": "higher_order_orientation_imbalance_detected",
    }
    stop_reasons = [failure_labels[name] for name, passed in pass_items.items() if not passed]
    if "residualization_increases_null_auc_p95" in stop_reasons:
        decision = "residualization_artifact_suspected"
    elif "higher_order_orientation_imbalance_detected" in stop_reasons:
        decision = "pairing_orientation_protocol_artifact_suspected"
    elif stop_reasons:
        decision = "representation_null_direction_persists_after_residualization_audit"
    else:
        decision = "residualization_pairing_protocol_clean"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_residualization_pairing_protocol.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_residualization_pairing_protocol_audit",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "claim_scope": CLAIM_SCOPE,
        "clean_dir": args.clean_dir,
        "source_dir": args.source_dir,
        "main_feature_count": int(len(clean_cols)),
        "max_residualized_oriented_auc_p95": max_residual_auc,
        "max_source_oriented_auc_p95": max_source_auc,
        "max_oriented_auc_p95_delta_residual_minus_source": max_delta,
        "score_std_ratio_max": ratio_max,
        "score_std_ratio_min": ratio_min,
        "higher_order_orientation_abs_sum_max": higher_order,
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "gate_passed": not stop_reasons,
        "decision": decision,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    rows.to_csv(output_dir / "v8a_residualization_pairing_null_rows.csv", index=False, lineterminator="\n")
    summary.to_csv(output_dir / "v8a_residualization_pairing_null_summary.csv", index=False, lineterminator="\n")
    comparison.to_csv(output_dir / "v8a_residualization_pairing_comparison.csv", index=False, lineterminator="\n")
    variance.to_csv(output_dir / "v8a_residualization_pairing_score_variance_by_group.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_residualization_pairing_protocol_gate.json", json_clean(gate))
    write_report(output_dir, gate, comparison)
    print(
        "decision={decision} gate_passed={passed} max_delta={delta:.4f} residual_auc={auc:.4f}".format(
            decision=decision,
            passed=str(gate["gate_passed"]).lower(),
            delta=max_delta,
            auc=max_residual_auc,
        )
    )


if __name__ == "__main__":
    main()
