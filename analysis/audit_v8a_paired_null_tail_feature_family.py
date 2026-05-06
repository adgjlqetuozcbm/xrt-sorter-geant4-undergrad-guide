from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnose_v8a_paired_clean_null_behavior import (
    apply_pair_orientations,
    magnetite_probability,
    model_specs,
    orientation_map_for_mode,
    pair_table,
    threshold_metrics,
)
from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only paired-clean null-tail feature-family anatomy for v8A H/M sidecar features; "
    "this is a null/root-cause audit, not training evidence, product accuracy, hardware validation, "
    "shadow/final validation, or manuscript-grade powder XRD"
)

THRESHOLDS = {
    "tail_threshold": 0.55,
    "family_abs_weight_share_stop": 0.45,
    "feature_abs_weight_share_stop": 0.25,
    "weight_direction_consistency_stop": 0.70,
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


def feature_family(col: str) -> str:
    name = col.removeprefix("diffraction_crystal_clean_").removeprefix("diffraction_")
    if "ratio" in name or "balance" in name:
        return "ratio_balance"
    if "window_all" in name:
        return "window_all_peaks"
    if "window_hematite_unique" in name:
        return "window_hematite_unique"
    if "window_magnetite_unique" in name:
        return "window_magnetite_unique"
    if "peak_hematite" in name:
        return "peak_hematite"
    if "peak_magnetite" in name:
        return "peak_magnetite"
    if "unique" in name:
        return "unique_window"
    if "window" in name:
        return "window"
    return "other"


def expected_model_name(model_name: str, seed: int, sk: dict[str, Any]) -> Any:
    for name, estimator in model_specs(sk, seed):
        if name == model_name:
            return deepcopy(estimator)
    raise ValueError(f"Unsupported null-tail model: {model_name}")


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("Missing scikit-learn in the active environment.") from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def train_split(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    if train.empty:
        raise RuntimeError("Null-tail feature anatomy requires source-on train rows.")
    return train


def eval_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    result = frame[frame["split"].astype(str).eq(split) & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    if result.empty:
        raise RuntimeError(f"Null-tail feature anatomy requires source-on rows for split={split}.")
    return result


def fit_tail_estimator(
    frame: pd.DataFrame,
    main_cols: list[str],
    tail_row: pd.Series,
    sk: dict[str, Any],
) -> tuple[Any, np.ndarray, pd.DataFrame, dict[str, Any]]:
    train = train_split(frame)
    pairs = pair_table(train)
    mode = str(tail_row["shuffle_mode"])
    seed = int(tail_row["shuffle_seed"])
    orientations = orientation_map_for_mode(pairs, seed, mode)
    y_train, effective_shuffle_fraction, orientation_diag = apply_pair_orientations(train, orientations)
    estimator = expected_model_name(str(tail_row["model"]), seed, sk)
    x_train = train[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    estimator.fit(x_train, y_train)
    diagnostics = {"effective_shuffle_fraction": effective_shuffle_fraction, **orientation_diag}
    return estimator, y_train, train, diagnostics


def extract_contributions(estimator: Any, main_cols: list[str], row_meta: dict[str, Any]) -> pd.DataFrame:
    if row_meta["model"] == "Logistic":
        # Pipeline(StandardScaler, LogisticRegression). Coefficients are read after scaling.
        logistic = estimator.named_steps["logisticregression"] if hasattr(estimator, "named_steps") else estimator
        raw = logistic.coef_[0].astype(np.float64)
        contribution_kind = "standardized_logistic_coefficient"
        contribution = raw
    elif row_meta["model"] == "ExtraTrees":
        contribution_kind = "extra_trees_feature_importance"
        contribution = estimator.feature_importances_.astype(np.float64)
    else:
        return pd.DataFrame()
    rows = []
    abs_total = float(np.sum(np.abs(contribution))) or 1.0
    for col, value in zip(main_cols, contribution):
        rows.append(
            {
                **row_meta,
                "feature": col,
                "feature_family": feature_family(col),
                "contribution_kind": contribution_kind,
                "contribution": float(value),
                "abs_contribution": float(abs(value)),
                "abs_contribution_share_within_model": float(abs(value) / abs_total),
                "contribution_sign": int(np.sign(value)),
            }
        )
    return pd.DataFrame(rows)


def material_difference_direction(train: pd.DataFrame, main_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in main_cols:
        hematite = train[train["material"].astype(str).eq("Hematite")][col].fillna(0.0).astype(float)
        magnetite = train[train["material"].astype(str).eq("Magnetite")][col].fillna(0.0).astype(float)
        pooled = float(np.sqrt(0.5 * (np.var(hematite) + np.var(magnetite)) + 1e-12))
        delta = float(np.mean(magnetite) - np.mean(hematite))
        rows.append(
            {
                "feature": col,
                "feature_family": feature_family(col),
                "train_magnetite_minus_hematite_mean": delta,
                "train_d_prime_signed": float(delta / pooled) if pooled else 0.0,
            }
        )
    return pd.DataFrame(rows)


def tail_rows_to_probe(null_rows: pd.DataFrame, tail_threshold: float, max_rows: int) -> pd.DataFrame:
    tail = null_rows[null_rows["hm_min_recall"].fillna(0.0).astype(float) > tail_threshold].copy()
    if tail.empty:
        return tail
    return tail.sort_values(["hm_min_recall", "shuffle_seed", "model"], ascending=[False, True, True]).head(max_rows)


def summarize_contributions(contrib: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if contrib.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    family = (
        contrib.groupby(["model", "feature_family"], sort=True)
        .agg(
            tail_model_rows=("shuffle_seed", "count"),
            abs_contribution_sum=("abs_contribution", "sum"),
            abs_contribution_share_mean=("abs_contribution_share_within_model", "mean"),
            abs_contribution_share_max=("abs_contribution_share_within_model", "max"),
            positive_sign_share=("contribution_sign", lambda values: float(np.mean(np.asarray(values) > 0))),
            negative_sign_share=("contribution_sign", lambda values: float(np.mean(np.asarray(values) < 0))),
        )
        .reset_index()
    )
    totals = family.groupby("model")["abs_contribution_sum"].transform("sum").replace(0.0, 1.0)
    family["family_abs_contribution_share_by_model"] = family["abs_contribution_sum"] / totals
    family = family.sort_values(["model", "family_abs_contribution_share_by_model"], ascending=[True, False])

    feature = (
        contrib.groupby(["model", "feature", "feature_family"], sort=True)
        .agg(
            tail_model_rows=("shuffle_seed", "count"),
            abs_contribution_sum=("abs_contribution", "sum"),
            abs_contribution_share_mean=("abs_contribution_share_within_model", "mean"),
            abs_contribution_share_max=("abs_contribution_share_within_model", "max"),
            positive_sign_share=("contribution_sign", lambda values: float(np.mean(np.asarray(values) > 0))),
            negative_sign_share=("contribution_sign", lambda values: float(np.mean(np.asarray(values) < 0))),
        )
        .reset_index()
    )
    feature_totals = feature.groupby("model")["abs_contribution_sum"].transform("sum").replace(0.0, 1.0)
    feature["feature_abs_contribution_share_by_model"] = feature["abs_contribution_sum"] / feature_totals
    feature = feature.sort_values(["model", "feature_abs_contribution_share_by_model"], ascending=[True, False])

    top_features = (
        contrib.sort_values(["model", "shuffle_seed", "abs_contribution"], ascending=[True, True, False])
        .groupby(["model", "shuffle_seed", "shuffle_mode", "eval_split", "threshold_policy"], sort=True)
        .head(5)
        .reset_index(drop=True)
    )
    return family, feature, top_features


def evaluate_tail_row(estimator: Any, frame: pd.DataFrame, main_cols: list[str], tail_row: pd.Series) -> dict[str, Any]:
    split_frame = eval_split(frame, str(tail_row["eval_split"]))
    x_eval = split_frame[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_true = split_frame["material"].astype(str).to_numpy()
    probabilities = magnetite_probability(estimator, x_eval)
    metrics = threshold_metrics(y_true, probabilities, float(tail_row["threshold"]))
    return {
        "recomputed_accuracy": metrics["accuracy"],
        "recomputed_hm_min_recall": metrics["hm_min_recall"],
        "recomputed_hematite_recall": metrics["hematite_recall"],
        "recomputed_magnetite_recall": metrics["magnetite_recall"],
    }


def write_report(output_dir: Path, gate: dict[str, Any], family_summary: pd.DataFrame, feature_summary: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame, cols: list[str], limit: int) -> str:
        if frame.empty:
            return "No rows."
        return frame.head(limit)[cols].to_csv(index=False, lineterminator="\n").rstrip()

    lines = [
        "# v8A paired-clean null-tail feature-family anatomy",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Tail rows probed: `{gate['tail_rows_probed']}`",
        f"- Top family share: `{gate['top_family_abs_weight_share']:.4f}`",
        f"- Top feature share: `{gate['top_feature_abs_weight_share']:.4f}`",
        f"- Top feature direction consistency: `{gate['top_feature_direction_consistency']:.4f}`",
        "",
        "## Family Summary",
        "",
        "```csv",
        table(
            family_summary,
            [
                "model",
                "feature_family",
                "family_abs_contribution_share_by_model",
                "abs_contribution_share_mean",
                "positive_sign_share",
                "negative_sign_share",
            ],
            20,
        ),
        "```",
        "",
        "## Feature Summary",
        "",
        "```csv",
        table(
            feature_summary,
            [
                "model",
                "feature",
                "feature_family",
                "feature_abs_contribution_share_by_model",
                "abs_contribution_share_mean",
                "positive_sign_share",
                "negative_sign_share",
            ],
            20,
        ),
        "```",
        "",
        "## Interpretation Boundary",
        "",
        "This is a null-failure anatomy. It may justify a preregistered feature rebuild or replication plan, but it does not unlock model training.",
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    (output_dir / "v8a_paired_null_tail_feature_family_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit feature-family contributors for paired-clean null-tail rows.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--null-dir", required=True)
    parser.add_argument("--tail-anatomy-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail-threshold", type=float, default=THRESHOLDS["tail_threshold"])
    parser.add_argument("--max-tail-rows", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    feature_dir = project_root / args.feature_dir
    null_dir = project_root / args.null_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    for name, payload in {
        "feature_schema_gate": load_json(feature_dir / "v8a_event_schema_gate.json"),
        "feature_manifest": load_json(feature_dir / "v8a_event_feature_manifest.json"),
        "null_gate": load_json(null_dir / "v8a_paired_clean_null_behavior_gate.json"),
    }.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing null-tail feature anatomy because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing null-tail feature anatomy because {name} reports existing XRT cube reads.")

    frame = pd.read_csv(feature_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features available.")
    null_rows = pd.read_csv(null_dir / "v8a_paired_clean_null_behavior_rows.csv")
    tail = tail_rows_to_probe(null_rows, float(args.tail_threshold), int(args.max_tail_rows))
    sk = require_sklearn()

    contribution_frames = []
    probe_records = []
    for _, tail_row in tail.iterrows():
        row_meta = {
            "shuffle_mode": str(tail_row["shuffle_mode"]),
            "shuffle_seed": int(tail_row["shuffle_seed"]),
            "model": str(tail_row["model"]),
            "eval_split": str(tail_row["eval_split"]),
            "threshold_policy": str(tail_row["threshold_policy"]),
            "threshold": float(tail_row["threshold"]),
            "reported_hm_min_recall": float(tail_row["hm_min_recall"]),
        }
        estimator, y_train, train, diagnostics = fit_tail_estimator(frame, main_cols, tail_row, sk)
        contribution = extract_contributions(estimator, main_cols, row_meta)
        if not contribution.empty:
            contribution_frames.append(contribution)
        recomputed = evaluate_tail_row(estimator, frame, main_cols, tail_row)
        pseudo_counts = Counter(str(item) for item in y_train)
        probe_records.append(
            {
                **row_meta,
                **diagnostics,
                **recomputed,
                "pseudo_hematite_count": int(pseudo_counts.get("Hematite", 0)),
                "pseudo_magnetite_count": int(pseudo_counts.get("Magnetite", 0)),
            }
        )

    contrib = pd.concat(contribution_frames, ignore_index=True) if contribution_frames else pd.DataFrame()
    family_summary, feature_summary, top_features = summarize_contributions(contrib)
    train = train_split(frame)
    material_direction = material_difference_direction(train, main_cols)
    if not feature_summary.empty:
        feature_summary = feature_summary.merge(material_direction, on=["feature", "feature_family"], how="left")
    if not top_features.empty:
        top_features = top_features.merge(material_direction, on=["feature", "feature_family"], how="left")

    contrib.to_csv(output_dir / "v8a_paired_null_tail_feature_contributions.csv", index=False, lineterminator="\n")
    family_summary.to_csv(output_dir / "v8a_paired_null_tail_feature_family_summary.csv", index=False, lineterminator="\n")
    feature_summary.to_csv(output_dir / "v8a_paired_null_tail_feature_summary.csv", index=False, lineterminator="\n")
    top_features.to_csv(output_dir / "v8a_paired_null_tail_top_features_by_row.csv", index=False, lineterminator="\n")
    pd.DataFrame(probe_records).to_csv(output_dir / "v8a_paired_null_tail_probe_rows.csv", index=False, lineterminator="\n")

    top_family_share = float(family_summary["family_abs_contribution_share_by_model"].max()) if not family_summary.empty else 0.0
    top_feature_share = float(feature_summary["feature_abs_contribution_share_by_model"].max()) if not feature_summary.empty else 0.0
    top_feature_direction_consistency = 0.0
    if not feature_summary.empty:
        top_feature = feature_summary.sort_values("feature_abs_contribution_share_by_model", ascending=False).iloc[0]
        top_feature_direction_consistency = float(max(top_feature["positive_sign_share"], top_feature["negative_sign_share"]))
    top_family = family_summary.sort_values("family_abs_contribution_share_by_model", ascending=False).head(1).to_dict(orient="records")
    top_feature = feature_summary.sort_values("feature_abs_contribution_share_by_model", ascending=False).head(1).to_dict(orient="records")

    pass_items = {
        "no_single_family_dominance": top_family_share < THRESHOLDS["family_abs_weight_share_stop"],
        "no_single_feature_dominance": top_feature_share < THRESHOLDS["feature_abs_weight_share_stop"],
        "no_stable_single_feature_direction": top_feature_direction_consistency < THRESHOLDS["weight_direction_consistency_stop"],
    }
    failure_labels = {
        "no_single_family_dominance": "null_tail_single_feature_family_dominates",
        "no_single_feature_dominance": "null_tail_single_feature_dominates",
        "no_stable_single_feature_direction": "null_tail_single_feature_direction_consistent",
    }
    stop_reasons = [failure_labels[name] for name, passed in pass_items.items() if not passed]
    if len(tail) > 0:
        stop_reasons.insert(0, "paired_clean_null_tail_still_above_ceiling")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if any(reason in stop_reasons for reason in ["null_tail_single_feature_family_dominates", "null_tail_single_feature_dominates"]):
        decision = "feature_family_rebuild_prereg_needed"
    elif len(tail) > 0:
        decision = "broad_null_tail_replication_or_protocol_review_needed"
    else:
        decision = "null_tail_feature_family_clean"
    gate = {
        "generated_by": "analysis/audit_v8a_paired_null_tail_feature_family.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_paired_clean_null_tail_feature_family_anatomy",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "feature_dir": args.feature_dir,
        "null_dir": args.null_dir,
        "tail_anatomy_dir": args.tail_anatomy_dir,
        "tail_threshold": float(args.tail_threshold),
        "tail_rows_available": int((null_rows["hm_min_recall"].fillna(0.0).astype(float) > float(args.tail_threshold)).sum()),
        "tail_rows_probed": int(len(tail)),
        "main_feature_count": int(len(main_cols)),
        "top_family_abs_weight_share": top_family_share,
        "top_feature_abs_weight_share": top_feature_share,
        "top_feature_direction_consistency": top_feature_direction_consistency,
        "top_family": json_clean(top_family),
        "top_feature": json_clean(top_feature),
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "gate_passed": not stop_reasons,
        "decision": decision,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(output_dir / "v8a_paired_null_tail_feature_family_gate.json", json_clean(gate))
    write_report(output_dir, gate, family_summary, feature_summary)
    print(
        "decision={decision} tail_rows={tail_rows} top_family_share={family:.4f} top_feature_share={feature:.4f}".format(
            decision=decision,
            tail_rows=len(tail),
            family=top_family_share,
            feature=top_feature_share,
        )
    )


if __name__ == "__main__":
    main()
