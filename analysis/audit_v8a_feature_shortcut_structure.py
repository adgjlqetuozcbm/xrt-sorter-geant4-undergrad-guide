from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only feature-shortcut structure audit for v8A H/M sidecar features; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)

THRESHOLDS = {
    "nonmaterial_balanced_accuracy_max": 0.75,
    "shuffle_top_family_share_max": 0.50,
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
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import balanced_accuracy_score
    except ImportError as exc:
        raise SystemExit("Missing scikit-learn in the active environment.") from exc
    return {"ExtraTreesClassifier": ExtraTreesClassifier, "balanced_accuracy_score": balanced_accuracy_score}


def feature_family(col: str) -> str:
    if "residual" in col:
        return "residual"
    if "ratio" in col or "balance" in col:
        return "ratio"
    if "unique" in col:
        return "unique_window"
    if "prop_peak" in col or "_peak_" in col:
        return "prop_peak"
    if "window" in col:
        return "window"
    return "other"


def add_targets(frame: pd.DataFrame, count_bin_width: float) -> pd.DataFrame:
    result = frame.copy()
    result["audit_count_bin"] = np.floor(result["control_total_count_norm"].fillna(0.0).to_numpy(dtype=np.float64) / count_bin_width).astype(int).astype(str)
    result["audit_seed_group"] = result["random_seed"].astype(str)
    result["audit_source_id"] = result["source_id"].astype(str)
    result["audit_source_mode"] = result["source_mode"].astype(str)
    result["audit_stress_label"] = result["stress_label"].astype(str)
    result["audit_thickness"] = result["thickness_mm"].astype(str)
    result["audit_pose"] = result["pose_index"].astype(str)
    if "combined_feature_origin" in result.columns:
        result["audit_combined_origin"] = result["combined_feature_origin"].astype(str)
    else:
        result["audit_combined_origin"] = "unknown"
    result["audit_medium_vs_extension"] = result["audit_combined_origin"].str.contains("extension", case=False, na=False).map({True: "extension", False: "medium_or_unknown"})
    return result


def usable_target(train: pd.Series, eval_values: pd.Series) -> bool:
    return train.nunique(dropna=True) >= 2 and eval_values.nunique(dropna=True) >= 2


def evaluate_target(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    family_name: str,
    sk: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    source_on = frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    train = source_on[source_on["split"].astype(str).eq("train")].copy()
    eval_frame = source_on[source_on["split"].astype(str).isin(["validation", "stress_holdout"])].copy()
    if not feature_cols or train.empty or eval_frame.empty or not usable_target(train[target_col], eval_frame[target_col]):
        return (
            {
                "target": target_col,
                "feature_family": family_name,
                "status": "not_evaluable",
                "train_samples": int(len(train)),
                "eval_samples": int(len(eval_frame)),
                "target_classes_train": int(train[target_col].nunique(dropna=True)) if target_col in train else 0,
                "target_classes_eval": int(eval_frame[target_col].nunique(dropna=True)) if target_col in eval_frame else 0,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "feature_count": int(len(feature_cols)),
            },
            pd.DataFrame(),
        )
    estimator = sk["ExtraTreesClassifier"](n_estimators=180, random_state=seed, class_weight="balanced", max_features="sqrt", n_jobs=-1)
    estimator.fit(train[feature_cols].fillna(0.0).to_numpy(dtype=np.float64), train[target_col].astype(str).to_numpy())
    predictions = estimator.predict(eval_frame[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)).astype(str)
    y_true = eval_frame[target_col].astype(str).to_numpy()
    importances = pd.DataFrame(
        {
            "target": target_col,
            "feature_family": family_name,
            "feature": feature_cols,
            "importance": estimator.feature_importances_,
            "feature_source_family": [feature_family(col) for col in feature_cols],
        }
    ).sort_values("importance", ascending=False)
    return (
        {
            "target": target_col,
            "feature_family": family_name,
            "status": "evaluated",
            "train_samples": int(len(train)),
            "eval_samples": int(len(eval_frame)),
            "target_classes_train": int(train[target_col].nunique(dropna=True)),
            "target_classes_eval": int(eval_frame[target_col].nunique(dropna=True)),
            "accuracy": float(np.mean(y_true == predictions)) if len(y_true) else 0.0,
            "balanced_accuracy": float(sk["balanced_accuracy_score"](y_true, predictions)),
            "feature_count": int(len(feature_cols)),
        },
        importances,
    )


def high_shuffle_feature_importance(
    frame: pd.DataFrame,
    main_cols: list[str],
    null_dir: Path | None,
    sk: dict[str, Any],
) -> pd.DataFrame:
    if null_dir is None or not (null_dir / "v8a_shuffled_label_null_behavior_rows.csv").exists():
        return pd.DataFrame()
    null_rows = pd.read_csv(null_dir / "v8a_shuffled_label_null_behavior_rows.csv")
    rows_to_probe = (
        null_rows[
            null_rows["model"].astype(str).eq("ExtraTrees")
            & null_rows["eval_split"].astype(str).eq("validation")
            & null_rows["threshold_policy"].astype(str).eq("fixed_0p5")
        ]
        .sort_values("hm_min_recall", ascending=False)
        .head(3)
    )
    if rows_to_probe.empty:
        return pd.DataFrame()
    train = frame[frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    x_train = train[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_true = train["material"].astype(str).to_numpy()
    rows = []
    for _, probe in rows_to_probe.iterrows():
        seed = int(probe["shuffle_seed"])
        y_train = np.random.default_rng(seed).permutation(y_true)
        estimator = sk["ExtraTreesClassifier"](n_estimators=180, random_state=seed, class_weight="balanced", max_features="sqrt", n_jobs=-1)
        estimator.fit(x_train, y_train)
        for col, importance in sorted(zip(main_cols, estimator.feature_importances_), key=lambda item: item[1], reverse=True)[:20]:
            rows.append(
                {
                    "shuffle_seed": seed,
                    "null_hm_min_recall": float(probe["hm_min_recall"]),
                    "feature": col,
                    "importance": float(importance),
                    "feature_source_family": feature_family(col),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 18) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.head(limit)[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], summary: pd.DataFrame, shuffle_importance_summary: pd.DataFrame) -> None:
    lines = [
        "# v8A feature shortcut structure audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Max non-material balanced accuracy: `{gate['max_nonmaterial_balanced_accuracy']:.4f}`",
        f"- Shortcut suspected: `{str(gate['feature_or_sampling_shortcut_suspected']).lower()}`",
        "",
        "## Target Predictability",
        "",
        markdown_table(
            summary.sort_values("balanced_accuracy", ascending=False),
            ["target", "feature_family", "status", "balanced_accuracy", "accuracy", "feature_count"],
        ),
        "",
        "## High-Null Feature Families",
        "",
        markdown_table(shuffle_importance_summary, ["feature_source_family", "top_feature_count", "importance_sum"], limit=10),
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_feature_shortcut_structure_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether v8A diffraction features predict non-material shortcut variables.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--null-diagnosis-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count-bin-width", type=float, default=0.003)
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
            raise RuntimeError(f"Refusing shortcut audit because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing shortcut audit because {name} reports existing XRT cube reads.")
    frame = add_targets(pd.read_csv(input_dir / "v8a_event_sidecar_features.csv"), float(args.count_bin_width))
    main_cols, _, _, _, _ = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features available.")
    family_cols: dict[str, list[str]] = {"all": main_cols}
    for family in sorted({feature_family(col) for col in main_cols}):
        family_cols[family] = [col for col in main_cols if feature_family(col) == family]
    targets = [
        "audit_count_bin",
        "audit_stress_label",
        "audit_source_id",
        "audit_source_mode",
        "audit_combined_origin",
        "audit_medium_vs_extension",
        "audit_thickness",
        "audit_pose",
        "audit_seed_group",
    ]
    sk = require_sklearn()
    summary_rows = []
    importance_tables = []
    for target in targets:
        for family_name, cols in family_cols.items():
            row, importance = evaluate_target(frame, cols, target, family_name, sk, seed=11001)
            summary_rows.append(row)
            if not importance.empty:
                importance_tables.append(importance)
    summary = pd.DataFrame(summary_rows)
    importances = pd.concat(importance_tables, ignore_index=True) if importance_tables else pd.DataFrame()
    null_dir = project_root / args.null_diagnosis_dir if args.null_diagnosis_dir else None
    shuffle_importance = high_shuffle_feature_importance(frame, main_cols, null_dir, sk)
    if not shuffle_importance.empty:
        shuffle_importance_summary = (
            shuffle_importance.groupby("feature_source_family", sort=True)
            .agg(top_feature_count=("feature", "count"), importance_sum=("importance", "sum"))
            .reset_index()
            .sort_values("importance_sum", ascending=False)
        )
        top_family_share = float(shuffle_importance_summary["top_feature_count"].max() / max(shuffle_importance_summary["top_feature_count"].sum(), 1))
    else:
        shuffle_importance_summary = pd.DataFrame(columns=["feature_source_family", "top_feature_count", "importance_sum"])
        top_family_share = 0.0
    max_nonmaterial = float(summary.loc[summary["status"].eq("evaluated"), "balanced_accuracy"].max()) if not summary.empty else 0.0
    top_targets = summary[summary["feature_family"].eq("all")].sort_values("balanced_accuracy", ascending=False).head(5)
    shortcut_suspected = bool(max_nonmaterial > THRESHOLDS["nonmaterial_balanced_accuracy_max"])
    high_null_concentrated = bool(top_family_share > THRESHOLDS["shuffle_top_family_share_max"])
    pass_items = {
        "nonmaterial_targets_below_ceiling": not shortcut_suspected,
        "high_null_importance_not_family_concentrated": not high_null_concentrated,
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_feature_shortcut_structure.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_feature_shortcut_structure_audit",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "null_diagnosis_dir": args.null_diagnosis_dir,
        "gate_passed": gate_passed,
        "decision": "feature_shortcut_structure_clean" if gate_passed else "feature_or_sampling_shortcut_found",
        "main_feature_count": int(len(main_cols)),
        "max_nonmaterial_balanced_accuracy": max_nonmaterial,
        "feature_or_sampling_shortcut_suspected": shortcut_suspected,
        "high_null_feature_family_concentration_suspected": high_null_concentrated,
        "top_high_null_family_share": top_family_share,
        "top_nonmaterial_targets": json_clean(top_targets.to_dict(orient="records")),
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    summary.to_csv(output_dir / "v8a_feature_shortcut_target_summary.csv", index=False, lineterminator="\n")
    importances.to_csv(output_dir / "v8a_feature_shortcut_importances.csv", index=False, lineterminator="\n")
    shuffle_importance.to_csv(output_dir / "v8a_high_null_seed_feature_importances.csv", index=False, lineterminator="\n")
    shuffle_importance_summary.to_csv(output_dir / "v8a_high_null_seed_family_summary.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_feature_shortcut_structure_gate.json", json_clean(gate))
    write_report(output_dir, gate, summary, shuffle_importance_summary)
    print(
        "decision={decision} gate_passed={passed} max_nonmaterial={score:.4f} top_family_share={share:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            score=max_nonmaterial,
            share=top_family_share,
        )
    )


if __name__ == "__main__":
    main()
