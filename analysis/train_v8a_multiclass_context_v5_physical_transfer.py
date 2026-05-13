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

from train_v8a_event_feature_smoke import feature_sets, load_json
from train_v8a_multiclass_context_model import (
    HM_PAIR,
    add_lineage_controls,
    build_models,
    ensure_output_dir,
    evaluate_estimator,
    fit_estimator,
    json_clean,
    require_sklearn,
    write_json,
)


THRESHOLDS = {
    "nominal_validation_hm_min_recall_min": 0.90,
    "worst_perturbed_validation_hm_min_recall_min": 0.70,
    "worst_perturbed_stress_hm_min_recall_min": 0.65,
    "worst_perturbed_validation_macro_f1_min": 0.50,
    "worst_perturbed_stress_macro_f1_min": 0.45,
    "total_count_only_worst_perturbed_hm_max": 0.65,
    "lineage_only_worst_perturbed_hm_max": 0.65,
}


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def fit_on_nominal_train(model: dict[str, Any], train: pd.DataFrame, labels: list[str]) -> Any | None:
    if not model["feature_cols"] or train.empty:
        return None
    return fit_estimator(model, train, labels)


def profile_eval_frames(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    result: list[tuple[str, str, pd.DataFrame]] = []
    for split in ["validation", "stress_holdout"]:
        split_frame = frame[frame["split"].astype(str).eq(split)].copy()
        for profile, group in split_frame.groupby("physical_perturbation_profile", sort=True):
            result.append((split, str(profile), group.copy()))
    return result


def summary_best(summary: pd.DataFrame, family: str = "main") -> pd.DataFrame:
    rows: list[pd.Series] = []
    subset = summary[summary["family"].astype(str).eq(family) & summary["status"].astype(str).eq("evaluated")].copy()
    for _, group in subset.groupby(["eval_split", "physical_perturbation_profile"], sort=True):
        rows.append(group.sort_values(["hm_min_recall", "macro_f1", "min_class_recall"], ascending=False).iloc[0])
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 32) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.head(limit)[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], best: pd.DataFrame, controls: pd.DataFrame) -> None:
    lines = [
        "# v8A v5 physical perturbation transfer report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Worst perturbed validation H/M min recall: `{gate['worst_perturbed_validation_hm_min_recall']:.4f}`",
        f"- Worst perturbed stress H/M min recall: `{gate['worst_perturbed_stress_hm_min_recall']:.4f}`",
        "",
        "## Best Main Model by Perturbation Profile",
        "",
        markdown_table(
            best.sort_values(["eval_split", "physical_perturbation_profile"]),
            ["eval_split", "physical_perturbation_profile", "method", "top1_accuracy", "macro_f1", "min_class_recall", "hematite_recall", "magnetite_recall", "hm_min_recall"],
            limit=32,
        ),
        "",
        "## Control Models",
        "",
        markdown_table(
            controls.sort_values(["eval_split", "physical_perturbation_profile", "method"]),
            ["eval_split", "physical_perturbation_profile", "method", "top1_accuracy", "macro_f1", "hm_min_recall"],
            limit=32,
        ),
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiclass_context_v5_physical_transfer_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on nominal v5 rows and evaluate transfer to physical perturbation profiles.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--final-audit-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    final_gate = load_json(as_project_path(project_root, args.final_audit_gate))
    labels = sorted(frame["material"].astype(str).unique().tolist())
    sk = require_sklearn()
    main_cols, _, total_count_cols, _, _ = feature_sets(frame)
    frame, lineage_cols = add_lineage_controls(frame)
    models = build_models(sk, main_cols, total_count_cols, lineage_cols)

    train = frame[
        frame["split"].astype(str).eq("train")
        & frame["physical_perturbation_profile"].astype(str).eq("nominal")
    ].copy()
    fitted = {str(model["method"]): fit_on_nominal_train(model, train, labels) for model in models}

    summary_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    for eval_split, profile, eval_frame in profile_eval_frames(frame):
        for model in models:
            summary, decisions, _ = evaluate_estimator(
                track="v5_nominal_train_transfer",
                model=model,
                estimator=fitted[str(model["method"])],
                eval_frame=eval_frame,
                eval_split=eval_split,
                labels=labels,
                sk=sk,
            )
            summary["physical_perturbation_profile"] = profile
            summary_rows.append(summary)
            if not decisions.empty:
                decisions["physical_perturbation_profile"] = profile
                decision_frames.append(decisions)

    summary = pd.DataFrame(summary_rows)
    best = summary_best(summary, "main")
    controls = summary[summary["family"].astype(str).eq("control")].copy()
    summary.to_csv(output_dir / "v8a_multiclass_context_v5_physical_transfer_summary.csv", index=False)
    best.to_csv(output_dir / "v8a_multiclass_context_v5_physical_transfer_best_main.csv", index=False)
    controls.to_csv(output_dir / "v8a_multiclass_context_v5_physical_transfer_controls.csv", index=False)
    if decision_frames:
        pd.concat(decision_frames, ignore_index=True).to_csv(output_dir / "v8a_multiclass_context_v5_physical_transfer_decisions.csv", index=False)

    nominal_val = best[
        best["eval_split"].astype(str).eq("validation")
        & best["physical_perturbation_profile"].astype(str).eq("nominal")
    ]
    perturbed = best[~best["physical_perturbation_profile"].astype(str).eq("nominal")]
    perturbed_val = perturbed[perturbed["eval_split"].astype(str).eq("validation")]
    perturbed_stress = perturbed[perturbed["eval_split"].astype(str).eq("stress_holdout")]
    def min_value(data: pd.DataFrame, column: str) -> float:
        return float(data[column].min()) if not data.empty else 0.0

    total_count_controls = controls[controls["method"].astype(str).eq("ExtraTreesTotalCountOnly")]
    lineage_controls = controls[controls["method"].astype(str).eq("ExtraTreesLineageOnly")]
    stop_reasons: list[str] = []
    if not bool(final_gate.get("training_unlocked", False)):
        stop_reasons.append("final_audit_did_not_unlock_training")
    nominal_val_hm = min_value(nominal_val, "hm_min_recall")
    worst_val_hm = min_value(perturbed_val, "hm_min_recall")
    worst_stress_hm = min_value(perturbed_stress, "hm_min_recall")
    worst_val_macro = min_value(perturbed_val, "macro_f1")
    worst_stress_macro = min_value(perturbed_stress, "macro_f1")
    total_count_worst = float(total_count_controls[~total_count_controls["physical_perturbation_profile"].astype(str).eq("nominal")]["hm_min_recall"].max()) if not total_count_controls.empty else 0.0
    lineage_worst = float(lineage_controls[~lineage_controls["physical_perturbation_profile"].astype(str).eq("nominal")]["hm_min_recall"].max()) if not lineage_controls.empty else 0.0
    if nominal_val_hm < THRESHOLDS["nominal_validation_hm_min_recall_min"]:
        stop_reasons.append("nominal_validation_hm_min_recall_below_threshold")
    if worst_val_hm < THRESHOLDS["worst_perturbed_validation_hm_min_recall_min"]:
        stop_reasons.append("worst_perturbed_validation_hm_min_recall_below_threshold")
    if worst_stress_hm < THRESHOLDS["worst_perturbed_stress_hm_min_recall_min"]:
        stop_reasons.append("worst_perturbed_stress_hm_min_recall_below_threshold")
    if worst_val_macro < THRESHOLDS["worst_perturbed_validation_macro_f1_min"]:
        stop_reasons.append("worst_perturbed_validation_macro_f1_below_threshold")
    if worst_stress_macro < THRESHOLDS["worst_perturbed_stress_macro_f1_min"]:
        stop_reasons.append("worst_perturbed_stress_macro_f1_below_threshold")
    if total_count_worst > THRESHOLDS["total_count_only_worst_perturbed_hm_max"]:
        stop_reasons.append("total_count_only_worst_perturbed_hm_above_shortcut_threshold")
    if lineage_worst > THRESHOLDS["lineage_only_worst_perturbed_hm_max"]:
        stop_reasons.append("lineage_only_worst_perturbed_hm_above_shortcut_threshold")

    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/train_v8a_multiclass_context_v5_physical_transfer.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "claim_scope": "development-only nominal-train physical-perturbation transfer diagnostic; not hardware validation or product accuracy",
        "gate_passed": gate_passed,
        "decision": "v5_physical_transfer_gate_passed_not_promoted" if gate_passed else "stop_v5_physical_transfer_gate",
        "input_dir": args.input_dir,
        "train_profile": "nominal",
        "sample_count": int(len(frame)),
        "train_samples": int(len(train)),
        "materials": labels,
        "main_feature_count": int(len(main_cols)),
        "nominal_validation_hm_min_recall": nominal_val_hm,
        "worst_perturbed_validation_hm_min_recall": worst_val_hm,
        "worst_perturbed_stress_hm_min_recall": worst_stress_hm,
        "worst_perturbed_validation_macro_f1": worst_val_macro,
        "worst_perturbed_stress_macro_f1": worst_stress_macro,
        "total_count_only_worst_perturbed_hm": total_count_worst,
        "lineage_only_worst_perturbed_hm": lineage_worst,
        "thresholds": THRESHOLDS,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(output_dir / "v8a_multiclass_context_v5_physical_transfer_gate.json", json_clean(gate))
    write_report(output_dir, gate, best, controls)
    print(
        "decision={decision} gate_passed={passed} worst_val_hm={val:.4f} worst_stress_hm={stress:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            val=worst_val_hm,
            stress=worst_stress_hm,
        )
    )


if __name__ == "__main__":
    main()
