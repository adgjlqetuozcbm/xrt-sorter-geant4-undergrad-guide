from __future__ import annotations

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THRESHOLDS = {
    "validation_overall_top1_min": 0.95,
    "stress_overall_top1_min": 0.95,
    "validation_overall_macro_f1_min": 0.95,
    "stress_overall_macro_f1_min": 0.95,
    "validation_overall_hm_min_recall_min": 0.95,
    "stress_overall_hm_min_recall_min": 0.80,
    "stress_nominal_hm_min_recall_min": 0.95,
    "stress_nominal_macro_f1_min": 0.95,
    "stress_worst_profile_macro_f1_min": 0.90,
    "total_count_only_worst_profile_hm_max": 0.65,
    "lineage_only_worst_profile_hm_max": 0.65,
    "shuffled_label_worst_profile_hm_p95_max": 0.55,
    "cross_split_exact_main_feature_hash_overlap_max": 0,
}


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def metric_row(best: pd.DataFrame, split: str, profile: str) -> dict[str, Any]:
    row = best[
        best["eval_split"].astype(str).eq(split)
        & best["physical_perturbation_profile"].astype(str).eq(profile)
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if pd.notna(value) else default


def stop_if(condition: bool, stop_reasons: list[str], reason: str) -> None:
    if condition:
        stop_reasons.append(reason)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 32) -> str:
    if frame.empty:
        return "No rows."
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.head(limit)[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(output_dir: Path, gate: dict[str, Any], best: pd.DataFrame) -> None:
    stress_profiles = best[
        best["eval_split"].astype(str).eq("stress_holdout")
        & ~best["physical_perturbation_profile"].astype(str).eq("__overall__")
    ].copy()
    lines = [
        "# v8 twenty-material reporting-grade extension gate",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Claim scope: {gate['claim_scope']}",
        f"- Strict hard-negative gate passed: `{str(gate['strict_hard_negative_gate_passed']).lower()}`",
        "",
        "## Key Metrics",
        "",
        f"- Validation overall top-1 / macro-F1 / H-M min recall: `{gate['validation_overall_top1']:.4f}` / `{gate['validation_overall_macro_f1']:.4f}` / `{gate['validation_overall_hm_min_recall']:.4f}`",
        f"- Stress overall top-1 / macro-F1 / H-M min recall: `{gate['stress_overall_top1']:.4f}` / `{gate['stress_overall_macro_f1']:.4f}` / `{gate['stress_overall_hm_min_recall']:.4f}`",
        f"- Stress worst-profile macro-F1: `{gate['stress_worst_profile_macro_f1']:.4f}`",
        f"- Strict stress worst-profile H-M min recall, report-only: `{gate['strict_stress_worst_profile_hm_min_recall_report_only']:.4f}`",
        "",
        "## Control Checks",
        "",
        f"- Total-count-only worst-profile H-M: `{gate['total_count_only_worst_profile_hm']:.4f}`",
        f"- Lineage-only worst-profile H-M: `{gate['lineage_only_worst_profile_hm']:.4f}`",
        f"- Shuffled-label H-M p95: `{gate['shuffled_label_worst_profile_hm_p95']:.4f}`",
        f"- Exact cross-split feature hash overlap: `{gate['cross_split_exact_main_feature_hash_overlap_count']}`",
        "",
        "## Stress Profiles",
        "",
        markdown_table(
            stress_profiles.sort_values("physical_perturbation_profile"),
            [
                "physical_perturbation_profile",
                "method",
                "top1_accuracy",
                "macro_f1",
                "hematite_recall",
                "magnetite_recall",
                "hm_min_recall",
            ],
        ),
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {item}" for item in gate["caveats"]) if gate["caveats"] else lines.append("- None.")
    lines.extend(["", "## Stop Reasons", ""])
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiclass_context_v8_reporting_grade_extension_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reporting-grade 20-material extension gate on v8 outputs.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-dir", default="results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_model")
    parser.add_argument(
        "--skeptical-dir",
        default="results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_skeptical_shortcut_audit",
    )
    parser.add_argument(
        "--diagnostic-dir",
        default="results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_hard_negative_diagnostics",
    )
    parser.add_argument(
        "--output-dir",
        default="results/accuracy_v3/v8a_multiclass_context_v8_reporting_grade_extension_gate",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    model_dir = as_project_path(project_root, args.model_dir)
    skeptical_dir = as_project_path(project_root, args.skeptical_dir)
    diagnostic_dir = as_project_path(project_root, args.diagnostic_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    model_gate = read_json(model_dir / "v8a_multiclass_context_v8_hard_negative_robust_gate.json")
    skeptical_gate = read_json(skeptical_dir / "v8a_multiclass_context_v8_hard_negative_robust_skeptical_shortcut_gate.json")
    diagnostic_gate = read_json(diagnostic_dir / "v8a_multiclass_context_v8_hard_negative_robust_diagnostics_gate.json")
    best = pd.read_csv(model_dir / "v8a_multiclass_context_v8_hard_negative_robust_best_main.csv")

    validation_overall = metric_row(best, "validation", "__overall__")
    stress_overall = metric_row(best, "stress_holdout", "__overall__")
    stress_nominal = metric_row(best, "stress_holdout", "nominal")
    stress_profiles = best[
        best["eval_split"].astype(str).eq("stress_holdout")
        & ~best["physical_perturbation_profile"].astype(str).eq("__overall__")
    ].copy()

    validation_overall_top1 = f(validation_overall, "top1_accuracy")
    stress_overall_top1 = f(stress_overall, "top1_accuracy")
    validation_overall_macro = f(validation_overall, "macro_f1")
    stress_overall_macro = f(stress_overall, "macro_f1")
    validation_overall_hm = f(validation_overall, "hm_min_recall")
    stress_overall_hm = f(stress_overall, "hm_min_recall")
    stress_nominal_hm = f(stress_nominal, "hm_min_recall")
    stress_nominal_macro = f(stress_nominal, "macro_f1")
    stress_worst_profile_macro = float(stress_profiles["macro_f1"].min()) if not stress_profiles.empty else 0.0
    strict_stress_worst_hm = float(stress_profiles["hm_min_recall"].min()) if not stress_profiles.empty else 0.0

    total_count_worst = float(model_gate.get("total_count_only_worst_profile_hm", 0.0))
    lineage_worst = float(model_gate.get("lineage_only_worst_profile_hm", 0.0))
    shuffled_p95 = float(model_gate.get("shuffled_label_worst_profile_hm_p95", 0.0))
    hash_overlap = int(skeptical_gate.get("cross_split_exact_main_feature_hash_overlap_count", 0))

    stop_reasons: list[str] = []
    stop_if(validation_overall_top1 < THRESHOLDS["validation_overall_top1_min"], stop_reasons, "validation_overall_top1_below_reporting_threshold")
    stop_if(stress_overall_top1 < THRESHOLDS["stress_overall_top1_min"], stop_reasons, "stress_overall_top1_below_reporting_threshold")
    stop_if(validation_overall_macro < THRESHOLDS["validation_overall_macro_f1_min"], stop_reasons, "validation_overall_macro_f1_below_reporting_threshold")
    stop_if(stress_overall_macro < THRESHOLDS["stress_overall_macro_f1_min"], stop_reasons, "stress_overall_macro_f1_below_reporting_threshold")
    stop_if(validation_overall_hm < THRESHOLDS["validation_overall_hm_min_recall_min"], stop_reasons, "validation_overall_hm_min_recall_below_reporting_threshold")
    stop_if(stress_overall_hm < THRESHOLDS["stress_overall_hm_min_recall_min"], stop_reasons, "stress_overall_hm_min_recall_below_reporting_threshold")
    stop_if(stress_nominal_hm < THRESHOLDS["stress_nominal_hm_min_recall_min"], stop_reasons, "stress_nominal_hm_min_recall_below_reporting_threshold")
    stop_if(stress_nominal_macro < THRESHOLDS["stress_nominal_macro_f1_min"], stop_reasons, "stress_nominal_macro_f1_below_reporting_threshold")
    stop_if(stress_worst_profile_macro < THRESHOLDS["stress_worst_profile_macro_f1_min"], stop_reasons, "stress_worst_profile_macro_f1_below_reporting_threshold")
    stop_if(total_count_worst > THRESHOLDS["total_count_only_worst_profile_hm_max"], stop_reasons, "total_count_control_above_shortcut_threshold")
    stop_if(lineage_worst > THRESHOLDS["lineage_only_worst_profile_hm_max"], stop_reasons, "lineage_control_above_shortcut_threshold")
    stop_if(shuffled_p95 > THRESHOLDS["shuffled_label_worst_profile_hm_p95_max"], stop_reasons, "shuffled_label_control_above_threshold")
    stop_if(hash_overlap > THRESHOLDS["cross_split_exact_main_feature_hash_overlap_max"], stop_reasons, "cross_split_exact_feature_hash_overlap_detected")

    caveats: list[str] = []
    if not bool(model_gate.get("gate_passed", False)):
        caveats.append("strict hard-negative robustness gate still fails; this reporting gate must not be described as strong hard-negative success")
    if strict_stress_worst_hm < 0.70:
        caveats.append("worst stress-profile H/M recall is below the strict robustness threshold and should be reported as a limitation")
    h_to_i_unique = int(diagnostic_gate.get("hematite_to_ilmenite_unique_samples_excluding_overall", 0))
    if h_to_i_unique:
        caveats.append(f"Hematite-to-Ilmenite errors remain in held-out stress profiles: {h_to_i_unique} unique samples")

    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_v8_reporting_grade_extension_gate.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "claim_scope": "development-only 20-material reporting extension; not a strict hard-negative robustness claim, not hardware validation, and not product accuracy",
        "gate_passed": gate_passed,
        "decision": "v8_twenty_material_reporting_extension_passed_with_caveats" if gate_passed else "stop_v8_twenty_material_reporting_extension_gate",
        "strict_hard_negative_gate_passed": bool(model_gate.get("gate_passed", False)),
        "validation_overall_top1": validation_overall_top1,
        "stress_overall_top1": stress_overall_top1,
        "validation_overall_macro_f1": validation_overall_macro,
        "stress_overall_macro_f1": stress_overall_macro,
        "validation_overall_hm_min_recall": validation_overall_hm,
        "stress_overall_hm_min_recall": stress_overall_hm,
        "stress_nominal_hm_min_recall": stress_nominal_hm,
        "stress_nominal_macro_f1": stress_nominal_macro,
        "stress_worst_profile_macro_f1": stress_worst_profile_macro,
        "strict_stress_worst_profile_hm_min_recall_report_only": strict_stress_worst_hm,
        "total_count_only_worst_profile_hm": total_count_worst,
        "lineage_only_worst_profile_hm": lineage_worst,
        "shuffled_label_worst_profile_hm_p95": shuffled_p95,
        "cross_split_exact_main_feature_hash_overlap_count": hash_overlap,
        "hematite_to_ilmenite_unique_samples_excluding_overall": h_to_i_unique,
        "thresholds": THRESHOLDS,
        "caveats": caveats,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    (output_dir / "v8a_multiclass_context_v8_reporting_grade_extension_gate.json").write_text(
        json.dumps(json_clean(gate), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    best.to_csv(output_dir / "v8a_multiclass_context_v8_reporting_grade_best_main.csv", index=False)
    write_report(output_dir, gate, best)
    print(
        "decision={decision} gate_passed={passed} stress_overall_hm={hm:.4f} stress_macro={macro:.4f} strict_worst_hm={worst:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            hm=stress_overall_hm,
            macro=stress_overall_macro,
            worst=strict_stress_worst_hm,
        )
    )


if __name__ == "__main__":
    main()
