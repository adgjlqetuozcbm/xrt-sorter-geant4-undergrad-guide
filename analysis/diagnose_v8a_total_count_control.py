from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def standardized_gap(left: pd.Series, right: pd.Series) -> float:
    left_values = left.fillna(0.0).to_numpy(dtype=np.float64)
    right_values = right.fillna(0.0).to_numpy(dtype=np.float64)
    pooled = np.sqrt(0.5 * (np.var(left_values) + np.var(right_values)) + 1e-12)
    return float(abs(np.mean(left_values) - np.mean(right_values)) / pooled)


def feature_summary(frame: pd.DataFrame, total_count_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eval_frame = frame[
        frame["split"].astype(str).isin(["validation", "stress_holdout"])
        & frame["source_mode"].astype(str).eq("custom_diffraction_on")
    ].copy()
    for split, split_frame in eval_frame.groupby("split", sort=True):
        hematite = split_frame[split_frame["material"].astype(str).eq("Hematite")]
        magnetite = split_frame[split_frame["material"].astype(str).eq("Magnetite")]
        for col in total_count_cols:
            rows.append(
                {
                    "split": split,
                    "feature": col,
                    "hematite_mean": float(hematite[col].mean()),
                    "magnetite_mean": float(magnetite[col].mean()),
                    "mean_difference_magnetite_minus_hematite": float(magnetite[col].mean() - hematite[col].mean()),
                    "hematite_std": float(hematite[col].std()),
                    "magnetite_std": float(magnetite[col].std()),
                    "standardized_gap_abs": standardized_gap(hematite[col], magnetite[col]),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "standardized_gap_abs"], ascending=[True, False])


def group_summary(frame: pd.DataFrame, total_count_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eval_frame = frame[
        frame["split"].astype(str).isin(["validation", "stress_holdout"])
        & frame["source_mode"].astype(str).eq("custom_diffraction_on")
    ].copy()
    group_columns = ["split", "stress_label", "thickness_mm", "pose_index"]
    for keys, group in eval_frame.groupby(group_columns + ["material"], sort=True):
        split, stress_label, thickness_mm, pose_index, material = keys
        row = {
            "split": split,
            "stress_label": stress_label,
            "thickness_mm": thickness_mm,
            "pose_index": pose_index,
            "material": material,
            "samples": int(len(group)),
        }
        for col in total_count_cols:
            row[f"{col}_mean"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, diagnostic: dict[str, Any], feature_table: pd.DataFrame) -> None:
    lines = [
        "# v8A total-count control diagnostic",
        "",
        f"Generated: {diagnostic['generated_at_utc']}",
        "",
        "Scope: development-only diagnostic explaining whether the medium Phase 4 model gate was stopped by a total-count-only shortcut. This is not product accuracy and not shadow/final evidence.",
        "",
        "## Decision",
        "",
        f"- Decision: `{diagnostic['decision']}`",
        f"- Total-count-only H/M min recall: `{diagnostic['total_count_only_hm_min_recall']:.4f}`",
        f"- Ceiling: `<{diagnostic['total_count_only_hm_min_recall_max']:.2f}`",
        f"- Max standardized material gap: `{diagnostic['max_standardized_gap_abs']:.4f}`",
        "",
        "## Largest Total-Count Gaps",
        "",
        markdown_table(
            feature_table.head(12),
            [
                "split",
                "feature",
                "hematite_mean",
                "magnetite_mean",
                "mean_difference_magnetite_minus_hematite",
                "standardized_gap_abs",
            ],
        ),
        "",
        "## Recommended Rework",
        "",
    ]
    for item in diagnostic["recommended_actions"]:
        lines.append(f"- {item}")
    lines.append("")
    (output_dir / "v8a_total_count_control_diagnostic_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose total-count-only control strength in the v8A medium development model gate.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_medium_event_to_feature")
    parser.add_argument("--model-dir", default="results/accuracy_v3/v8a_medium_development_model")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_medium_total_count_control_diagnostic")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    model_dir = project_root / args.model_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    model_gate = load_json(model_dir / "v8a_medium_development_model_gate.json")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    _, _, total_count_cols, _, _ = feature_sets(frame)
    feature_table = feature_summary(frame, total_count_cols)
    groups = group_summary(frame, total_count_cols)
    total_count_value = float(model_gate.get("total_count_only_hm_min_recall", 0.0))
    total_count_ceiling = float(model_gate.get("thresholds", {}).get("total_count_only_hm_min_recall_max", 0.6))
    gate_stopped_for_total_count = "total_count_only_below_ceiling" in model_gate.get("stop_reasons", [])
    max_gap = float(feature_table["standardized_gap_abs"].max()) if not feature_table.empty else 0.0
    decision = (
        "rework_total_count_confounding_before_any_shadow_final_or_product_claim"
        if gate_stopped_for_total_count and total_count_value >= total_count_ceiling
        else "total_count_diagnostic_informational_only"
    )
    recommended_actions = [
        "Do not open shadow/final and do not claim product accuracy.",
        "Add a total-count-normalized or count-residualized main-feature variant and rerun Phase 4.",
        "Add source-intensity/material balancing or per-run count equalization before the next medium matrix if the shortcut persists.",
        "Keep total-count-only as a negative control ceiling in all successor gates.",
    ]
    diagnostic = {
        "generated_by": "analysis/diagnose_v8a_total_count_control.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": "development-only total-count confounder diagnostic; not product accuracy or shadow/final validation",
        "input_dir": args.input_dir,
        "model_dir": args.model_dir,
        "decision": decision,
        "model_gate_decision": model_gate.get("decision"),
        "model_gate_passed": bool(model_gate.get("gate_passed", False)),
        "total_count_only_hm_min_recall": total_count_value,
        "total_count_only_hm_min_recall_max": total_count_ceiling,
        "gate_stopped_for_total_count": gate_stopped_for_total_count,
        "total_count_feature_count": int(len(total_count_cols)),
        "max_standardized_gap_abs": max_gap,
        "recommended_actions": recommended_actions,
    }
    feature_table.to_csv(output_dir / "v8a_total_count_feature_summary.csv", index=False, lineterminator="\n")
    groups.to_csv(output_dir / "v8a_total_count_group_summary.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_total_count_control_diagnostic.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(output_dir, diagnostic, feature_table)
    print(
        "decision={decision} total_count_only={value:.4f} max_gap={gap:.4f}".format(
            decision=decision,
            value=total_count_value,
            gap=max_gap,
        )
    )


if __name__ == "__main__":
    main()
