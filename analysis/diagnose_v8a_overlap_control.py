from __future__ import annotations

import argparse
import json
import platform
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


def summarize_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    validation = frame["split"].astype(str).eq("validation") & frame["source_mode"].astype(str).eq("custom_diffraction_on")
    y = frame.loc[validation, "material"].astype(str)
    for col in columns:
        values = frame.loc[validation, col].fillna(0.0).astype(float)
        h = values[y.eq("Hematite")]
        m = values[y.eq("Magnetite")]
        pooled = float(np.sqrt(0.5 * (np.var(h) + np.var(m)) + 1e-12))
        rows.append(
            {
                "feature": col,
                "hematite_mean": float(np.mean(h)) if len(h) else 0.0,
                "magnetite_mean": float(np.mean(m)) if len(m) else 0.0,
                "mean_abs_delta": abs(float(np.mean(h) - np.mean(m))) if len(h) and len(m) else 0.0,
                "d_prime_abs": abs(float(np.mean(h) - np.mean(m))) / pooled if len(h) and len(m) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["d_prime_abs", "mean_abs_delta"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose overlap-only control strength in the v8A event-feature stress gate.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_event_to_feature_smoke")
    parser.add_argument("--stress-dir", default="results/accuracy_v3/v8a_event_feature_stress_gate")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_overlap_control_diagnostic")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    stress_dir = project_root / args.stress_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    stress_gate = load_json(stress_dir / "v8a_event_feature_stress_gate.json")
    feature_manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, overlap_cols, _ = feature_sets(frame)
    overlap_like_main = [col for col in main_cols if "35p" in col or "62p" in col or "unique" in col]
    overlap_summary = summarize_columns(frame, overlap_cols)
    overlap_like_main_summary = summarize_columns(frame, overlap_like_main)

    overlap_threshold = float(stress_gate.get("thresholds", {}).get("overlap_only_hm_min_recall_max", 0.6))
    overlap_value = float(stress_gate.get("worst_overlap_only_hm_min_recall", 0.0))
    stop_reasons = []
    if stress_gate.get("decision") != "stop_or_rework_v8a_stress_gate":
        stop_reasons.append("Stress gate did not stop; diagnostic is informational only.")
    if overlap_value < overlap_threshold:
        stop_reasons.append("Overlap-only control is already below the stricter ceiling.")
    decision = (
        "rework_overlap_feature_definition_before_medium_matrix"
        if stress_gate.get("decision") == "stop_or_rework_v8a_stress_gate"
        and overlap_value >= overlap_threshold
        else "overlap_diagnostic_informational_only"
    )

    gate = {
        "generated_by": "analysis/diagnose_v8a_overlap_control.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "claim_scope": "development-only overlap-control diagnostic; not a training or product metric",
        "stress_gate_decision": stress_gate.get("decision"),
        "stress_gate_passed": stress_gate.get("gate_passed"),
        "overlap_only_hm_min_recall_max": overlap_threshold,
        "worst_overlap_only_hm_min_recall": overlap_value,
        "overlap_feature_count": len(overlap_cols),
        "overlap_like_main_feature_count": len(overlap_like_main),
        "input_peak_table_id": feature_manifest.get("peak_table_id"),
        "decision": decision,
        "recommended_actions": [
            "Separate overlap-window diagnostics from main non-overlap peak features in the next event-to-feature revision.",
            "Add a non-overlap-only main gate and require it to preserve H/M recall without broad overlap windows.",
            "Review q-window tolerance around shared H/M peaks near 35.5 and 62.5 degrees.",
            "Rerun the stricter stress gate without relaxing the overlap-only ceiling."
        ],
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }

    overlap_summary.to_csv(output_dir / "v8a_overlap_control_feature_summary.csv", index=False, lineterminator="\n")
    overlap_like_main_summary.to_csv(output_dir / "v8a_overlap_like_main_feature_summary.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_overlap_control_diagnostic.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if decision == "rework_overlap_feature_definition_before_medium_matrix":
        scope = "development-only diagnostic explaining why the stricter stress gate stopped. This is not a training or product metric."
        actions = gate["recommended_actions"]
    else:
        scope = "development-only informational overlap diagnostic. This is not a training or product metric."
        actions = ["No overlap rework is required by this diagnostic decision."]
    lines = [
        "# v8A overlap-control diagnostic report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {scope}",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Stress gate decision: `{gate['stress_gate_decision']}`",
        f"- Worst overlap-only H/M min recall: `{gate['worst_overlap_only_hm_min_recall']}`",
        f"- Overlap control feature count: `{gate['overlap_feature_count']}`",
        f"- Overlap-like main feature count: `{gate['overlap_like_main_feature_count']}`",
        "",
        "## Recommended Actions",
        "",
    ]
    lines.extend(f"- {action}" for action in actions)
    lines.append("")
    (output_dir / "v8a_overlap_control_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"decision={gate['decision']} overlap_only={gate['worst_overlap_only_hm_min_recall']}")


if __name__ == "__main__":
    main()
