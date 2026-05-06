from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import load_json


CLAIM_SCOPE = (
    "development-only summary of v8A H/M low-freedom tail-rebuild candidate gates; "
    "not training evidence, product accuracy, hardware validation, shadow/final validation, "
    "full ten-material matrix, or manuscript-grade powder XRD"
)


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


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def view_row(project_root: Path, view_id: str, prefix: str) -> dict[str, Any]:
    view_dir = project_root / f"{prefix}_{view_id}"
    shortcut_dir = project_root / f"{prefix}_{view_id}_feature_shortcut"
    null_dir = project_root / f"{prefix}_{view_id}_paired_null"
    tail_dir = project_root / f"{prefix}_{view_id}_null_tail_feature_family"
    admission_dir = project_root / f"{prefix}_{view_id}_admission"
    view_gate = load_json(view_dir / "v8a_event_schema_gate.json")
    shortcut = load_json(shortcut_dir / "v8a_feature_shortcut_structure_gate.json")
    null = load_json(null_dir / "v8a_paired_clean_null_behavior_gate.json")
    tail = load_json(tail_dir / "v8a_paired_null_tail_feature_family_gate.json")
    admission = load_json(admission_dir / "v8a_crystal_clean_admission_gate.json")
    for name, payload in {"view": view_gate, "shortcut": shortcut, "null": null, "tail": tail, "admission": admission}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing summary because {view_id}/{name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing summary because {view_id}/{name} reports existing XRT cube reads.")
    return {
        "view_id": view_id,
        "view_gate_passed": bool(view_gate.get("gate_passed", False)),
        "main_feature_count": int(view_gate.get("main_feature_count", 0)),
        "shortcut_gate_passed": bool(shortcut.get("gate_passed", False)),
        "max_nonmaterial_balanced_accuracy": float(shortcut.get("max_nonmaterial_balanced_accuracy", 1.0)),
        "paired_null_gate_passed": bool(null.get("gate_passed", False)),
        "primary_fixed_p95": float(null.get("primary_fixed_threshold_hm_min_recall_p95", 1.0)),
        "primary_selected_p95": float(null.get("primary_selected_threshold_hm_min_recall_p95", 1.0)),
        "all_modes_fixed_p95": float(null.get("all_modes_fixed_threshold_hm_min_recall_p95", 1.0)),
        "all_modes_selected_p95": float(null.get("all_modes_selected_threshold_hm_min_recall_p95", 1.0)),
        "primary_fixed_max": float(null.get("primary_fixed_threshold_hm_min_recall_max", 1.0)),
        "primary_selected_max": float(null.get("primary_selected_threshold_hm_min_recall_max", 1.0)),
        "tail_feature_family_decision": str(tail.get("decision", "")),
        "tail_feature_family_gate_passed": bool(tail.get("gate_passed", False)),
        "tail_rows_available": int(tail.get("tail_rows_available", 0)),
        "tail_rows_probed": int(tail.get("tail_rows_probed", 0)),
        "top_family_abs_weight_share": float(tail.get("top_family_abs_weight_share", 0.0)),
        "top_feature_abs_weight_share": float(tail.get("top_feature_abs_weight_share", 0.0)),
        "admission_gate_passed": bool(admission.get("gate_passed", False)),
        "training_unlocked": bool(admission.get("training_unlocked", False)),
        "admission_stop_reasons": ";".join(str(item) for item in admission.get("stop_reasons", [])),
    }


def write_report(output_dir: Path, gate: dict[str, Any], summary: pd.DataFrame) -> None:
    columns = [
        "view_id",
        "main_feature_count",
        "shortcut_gate_passed",
        "max_nonmaterial_balanced_accuracy",
        "paired_null_gate_passed",
        "primary_fixed_p95",
        "all_modes_fixed_p95",
        "primary_fixed_max",
        "tail_feature_family_decision",
        "tail_rows_available",
        "top_family_abs_weight_share",
        "training_unlocked",
    ]
    lines = [
        "# v8A tail rebuild v1 stage report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Training unlocked: `{str(gate['training_unlocked']).lower()}`",
        f"- Candidate count: `{gate['candidate_count']}`",
        f"- Passed candidate count: `{gate['passed_candidate_count']}`",
        "",
        "## Candidate Summary",
        "",
        "```csv",
        summary[columns].to_csv(index=False, lineterminator="\n").rstrip(),
        "```",
        "",
        "## Interpretation",
        "",
        "All candidates kept visible non-material shortcut scores below the ceiling, but none passed paired-clean null/admission. "
        "This means the next step is not training. The failure moved from a single visible source/stress/origin shortcut to a harder representation/null-protocol problem.",
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
            "No result here is product accuracy, hardware validation, shadow/final validation, full ten-material evidence, or manuscript-grade powder-XRD evidence.",
            "",
        ]
    )
    (output_dir / "v8a_tail_rebuild_v1_stage_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v8A tail rebuild v1 candidate gates.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(project_root / args.config)
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    prefix = str(config["output_prefix"])
    views = [str(item["view_id"]) for item in config["candidate_views"]]
    rows = [view_row(project_root, view_id, prefix) for view_id in views]
    summary = pd.DataFrame(rows)
    passed = summary[
        summary["view_gate_passed"]
        & summary["shortcut_gate_passed"]
        & summary["paired_null_gate_passed"]
        & summary["tail_feature_family_gate_passed"]
        & summary["admission_gate_passed"]
        & summary["training_unlocked"]
    ].copy()
    stop_reasons = []
    if passed.empty:
        stop_reasons.append("no_tail_rebuild_candidate_passed_full_clean_admission")
    if bool(summary["paired_null_gate_passed"].any()) is False:
        stop_reasons.append("all_tail_rebuild_candidates_failed_paired_clean_null")
    if bool(summary["training_unlocked"].any()) is False:
        stop_reasons.append("training_remains_locked")
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/summarize_v8a_tail_rebuild_v1.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_tail_rebuild_v1_stage_summary",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "config": args.config,
        "candidate_count": int(len(summary)),
        "passed_candidate_count": int(len(passed)),
        "training_unlocked": bool(not passed.empty),
        "gate_passed": bool(not stop_reasons),
        "decision": "tail_rebuild_candidate_ready_for_development_training_diagnostics" if not stop_reasons else "stop_tail_rebuild_v1_before_training",
        "stop_reasons": stop_reasons,
        "candidate_rows": json_clean(summary.to_dict(orient="records")),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    summary.to_csv(output_dir / "v8a_tail_rebuild_v1_candidate_summary.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_tail_rebuild_v1_stage_gate.json", json_clean(gate))
    write_report(output_dir, gate, summary)
    print(
        "decision={decision} passed_candidates={passed} training_unlocked={unlocked}".format(
            decision=gate["decision"],
            passed=len(passed),
            unlocked=str(gate["training_unlocked"]).lower(),
        )
    )


if __name__ == "__main__":
    main()
