from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLAIM_SCOPE = (
    "development-only v8A count-robust v2 candidate review before Phase 4; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)

CANDIDATES = [
    {
        "candidate_id": "v2_proportion_only",
        "feature_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_event_to_feature",
        "count_stratified_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_count_stratified_event_to_feature",
        "training_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_count_stratified_event_training",
        "stress_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_count_stratified_event_feature_stress_gate",
        "multiseed_shuffle_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_multiseed_shuffled_label",
    },
    {
        "candidate_id": "v2_residualized_no_absolute_windows",
        "feature_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_event_to_feature",
        "count_stratified_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_count_stratified_event_to_feature",
        "training_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_count_stratified_event_training",
        "stress_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_count_stratified_event_feature_stress_gate",
        "multiseed_shuffle_dir": "results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_multiseed_shuffled_label",
    },
]


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


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


def blocked_by(payload: dict[str, Any], field: str = "stop_reasons") -> str:
    reasons = payload.get(field, [])
    if not reasons:
        return ""
    return "; ".join(str(item) for item in reasons)


def forbidden_input_flags(payloads: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for name, payload in payloads.items():
        if bool(payload.get("missing")):
            failures.append(f"{name}_missing")
        if bool(payload.get("shadow_or_final_used", False)):
            failures.append(f"{name}_shadow_or_final_used")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            failures.append(f"{name}_reads_existing_xrt_cubes")
        if bool(payload.get("runs_geant4", False)):
            failures.append(f"{name}_runs_geant4")
    return failures


def summarize_candidate(project_root: Path, candidate: dict[str, str]) -> dict[str, Any]:
    feature_manifest = load_json_if_exists(project_root / candidate["feature_dir"] / "v8a_event_feature_manifest.json")
    feature_gate = load_json_if_exists(project_root / candidate["feature_dir"] / "v8a_event_schema_gate.json")
    view_gate = load_json_if_exists(project_root / candidate["count_stratified_dir"] / "v8a_event_schema_gate.json")
    training_gate = load_json_if_exists(project_root / candidate["training_dir"] / "v8a_event_training_gate.json")
    stress_gate = load_json_if_exists(project_root / candidate["stress_dir"] / "v8a_event_feature_stress_gate.json")
    shuffle_gate = load_json_if_exists(project_root / candidate["multiseed_shuffle_dir"] / "v8a_multiseed_shuffled_label_gate.json")
    inputs = {
        "feature_manifest": feature_manifest,
        "feature_gate": feature_gate,
        "count_stratified_view_gate": view_gate,
        "training_gate": training_gate,
        "stress_gate": stress_gate,
        "multiseed_shuffle_gate": shuffle_gate,
    }
    forbidden = forbidden_input_flags(inputs)
    stress_stop = blocked_by(stress_gate)
    shuffle_stop = blocked_by(shuffle_gate)
    pass_items = {
        "feature_schema_gate_passed": bool(feature_gate.get("gate_passed", False)),
        "count_stratified_view_passed": bool(view_gate.get("gate_passed", False)),
        "baseline_training_gate_passed": bool(training_gate.get("gate_passed", False)),
        "stress_gate_passed": bool(stress_gate.get("gate_passed", False)),
        "multiseed_shuffled_label_gate_passed": bool(shuffle_gate.get("gate_passed", False)),
        "no_forbidden_inputs": not forbidden,
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    if forbidden:
        stop_reasons.extend(forbidden)
    return {
        "candidate_id": candidate["candidate_id"],
        "feature_dir": candidate["feature_dir"],
        "count_stratified_dir": candidate["count_stratified_dir"],
        "training_dir": candidate["training_dir"],
        "stress_dir": candidate["stress_dir"],
        "multiseed_shuffle_dir": candidate["multiseed_shuffle_dir"],
        "candidate_passed": not stop_reasons,
        "phase4_unlocked_for_candidate": bool(
            pass_items["feature_schema_gate_passed"]
            and pass_items["count_stratified_view_passed"]
            and pass_items["baseline_training_gate_passed"]
            and pass_items["stress_gate_passed"]
            and pass_items["multiseed_shuffled_label_gate_passed"]
            and pass_items["no_forbidden_inputs"]
        ),
        "main_feature_count": int(feature_manifest.get("main_feature_count", 0) or 0),
        "count_stratified_pairs_train": int(view_gate.get("matched_pair_counts", {}).get("train", 0) or 0),
        "count_stratified_pairs_validation": int(view_gate.get("matched_pair_counts", {}).get("validation", 0) or 0),
        "count_stratified_pairs_stress_holdout": int(view_gate.get("matched_pair_counts", {}).get("stress_holdout", 0) or 0),
        "max_standardized_count_gap_abs": float(view_gate.get("max_standardized_count_gap_abs", 0.0) or 0.0),
        "training_main_hm_min_recall": float(training_gate.get("best_main_hm_min_recall", 0.0) or 0.0),
        "training_source_off_hm_min_recall": float(training_gate.get("source_off_hm_min_recall", 0.0) or 0.0),
        "stress_worst_main_hm_min_recall": float(stress_gate.get("worst_main_hm_min_recall", 0.0) or 0.0),
        "stress_worst_total_count_hm_min_recall": float(stress_gate.get("worst_total_count_hm_min_recall", 0.0) or 0.0),
        "stress_worst_overlap_only_hm_min_recall": float(stress_gate.get("worst_overlap_only_hm_min_recall", 0.0) or 0.0),
        "stress_worst_shuffled_label_hm_min_recall": float(stress_gate.get("worst_shuffled_label_hm_min_recall", 0.0) or 0.0),
        "stress_worst_source_off_hm_min_recall": float(stress_gate.get("worst_source_off_hm_min_recall", 0.0) or 0.0),
        "multiseed_shuffle_max_validation_hm_min_recall": float(shuffle_gate.get("max_validation_hm_min_recall", 0.0) or 0.0),
        "multiseed_shuffle_max_stress_holdout_hm_min_recall": float(shuffle_gate.get("max_stress_holdout_hm_min_recall", 0.0) or 0.0),
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "stress_stop_reasons": stress_stop,
        "multiseed_shuffle_stop_reasons": shuffle_stop,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        "# v8A count-robust v2 candidate review",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Phase 4 unlocked: `{str(gate['phase4_unlocked']).lower()}`",
        "",
        "## Candidate Summary",
        "",
        markdown_table(
            summary,
            [
                "candidate_id",
                "candidate_passed",
                "count_stratified_pairs_train",
                "count_stratified_pairs_validation",
                "count_stratified_pairs_stress_holdout",
                "max_standardized_count_gap_abs",
                "training_main_hm_min_recall",
                "stress_worst_main_hm_min_recall",
                "stress_worst_total_count_hm_min_recall",
                "stress_worst_shuffled_label_hm_min_recall",
                "multiseed_shuffle_max_validation_hm_min_recall",
                "multiseed_shuffle_max_stress_holdout_hm_min_recall",
            ],
        ),
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This review is a stop/go decision for development-only candidate promotion. It does not run Geant4, does not open shadow/final, and does not unlock a large development matrix.",
            "",
        ]
    )
    (output_dir / "v8a_count_robust_v2_candidate_review_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review v8A count-robust v2 candidates before Phase 4 promotion.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_count_robust_v2_candidate_review")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    rows = [summarize_candidate(project_root, candidate) for candidate in CANDIDATES]
    summary = pd.DataFrame(rows)
    phase4_unlocked = bool(summary["phase4_unlocked_for_candidate"].any()) if not summary.empty else False
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common_stop_reasons: list[str] = []
    if not phase4_unlocked:
        common_stop_reasons.append("no_v2_candidate_passed_pre_phase4_gates")
    if bool(summary["multiseed_shuffle_max_validation_hm_min_recall"].ge(0.55).any()) or bool(
        summary["multiseed_shuffle_max_stress_holdout_hm_min_recall"].ge(0.55).any()
    ):
        common_stop_reasons.append("multiseed_shuffled_label_sanity_failed")
    if bool(summary["stress_worst_shuffled_label_hm_min_recall"].ge(0.55).any()):
        common_stop_reasons.append("stress_gate_shuffled_label_sanity_failed")
    gate = {
        "generated_by": "analysis/audit_v8a_count_robust_v2_candidates.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_count_robust_v2_candidate_review_gate",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "output_dir": args.output_dir,
        "gate_passed": phase4_unlocked,
        "phase4_unlocked": phase4_unlocked,
        "decision": "v8a_count_robust_v2_candidate_passed_ready_for_phase4"
        if phase4_unlocked
        else "stop_v8a_count_robust_v2_before_phase4",
        "candidate_count": int(len(summary)),
        "passed_candidate_count": int(summary["candidate_passed"].sum()) if not summary.empty else 0,
        "candidate_rows": json_clean(rows),
        "stop_reasons": common_stop_reasons,
        "next_recommended_action": (
            "Run Phase 4 only for the passing v2 candidate, keeping shadow/final sealed."
            if phase4_unlocked
            else "Do not run Phase 4, count-balanced retest, stability replication, or development matrix preregistration from v2; diagnose shuffled-label null behavior and feature shortcut structure first."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    summary.to_csv(output_dir / "v8a_count_robust_v2_candidate_review_summary.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_count_robust_v2_candidate_review_gate.json", json_clean(gate))
    write_report(output_dir, gate, summary)
    print(
        "decision={decision} gate_passed={passed} phase4_unlocked={unlocked} passed_candidates={passed_count}".format(
            decision=gate["decision"],
            passed=str(gate["gate_passed"]).lower(),
            unlocked=str(phase4_unlocked).lower(),
            passed_count=gate["passed_candidate_count"],
        )
    )


if __name__ == "__main__":
    main()
