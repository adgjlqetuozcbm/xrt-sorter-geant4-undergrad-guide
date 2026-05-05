from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CLAIM_SCOPE = (
    "development-only admission audit for an H/M-only v8A development matrix preregistration; "
    "not product accuracy, hardware validation, shadow/final validation, full ten-material validation, "
    "or manuscript-grade powder XRD"
)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def flag(payload: dict[str, Any], name: str, default: bool = False) -> bool:
    return bool(payload.get(name, default))


def no_forbidden_inputs(payloads: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for name, payload in payloads.items():
        if payload.get("missing"):
            continue
        if flag(payload, "shadow_or_final_used"):
            failures.append(f"{name}_shadow_or_final_used")
        if flag(payload, "reads_existing_xrt_cubes"):
            failures.append(f"{name}_reads_existing_xrt_cubes")
        if flag(payload, "full_ten_material_matrix"):
            failures.append(f"{name}_full_ten_material_matrix")
    return not failures, failures


def lineage_clean(feature_manifest: dict[str, Any], phase4_gate: dict[str, Any], count_balanced_gate: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    manifest_lineage = feature_manifest.get("lineage_like_main_features", [])
    if manifest_lineage:
        failures.append("feature_manifest_lineage_like_main_features")
    for name, payload in {"phase4_gate": phase4_gate, "count_balanced_gate": count_balanced_gate}.items():
        integrity = payload.get("integrity_summary", {}) if isinstance(payload, dict) else {}
        if integrity.get("lineage_like_main_features"):
            failures.append(f"{name}_lineage_like_main_features")
    return not failures, failures


def stage_report_has_boundary(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    required_groups = [
        ["ordinary xrt does not solve h/m", "ordinary xrt solves h/m"],
        ["shadow/final remains sealed", "shadow/final"],
        ["full ten-material v8a matrix remains locked", "full ten-material"],
        ["not product accuracy", "no product accuracy", "product accuracy"],
    ]
    return all(any(item in text for item in group) for group in required_groups)


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A development matrix admission audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Preregistration unlocked: `{str(gate['development_matrix_prereg_unlocked']).lower()}`",
        "",
        "## Required Inputs",
        "",
    ]
    for key, passed in gate["pass_items"].items():
        lines.append(f"- `{key}`: `{str(passed).lower()}`")
    lines.extend(["", "## Stop Reasons", ""])
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Default Matrix Design If Later Unlocked",
            "",
            "- H/M-only development matrix.",
            "- Target rows: 3000-6000.",
            "- Balanced H/M, source-on/source-off, thickness, pose, seed, default/stress source variants.",
            "- Generated results stay under `results/accuracy_v3/` and are not committed by default.",
            "",
        ]
    )
    (output_dir / "v8a_development_matrix_admission_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether v8A H/M development matrix preregistration is unlocked.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--feature-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_event_to_feature")
    parser.add_argument("--peak-gate", default="results/accuracy_v3/v8a_peak_provenance_audit/v8a_peak_provenance_gate.json")
    parser.add_argument("--phase4-gate", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_development_model/v8a_medium_development_model_gate.json")
    parser.add_argument("--count-balanced-gate", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_count_balanced_retest/v8a_count_balanced_retest_gate.json")
    parser.add_argument("--stability-gate", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_stability/v8a_count_robust_stability_gate.json")
    parser.add_argument("--stage-report", default="docs/ACCURACY_SPRINT_V8A_COUNT_BALANCED_STAGE_REPORT_zh.md")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_development_matrix_admission")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    feature_manifest = load_json_if_exists(project_root / args.feature_dir / "v8a_event_feature_manifest.json")
    schema_gate = load_json_if_exists(project_root / args.feature_dir / "v8a_event_schema_gate.json")
    peak_gate = load_json_if_exists(project_root / args.peak_gate)
    phase4_gate = load_json_if_exists(project_root / args.phase4_gate)
    count_balanced_gate = load_json_if_exists(project_root / args.count_balanced_gate)
    stability_gate = load_json_if_exists(project_root / args.stability_gate)
    payloads = {
        "feature_manifest": feature_manifest,
        "schema_gate": schema_gate,
        "peak_gate": peak_gate,
        "phase4_gate": phase4_gate,
        "count_balanced_gate": count_balanced_gate,
        "stability_gate": stability_gate,
    }
    forbidden_ok, forbidden_failures = no_forbidden_inputs(payloads)
    lineage_ok, lineage_failures = lineage_clean(feature_manifest, phase4_gate, count_balanced_gate)
    pass_items = {
        "peak_provenance_gate_passed": bool(peak_gate.get("gate_passed", False)),
        "feature_schema_gate_passed": bool(schema_gate.get("gate_passed", False)),
        "ordinary_reworked_phase4_passed": bool(phase4_gate.get("gate_passed", False)),
        "strict_count_balanced_retest_passed": bool(count_balanced_gate.get("gate_passed", False)),
        "stability_replication_passed": bool(stability_gate.get("gate_passed", False)),
        "no_forbidden_inputs": forbidden_ok,
        "no_lineage_leakage": lineage_ok,
        "stage_report_claim_boundary_present": stage_report_has_boundary(project_root / args.stage_report),
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    stop_reasons.extend(forbidden_failures)
    stop_reasons.extend(lineage_failures)
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_development_matrix_admission.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_development_matrix_admission_audit",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "full_ten_material_matrix": False,
        "claim_scope": CLAIM_SCOPE,
        "gate_passed": gate_passed,
        "decision": "development_matrix_preregistration_unlocked_keep_shadow_final_sealed" if gate_passed else "stop_development_matrix_preregistration_not_unlocked",
        "development_matrix_prereg_unlocked": gate_passed,
        "default_matrix_design_if_unlocked": {
            "matrix_scope": "H/M-only development matrix",
            "target_rows": "3000-6000",
            "balanced_materials": True,
            "source_on_off_controls": True,
            "thickness_pose_seed_coverage": True,
            "default_and_stress_source_variants": True,
            "shadow_final_allowed": False,
            "full_ten_material_matrix_allowed": False,
        },
        "pass_items": pass_items,
        "stop_reasons": sorted(set(stop_reasons)),
        "input_decisions": {
            "schema_gate": schema_gate.get("decision"),
            "peak_gate": peak_gate.get("decision"),
            "phase4_gate": phase4_gate.get("decision"),
            "count_balanced_gate": count_balanced_gate.get("decision"),
            "stability_gate": stability_gate.get("decision"),
        },
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    write_json(output_dir / "v8a_development_matrix_admission_gate.json", gate)
    write_report(output_dir, gate)
    print(
        "decision={decision} gate_passed={passed} prereg_unlocked={unlocked}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            unlocked=str(gate_passed).lower(),
        )
    )


if __name__ == "__main__":
    main()
