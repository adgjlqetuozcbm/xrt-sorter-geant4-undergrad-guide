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
    "development-only root-cause decision for v8A shuffled-label/null anomaly; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path), "gate_passed": False}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def choose_decision(null_gate: dict[str, Any], shortcut_gate: dict[str, Any], stress_gate: dict[str, Any], probe_gate: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if bool(null_gate.get("gate_protocol_artifact_suspected", False)):
        reasons.append("validation-selected threshold inflated shuffled-label null performance")
        return "gate_protocol_artifact_found", reasons
    if bool(null_gate.get("tree_null_overfit_suspected", False)):
        reasons.append("ExtraTrees null performance exceeded Logistic under fixed threshold")
        return "tree_null_overfit_found", reasons
    if bool(shortcut_gate.get("feature_or_sampling_shortcut_suspected", False)):
        reasons.append("main features predict non-material audit targets above ceiling")
        return "sampling_or_origin_shortcut_found", reasons
    if bool(shortcut_gate.get("high_null_feature_family_concentration_suspected", False)):
        reasons.append("high-null shuffled models concentrate importance in one feature family")
        return "feature_shortcut_found", reasons
    if bool(stress_gate.get("stress_generator_artifact_suspected", False)):
        reasons.append("stress scenarios increased null performance or material-correlated deltas")
        return "stress_generator_artifact_found", reasons
    if bool(stress_gate.get("residual_stress_artifact_suspected", False)):
        reasons.append("residual features show material-correlated stress deltas")
        return "stress_generator_artifact_found", reasons
    if bool(probe_gate.get("clean_signal_confirmed", False)) and all(bool(g.get("gate_passed", False)) for g in [null_gate, shortcut_gate, stress_gate]):
        reasons.append("null, shortcut, and stress audits passed and probe found real-vs-null margin")
        return "clean_signal_confirmed_ready_for_v3_prereg", reasons
    reasons.append("no clean signal confirmation after root-cause audits")
    return "representation_insufficient", reasons


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A root-cause decision report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Root-cause decision: `{gate['root_cause_decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- v3 prereg unlocked: `{str(gate['v3_prereg_unlocked']).lower()}`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["decision_reasons"])
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is a development-only diagnostic decision. It does not unlock shadow/final, full ten-material matrix, H/M development large run, product accuracy, hardware validation, or manuscript-grade powder-XRD claims.",
            "",
        ]
    )
    (output_dir / "v8a_root_cause_decision_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide v8A root cause after null/shortcut/stress/probe audits.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--null-gate", required=True)
    parser.add_argument("--shortcut-gate", required=True)
    parser.add_argument("--stress-gate", required=True)
    parser.add_argument("--probe-gate", required=True)
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_root_cause_decision")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    null_gate = load_json(project_root / args.null_gate)
    shortcut_gate = load_json(project_root / args.shortcut_gate)
    stress_gate = load_json(project_root / args.stress_gate)
    probe_gate = load_json(project_root / args.probe_gate)
    decision, reasons = choose_decision(null_gate, shortcut_gate, stress_gate, probe_gate)
    v3_unlocked = decision == "clean_signal_confirmed_ready_for_v3_prereg"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/decide_v8a_root_cause.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_root_cause_decision",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "gate_passed": v3_unlocked,
        "v3_prereg_unlocked": v3_unlocked,
        "root_cause_decision": decision,
        "decision_reasons": reasons,
        "forbidden_next_steps": [
            "shadow_final",
            "full_ten_material_v8a_matrix",
            "hm_development_matrix_large_run",
            "product_accuracy_claim",
            "hardware_validation_claim",
            "manuscript_grade_powder_xrd_claim",
        ],
        "input_gate_decisions": {
            "null": null_gate.get("decision"),
            "shortcut": shortcut_gate.get("decision"),
            "stress": stress_gate.get("decision"),
            "probe": probe_gate.get("decision"),
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(output_dir / "v8a_root_cause_decision_gate.json", gate)
    write_report(output_dir, gate)
    print(
        "root_cause_decision={decision} v3_unlocked={unlocked}".format(
            decision=decision,
            unlocked=str(v3_unlocked).lower(),
        )
    )


if __name__ == "__main__":
    main()
