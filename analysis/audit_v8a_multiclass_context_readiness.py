from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
    if not path.exists():
        if missing_ok:
            return {"missing": True, "path": path.as_posix()}
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def material_catalog_status(project_root: Path, target_materials: list[str]) -> dict[str, Any]:
    catalog_path = project_root / "source_models" / "materials" / "material_catalog.csv"
    frame = pd.read_csv(catalog_path)
    enabled = set(frame.loc[frame["enabled_for_undergrad"].astype(str).str.lower().eq("true"), "material_name"].astype(str))
    target_set = set(target_materials)
    return {
        "catalog_path": catalog_path.relative_to(project_root).as_posix(),
        "target_materials": target_materials,
        "enabled_target_materials": sorted(target_set & enabled),
        "missing_from_enabled_catalog": sorted(target_set - enabled),
    }


def peak_manifest_status(path: Path, target_materials: list[str]) -> dict[str, Any]:
    manifest = load_json(path, missing_ok=True)
    if manifest.get("missing"):
        return {
            "path": path.as_posix(),
            "exists": False,
            "peak_table_id": "",
            "status": "",
            "materials": [],
            "target_materials_present": [],
            "target_materials_missing": target_materials,
            "peak_count": 0,
        }
    materials = []
    peak_count = 0
    for block in manifest.get("materials", []):
        material = str(block.get("material", ""))
        materials.append(material)
        peaks = block.get("peaks", [])
        peak_count += len(peaks) if isinstance(peaks, list) else 0
    material_set = set(materials)
    target_set = set(target_materials)
    return {
        "path": path.as_posix(),
        "exists": True,
        "peak_table_id": str(manifest.get("peak_table_id", "")),
        "status": str(manifest.get("status", "")),
        "materials": sorted(materials),
        "target_materials_present": sorted(target_set & material_set),
        "target_materials_missing": sorted(target_set - material_set),
        "peak_count": peak_count,
    }


def csv_summary(path: Path, material_filter: set[str] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": path.as_posix()}
    frame = pd.read_csv(path)
    if material_filter is not None and "material" in frame.columns:
        frame = frame[frame["material"].astype(str).isin(material_filter)].copy()
    return {
        "missing": False,
        "path": path.as_posix(),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "records": frame.to_dict(orient="records"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A multiclass context readiness audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {gate['claim_scope']}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Matrix generation unlocked: `{str(gate['matrix_generation_unlocked']).lower()}`",
        f"- Training unlocked: `{str(gate['training_unlocked']).lower()}`",
        f"- Target materials: `{len(gate['target_materials'])}`",
        f"- Required peak manifest exists: `{str(gate['required_peak_manifest_status']['exists']).lower()}`",
        f"- Missing peak materials: `{';'.join(gate['required_peak_manifest_status']['target_materials_missing'])}`",
        f"- Peak provenance gate passed: `{str(gate['ten_material_peak_provenance_gate']['gate_passed']).lower()}`",
        "",
        "## Current Benchmarks",
        "",
        f"- H/M final audit decision: `{gate['hm_countmatched_final_audit'].get('decision')}`",
        f"- H/M training gate decision: `{gate['hm_countmatched_training_gate'].get('decision')}`",
        f"- Existing context probe H/M min recall: `{gate['existing_context_probe_summary'].get('hm_min_recall', '')}`",
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if gate["warnings"]:
        lines.extend(f"- {warning}" for warning in gate["warnings"])
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiclass_context_readiness_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether the formal v8A ten-material context matrix can be generated.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_multiclass_context_v1_config.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v1_readiness")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = as_project_path(project_root, args.config)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    config = load_json(config_path)
    target_materials = [str(item) for item in config["target_materials"]]
    hm_pair = set(str(item) for item in config["hm_pair"])
    required_inputs = config["required_inputs"]
    required_peak_manifest = as_project_path(project_root, config["required_peak_manifest"])
    fallback_hm_peak_manifest = as_project_path(project_root, config["fallback_hm_peak_manifest"])
    hm_training_gate = load_json(as_project_path(project_root, required_inputs["hm_countmatched_training_gate"]), missing_ok=True)
    hm_final_audit = load_json(as_project_path(project_root, required_inputs["hm_countmatched_final_audit_gate"]), missing_ok=True)
    peak_provenance_gate = load_json(
        as_project_path(project_root, required_inputs["ten_material_peak_provenance_gate"]),
        missing_ok=True,
    )
    context_probe_dir = as_project_path(project_root, required_inputs["existing_context_probe_dir"])
    context_summary_path = context_probe_dir / "final_test_summary.csv"
    context_recall_path = context_probe_dir / "per_class_recall_final_test.csv"
    context_summary = csv_summary(context_summary_path)
    context_hm_recall = csv_summary(context_recall_path, material_filter=hm_pair)

    existing_context_probe_summary: dict[str, Any] = {"available": False}
    if not context_summary.get("missing") and context_summary["records"]:
        selected = context_summary["records"][0]
        hm_records = context_hm_recall.get("records", [])
        hm_min = min((float(row.get("recall", 0.0)) for row in hm_records), default=0.0)
        existing_context_probe_summary = {
            "available": True,
            "method": selected.get("method"),
            "top1_accuracy": selected.get("top1_accuracy"),
            "macro_f1": selected.get("macro_f1"),
            "min_class_recall": selected.get("min_class_recall"),
            "hm_min_recall": hm_min,
            "hm_recall_records": hm_records,
        }

    catalog = material_catalog_status(project_root, target_materials)
    required_peak_status = peak_manifest_status(required_peak_manifest, target_materials)
    fallback_hm_peak_status = peak_manifest_status(fallback_hm_peak_manifest, target_materials)

    stop_reasons: list[str] = []
    warnings: list[str] = []
    if config.get("status") != "development_readiness_review":
        stop_reasons.append("config_status_not_development_readiness_review")
    if catalog["missing_from_enabled_catalog"]:
        stop_reasons.append("target_materials_missing_from_enabled_catalog")
    if not required_peak_status["exists"]:
        stop_reasons.append("required_ten_material_peak_manifest_missing")
    if required_peak_status["exists"] and required_peak_status["peak_table_id"] != str(config["required_peak_table_id"]):
        stop_reasons.append("required_peak_table_id_mismatch")
    if required_peak_status["target_materials_missing"]:
        stop_reasons.append("required_peak_manifest_missing_target_materials")
    if bool(config["hard_guards"].get("requires_peak_provenance_audit_before_matrix_generation", False)):
        if peak_provenance_gate.get("missing"):
            stop_reasons.append("ten_material_peak_provenance_audit_missing")
        else:
            if peak_provenance_gate.get("peak_table_id") != str(config["required_peak_table_id"]):
                stop_reasons.append("ten_material_peak_provenance_peak_table_id_mismatch")
            if not bool(peak_provenance_gate.get("gate_passed", False)):
                stop_reasons.append("ten_material_peak_provenance_audit_failed")
    if hm_final_audit.get("missing"):
        stop_reasons.append("hm_countmatched_final_audit_missing")
    elif not bool(hm_final_audit.get("training_unlocked", False)):
        warnings.append("hm_countmatched_final_audit_did_not_unlock_training_diagnostics")
    if hm_training_gate.get("missing"):
        warnings.append("hm_countmatched_training_gate_missing")
    elif not bool(hm_training_gate.get("gate_passed", False)):
        warnings.append("hm_countmatched_training_gate_failed_controls_remain_a_benchmark_risk")
    if not existing_context_probe_summary.get("available", False):
        warnings.append("existing_context_probe_missing")
    elif float(existing_context_probe_summary.get("hm_min_recall", 0.0)) <= 0.5:
        warnings.append("existing_context_probe_did_not_improve_hm_min_recall")

    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_readiness.py",
        "generated_at_utc": generated_at,
        "config": args.config,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "matrix_generation_unlocked": gate_passed,
        "claim_scope": config["claim_scope"],
        "gate_passed": gate_passed,
        "decision": "multiclass_context_matrix_generation_unlocked_not_run" if gate_passed else "stop_multiclass_context_before_matrix_generation",
        "profile": config["profile"],
        "target_materials": target_materials,
        "material_catalog_status": catalog,
        "required_peak_manifest_status": required_peak_status,
        "fallback_hm_peak_manifest_status": fallback_hm_peak_status,
        "ten_material_peak_provenance_gate": {
            "missing": bool(peak_provenance_gate.get("missing", False)),
            "decision": peak_provenance_gate.get("decision"),
            "gate_passed": bool(peak_provenance_gate.get("gate_passed", False)),
            "peak_table_id": peak_provenance_gate.get("peak_table_id"),
            "peak_count": peak_provenance_gate.get("peak_count"),
            "max_q_error_a_inv": peak_provenance_gate.get("max_q_error_a_inv"),
            "max_d_error_a": peak_provenance_gate.get("max_d_error_a"),
            "stop_reasons": peak_provenance_gate.get("stop_reasons", []),
        },
        "hm_countmatched_final_audit": {
            "missing": bool(hm_final_audit.get("missing", False)),
            "decision": hm_final_audit.get("decision"),
            "gate_passed": hm_final_audit.get("gate_passed"),
            "training_unlocked": hm_final_audit.get("training_unlocked"),
            "count_controls_required": hm_final_audit.get("count_controls_required"),
        },
        "hm_countmatched_training_gate": {
            "missing": bool(hm_training_gate.get("missing", False)),
            "decision": hm_training_gate.get("decision"),
            "gate_passed": hm_training_gate.get("gate_passed"),
            "ordinary_gate_passed": hm_training_gate.get("ordinary_gate", {}).get("gate_passed") if isinstance(hm_training_gate.get("ordinary_gate"), dict) else None,
            "count_balanced_gate_passed": hm_training_gate.get("count_balanced_gate", {}).get("gate_passed") if isinstance(hm_training_gate.get("count_balanced_gate"), dict) else None,
            "stop_reasons": hm_training_gate.get("stop_reasons", []),
        },
        "existing_context_probe_summary": existing_context_probe_summary,
        "hard_guards": config["hard_guards"],
        "next_if_ready": config["next_if_ready"],
        "next_if_blocked": config["next_if_blocked"],
        "stop_reasons": stop_reasons,
        "warnings": warnings,
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    write_json(output_dir / "v8a_multiclass_context_readiness_gate.json", gate)
    write_report(output_dir, gate)
    print(
        "decision={decision} gate_passed={passed} matrix_generation_unlocked={unlocked}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            unlocked=str(gate["matrix_generation_unlocked"]).lower(),
        )
    )


if __name__ == "__main__":
    main()
