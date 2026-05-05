from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HC_KEV_A = 12.398419843320026
REQUIRED_TOP_LEVEL = {
    "peak_table_id",
    "version",
    "status",
    "reference_type",
    "reference_citation",
    "reference_url_or_doi",
    "wavelength_a",
    "intensity_normalization",
    "materials",
    "known_limitations",
    "upgrade_required_before",
}
REQUIRED_MATERIAL_FIELDS = {
    "material",
    "phase_name",
    "chemical_formula",
    "structure_note",
    "reference_citation",
    "reference_url_or_doi",
    "peaks",
}
REQUIRED_PEAK_FIELDS = {
    "peak_id",
    "hkl",
    "two_theta_deg",
    "q_a_inv",
    "d_a",
    "relative_intensity",
    "reference_citation",
    "reference_url_or_doi",
}
REQUIRED_MATERIALS = {"Hematite", "Magnetite"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def q_from_two_theta(two_theta_deg: float, wavelength_a: float) -> float:
    return 4.0 * math.pi * math.sin(math.radians(two_theta_deg) / 2.0) / wavelength_a


def d_from_q(q_a_inv: float) -> float:
    return 2.0 * math.pi / q_a_inv


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if value not in (None, "") else []


def has_external_reference(value: Any) -> bool:
    return any(str(item).strip().startswith("http") or "doi" in str(item).strip().lower() for item in as_list(value))


def audit_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stop_reasons: list[str] = []
    warnings: list[str] = []
    peak_rows: list[dict[str, Any]] = []

    missing_top = sorted(REQUIRED_TOP_LEVEL.difference(manifest))
    if missing_top:
        stop_reasons.append(f"Missing top-level fields: {missing_top}")

    if manifest.get("status") != "development_reference_candidate":
        stop_reasons.append("Manifest status must be development_reference_candidate for this phase.")

    reference_values = [str(item).strip() for item in as_list(manifest.get("reference_url_or_doi"))]
    if not has_external_reference(reference_values):
        stop_reasons.append("No external reference URL or DOI recorded.")

    citation = str(manifest.get("reference_citation", "")).strip().lower()
    if "internal" in citation and "candidate" not in citation:
        stop_reasons.append("Reference citation still appears to be internal-only.")

    wavelength_a = float(manifest.get("wavelength_a", 0.0) or 0.0)
    if wavelength_a <= 0.0:
        stop_reasons.append("wavelength_a must be positive.")

    materials = manifest.get("materials", [])
    if not isinstance(materials, list):
        stop_reasons.append("materials must be a list of material reference blocks.")
        materials = []
    materials_seen = {str(block.get("material", "")) for block in materials if isinstance(block, dict)}
    missing_materials = sorted(REQUIRED_MATERIALS.difference(materials_seen))
    if missing_materials:
        stop_reasons.append(f"Missing required materials: {missing_materials}")

    for block in materials:
        if not isinstance(block, dict):
            stop_reasons.append("Each material entry must be an object.")
            continue
        missing_material = sorted(REQUIRED_MATERIAL_FIELDS.difference(block))
        if missing_material:
            material_name = str(block.get("material", "unknown"))
            stop_reasons.append(f"{material_name}: missing material fields {missing_material}.")
        for field in REQUIRED_MATERIAL_FIELDS.difference({"peaks"}):
            if not str(block.get(field, "")).strip():
                material_name = str(block.get("material", "unknown"))
                stop_reasons.append(f"{material_name}: material field {field} is empty.")
        material = str(block.get("material", "unknown"))
        material_ref = str(block.get("reference_url_or_doi", manifest.get("reference_url_or_doi", ""))).strip()
        material_citation = str(block.get("reference_citation", manifest.get("reference_citation", ""))).strip()
        if not has_external_reference(material_ref):
            stop_reasons.append(f"{material}: missing external material-level reference URL/DOI.")
        if not material_citation:
            stop_reasons.append(f"{material}: missing material-level citation.")
        peaks = block.get("peaks", [])
        if not isinstance(peaks, list):
            stop_reasons.append(f"{material}: peaks must be a list.")
            peaks = []
        if len(peaks) < 5:
            stop_reasons.append(f"{material}: fewer than 5 peaks.")
        intensities = []
        for peak in peaks:
            if not isinstance(peak, dict):
                stop_reasons.append(f"{material}: each peak entry must be an object.")
                continue
            missing_peak = sorted(REQUIRED_PEAK_FIELDS.difference(peak))
            peak_id = str(peak.get("peak_id", "unknown_peak"))
            if missing_peak:
                stop_reasons.append(f"{material}/{peak_id}: missing peak fields {missing_peak}")
                continue
            peak_ref = str(peak.get("reference_url_or_doi", "")).strip()
            peak_citation = str(peak.get("reference_citation", "")).strip()
            if not has_external_reference(peak_ref):
                stop_reasons.append(f"{material}/{peak_id}: missing external peak-level reference URL/DOI.")
            if not peak_citation:
                stop_reasons.append(f"{material}/{peak_id}: missing peak-level citation.")
            two_theta = float(peak["two_theta_deg"])
            q_recorded = float(peak["q_a_inv"])
            d_recorded = float(peak["d_a"])
            intensity = float(peak["relative_intensity"])
            intensities.append(intensity)
            q_expected = q_from_two_theta(two_theta, wavelength_a)
            d_expected = d_from_q(q_expected)
            q_error = abs(q_recorded - q_expected)
            d_error = abs(d_recorded - d_expected)
            if q_error > 0.002:
                stop_reasons.append(f"{material}/{peak_id}: q mismatch {q_error:.6f}.")
            if d_error > 0.003:
                stop_reasons.append(f"{material}/{peak_id}: d-spacing mismatch {d_error:.6f}.")
            if not (0.0 < intensity <= 1.0):
                stop_reasons.append(f"{material}/{peak_id}: relative_intensity must be in (0, 1].")
            peak_rows.append(
                {
                    "material": material,
                    "peak_id": peak_id,
                    "two_theta_deg": two_theta,
                    "q_a_inv": q_recorded,
                    "q_expected_a_inv": q_expected,
                    "q_error_a_inv": q_error,
                    "d_a": d_recorded,
                    "d_expected_a": d_expected,
                    "d_error_a": d_error,
                    "relative_intensity": intensity,
                    "reference_url_or_doi": peak_ref,
                }
            )
        if intensities and abs(max(intensities) - 1.0) > 1e-9:
            stop_reasons.append(f"{material}: intensities must be normalized to max 1.0.")

    limitations = " ".join(str(item).lower() for item in as_list(manifest.get("known_limitations")))
    for required_phrase in ["manuscript", "development", "instrument"]:
        if required_phrase not in limitations:
            warnings.append(f"known_limitations does not mention {required_phrase}.")

    required_upgrade = set(str(item) for item in as_list(manifest.get("upgrade_required_before")))
    for gate_name in ["full_v8a_matrix", "shadow_validation", "final_validation", "manuscript_grade_powder_xrd_claim"]:
        if gate_name not in required_upgrade:
            stop_reasons.append(f"upgrade_required_before missing {gate_name}.")

    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/audit_v8a_peak_provenance.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "shadow_or_final_used": False,
        "claim_scope": "development-only peak provenance audit; not manuscript-grade powder XRD validation",
        "peak_table_id": manifest.get("peak_table_id"),
        "status": manifest.get("status"),
        "gate_passed": gate_passed,
        "decision": "proceed_to_v8a_event_feature_stress_gate" if gate_passed else "stop_peak_provenance_upgrade",
        "materials": sorted(materials_seen),
        "peak_count": len(peak_rows),
        "external_reference_count": len(reference_values),
        "max_q_error_a_inv": max((row["q_error_a_inv"] for row in peak_rows), default=0.0),
        "max_d_error_a": max((row["d_error_a"] for row in peak_rows), default=0.0),
        "warnings": warnings,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version()},
    }
    return gate, peak_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A peak provenance audit report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only peak provenance audit. Passing this gate does not create manuscript-grade powder-XRD evidence.",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Peak table: `{gate['peak_table_id']}`",
        f"- Status: `{gate['status']}`",
        f"- Peak count: `{gate['peak_count']}`",
        f"- External reference count: `{gate['external_reference_count']}`",
        f"- Max q error: `{gate['max_q_error_a_inv']:.6f}`",
        f"- Max d error: `{gate['max_d_error_a']:.6f}`",
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
    (output_dir / "v8a_peak_provenance_audit_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the v8A H/M peak provenance manifest.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default="source_models/config/diffraction_peak_tables/hm_powder_peaks_cif_or_literature_v8a_manifest.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_peak_provenance_audit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    manifest = load_json(project_root / args.manifest)
    gate, peak_rows = audit_manifest(manifest)
    (output_dir / "v8a_peak_provenance_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_csv(output_dir / "v8a_peak_provenance_peak_audit.csv", peak_rows)
    write_report(output_dir, gate)
    print(f"decision={gate['decision']} gate_passed={str(gate['gate_passed']).lower()} peak_count={gate['peak_count']}")


if __name__ == "__main__":
    main()
