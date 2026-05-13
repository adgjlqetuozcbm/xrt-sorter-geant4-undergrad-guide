from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile


DEFAULT_RRUFF_DIF_ZIP = Path("/tmp/rruff_xrd/RRUFF_powder.zip")
DEFAULT_OUTPUT_MANIFEST = Path(
    "source_models/config/diffraction_peak_tables/ten_material_powder_peaks_cif_or_literature_v8a_manifest.json"
)
PEAK_TABLE_ID = "ten_material_powder_peaks_cif_or_literature_v8a"
MAX_PEAKS_PER_MATERIAL = 12
MIN_PEAKS_PER_MATERIAL = 5

TARGET_MATERIALS = [
    "Quartz",
    "Calcite",
    "Orthoclase",
    "Albite",
    "Dolomite",
    "Pyrite",
    "Hematite",
    "Magnetite",
    "Chalcopyrite",
    "Galena",
]

MATERIAL_METADATA = {
    "Quartz": {
        "phase_name": "quartz",
        "chemical_formula": "SiO2",
        "structure_note": "alpha-quartz silicate reference phase from RRUFF powder DIF data",
    },
    "Calcite": {
        "phase_name": "calcite",
        "chemical_formula": "CaCO3",
        "structure_note": "calcite carbonate reference phase from RRUFF powder DIF data",
    },
    "Orthoclase": {
        "phase_name": "orthoclase",
        "chemical_formula": "KAlSi3O8",
        "structure_note": "orthoclase potassium feldspar reference phase from RRUFF powder DIF data",
    },
    "Albite": {
        "phase_name": "albite",
        "chemical_formula": "NaAlSi3O8",
        "structure_note": "albite sodium feldspar reference phase from RRUFF powder DIF data",
    },
    "Dolomite": {
        "phase_name": "dolomite",
        "chemical_formula": "CaMg(CO3)2",
        "structure_note": "dolomite carbonate reference phase from RRUFF powder DIF data",
    },
    "Pyrite": {
        "phase_name": "pyrite",
        "chemical_formula": "FeS2",
        "structure_note": "pyrite sulfide reference phase from RRUFF powder DIF data",
    },
    "Hematite": {
        "phase_name": "hematite",
        "chemical_formula": "Fe2O3",
        "structure_note": "alpha-Fe2O3 iron oxide reference phase from RRUFF powder DIF data",
    },
    "Magnetite": {
        "phase_name": "magnetite",
        "chemical_formula": "Fe3O4",
        "structure_note": "Fe3O4 inverse-spinel iron oxide reference phase from RRUFF powder DIF data",
    },
    "Chalcopyrite": {
        "phase_name": "chalcopyrite",
        "chemical_formula": "CuFeS2",
        "structure_note": "chalcopyrite copper iron sulfide reference phase from RRUFF powder DIF data",
    },
    "Galena": {
        "phase_name": "galena",
        "chemical_formula": "PbS",
        "structure_note": "galena lead sulfide reference phase from RRUFF powder DIF data",
    },
}

PEAK_ROW_RE = re.compile(
    r"^\s*"
    r"(?P<two_theta>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<intensity>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<d_spacing>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<h>-?\d+)\s+"
    r"(?P<k>-?\d+)\s+"
    r"(?P<l>-?\d+)\s*$"
)
WAVELENGTH_RE = re.compile(r"X-RAY WAVELENGTH:\s*(?P<wavelength>[0-9.]+)", re.IGNORECASE)
CELL_RE = re.compile(r"CELL PARAMETERS:", re.IGNORECASE)


@dataclass(frozen=True)
class RawPeak:
    two_theta_deg: float
    intensity: float
    d_a: float
    hkl: str


@dataclass(frozen=True)
class DifRecord:
    material: str
    archive_name: str
    sample_id: str
    wavelength_a: float
    citation: str
    raw_peaks: list[RawPeak]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def q_from_two_theta(two_theta_deg: float, wavelength_a: float) -> float:
    return 4.0 * math.pi * math.sin(math.radians(two_theta_deg) / 2.0) / wavelength_a


def d_from_q(q_a_inv: float) -> float:
    return 2.0 * math.pi / q_a_inv


def sample_id_from_archive_name(name: str) -> str:
    parts = name.split("__")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse RRUFF sample id from archive member: {name}")
    return parts[1]


def sample_url(sample_id: str) -> str:
    return f"https://rruff.info/{sample_id}"


def extract_citation(material: str, sample_id: str, text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    citation_lines: list[str] = []
    for line in lines[1:]:
        if CELL_RE.search(line):
            break
        if line and not line.startswith("Diffraction data computed"):
            citation_lines.append(line)
    citation = "; ".join(citation_lines[:8])
    if citation:
        return f"RRUFF {material} sample {sample_id} powder DIF computed pattern; source paper metadata: {citation}."
    return f"RRUFF {material} sample {sample_id} powder DIF computed pattern."


def parse_dif_record(material: str, archive_name: str, text: str) -> DifRecord:
    sample_id = sample_id_from_archive_name(archive_name)
    wavelength_match = WAVELENGTH_RE.search(text)
    if wavelength_match is None:
        raise ValueError(f"{archive_name}: no X-ray wavelength found")
    peaks: list[RawPeak] = []
    for line in text.splitlines():
        match = PEAK_ROW_RE.match(line)
        if match is None:
            continue
        hkl = " ".join(match.group(name) for name in ("h", "k", "l"))
        peaks.append(
            RawPeak(
                two_theta_deg=float(match.group("two_theta")),
                intensity=float(match.group("intensity")),
                d_a=float(match.group("d_spacing")),
                hkl=hkl,
            )
        )
    return DifRecord(
        material=material,
        archive_name=archive_name,
        sample_id=sample_id,
        wavelength_a=float(wavelength_match.group("wavelength")),
        citation=extract_citation(material, sample_id, text),
        raw_peaks=peaks,
    )


def select_record(zip_file: ZipFile, material: str) -> DifRecord:
    candidates = [
        name
        for name in zip_file.namelist()
        if name.startswith(f"{material}__") and "__Powder__DIF_File__" in name and name.endswith(".txt")
    ]
    if not candidates:
        raise RuntimeError(f"No RRUFF powder DIF file found for {material}")

    parsed: list[DifRecord] = []
    for name in candidates:
        text = zip_file.read(name).decode("utf-8", errors="replace")
        try:
            parsed.append(parse_dif_record(material, name, text))
        except ValueError:
            continue
    parsed = [record for record in parsed if len(record.raw_peaks) >= MIN_PEAKS_PER_MATERIAL]
    if not parsed:
        raise RuntimeError(f"No usable RRUFF powder DIF file with at least {MIN_PEAKS_PER_MATERIAL} peaks for {material}")
    return sorted(parsed, key=lambda item: (-len(item.raw_peaks), item.archive_name))[0]


def grouped_top_peaks(raw_peaks: list[RawPeak], wavelength_a: float) -> list[dict[str, Any]]:
    groups: dict[float, dict[str, Any]] = {}
    for peak in raw_peaks:
        key = round(peak.two_theta_deg, 2)
        group = groups.setdefault(
            key,
            {"intensity_sum": 0.0, "weighted_two_theta": 0.0, "weighted_d": 0.0, "hkls": []},
        )
        group["intensity_sum"] += peak.intensity
        group["weighted_two_theta"] += peak.two_theta_deg * peak.intensity
        group["weighted_d"] += peak.d_a * peak.intensity
        if peak.hkl not in group["hkls"]:
            group["hkls"].append(peak.hkl)

    collapsed = []
    for group in groups.values():
        intensity = float(group["intensity_sum"])
        two_theta = float(group["weighted_two_theta"]) / intensity
        d_a = float(group["weighted_d"]) / intensity
        collapsed.append(
            {
                "two_theta_deg": two_theta,
                "raw_intensity": intensity,
                "rruff_d_a": d_a,
                "hkl": ";".join(group["hkls"]),
            }
        )
    selected = sorted(collapsed, key=lambda item: (-item["raw_intensity"], item["two_theta_deg"]))[:MAX_PEAKS_PER_MATERIAL]
    max_intensity = max(float(item["raw_intensity"]) for item in selected)
    normalized = []
    for item in sorted(selected, key=lambda item: item["two_theta_deg"]):
        q_a_inv = q_from_two_theta(float(item["two_theta_deg"]), wavelength_a)
        normalized.append(
            {
                "hkl": item["hkl"],
                "two_theta_deg": round(float(item["two_theta_deg"]), 4),
                "q_a_inv": round(q_a_inv, 6),
                "d_a": round(d_from_q(q_a_inv), 6),
                "relative_intensity": round(float(item["raw_intensity"]) / max_intensity, 6),
            }
        )
    return normalized


def build_material_block(record: DifRecord, wavelength_a: float) -> dict[str, Any]:
    metadata = MATERIAL_METADATA[record.material]
    reference_url = sample_url(record.sample_id)
    peak_citation = f"{record.citation} Peaks parsed from RRUFF powder DIF file {record.archive_name}."
    peaks = []
    for index, peak in enumerate(grouped_top_peaks(record.raw_peaks, wavelength_a), start=1):
        two_theta_tag = str(peak["two_theta_deg"]).replace(".", "p")
        peaks.append(
            {
                "peak_id": f"{slug(record.material)}_{two_theta_tag}_{index:02d}",
                "hkl": peak["hkl"],
                "two_theta_deg": peak["two_theta_deg"],
                "q_a_inv": peak["q_a_inv"],
                "d_a": peak["d_a"],
                "relative_intensity": peak["relative_intensity"],
                "reference_citation": peak_citation,
                "reference_url_or_doi": reference_url,
            }
        )
    return {
        "material": record.material,
        "phase_name": metadata["phase_name"],
        "chemical_formula": metadata["chemical_formula"],
        "structure_note": metadata["structure_note"],
        "reference_citation": record.citation,
        "reference_url_or_doi": reference_url,
        "source_archive_member": record.archive_name,
        "source_sample_id": record.sample_id,
        "raw_peak_count": len(record.raw_peaks),
        "selected_peak_count": len(peaks),
        "peaks": peaks,
    }


def build_manifest(rruff_dif_zip: Path) -> dict[str, Any]:
    with ZipFile(rruff_dif_zip) as zip_file:
        records = [select_record(zip_file, material) for material in TARGET_MATERIALS]
    wavelengths = sorted({round(record.wavelength_a, 6) for record in records})
    if len(wavelengths) != 1:
        raise RuntimeError(f"Selected RRUFF DIF records use inconsistent wavelengths: {wavelengths}")
    wavelength_a = float(wavelengths[0])
    return {
        "peak_table_id": PEAK_TABLE_ID,
        "version": "2026-05-10",
        "status": "development_reference_candidate",
        "reference_type": "rruff_powder_dif_computed_reference_pattern_candidate",
        "reference_citation": (
            "RRUFF Project powder diffraction DIF files computed with XPOW from structure/cell data; "
            "development-only candidate peak table for v8A ten-material context experiments."
        ),
        "reference_url_or_doi": [
            "https://rruff.info/zipped_data_files/powder/DIF.zip",
            "https://rruff.info/",
        ],
        "wavelength_a": wavelength_a,
        "intensity_normalization": (
            "For each material, RRUFF DIF rows with identical 0.01 degree two-theta are collapsed by summed intensity; "
            f"the strongest {MAX_PEAKS_PER_MATERIAL} collapsed peaks are retained and normalized to max 1.0."
        ),
        "source_selection_policy": (
            "For each target material, choose the RRUFF powder DIF archive member with the largest parsed peak count "
            "and at least five peaks; ties are resolved by archive member name."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "analysis/build_v8a_ten_material_peak_manifest_from_rruff.py",
        "known_limitations": [
            "This is a development reference candidate built from RRUFF computed powder DIF files, not a project-specific measured reference scan.",
            "Relative intensities are normalized development features and may vary with instrument response, texture, grain size, impurities, preferred orientation, and sample preparation.",
            f"Peak lists are truncated to the strongest {MAX_PEAKS_PER_MATERIAL} collapsed peaks per material for the v8A context experiment.",
            "The manifest does not encode mixture, fluorescence, detector-response, throughput, or industrial sorter calibration provenance.",
            "It must not be used as manuscript-grade powder-XRD validation without independent CIF/Rietveld or measured-reference review.",
        ],
        "upgrade_required_before": [
            "full_v8a_matrix",
            "shadow_validation",
            "final_validation",
            "manuscript_grade_powder_xrd_claim",
        ],
        "materials": [build_material_block(record, wavelength_a) for record in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a development-only v8A ten-material peak manifest from RRUFF DIF data.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rruff-dif-zip", default=DEFAULT_RRUFF_DIF_ZIP)
    parser.add_argument("--output-manifest", default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    rruff_dif_zip = Path(args.rruff_dif_zip).resolve()
    output_manifest = Path(args.output_manifest)
    if not output_manifest.is_absolute():
        output_manifest = project_root / output_manifest
    if not rruff_dif_zip.exists():
        raise FileNotFoundError(f"RRUFF DIF zip not found: {rruff_dif_zip}")
    if output_manifest.exists() and not args.overwrite:
        raise SystemExit(f"Output manifest already exists: {output_manifest}. Use --overwrite.")

    manifest = build_manifest(rruff_dif_zip)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    material_summary = ", ".join(
        f"{block['material']}:{block['selected_peak_count']}/{block['raw_peak_count']}"
        for block in manifest["materials"]
    )
    print(f"wrote={output_manifest} peak_table_id={manifest['peak_table_id']} materials={len(manifest['materials'])}")
    print(f"selected_peaks={material_summary}")


if __name__ == "__main__":
    main()
