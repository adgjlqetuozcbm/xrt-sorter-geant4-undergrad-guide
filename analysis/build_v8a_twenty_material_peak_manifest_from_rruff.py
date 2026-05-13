from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import build_v8a_ten_material_peak_manifest_from_rruff as rruff


DEFAULT_RRUFF_DIF_ZIP = Path("/tmp/rruff_xrd/RRUFF_powder.zip")
DEFAULT_OUTPUT_MANIFEST = Path(
    "source_models/config/diffraction_peak_tables/twenty_material_powder_peaks_rruff_v8a_manifest.json"
)
PEAK_TABLE_ID = "twenty_material_powder_peaks_rruff_v8a"

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
    "Aragonite",
    "Magnesite",
    "Siderite",
    "Goethite",
    "Ilmenite",
    "Rutile",
    "Sphalerite",
    "Fluorite",
    "Gypsum",
    "Baryte",
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
    "Aragonite": {
        "phase_name": "aragonite",
        "chemical_formula": "CaCO3",
        "structure_note": "aragonite carbonate polymorph included as a calcite hard-negative",
    },
    "Magnesite": {
        "phase_name": "magnesite",
        "chemical_formula": "MgCO3",
        "structure_note": "magnesium carbonate reference phase included for carbonate scalability",
    },
    "Siderite": {
        "phase_name": "siderite",
        "chemical_formula": "FeCO3",
        "structure_note": "iron carbonate reference phase included for carbonate/iron-bearing hard negatives",
    },
    "Goethite": {
        "phase_name": "goethite",
        "chemical_formula": "FeO(OH)",
        "structure_note": "iron oxyhydroxide reference phase included as an iron-oxide family hard-negative",
    },
    "Ilmenite": {
        "phase_name": "ilmenite",
        "chemical_formula": "FeTiO3",
        "structure_note": "iron titanium oxide reference phase included as an oxide hard-negative",
    },
    "Rutile": {
        "phase_name": "rutile",
        "chemical_formula": "TiO2",
        "structure_note": "titanium oxide reference phase included for oxide-family scalability",
    },
    "Sphalerite": {
        "phase_name": "sphalerite",
        "chemical_formula": "ZnS",
        "structure_note": "zinc sulfide reference phase included for sulfide-family scalability",
    },
    "Fluorite": {
        "phase_name": "fluorite",
        "chemical_formula": "CaF2",
        "structure_note": "calcium fluoride reference phase included as a halide control mineral",
    },
    "Gypsum": {
        "phase_name": "gypsum",
        "chemical_formula": "CaSO4·2H2O",
        "structure_note": "hydrated calcium sulfate reference phase included for sulfate/hydrate contrast",
    },
    "Baryte": {
        "phase_name": "baryte",
        "chemical_formula": "BaSO4",
        "structure_note": "barium sulfate reference phase included as a high-density sulfate mineral",
    },
}


def build_manifest(rruff_dif_zip: Path) -> dict[str, object]:
    rruff.MATERIAL_METADATA = MATERIAL_METADATA
    with ZipFile(rruff_dif_zip) as zip_file:
        records = [rruff.select_record(zip_file, material) for material in TARGET_MATERIALS]
    wavelengths = sorted({round(record.wavelength_a, 6) for record in records})
    if len(wavelengths) != 1:
        raise RuntimeError(f"Selected RRUFF DIF records use inconsistent wavelengths: {wavelengths}")
    wavelength_a = float(wavelengths[0])
    return {
        "peak_table_id": PEAK_TABLE_ID,
        "version": "2026-05-11",
        "status": "development_reference_candidate",
        "reference_type": "rruff_powder_dif_computed_reference_pattern_candidate",
        "reference_citation": (
            "RRUFF Project powder diffraction DIF files computed with XPOW from structure/cell data; "
            "development-only candidate peak table for v8A twenty-material scalability scout."
        ),
        "reference_url_or_doi": [
            "https://rruff.info/zipped_data_files/powder/DIF.zip",
            "https://rruff.info/",
        ],
        "wavelength_a": wavelength_a,
        "intensity_normalization": (
            "For each material, RRUFF DIF rows with identical 0.01 degree two-theta are collapsed by summed intensity; "
            f"the strongest {rruff.MAX_PEAKS_PER_MATERIAL} collapsed peaks are retained and normalized to max 1.0."
        ),
        "source_selection_policy": (
            "For each target material, choose the RRUFF powder DIF archive member with the largest parsed peak count "
            "and at least five peaks; ties are resolved by archive member name."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "analysis/build_v8a_twenty_material_peak_manifest_from_rruff.py",
        "known_limitations": [
            "This is a development reference candidate built from RRUFF computed powder DIF files, not a project-specific measured reference scan.",
            "Relative intensities are normalized development features and may vary with instrument response, texture, grain size, impurities, preferred orientation, and sample preparation.",
            f"Peak lists are truncated to the strongest {rruff.MAX_PEAKS_PER_MATERIAL} collapsed peaks per material for the v8A scalability scout.",
            "The manifest does not encode mixture, fluorescence, detector-response, throughput, or industrial sorter calibration provenance.",
            "It must not be used as manuscript-grade powder-XRD validation without independent CIF/Rietveld or measured-reference review.",
        ],
        "upgrade_required_before": [
            "shadow_validation",
            "final_validation",
            "manuscript_grade_powder_xrd_claim",
        ],
        "materials": [rruff.build_material_block(record, wavelength_a) for record in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a development-only v8A twenty-material peak manifest from RRUFF DIF data.")
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
