from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LINEAGE_COLUMNS = [
    "sample_id",
    "split",
    "material",
    "source_id",
    "source_mode",
    "peak_table_id",
    "source_peak_table_id",
    "source_energy_kev",
    "source_wavelength_a",
    "bin_axis",
    "random_seed",
    "thickness_mm",
    "pose_index",
    "stress_label",
    "development_only",
    "shadow_or_final_used",
    "row_index",
    "clean_matrix_origin",
    "source_family",
    "seed_block",
    "seed_block_seed",
    "count_target_bin",
    "count_target_photons",
    "clean_context_cell_id",
    "nuisance_cell_id",
    "context_replicate_index",
    "context_material_count",
]


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def peak_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith("diffraction_peak_") and col.endswith("_norm"))


def build_count_invariant(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    peaks = peak_columns(frame)
    if not peaks:
        raise RuntimeError("No diffraction peak columns found.")
    result = frame[[col for col in LINEAGE_COLUMNS if col in frame.columns]].copy()
    peak_values = frame[peaks].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    peak_sum = peak_values.sum(axis=1).replace(0.0, np.nan)
    sqrt_sum = np.sqrt(peak_values).sum(axis=1).replace(0.0, np.nan)
    compositional_cols: list[str] = []
    sqrt_cols: list[str] = []
    for col in peaks:
        material_col = col.replace("diffraction_peak_", "diffraction_comp_")
        sqrt_col = col.replace("diffraction_peak_", "diffraction_sqrtcomp_")
        result[material_col] = (peak_values[col] / peak_sum).fillna(0.0)
        result[sqrt_col] = (np.sqrt(peak_values[col]) / sqrt_sum).fillna(0.0)
        compositional_cols.append(material_col)
        sqrt_cols.append(sqrt_col)
    if {"diffraction_window_hematite_unique_sum", "diffraction_window_magnetite_unique_sum"}.issubset(frame.columns):
        h = pd.to_numeric(frame["diffraction_window_hematite_unique_sum"], errors="coerce").fillna(0.0)
        m = pd.to_numeric(frame["diffraction_window_magnetite_unique_sum"], errors="coerce").fillna(0.0)
        denom = (h + m).replace(0.0, np.nan)
        result["diffraction_comp_hm_unique_balance"] = ((h - m) / denom).fillna(0.0)
        result["diffraction_comp_hm_hematite_share"] = (h / denom).fillna(0.0)
        result["diffraction_comp_hm_magnetite_share"] = (m / denom).fillna(0.0)
        compositional_cols.extend(
            [
                "diffraction_comp_hm_unique_balance",
                "diffraction_comp_hm_hematite_share",
                "diffraction_comp_hm_magnetite_share",
            ]
        )
    control_cols = [
        col
        for col in [
            "control_total_count_hit_count",
            "control_total_count_primary_hit_count",
            "control_total_count_norm",
            "control_high_angle_primary_norm",
            "control_direct_primary_norm",
            "control_scattered_primary_norm",
            "control_thickness_pose_thickness_mm",
            "control_thickness_pose_pose_index",
            "control_source_off_flag",
            "control_overlap_only_peak_norm",
        ]
        if col in frame.columns
    ]
    for col in control_cols:
        result[col] = frame[col]
    return result, compositional_cols, sqrt_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Build count-invariant compositional diffraction features from v8A multiclass context outputs.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_multiclass_context_v1_event_to_feature")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v2_count_invariant_event_to_feature")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    source_manifest = read_json(input_dir / "v8a_event_feature_manifest.json")
    features, comp_cols, sqrt_cols = build_count_invariant(frame)
    features.to_csv(output_dir / "v8a_event_sidecar_features.csv", index=False)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "generated_by": "analysis/build_v8a_multiclass_context_count_invariant_features.py",
        "generated_at_utc": generated_at,
        "source_input_dir": args.input_dir,
        "development_only": True,
        "shadow_or_final_used": bool(source_manifest.get("shadow_or_final_used", False)),
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "peak_table_id": source_manifest.get("peak_table_id"),
        "source_peak_table_ids": source_manifest.get("source_peak_table_ids"),
        "sample_count": int(len(features)),
        "feature_column_count": int(len(features.columns)),
        "main_feature_columns": comp_cols + sqrt_cols,
        "control_feature_columns": [col for col in features.columns if col.startswith("control_")],
        "claim_scope": "development-only count-invariant compositional feature view from completed v8A multiclass context Geant4 outputs",
    }
    gate = {
        "generated_by": "analysis/build_v8a_multiclass_context_count_invariant_features.py",
        "generated_at_utc": generated_at,
        "gate_passed": True,
        "decision": "count_invariant_feature_view_ready_for_development_diagnostics",
        "development_only": True,
        "shadow_or_final_used": bool(source_manifest.get("shadow_or_final_used", False)),
        "training_unlocked": False,
        "sample_count": int(len(features)),
        "main_feature_count": int(len(comp_cols) + len(sqrt_cols)),
    }
    write_json(output_dir / "v8a_event_feature_manifest.json", manifest)
    write_json(output_dir / "v8a_event_schema_gate.json", gate)
    print(
        f"decision={gate['decision']} samples={gate['sample_count']} main_feature_count={gate['main_feature_count']}"
    )


if __name__ == "__main__":
    main()
