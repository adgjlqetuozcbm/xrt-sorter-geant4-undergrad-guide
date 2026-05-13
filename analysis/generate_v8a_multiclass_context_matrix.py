from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from generate_v8a_clean_hm_development_matrix import (
    stable_seed,
    two_theta_from_q,
    unit_vector_from_angles,
    wavelength_from_energy,
    weighted_index_from_unit,
    write_config,
    write_phase_space,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def q_from_two_theta(two_theta_deg: float, wavelength_a: float) -> float:
    return 4.0 * math.pi * math.sin(math.radians(two_theta_deg) / 2.0) / wavelength_a


def load_material_peaks(manifest: dict[str, Any]) -> tuple[float, dict[str, list[tuple[float, float]]]]:
    wavelength_a = float(manifest["wavelength_a"])
    result: dict[str, list[tuple[float, float]]] = {}
    for block in manifest.get("materials", []):
        material = str(block.get("material", ""))
        peaks = []
        for peak in block.get("peaks", []):
            peaks.append((float(peak["two_theta_deg"]), float(peak["relative_intensity"])))
        if peaks:
            result[material] = peaks
    return wavelength_a, result


def build_context_phase_space_rows(
    *,
    peaks_by_material: dict[str, list[tuple[float, float]]],
    reference_wavelength_a: float,
    material: str,
    thickness_mm: float,
    pose_index: int,
    context_seed: int,
    source_energy_kev: float,
    photons: int,
) -> list[dict[str, float | int]]:
    """Generate a source-on/default phase space with nuisance randomness shared across all materials in a context cell."""
    source_wavelength_a = wavelength_from_energy(source_energy_kev)
    rng = random.Random(
        stable_seed(
            int(thickness_mm * 100),
            int(pose_index),
            int(context_seed),
            int(photons),
            20260510,
        )
    )
    continuum_fraction = 0.30
    continuum_count = int(round(photons * continuum_fraction))
    peak_count = max(0, photons - continuum_count)
    pose_phi_offset = (pose_index % 8) * (math.pi / 8.0)
    rows: list[dict[str, float | int]] = []

    def add_row(event_id: int, energy_kev: float, x_mm: float, y_mm: float, z_mm: float, direction: tuple[float, float, float]) -> None:
        rows.append(
            {
                "event_id": event_id,
                "energy_keV": energy_kev,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": z_mm,
                "dir_x": direction[0],
                "dir_y": direction[1],
                "dir_z": direction[2],
            }
        )

    for event_id in range(continuum_count):
        phi = rng.uniform(0.0, 2.0 * math.pi) + pose_phi_offset
        theta = max(0.0, rng.gauss(0.0, 0.35))
        add_row(
            event_id,
            source_energy_kev,
            thickness_mm / 2.0 + 0.05,
            rng.uniform(-4.0, 4.0),
            rng.uniform(-4.0, 4.0),
            unit_vector_from_angles(theta, phi),
        )

    peaks = peaks_by_material[material]
    weights = [max(float(weight), 0.0) for _, weight in peaks]
    peak_units = [rng.random() for _ in range(peak_count)]
    for offset, unit_value in enumerate(peak_units, start=continuum_count):
        peak_index = weighted_index_from_unit(float(unit_value), weights)
        two_theta_reference, _ = peaks[peak_index]
        q_a_inv = q_from_two_theta(two_theta_reference, reference_wavelength_a)
        two_theta_source = two_theta_from_q(q_a_inv, source_wavelength_a)
        if two_theta_source is None:
            two_theta_source = two_theta_reference
        theta = max(0.0, rng.gauss(two_theta_source, 0.18))
        phi = rng.uniform(0.0, 2.0 * math.pi) + pose_phi_offset
        add_row(
            offset,
            source_energy_kev,
            rng.uniform(-thickness_mm / 2.0, thickness_mm / 2.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
            unit_vector_from_angles(theta, phi),
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the v8A ten-material context matrix without running Geant4.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_multiclass_context_v1_config.json")
    parser.add_argument("--readiness-gate", default="results/accuracy_v3/v8a_multiclass_context_v1_readiness/v8a_multiclass_context_readiness_gate.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(as_project_path(project_root, args.config))
    readiness_gate = load_json(as_project_path(project_root, args.readiness_gate))
    if not bool(readiness_gate.get("gate_passed", False)) or not bool(readiness_gate.get("matrix_generation_unlocked", False)):
        raise RuntimeError("Multiclass context readiness gate has not unlocked matrix generation.")

    peak_manifest = load_json(as_project_path(project_root, config["required_peak_manifest"]))
    peak_table_id = str(peak_manifest.get("peak_table_id", ""))
    required_peak_table_id = str(config["required_peak_table_id"])
    if peak_table_id != required_peak_table_id:
        raise RuntimeError(f"Peak manifest mismatch: {peak_table_id} != {required_peak_table_id}")
    if config.get("status") != "development_readiness_review":
        raise RuntimeError("Multiclass context config must remain development_readiness_review for this phase.")
    if config.get("source_modes") != [{"source_mode": "on", "stress_label": "default"}]:
        raise RuntimeError("Multiclass context matrix must be source-on/default only.")

    profile = str(config["profile"])
    matrix_root = project_root / "source_models" / "config" / "material_sorting_matrix"
    profile_dir = matrix_root / profile
    if profile_dir.exists() and any(profile_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Profile directory is not empty: {profile_dir}. Use --overwrite to replace development artifacts.")
    if profile_dir.exists() and any(profile_dir.iterdir()) and args.overwrite:
        resolved_profile = profile_dir.resolve()
        resolved_matrix_root = matrix_root.resolve()
        if not resolved_profile.is_relative_to(resolved_matrix_root):
            raise RuntimeError(f"Refusing to clean unexpected profile path: {resolved_profile}")
        shutil.rmtree(resolved_profile)
    profile_dir.mkdir(parents=True, exist_ok=True)

    reference_wavelength_a, peaks_by_material = load_material_peaks(peak_manifest)
    materials = [str(item) for item in config["target_materials"]]
    missing_peak_materials = sorted(set(materials) - set(peaks_by_material))
    if missing_peak_materials:
        raise RuntimeError(f"Peak manifest is missing materials: {missing_peak_materials}")

    rows: list[dict[str, Any]] = []
    row_index = 0
    context_replicates_per_cell = int(config.get("context_replicates_per_cell", 1))
    if context_replicates_per_cell < 1:
        raise RuntimeError("context_replicates_per_cell must be >= 1.")

    count_bins = config["planned_count_target_bins"]
    for split, split_config in config["splits"].items():
        for seed_block_config in split_config["seed_blocks"]:
            seed_block = str(seed_block_config["seed_block"])
            seed_block_seed = int(seed_block_config["seed"])
            for thickness in config["planned_thickness_mm"]:
                for pose_index in config["planned_pose_indices"]:
                    for count_index, count_config in enumerate(count_bins):
                        count_target_bin = str(count_config["count_target_bin"])
                        photons = int(count_config["photons_per_row"])
                        for context_replicate_index in range(context_replicates_per_cell):
                            context_seed = stable_seed(
                                seed_block_seed,
                                int(float(thickness) * 100),
                                int(pose_index),
                                count_index,
                                context_replicate_index,
                                83,
                            )
                            replicate_tag = f"r{context_replicate_index + 1:02d}"
                            context_cell_id = (
                                f"{profile}_{split}_{seed_block}_t{float(thickness):g}_p{int(pose_index)}_"
                                f"c{count_target_bin}_{replicate_tag}"
                            )
                            nuisance_cell_id = (
                                f"{profile}|{split}|{config['clean_matrix_origin']}|{config['source_family']}|"
                                f"t{float(thickness):g}|p{int(pose_index)}|c{count_target_bin}|{seed_block}|{replicate_tag}"
                            )
                            for material in materials:
                                source_energy = float(config["planned_source_energy_kev"])
                                output_prefix = (
                                    f"{profile}_{split}_{seed_block}_t{float(thickness):g}mm_pose{int(pose_index)}_"
                                    f"count{count_target_bin}_{replicate_tag}_{material}"
                                )
                                phase_rel = Path("phase_space") / f"{output_prefix}.csv"
                                phase_path = profile_dir / phase_rel
                                phase_rows = build_context_phase_space_rows(
                                    peaks_by_material=peaks_by_material,
                                    reference_wavelength_a=reference_wavelength_a,
                                    material=material,
                                    thickness_mm=float(thickness),
                                    pose_index=int(pose_index),
                                    context_seed=int(context_seed),
                                    source_energy_kev=source_energy,
                                    photons=photons,
                                )
                                write_phase_space(phase_path, phase_rows)
                                config_rel = Path("source_models") / "config" / "material_sorting_matrix" / profile / f"{output_prefix}.txt"
                                write_config(
                                    project_root / config_rel,
                                    {
                                        "run_id": output_prefix,
                                        "experiment_label": profile,
                                        "output_prefix": output_prefix,
                                        "output_dir": f"material_sorting_runs/{profile}",
                                        "benchmark_suite": "accuracy_v3",
                                        "research_route": "v8a_multiclass_context",
                                        "prediction_stage": "ten_material_diffraction_sidecar_context_sampling",
                                        "run_role": "material",
                                        "source_variant": config["source_id"],
                                        "sample_photons": photons,
                                        "random_seed": int(context_seed),
                                        "source_mode": "phase_space",
                                        "phase_space_file": phase_rel.as_posix(),
                                        "source_x_cm": -30.0,
                                        "source_y_mm": 0.0,
                                        "source_z_mm": 0.0,
                                        "dir_x": 1.0,
                                        "dir_y": 0.0,
                                        "dir_z": 0.0,
                                        "ore_material_mode": "single",
                                        "ore_primary_material": material,
                                        "ore_shape": "slab",
                                        "ore_thickness_mm": float(thickness),
                                        "pose_index": int(pose_index),
                                        "ore_half_y_mm": 10.0,
                                        "ore_half_z_mm": 10.0,
                                        "detector_layout": "transmission_plus_side_scatter",
                                        "detector_x_cm": 25.0,
                                        "detector_half_y_mm": 120.0,
                                        "detector_half_z_mm": 120.0,
                                        "side_detector_y_cm": 12.0,
                                        "side_detector_half_x_mm": 140.0,
                                        "side_detector_half_z_mm": 120.0,
                                        "clean_matrix_origin": config["clean_matrix_origin"],
                                        "source_family": config["source_family"],
                                        "seed_block": seed_block,
                                        "seed_block_seed": seed_block_seed,
                                        "count_target_bin": count_target_bin,
                                        "count_target_photons": photons,
                                        "clean_context_cell_id": context_cell_id,
                                        "nuisance_cell_id": nuisance_cell_id,
                                        "context_replicate_index": context_replicate_index + 1,
                                        "context_material_count": len(materials),
                                        "stress_label": "default",
                                    },
                                )
                                rows.append(
                                    {
                                        "row_index": row_index,
                                        "profile": profile,
                                        "split": split,
                                        "run_role": "material",
                                        "material": material,
                                        "source_id": config["source_id"],
                                        "source_family": config["source_family"],
                                        "source_mode": "on",
                                        "stress_label": "default",
                                        "clean_matrix_origin": config["clean_matrix_origin"],
                                        "source_energy_kev": source_energy,
                                        "thickness_mm": float(thickness),
                                        "pose_index": int(pose_index),
                                        "count_target_bin": count_target_bin,
                                        "count_target_photons": photons,
                                        "seed_block": seed_block,
                                        "seed_block_seed": seed_block_seed,
                                        "random_seed": int(context_seed),
                                        "clean_context_cell_id": context_cell_id,
                                        "nuisance_cell_id": nuisance_cell_id,
                                        "context_replicate_index": context_replicate_index + 1,
                                        "context_material_count": len(materials),
                                        "phase_space_file": phase_rel.as_posix(),
                                        "config_path": config_rel.as_posix(),
                                        "output_prefix": output_prefix,
                                        "peak_table_id": peak_table_id,
                                        "development_only": True,
                                        "shadow_or_final_used": False,
                                    }
                                )
                                row_index += 1

    expected_total = int(config["expected_rows"]["total"])
    if len(rows) != expected_total:
        raise RuntimeError(f"Generated row count mismatch: {len(rows)} != expected {expected_total}")

    matrix_path = profile_dir / "material_sorting_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "generated_by": "analysis/generate_v8a_multiclass_context_matrix.py",
        "config": args.config,
        "readiness_gate": args.readiness_gate,
        "profile": profile,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "full_ten_material_context_matrix": True,
        "runs_geant4": False,
        "training_unlocked": False,
        "clean_matrix_origin": config["clean_matrix_origin"],
        "source_family": config["source_family"],
        "source_modes": ["on"],
        "stress_labels": ["default"],
        "peak_table_id": peak_table_id,
        "materials": materials,
        "rows": len(rows),
        "expected_rows": config["expected_rows"],
        "expected_context_cells": config["expected_context_cells"],
        "expected_hm_pairs": config["expected_hm_pairs"],
        "context_replicates_per_cell": context_replicates_per_cell,
        "source_energy_kev": float(config["planned_source_energy_kev"]),
        "count_target_bins": count_bins,
        "context_training_intent": config["context_training_intent"],
    }
    (profile_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote profile={profile} rows={len(rows)} matrix={matrix_path} runs_geant4=false training_unlocked=false")


if __name__ == "__main__":
    main()
