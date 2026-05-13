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
from generate_v8a_multiclass_context_matrix import load_material_peaks, q_from_two_theta


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower()).strip("_")


def split_seed_blocks(config: dict[str, Any], split: str) -> list[tuple[str, int]]:
    plan = config["split_design"][split]
    prefix = str(plan["seed_block_prefix"])
    seed_start = int(plan["seed_start"])
    count = int(plan["seed_block_count"])
    return [(f"{prefix}_{index + 1:03d}", seed_start + index) for index in range(count)]


def profile_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["physical_perturbation_profile"]): item for item in config["perturbation_profiles"]}


def build_perturbed_phase_space_rows(
    *,
    peaks_by_material: dict[str, list[tuple[float, float]]],
    reference_wavelength_a: float,
    material: str,
    thickness_mm: float,
    pose_index: int,
    context_seed: int,
    source_energy_kev: float,
    photons: int,
    perturbation: dict[str, Any],
) -> list[dict[str, float | int]]:
    source_energy = source_energy_kev * float(perturbation.get("source_energy_scale", 1.0))
    source_wavelength_a = wavelength_from_energy(source_energy)
    rng = random.Random(
        stable_seed(
            int(thickness_mm * 100),
            int(pose_index),
            int(context_seed),
            int(photons),
            int(abs(float(perturbation.get("peak_q_shift_fraction", 0.0))) * 1_000_000),
            int(float(perturbation.get("peak_theta_sigma_deg", 0.18)) * 10_000),
            20260510,
        )
    )
    continuum_fraction = min(max(float(perturbation.get("continuum_fraction", 0.30)), 0.0), 0.90)
    continuum_count = int(round(photons * continuum_fraction))
    peak_count = max(0, photons - continuum_count)
    pose_phi_offset = (pose_index % 8) * (math.pi / 8.0)
    q_shift = float(perturbation.get("peak_q_shift_fraction", 0.0))
    theta_sigma = max(float(perturbation.get("peak_theta_sigma_deg", 0.18)), 0.01)
    jitter_sigma = max(float(perturbation.get("peak_intensity_jitter_sigma", 0.0)), 0.0)
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
        continuum_sigma = 0.35 + 0.5 * max(theta_sigma - 0.18, 0.0)
        theta = max(0.0, rng.gauss(0.0, continuum_sigma))
        add_row(
            event_id,
            source_energy,
            thickness_mm / 2.0 + 0.05,
            rng.uniform(-4.0, 4.0),
            rng.uniform(-4.0, 4.0),
            unit_vector_from_angles(theta, phi),
        )

    peaks = peaks_by_material[material]
    weights = []
    for _, weight in peaks:
        base = max(float(weight), 0.0)
        if jitter_sigma > 0.0:
            base *= rng.lognormvariate(0.0, jitter_sigma)
        weights.append(base)
    peak_units = [rng.random() for _ in range(peak_count)]
    for offset, unit_value in enumerate(peak_units, start=continuum_count):
        peak_index = weighted_index_from_unit(float(unit_value), weights)
        two_theta_reference, _ = peaks[peak_index]
        q_a_inv = q_from_two_theta(two_theta_reference, reference_wavelength_a) * (1.0 + q_shift)
        two_theta_source = two_theta_from_q(q_a_inv, source_wavelength_a)
        if two_theta_source is None:
            two_theta_source = two_theta_reference
        theta = max(0.0, rng.gauss(two_theta_source, theta_sigma))
        phi = rng.uniform(0.0, 2.0 * math.pi) + pose_phi_offset
        add_row(
            offset,
            source_energy,
            rng.uniform(-thickness_mm / 2.0, thickness_mm / 2.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
            unit_vector_from_angles(theta, phi),
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v8A v5 full-10 physical perturbation scout matrix.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_multiclass_context_v5_physical_perturbation_scout_config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = as_project_path(project_root, args.config)
    config = load_json(config_path)
    if config.get("status") != "development_physical_perturbation_scout_preregistration":
        raise RuntimeError("v5 scout config must remain development_physical_perturbation_scout_preregistration.")

    peak_manifest = load_json(as_project_path(project_root, config["peak_manifest"]))
    peak_table_id = str(peak_manifest.get("peak_table_id", ""))
    if peak_table_id != str(config["required_peak_table_id"]):
        raise RuntimeError(f"Peak manifest mismatch: {peak_table_id} != {config['required_peak_table_id']}")
    reference_wavelength_a, peaks_by_material = load_material_peaks(peak_manifest)
    materials = [str(item) for item in config["target_materials"]]
    missing = sorted(set(materials) - set(peaks_by_material))
    if missing:
        raise RuntimeError(f"Peak manifest missing materials: {missing}")

    profiles = profile_map(config)
    profile = str(config["profile"])
    matrix_root = project_root / "source_models" / "config" / "material_sorting_matrix"
    profile_dir = matrix_root / profile
    if profile_dir.exists() and any(profile_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Profile directory is not empty: {profile_dir}. Use --overwrite.")
    if profile_dir.exists() and any(profile_dir.iterdir()) and args.overwrite:
        resolved_profile = profile_dir.resolve()
        resolved_root = matrix_root.resolve()
        if not resolved_profile.is_relative_to(resolved_root):
            raise RuntimeError(f"Refusing to clean unexpected profile path: {resolved_profile}")
        shutil.rmtree(resolved_profile)
    profile_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    row_index = 0
    source_energy = float(config["source_energy_kev"])
    energy_tag = f"{source_energy:g}kev".replace(".", "p")
    for split in ["train", "validation", "stress_holdout"]:
        split_profiles = [str(item) for item in config["split_design"][split]["profiles"]]
        for seed_block, seed_block_seed in split_seed_blocks(config, split):
            for thickness in config["thickness_mm"]:
                thickness = float(thickness)
                for pose_index in config["pose_indices"]:
                    pose_index = int(pose_index)
                    for count_index, count_config in enumerate(config["count_target_bins"]):
                        count_target_bin = str(count_config["count_target_bin"])
                        photons = int(count_config["photons_per_row"])
                        for perturbation_profile in split_profiles:
                            perturbation = profiles[perturbation_profile]
                            perturb_tag = safe_tag(perturbation_profile)
                            context_seed = stable_seed(
                                seed_block_seed,
                                int(source_energy * 100),
                                int(thickness * 100),
                                pose_index,
                                count_index,
                                len(perturb_tag),
                                20260510,
                            )
                            context_cell_id = (
                                f"{profile}_{split}_{seed_block}_{perturb_tag}_{energy_tag}_"
                                f"t{thickness:g}_p{pose_index}_c{count_target_bin}"
                            )
                            nuisance_cell_id = (
                                f"{profile}|{split}|{config['clean_matrix_origin']}|{config['source_family']}|"
                                f"{perturb_tag}|e{source_energy:g}|t{thickness:g}|p{pose_index}|c{count_target_bin}|{seed_block}"
                            )
                            source_id = f"v8a_full10_v5_{perturb_tag}_{energy_tag}"
                            for material in materials:
                                output_prefix = (
                                    f"{profile}_{split}_{seed_block}_{perturb_tag}_{energy_tag}_"
                                    f"t{thickness:g}mm_pose{pose_index}_count{count_target_bin}_{material}"
                                )
                                phase_rel = Path("phase_space") / f"{output_prefix}.csv"
                                phase_rows = build_perturbed_phase_space_rows(
                                    peaks_by_material=peaks_by_material,
                                    reference_wavelength_a=reference_wavelength_a,
                                    material=material,
                                    thickness_mm=thickness,
                                    pose_index=pose_index,
                                    context_seed=context_seed,
                                    source_energy_kev=source_energy,
                                    photons=photons,
                                    perturbation=perturbation,
                                )
                                write_phase_space(profile_dir / phase_rel, phase_rows)
                                config_rel = (
                                    Path("source_models")
                                    / "config"
                                    / "material_sorting_matrix"
                                    / profile
                                    / f"{output_prefix}.txt"
                                )
                                write_config(
                                    project_root / config_rel,
                                    {
                                        "run_id": output_prefix,
                                        "experiment_label": profile,
                                        "output_prefix": output_prefix,
                                        "output_dir": f"material_sorting_runs/{profile}",
                                        "benchmark_suite": "accuracy_v3",
                                        "research_route": "v8a_multiclass_context_physical_perturbation",
                                        "prediction_stage": "full10_physical_perturbation_scout",
                                        "run_role": "material",
                                        "source_variant": source_id,
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
                                        "ore_thickness_mm": thickness,
                                        "pose_index": pose_index,
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
                                        "context_replicate_index": 1,
                                        "context_material_count": len(materials),
                                        "stress_label": perturbation_profile if split != "train" else "nominal_train",
                                        "physical_perturbation_profile": perturbation_profile,
                                        "perturbation_family": perturbation["perturbation_family"],
                                        "peak_q_shift_fraction": perturbation["peak_q_shift_fraction"],
                                        "peak_theta_sigma_deg": perturbation["peak_theta_sigma_deg"],
                                        "continuum_fraction": perturbation["continuum_fraction"],
                                        "peak_intensity_jitter_sigma": perturbation["peak_intensity_jitter_sigma"],
                                        "source_energy_scale": perturbation["source_energy_scale"],
                                        "phase_space_profile_version": "v5_physical_perturbation_scout_v1",
                                    },
                                )
                                rows.append(
                                    {
                                        "row_index": row_index,
                                        "profile": profile,
                                        "split": split,
                                        "run_role": "material",
                                        "material": material,
                                        "source_id": source_id,
                                        "source_family": config["source_family"],
                                        "source_mode": "on",
                                        "stress_label": perturbation_profile if split != "train" else "nominal_train",
                                        "physical_perturbation_profile": perturbation_profile,
                                        "perturbation_family": perturbation["perturbation_family"],
                                        "peak_q_shift_fraction": perturbation["peak_q_shift_fraction"],
                                        "peak_theta_sigma_deg": perturbation["peak_theta_sigma_deg"],
                                        "continuum_fraction": perturbation["continuum_fraction"],
                                        "peak_intensity_jitter_sigma": perturbation["peak_intensity_jitter_sigma"],
                                        "source_energy_scale": perturbation["source_energy_scale"],
                                        "phase_space_profile_version": "v5_physical_perturbation_scout_v1",
                                        "clean_matrix_origin": config["clean_matrix_origin"],
                                        "source_energy_kev": source_energy * float(perturbation["source_energy_scale"]),
                                        "base_source_energy_kev": source_energy,
                                        "thickness_mm": thickness,
                                        "pose_index": pose_index,
                                        "count_target_bin": count_target_bin,
                                        "count_target_photons": photons,
                                        "seed_block": seed_block,
                                        "seed_block_seed": seed_block_seed,
                                        "random_seed": int(context_seed),
                                        "clean_context_cell_id": context_cell_id,
                                        "nuisance_cell_id": nuisance_cell_id,
                                        "context_replicate_index": 1,
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
        "generated_by": "analysis/generate_v8a_multiclass_context_v5_physical_perturbation_scout_matrix.py",
        "config": args.config,
        "profile": profile,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "peak_table_id": peak_table_id,
        "materials": materials,
        "rows": len(rows),
        "expected_rows": config["expected_rows"],
        "expected_context_cells": config["expected_context_cells"],
        "split_design": config["split_design"],
        "perturbation_profiles": config["perturbation_profiles"],
        "training_protocol_hint": config["training_protocol_hint"],
    }
    (profile_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote profile={profile} rows={len(rows)} matrix={matrix_path} runs_geant4=false training_unlocked=false")


if __name__ == "__main__":
    main()
