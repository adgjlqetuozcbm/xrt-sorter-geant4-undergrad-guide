from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from generate_v8a_clean_hm_development_matrix import stable_seed, write_config, write_phase_space
from generate_v8a_multiclass_context_matrix import load_material_peaks
from generate_v8a_multiclass_context_v5_physical_perturbation_scout_matrix import (
    as_project_path,
    build_perturbed_phase_space_rows,
    load_json,
    profile_map,
    safe_tag,
    split_seed_blocks,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_text_seed(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v8A v6 full-10 perturbation-augmented physical robustness matrix.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_multiclass_context_v6_physical_robust_config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = as_project_path(project_root, args.config)
    config = load_json(config_path)
    if config.get("status") != "development_physical_robust_preregistration":
        raise RuntimeError("v6 robust config must remain development_physical_robust_preregistration.")

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
            for thickness_value in config["thickness_mm"]:
                thickness = float(thickness_value)
                for pose_value in config["pose_indices"]:
                    pose_index = int(pose_value)
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
                                stable_text_seed(perturb_tag),
                                20260511,
                            )
                            context_cell_id = (
                                f"{profile}_{split}_{seed_block}_{perturb_tag}_{energy_tag}_"
                                f"t{thickness:g}_p{pose_index}_c{count_target_bin}"
                            )
                            nuisance_cell_id = (
                                f"{profile}|{split}|{config['clean_matrix_origin']}|{config['source_family']}|"
                                f"{perturb_tag}|e{source_energy:g}|t{thickness:g}|p{pose_index}|c{count_target_bin}|{seed_block}"
                            )
                            source_id = f"v8a_full10_v6_{perturb_tag}_{energy_tag}"
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
                                        "research_route": "v8a_multiclass_context_physical_robust",
                                        "prediction_stage": "full10_physical_robust",
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
                                        "stress_label": perturbation_profile,
                                        "physical_perturbation_profile": perturbation_profile,
                                        "perturbation_family": perturbation["perturbation_family"],
                                        "peak_q_shift_fraction": perturbation["peak_q_shift_fraction"],
                                        "peak_theta_sigma_deg": perturbation["peak_theta_sigma_deg"],
                                        "continuum_fraction": perturbation["continuum_fraction"],
                                        "peak_intensity_jitter_sigma": perturbation["peak_intensity_jitter_sigma"],
                                        "source_energy_scale": perturbation["source_energy_scale"],
                                        "phase_space_profile_version": "v6_physical_robust_v1",
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
                                        "stress_label": perturbation_profile,
                                        "physical_perturbation_profile": perturbation_profile,
                                        "perturbation_family": perturbation["perturbation_family"],
                                        "peak_q_shift_fraction": perturbation["peak_q_shift_fraction"],
                                        "peak_theta_sigma_deg": perturbation["peak_theta_sigma_deg"],
                                        "continuum_fraction": perturbation["continuum_fraction"],
                                        "peak_intensity_jitter_sigma": perturbation["peak_intensity_jitter_sigma"],
                                        "source_energy_scale": perturbation["source_energy_scale"],
                                        "phase_space_profile_version": "v6_physical_robust_v1",
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
    write_csv(matrix_path, rows)
    manifest = {
        "generated_by": "analysis/generate_v8a_multiclass_context_v6_physical_robust_matrix.py",
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
