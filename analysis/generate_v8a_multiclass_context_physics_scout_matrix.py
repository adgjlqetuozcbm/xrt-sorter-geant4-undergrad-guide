from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from generate_v8a_multiclass_context_matrix import build_context_phase_space_rows, load_material_peaks
from generate_v8a_clean_hm_development_matrix import stable_seed, write_config, write_phase_space


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the v8A full-10 physics scout matrix without running Geant4.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_multiclass_context_v3_physics_scout_config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(as_project_path(project_root, args.config))
    if config.get("status") != "development_scout_preregistration":
        raise RuntimeError("Scout config must remain development_scout_preregistration.")
    peak_manifest = load_json(as_project_path(project_root, config["peak_manifest"]))
    peak_table_id = str(peak_manifest.get("peak_table_id", ""))
    if peak_table_id != str(config["required_peak_table_id"]):
        raise RuntimeError(f"Peak manifest mismatch: {peak_table_id} != {config['required_peak_table_id']}")
    reference_wavelength_a, peaks_by_material = load_material_peaks(peak_manifest)
    materials = [str(item) for item in config["target_materials"]]
    missing = sorted(set(materials) - set(peaks_by_material))
    if missing:
        raise RuntimeError(f"Peak manifest missing materials: {missing}")

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
    for split, split_config in config["splits"].items():
        for seed_block_config in split_config["seed_blocks"]:
            seed_block = str(seed_block_config["seed_block"])
            seed_block_seed = int(seed_block_config["seed"])
            for source_energy in config["source_energies_kev"]:
                source_energy = float(source_energy)
                energy_tag = f"{source_energy:g}kev".replace(".", "p")
                source_id = f"v8a_full10_physics_scout_{energy_tag}"
                for thickness in config["thickness_mm"]:
                    thickness = float(thickness)
                    for pose_index in config["pose_indices"]:
                        pose_index = int(pose_index)
                        for count_index, count_config in enumerate(config["count_target_bins"]):
                            count_target_bin = str(count_config["count_target_bin"])
                            photons = int(count_config["photons_per_row"])
                            for context_replicate_index in range(int(config.get("context_replicates_per_cell", 1))):
                                replicate_tag = f"r{context_replicate_index + 1:02d}"
                                context_seed = stable_seed(
                                    seed_block_seed,
                                    int(source_energy * 100),
                                    int(thickness * 100),
                                    pose_index,
                                    count_index,
                                    context_replicate_index,
                                    97,
                                )
                                context_cell_id = (
                                    f"{profile}_{split}_{seed_block}_{energy_tag}_t{thickness:g}_p{pose_index}_"
                                    f"c{count_target_bin}_{replicate_tag}"
                                )
                                nuisance_cell_id = (
                                    f"{profile}|{split}|{config['clean_matrix_origin']}|{config['source_family']}|"
                                    f"e{source_energy:g}|t{thickness:g}|p{pose_index}|c{count_target_bin}|{seed_block}|{replicate_tag}"
                                )
                                for material in materials:
                                    output_prefix = (
                                        f"{profile}_{split}_{seed_block}_{energy_tag}_t{thickness:g}mm_pose{pose_index}_"
                                        f"count{count_target_bin}_{replicate_tag}_{material}"
                                    )
                                    phase_rel = Path("phase_space") / f"{output_prefix}.csv"
                                    phase_rows = build_context_phase_space_rows(
                                        peaks_by_material=peaks_by_material,
                                        reference_wavelength_a=reference_wavelength_a,
                                        material=material,
                                        thickness_mm=thickness,
                                        pose_index=pose_index,
                                        context_seed=context_seed,
                                        source_energy_kev=source_energy,
                                        photons=photons,
                                    )
                                    write_phase_space(profile_dir / phase_rel, phase_rows)
                                    config_rel = Path("source_models") / "config" / "material_sorting_matrix" / profile / f"{output_prefix}.txt"
                                    write_config(
                                        project_root / config_rel,
                                        {
                                            "run_id": output_prefix,
                                            "experiment_label": profile,
                                            "output_prefix": output_prefix,
                                            "output_dir": f"material_sorting_runs/{profile}",
                                            "benchmark_suite": "accuracy_v3",
                                            "research_route": "v8a_multiclass_context_physics_scout",
                                            "prediction_stage": "full10_output_count_overlap_scout",
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
                                            "source_id": source_id,
                                            "source_family": config["source_family"],
                                            "source_mode": "on",
                                            "stress_label": "default",
                                            "clean_matrix_origin": config["clean_matrix_origin"],
                                            "source_energy_kev": source_energy,
                                            "thickness_mm": thickness,
                                            "pose_index": pose_index,
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
        "generated_by": "analysis/generate_v8a_multiclass_context_physics_scout_matrix.py",
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
        "source_energies_kev": config["source_energies_kev"],
        "thickness_mm": config["thickness_mm"],
        "scout_questions": config["scout_questions"],
    }
    (profile_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote profile={profile} rows={len(rows)} matrix={matrix_path} runs_geant4=false training_unlocked=false")


if __name__ == "__main__":
    main()
