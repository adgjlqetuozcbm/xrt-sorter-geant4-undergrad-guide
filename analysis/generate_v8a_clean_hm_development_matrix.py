from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

CU_K_ALPHA_WAVELENGTH_A = 1.5406
HC_KEV_A = 12.398419843320026
EPS = 1e-9


def unit_vector_from_angles(theta_deg: float, phi_rad: float) -> tuple[float, float, float]:
    theta_rad = math.radians(theta_deg)
    return (
        math.cos(theta_rad),
        math.sin(theta_rad) * math.cos(phi_rad),
        math.sin(theta_rad) * math.sin(phi_rad),
    )


def stable_seed(*parts: int) -> int:
    value = 1729
    for part in parts:
        value = (value * 1000003 + int(part)) % (2**32 - 1)
    return value


def write_phase_space(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["event_id", "energy_keV", "x_mm", "y_mm", "z_mm", "dir_x", "dir_y", "dir_z"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_config(path: Path, values: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {value}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def wavelength_from_energy(energy_kev: float) -> float:
    return HC_KEV_A / energy_kev


def q_from_two_theta(two_theta_deg: float, wavelength_a: float = CU_K_ALPHA_WAVELENGTH_A) -> float:
    theta_rad = math.radians(two_theta_deg / 2.0)
    return 4.0 * math.pi * math.sin(theta_rad) / wavelength_a


def two_theta_from_q(q_a_inv: float, wavelength_a: float) -> float | None:
    argument = q_a_inv * wavelength_a / (4.0 * math.pi)
    if argument > 1.0:
        return None
    return math.degrees(2.0 * math.asin(max(argument, 0.0)))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_material_peaks(manifest: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {}
    for block in manifest.get("materials", []):
        material = str(block.get("material", ""))
        peaks = []
        for peak in block.get("peaks", []):
            peaks.append((float(peak["two_theta_deg"]), float(peak["relative_intensity"])))
        if peaks:
            result[material] = peaks
    return result


def weighted_index_from_unit(unit_value: float, weights: list[float]) -> int:
    total = sum(weights)
    cdf = []
    running = 0.0
    for weight in weights:
        running += weight / max(total, EPS)
        cdf.append(running)
    clipped = min(max(float(unit_value), 0.0), math.nextafter(1.0, 0.0))
    for index, threshold in enumerate(cdf):
        if clipped < threshold:
            return index
    return len(cdf) - 1


def build_clean_phase_space_rows(
    *,
    peaks_by_material: dict[str, list[tuple[float, float]]],
    material: str,
    thickness_mm: float,
    pose_index: int,
    pair_seed: int,
    source_energy_kev: float,
    photons: int,
) -> list[dict[str, float | int]]:
    """Generate paired source-on/default rows with nuisance randomness independent of material."""
    wavelength_a = wavelength_from_energy(source_energy_kev)
    rng = random.Random(
        stable_seed(
            int(thickness_mm * 100),
            int(pose_index),
            int(pair_seed),
            int(photons),
            20260506,
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
        two_theta_cu, _ = peaks[peak_index]
        q_a_inv = q_from_two_theta(two_theta_cu)
        two_theta = two_theta_from_q(q_a_inv, wavelength_a)
        if two_theta is None:
            two_theta = two_theta_cu
        theta = max(0.0, rng.gauss(two_theta, 0.18))
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
    parser = argparse.ArgumentParser(description="Generate the v8A clean H/M source-on/default development matrix without running Geant4.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_clean_hm_development_matrix_config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(project_root / args.config)
    peak_manifest = load_json(project_root / config["peak_manifest"])
    peak_table_id = str(peak_manifest.get("peak_table_id", ""))
    required_peak_table_id = str(config["required_peak_table_id"])
    if peak_table_id != required_peak_table_id:
        raise RuntimeError(f"Peak manifest mismatch: {peak_table_id} != {required_peak_table_id}")
    if config.get("status") != "development_preregistration":
        raise RuntimeError("Clean H/M matrix config must remain development_preregistration.")
    if config.get("source_modes") != [{"source_mode": "on", "stress_label": "default"}]:
        raise RuntimeError("Clean H/M matrix must be source-on/default only.")

    profile = str(config["profile"])
    matrix_root = project_root / "source_models" / "config" / "material_sorting_matrix"
    profile_dir = matrix_root / profile
    if profile_dir.exists() and any(profile_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Profile directory is not empty: {profile_dir}. Use --overwrite to replace preregistration artifacts.")
    if profile_dir.exists() and any(profile_dir.iterdir()) and args.overwrite:
        resolved_profile = profile_dir.resolve()
        resolved_matrix_root = matrix_root.resolve()
        if not resolved_profile.is_relative_to(resolved_matrix_root):
            raise RuntimeError(f"Refusing to clean unexpected profile path: {resolved_profile}")
        shutil.rmtree(resolved_profile)
    profile_dir.mkdir(parents=True, exist_ok=True)

    peaks_by_material = load_material_peaks(peak_manifest)
    rows: list[dict[str, Any]] = []
    row_index = 0
    count_bins = config["count_target_bins"]
    for split, split_config in config["splits"].items():
        for seed_block_config in split_config["seed_blocks"]:
            seed_block = str(seed_block_config["seed_block"])
            seed_block_seed = int(seed_block_config["seed"])
            for thickness in config["thickness_mm"]:
                for pose_index in config["pose_indices"]:
                    for count_index, count_config in enumerate(count_bins):
                        count_target_bin = str(count_config["count_target_bin"])
                        photons = int(count_config["photons_per_row"])
                        pair_seed = stable_seed(seed_block_seed, int(float(thickness) * 100), int(pose_index), count_index, 41)
                        nuisance_cell_id = (
                            f"{profile}|{split}|{config['clean_matrix_origin']}|{config['source_family']}|"
                            f"t{float(thickness):g}|p{int(pose_index)}|c{count_target_bin}|{seed_block}"
                        )
                        clean_pair_id = f"{profile}_{split}_{seed_block}_t{float(thickness):g}_p{int(pose_index)}_c{count_target_bin}"
                        for material in config["materials"]:
                            source_energy = float(config["source_energy_kev"])
                            output_prefix = (
                                f"{profile}_{split}_{seed_block}_t{float(thickness):g}mm_pose{int(pose_index)}_"
                                f"count{count_target_bin}_{material}"
                            )
                            phase_rel = Path("phase_space") / f"{output_prefix}.csv"
                            phase_path = profile_dir / phase_rel
                            phase_path.parent.mkdir(parents=True, exist_ok=True)
                            phase_rows = build_clean_phase_space_rows(
                                peaks_by_material=peaks_by_material,
                                material=str(material),
                                thickness_mm=float(thickness),
                                pose_index=int(pose_index),
                                pair_seed=int(pair_seed),
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
                                    "research_route": "v8a_clean_hm_development",
                                    "prediction_stage": "hm_diffraction_sidecar_clean_sampling",
                                    "run_role": "material",
                                    "source_variant": config["source_id"],
                                    "sample_photons": photons,
                                    "random_seed": int(pair_seed),
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
                                    "clean_pair_id": clean_pair_id,
                                    "nuisance_cell_id": nuisance_cell_id,
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
                                    "random_seed": int(pair_seed),
                                    "clean_pair_id": clean_pair_id,
                                    "nuisance_cell_id": nuisance_cell_id,
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
        "generated_by": "analysis/generate_v8a_clean_hm_development_matrix.py",
        "config": args.config,
        "profile": profile,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "full_ten_material_matrix": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "clean_matrix_origin": config["clean_matrix_origin"],
        "source_family": config["source_family"],
        "source_modes": ["on"],
        "stress_labels": ["default"],
        "peak_table_id": peak_table_id,
        "rows": len(rows),
        "expected_rows": config["expected_rows"],
        "expected_strict_pairs": config["expected_strict_pairs"],
        "minimum_strict_pairs": config["minimum_strict_pairs"],
        "source_energy_kev": float(config["source_energy_kev"]),
        "count_target_bins": config["count_target_bins"],
        "development_run_unlock_conditions": config["development_run_unlock_conditions"],
        "training_unlock_conditions": config["training_unlock_conditions"],
    }
    (profile_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote profile={profile} rows={len(rows)} matrix={matrix_path} runs_geant4=false training_unlocked=false")


if __name__ == "__main__":
    main()
