from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from generate_v8a_diffraction_g4_smoke_matrix import (
    stable_seed,
    unit_vector_from_angles,
    wavelength_from_energy,
    write_config,
    write_phase_space,
)
from v8a_transport_sidecar_smoke import q_from_two_theta, two_theta_from_q


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def material_index(material: str) -> int:
    return 1 if material == "Magnetite" else 0


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


def build_phase_space_rows(
    *,
    peaks_by_material: dict[str, list[tuple[float, float]]],
    material: str,
    thickness_mm: float,
    pose_index: int,
    random_seed: int,
    source_energy_kev: float,
    photons: int,
    source_mode: str,
    stress: bool,
) -> list[dict[str, float | int]]:
    wavelength_a = wavelength_from_energy(source_energy_kev)
    rng = np.random.default_rng(
        stable_seed(
            material_index(material),
            int(thickness_mm * 100),
            pose_index,
            random_seed,
            1 if stress else 0,
            1 if source_mode == "on" else 0,
        )
    )
    continuum_fraction = 0.30 if source_mode == "on" else 1.0
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
        theta = max(0.0, rng.normal(0.0, 0.35 if not stress else 0.60))
        x_mm = thickness_mm / 2.0 + 0.05
        add_row(
            event_id,
            source_energy_kev,
            x_mm,
            rng.uniform(-4.0, 4.0),
            rng.uniform(-4.0, 4.0),
            unit_vector_from_angles(theta, phi),
        )

    if source_mode == "off":
        return rows

    peaks = peaks_by_material[material]
    weights = np.array([max(weight, 0.0) for _, weight in peaks], dtype=np.float64)
    weights = weights / weights.sum()
    chosen = rng.choice(len(peaks), size=peak_count, p=weights)
    for offset, peak_index in enumerate(chosen, start=continuum_count):
        two_theta_cu, _ = peaks[int(peak_index)]
        q_a_inv = q_from_two_theta(two_theta_cu)
        two_theta = two_theta_from_q(q_a_inv, wavelength_a)
        if two_theta is None:
            two_theta = two_theta_cu
        sigma = 0.18 if not stress else 0.32
        theta = max(0.0, rng.normal(two_theta, sigma))
        phi = rng.uniform(0.0, 2.0 * math.pi) + pose_phi_offset
        x_mm = rng.uniform(-thickness_mm / 2.0, thickness_mm / 2.0)
        add_row(
            offset,
            source_energy_kev,
            x_mm,
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
            unit_vector_from_angles(theta, phi),
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the v8A H/M medium development matrix from the successor peak manifest.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_medium_development_matrix_config.json")
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
        raise RuntimeError("Medium matrix config must remain development_preregistration.")

    profile = str(config["profile"])
    profile_dir = project_root / "source_models" / "config" / "material_sorting_matrix" / profile
    if profile_dir.exists() and any(profile_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Profile directory is not empty: {profile_dir}. Use --overwrite to replace generated development matrix files.")
    profile_dir.mkdir(parents=True, exist_ok=True)

    peaks_by_material = load_material_peaks(peak_manifest)
    rows: list[dict[str, Any]] = []
    row_index = 0
    for split, split_config in config["splits"].items():
        for seed in split_config["seeds"]:
            for material in config["materials"]:
                for thickness in config["thickness_mm"]:
                    for pose_index in config["pose_indices"]:
                        for source in config["source_modes"]:
                            source_mode = str(source["source_mode"])
                            stress_label = str(source["stress_label"])
                            stress = stress_label == "stress"
                            source_energy = float(config["source_energy_kev"])
                            source_id = f"v8a_medium_{source_mode}_{stress_label}_{source_energy:g}kev_pose{pose_index}"
                            output_prefix = f"{profile}_{split}_{material}_{thickness:g}mm_pose{pose_index}_seed{seed}_{source_mode}_{stress_label}"
                            phase_rel = Path("phase_space") / f"{output_prefix}.csv"
                            phase_path = profile_dir / phase_rel
                            phase_rows = build_phase_space_rows(
                                peaks_by_material=peaks_by_material,
                                material=material,
                                thickness_mm=float(thickness),
                                pose_index=int(pose_index),
                                random_seed=int(seed),
                                source_energy_kev=source_energy,
                                photons=int(config["photons_per_row"]),
                                source_mode=source_mode,
                                stress=stress,
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
                                    "research_route": "v8a_hm_medium_development",
                                    "prediction_stage": "hm_diffraction_sidecar",
                                    "run_role": "material",
                                    "source_variant": source_id,
                                    "sample_photons": int(config["photons_per_row"]),
                                    "random_seed": int(seed),
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
                                    "source_mode": source_mode,
                                    "stress_label": stress_label,
                                    "source_energy_kev": source_energy,
                                    "thickness_mm": float(thickness),
                                    "pose_index": int(pose_index),
                                    "random_seed": int(seed),
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
        "generated_by": "analysis/generate_v8a_medium_development_matrix.py",
        "config": args.config,
        "profile": profile,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "full_ten_material_matrix": False,
        "peak_table_id": peak_table_id,
        "rows": len(rows),
        "expected_rows": config["expected_rows"],
        "photons_per_row": int(config["photons_per_row"]),
        "source_energy_kev": float(config["source_energy_kev"]),
        "training_unlock_conditions": config["training_unlock_conditions"],
    }
    (profile_dir / "matrix_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote profile={profile} rows={len(rows)} matrix={matrix_path}")


if __name__ == "__main__":
    main()
