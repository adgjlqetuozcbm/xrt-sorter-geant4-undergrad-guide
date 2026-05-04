from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from v8a_transport_sidecar_smoke import (
    HC_KEV_A,
    HM_PAIR,
    POWDER_PEAKS,
    q_from_two_theta,
    two_theta_from_q,
    wavelength_from_energy,
)


PROFILE = "v8a_custom_diffraction_g4_smoke"
PEAK_TABLE_ID = "hm_powder_peaks_project_scan_v8a"


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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


def material_index(material: str) -> int:
    return 1 if material == "Magnetite" else 0


def build_phase_space_rows(
    *,
    material: str,
    thickness_mm: float,
    random_seed: int,
    source_energy_kev: float,
    photons: int,
    source_mode: str,
    stress: bool,
) -> list[dict[str, float | int]]:
    wavelength_a = wavelength_from_energy(source_energy_kev)
    rng = np.random.default_rng(stable_seed(material_index(material), int(thickness_mm * 100), random_seed, 1 if stress else 0, 1 if source_mode == "on" else 0))
    continuum_fraction = 0.30 if source_mode == "on" else 1.0
    continuum_count = int(round(photons * continuum_fraction))
    peak_count = max(0, photons - continuum_count)
    rows: list[dict[str, float | int]] = []

    def add_row(event_id: int, energy_kev: float, x_mm: float, y_mm: float, z_mm: float, direction: tuple[float, float, float]) -> None:
        rows.append(
            {
                "event_id": event_id,
                "energy_keV": energy_kev,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": y_mm * 0.0 + z_mm,
                "dir_x": direction[0],
                "dir_y": direction[1],
                "dir_z": direction[2],
            }
        )

    for event_id in range(continuum_count):
        phi = rng.uniform(0.0, 2.0 * math.pi)
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

    peaks = POWDER_PEAKS[material]
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
        phi = rng.uniform(0.0, 2.0 * math.pi)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v8A custom diffraction Geant4 smoke configs and phase-space photons.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--train-seeds", default="6701,6702,6703")
    parser.add_argument("--validation-seeds", default="6801,6802")
    parser.add_argument("--thickness-list", default="3,10,30")
    parser.add_argument("--source-energy-kev", type=float, default=35.0)
    parser.add_argument("--photons-per-row", type=int, default=4000)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    profile_dir = project_root / "source_models" / "config" / "material_sorting_matrix" / args.profile
    rows = []
    row_index = 0
    split_seed_map = {"train": parse_csv_ints(args.train_seeds), "validation": parse_csv_ints(args.validation_seeds)}
    thicknesses = parse_csv_floats(args.thickness_list)
    modes = [("on", "default"), ("on", "stress"), ("off", "leakage_off")]
    for split, seeds in split_seed_map.items():
        for seed in seeds:
            for material in HM_PAIR:
                for thickness in thicknesses:
                    for source_mode, stress_label in modes:
                        stress = stress_label == "stress"
                        source_id = f"v8a_g4_{source_mode}_{stress_label}_{args.source_energy_kev:g}kev"
                        output_prefix = f"{args.profile}_{split}_{material}_{thickness:g}mm_seed{seed}_{source_mode}_{stress_label}"
                        phase_rel = Path("phase_space") / f"{output_prefix}.csv"
                        phase_path = profile_dir / phase_rel
                        phase_rows = build_phase_space_rows(
                            material=material,
                            thickness_mm=thickness,
                            random_seed=seed,
                            source_energy_kev=args.source_energy_kev,
                            photons=args.photons_per_row,
                            source_mode=source_mode,
                            stress=stress,
                        )
                        write_phase_space(phase_path, phase_rows)
                        config_rel = Path("source_models") / "config" / "material_sorting_matrix" / args.profile / f"{output_prefix}.txt"
                        config_path = project_root / config_rel
                        write_config(
                            config_path,
                            {
                                "run_id": output_prefix,
                                "experiment_label": args.profile,
                                "output_prefix": output_prefix,
                                "output_dir": f"material_sorting_runs/{args.profile}",
                                "benchmark_suite": "accuracy_v3",
                                "research_route": "v8a_custom_diffraction_g4_smoke",
                                "prediction_stage": "hm_diffraction_sidecar",
                                "run_role": "material",
                                "source_variant": source_id,
                                "sample_photons": args.photons_per_row,
                                "random_seed": seed,
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
                                "profile": args.profile,
                                "split": split,
                                "run_role": "material",
                                "material": material,
                                "source_id": source_id,
                                "source_mode": source_mode,
                                "stress_label": stress_label,
                                "source_energy_kev": args.source_energy_kev,
                                "thickness_mm": thickness,
                                "random_seed": seed,
                                "phase_space_file": phase_rel.as_posix(),
                                "config_path": config_rel.as_posix(),
                                "output_prefix": output_prefix,
                                "peak_table_id": PEAK_TABLE_ID,
                                "development_only": True,
                                "shadow_or_final_used": False,
                            }
                        )
                        row_index += 1
    profile_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = profile_dir / "material_sorting_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "generated_by": "analysis/generate_v8a_diffraction_g4_smoke_matrix.py",
        "profile": args.profile,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "bin_axis": "q_a_inv",
        "peak_table_id": PEAK_TABLE_ID,
        "rows": len(rows),
        "photons_per_row": args.photons_per_row,
        "source_energy_kev": args.source_energy_kev,
        "hc_kev_a": HC_KEV_A,
    }
    (profile_dir / "matrix_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote profile={args.profile} rows={len(rows)} matrix={matrix_path}")


if __name__ == "__main__":
    main()
