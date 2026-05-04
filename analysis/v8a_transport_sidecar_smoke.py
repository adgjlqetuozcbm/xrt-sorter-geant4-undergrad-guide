from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v8a_diffraction_observability import HM_PAIR, POWDER_PEAKS, require_sklearn


CU_K_ALPHA_WAVELENGTH_A = 1.5406
HC_KEV_A = 12.398419843320026
EPS = 1e-9

PASS_FEATURE_AUC = 0.95
PASS_FEATURE_D_PRIME = 3.0
PASS_HM_MIN_RECALL = 0.80
PASS_WORST_THICKNESS_HM_MIN_RECALL = 0.78
MAX_CONTROL_HM_MIN_RECALL = 0.75
MAX_SHUFFLED_LABEL_HM_MIN_RECALL = 0.65
MAX_OVERLAP_ONLY_HM_MIN_RECALL = 0.75
MAX_TOTAL_COUNT_HM_MIN_RECALL = 0.75
NO_GO_HM_MIN_RECALL = 0.75

UNIQUE_TWO_THETA_DEG = {
    "hematite_unique": [24.1, 33.2, 40.9, 49.5, 54.1, 64.0],
    "magnetite_unique": [18.3, 30.1, 37.0, 43.1, 53.4, 74.0],
    "overlap": [35.55, 57.25, 62.55],
}

SECTOR_NAMES = ("sector_00", "sector_01", "sector_02", "sector_03")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def wavelength_from_energy(energy_kev: float) -> float:
    return HC_KEV_A / energy_kev


def q_from_two_theta(two_theta_deg: float, wavelength_a: float = CU_K_ALPHA_WAVELENGTH_A) -> float:
    theta_rad = math.radians(two_theta_deg / 2.0)
    return 4.0 * math.pi * math.sin(theta_rad) / wavelength_a


def d_from_q(q_a_inv: float) -> float:
    return 2.0 * math.pi / max(q_a_inv, EPS)


def two_theta_from_q(q_a_inv: float, wavelength_a: float) -> float | None:
    argument = q_a_inv * wavelength_a / (4.0 * math.pi)
    if argument > 1.0:
        return None
    return math.degrees(2.0 * math.asin(max(argument, 0.0)))


def make_q_axis(q_min: float, q_max: float, q_bin_width: float) -> np.ndarray:
    bins = int(round((q_max - q_min) / q_bin_width)) + 1
    return np.linspace(q_min, q_max, bins, dtype=np.float64)


def gaussian(axis: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((axis - center) / max(sigma, EPS)) ** 2)


def stable_rng(random_seed: int, material: str, split_seed: int, thickness: float, pose_index: int, source_index: int) -> np.random.Generator:
    material_offset = 200000 if material == "Magnetite" else 0
    thickness_offset = int(round(thickness * 100.0))
    seed = random_seed + material_offset + split_seed * 97 + thickness_offset * 31 + pose_index * 1009 + source_index * 7919
    return np.random.default_rng(seed)


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Use --overwrite to replace smoke artifacts.")
    output_dir.mkdir(parents=True, exist_ok=True)


def q_resolution_sigma(
    *,
    q_a_inv: float,
    source_wavelength_a: float,
    detector_resolution_deg: float,
    angular_bin_width_deg: float,
    source_bandwidth_fraction: float,
    intrinsic_q_sigma: float,
) -> float:
    two_theta = two_theta_from_q(q_a_inv, source_wavelength_a)
    if two_theta is None:
        return intrinsic_q_sigma
    theta_rad = math.radians(two_theta / 2.0)
    dq_d_twotheta = (2.0 * math.pi / source_wavelength_a) * math.cos(theta_rad)
    angular_sigma_rad = math.radians(math.sqrt(detector_resolution_deg**2 + angular_bin_width_deg**2 / 12.0))
    angular_q_sigma = abs(dq_d_twotheta) * angular_sigma_rad
    bandwidth_q_sigma = abs(q_a_inv) * source_bandwidth_fraction
    return math.sqrt(intrinsic_q_sigma**2 + angular_q_sigma**2 + bandwidth_q_sigma**2)


def sector_response(sector_index: int, rng: np.random.Generator, sector_count: int) -> dict[str, float]:
    base_angles = np.linspace(0.0, 2.0 * math.pi, sector_count, endpoint=False)
    angle = float(base_angles[sector_index])
    orientation = 1.0 + 0.18 * math.cos(angle) + rng.normal(0.0, 0.05)
    return {
        "sector_angle_deg": math.degrees(angle),
        "sector_throughput": float(max(0.35, rng.lognormal(mean=0.0, sigma=0.10))),
        "sector_orientation_factor": float(max(0.10, orientation)),
        "sector_background_factor": float(max(0.20, rng.lognormal(mean=0.0, sigma=0.08))),
    }


def simulate_sector_spectrum(
    *,
    q_axis: np.ndarray,
    material: str,
    thickness_mm: float,
    source_energy_kev: float,
    sector_index: int,
    sector_count: int,
    rng: np.random.Generator,
    detector_resolution_deg: float,
    angular_bin_width_deg: float,
    source_bandwidth_fraction: float,
    intrinsic_q_sigma: float,
    q_calibration_jitter: float,
    orientation_sigma: float,
    absorption_strength: float,
    background_level: float,
    background_slope_sigma: float,
    counts_scale: float,
    read_noise_sigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    source_wavelength_a = wavelength_from_energy(source_energy_kev)
    sector = sector_response(sector_index, rng, sector_count)
    centered_q = (q_axis - float(q_axis.mean())) / max(float(np.ptp(q_axis)), EPS)
    background_slope = rng.normal(0.0, background_slope_sigma)
    continuum = background_level * sector["sector_background_factor"] * (1.0 + background_slope * centered_q)
    continuum += 0.018 * np.exp(-0.22 * (q_axis - float(q_axis.min()))) * rng.lognormal(mean=0.0, sigma=0.10)
    continuum = np.clip(continuum, background_level * 0.10, None)

    attenuation = math.exp(-absorption_strength * thickness_mm)
    self_absorption = math.exp(-0.50 * absorption_strength * thickness_mm)
    q_shift = rng.normal(0.0, q_calibration_jitter)
    signal = np.zeros_like(q_axis, dtype=np.float64)

    for peak_index, (two_theta_deg, relative_intensity) in enumerate(POWDER_PEAKS[material]):
        q_center = q_from_two_theta(two_theta_deg)
        peak_two_theta = two_theta_from_q(q_center, source_wavelength_a)
        if peak_two_theta is None:
            continue
        peak_sigma = q_resolution_sigma(
            q_a_inv=q_center,
            source_wavelength_a=source_wavelength_a,
            detector_resolution_deg=detector_resolution_deg,
            angular_bin_width_deg=angular_bin_width_deg,
            source_bandwidth_fraction=source_bandwidth_fraction,
            intrinsic_q_sigma=intrinsic_q_sigma,
        )
        orientation = rng.lognormal(mean=0.0, sigma=orientation_sigma)
        texture_phase = 0.12 * math.sin((sector_index + 1) * (peak_index + 1))
        peak_scale = relative_intensity * sector["sector_orientation_factor"] * orientation * (1.0 + texture_phase)
        signal += peak_scale * gaussian(q_axis, q_center + q_shift, peak_sigma)

    signal *= attenuation * self_absorption * sector["sector_throughput"]
    expected_counts = np.clip((continuum + signal) * counts_scale, 0.0, None)
    observed_counts = rng.poisson(expected_counts).astype(np.float64)
    intensity = observed_counts / max(counts_scale, EPS)
    intensity += rng.normal(0.0, read_noise_sigma, size=intensity.shape)
    intensity = np.clip(intensity, 0.0, None)
    controls = {
        **sector,
        "source_wavelength_a": float(source_wavelength_a),
        "source_energy_kev": float(source_energy_kev),
        "absorption_factor": float(attenuation),
        "self_absorption_factor": float(self_absorption),
        "background_level_effective": float(np.median(continuum)),
        "expected_total_counts": float(expected_counts.sum()),
        "observed_total_counts": float(observed_counts.sum()),
        "q_calibration_shift_a_inv": float(q_shift),
        "background_slope": float(background_slope),
    }
    return intensity.astype(np.float64), controls


def baseline_correct_and_normalize(values: np.ndarray) -> np.ndarray:
    baseline = float(np.quantile(values, 0.08))
    net = np.clip(values - baseline, 0.0, None)
    area = float(net.sum())
    if area <= EPS:
        return net
    return net / area


def peak_q_windows() -> dict[str, list[float]]:
    return {
        family: [q_from_two_theta(two_theta) for two_theta in centers]
        for family, centers in UNIQUE_TWO_THETA_DEG.items()
    }


def build_sidecar(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    q_axis = make_q_axis(args.q_min, args.q_max, args.q_bin_width)
    source_energies = parse_csv_floats(args.source_energies_kev)
    thicknesses = parse_csv_floats(args.thickness_list)
    split_seed_map = {"train": parse_csv_ints(args.train_seeds), "validation": parse_csv_ints(args.validation_seeds)}
    sector_count = args.detector_sector_count
    sector_names = [f"sector_{idx:02d}" for idx in range(sector_count)]
    q_windows = peak_q_windows()
    long_rows: list[dict] = []
    feature_rows: list[dict] = []

    for split, seeds in split_seed_map.items():
        for split_seed in seeds:
            for material in HM_PAIR:
                for thickness in thicknesses:
                    for pose_index in range(args.poses_per_condition):
                        for source_index, source_energy in enumerate(source_energies):
                            sample_id = f"v8a_transport_{split}_{material}_{split_seed}_{thickness:g}_{pose_index}_src{source_index}"
                            sample_sector_arrays: dict[str, np.ndarray] = {}
                            sample_controls: list[dict[str, float]] = []
                            for sector_index, sector_name in enumerate(sector_names):
                                rng = stable_rng(args.random_seed, material, split_seed, thickness, pose_index, source_index * 100 + sector_index)
                                intensity, controls = simulate_sector_spectrum(
                                    q_axis=q_axis,
                                    material=material,
                                    thickness_mm=thickness,
                                    source_energy_kev=source_energy,
                                    sector_index=sector_index,
                                    sector_count=sector_count,
                                    rng=rng,
                                    detector_resolution_deg=args.detector_resolution_deg,
                                    angular_bin_width_deg=args.angular_bin_width_deg,
                                    source_bandwidth_fraction=args.source_bandwidth_fraction,
                                    intrinsic_q_sigma=args.intrinsic_q_sigma,
                                    q_calibration_jitter=args.q_calibration_jitter,
                                    orientation_sigma=args.orientation_sigma,
                                    absorption_strength=args.absorption_strength,
                                    background_level=args.background_level,
                                    background_slope_sigma=args.background_slope_sigma,
                                    counts_scale=args.counts_scale,
                                    read_noise_sigma=args.read_noise_sigma,
                                )
                                normalized = baseline_correct_and_normalize(intensity)
                                sample_sector_arrays[sector_name] = normalized
                                sample_controls.append(controls)
                                for q_index, (q_value, raw_value, normalized_value) in enumerate(zip(q_axis, intensity, normalized)):
                                    long_rows.append(
                                        {
                                            "protocol": "v8A_transport_sidecar_smoke",
                                            "development_only": True,
                                            "shadow_or_final_used": False,
                                            "reads_existing_xrt_cubes": False,
                                            "sample_id": sample_id,
                                            "split": split,
                                            "material": material,
                                            "random_seed": split_seed,
                                            "thickness_mm": thickness,
                                            "pose_index": pose_index,
                                            "source_id": f"mono_{source_energy:g}kev",
                                            "source_energy_kev": source_energy,
                                            "source_wavelength_a": controls["source_wavelength_a"],
                                            "peak_table_id": "hm_powder_peaks_project_scan_v8a",
                                            "bin_axis": "q_a_inv",
                                            "q_bin_index": q_index,
                                            "q_bin_center_a_inv": float(q_value),
                                            "d_bin_center_a": d_from_q(float(q_value)),
                                            "detector_sector": sector_name,
                                            "detector_sector_angle_deg": controls["sector_angle_deg"],
                                            "angular_bin_width_deg": args.angular_bin_width_deg,
                                            "detector_resolution_deg": args.detector_resolution_deg,
                                            "source_bandwidth_fraction": args.source_bandwidth_fraction,
                                            "background_level_effective": controls["background_level_effective"],
                                            "throughput": controls["sector_throughput"],
                                            "absorption_factor": controls["absorption_factor"],
                                            "raw_intensity": float(raw_value),
                                            "sidecar_intensity_norm": float(normalized_value),
                                        }
                                    )

                            feature_row: dict[str, float | int | str | bool] = {
                                "protocol": "v8A_transport_sidecar_smoke",
                                "development_only": True,
                                "shadow_or_final_used": False,
                                "reads_existing_xrt_cubes": False,
                                "sample_id": sample_id,
                                "split": split,
                                "material": material,
                                "random_seed": split_seed,
                                "thickness_mm": thickness,
                                "pose_index": pose_index,
                                "source_id": f"mono_{source_energy:g}kev",
                                "source_energy_kev": source_energy,
                                "source_wavelength_a": wavelength_from_energy(source_energy),
                                "detector_resolution_deg": args.detector_resolution_deg,
                                "angular_bin_width_deg": args.angular_bin_width_deg,
                                "source_bandwidth_fraction": args.source_bandwidth_fraction,
                                "observed_total_counts": float(sum(item["observed_total_counts"] for item in sample_controls)),
                                "expected_total_counts": float(sum(item["expected_total_counts"] for item in sample_controls)),
                                "mean_background_level": float(np.mean([item["background_level_effective"] for item in sample_controls])),
                                "mean_throughput": float(np.mean([item["sector_throughput"] for item in sample_controls])),
                                "mean_absorption_factor": float(np.mean([item["absorption_factor"] for item in sample_controls])),
                            }
                            for sector_name, normalized in sample_sector_arrays.items():
                                stride = max(1, int(round(args.feature_q_stride / args.q_bin_width)))
                                for q_index in range(0, len(q_axis), stride):
                                    q_token = f"{q_axis[q_index]:.2f}".replace(".", "p")
                                    feature_row[f"{sector_name}_q_{q_token}_norm"] = float(normalized[q_index])
                                for family, centers in q_windows.items():
                                    window_values = []
                                    for center_q in centers:
                                        mask = np.abs(q_axis - center_q) <= args.feature_q_half_width
                                        window_values.append(float(normalized[mask].sum()) if mask.any() else 0.0)
                                    feature_row[f"{sector_name}_{family}_area_sum"] = float(np.sum(window_values))
                                    feature_row[f"{sector_name}_{family}_area_max"] = float(np.max(window_values) if window_values else 0.0)

                            h_values = [
                                float(feature_row[f"{sector_name}_hematite_unique_area_sum"])
                                for sector_name in sector_names
                            ]
                            m_values = [
                                float(feature_row[f"{sector_name}_magnetite_unique_area_sum"])
                                for sector_name in sector_names
                            ]
                            overlap_values = [
                                float(feature_row[f"{sector_name}_overlap_area_sum"])
                                for sector_name in sector_names
                            ]
                            feature_row["signature_hematite_unique_area_sum"] = float(np.sum(h_values))
                            feature_row["signature_magnetite_unique_area_sum"] = float(np.sum(m_values))
                            feature_row["signature_overlap_area_sum"] = float(np.sum(overlap_values))
                            feature_row["signature_h_over_m_log_ratio"] = float(math.log((np.sum(h_values) + EPS) / (np.sum(m_values) + EPS)))
                            feature_row["signature_h_minus_m_area"] = float(np.sum(h_values) - np.sum(m_values))
                            feature_rows.append(feature_row)

    long_frame = pd.DataFrame(long_rows)
    feature_frame = pd.DataFrame(feature_rows)
    manifest = {
        "protocol_name": "v8A_transport_sidecar_smoke",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_shadow_or_final": False,
        "reads_existing_xrt_cubes": False,
        "input_data": "hardcoded_reference_powder_peak_table_only",
        "diffraction_representation": "tabulated_powder_pattern_sidecar_project_scan",
        "geant4_role": "future_transport_absorption_background_detector_response_only",
        "ordinary_geant4_rayleigh_used_as_powder_xrd": False,
        "g4_xray_reflection_used_as_bulk_powder_xrd": False,
        "bin_axis": "q_a_inv",
        "multi_energy_fusion_rule": "q_or_d_spacing_only_no_cross_energy_fixed_2theta_fusion",
        "peak_table_id": "hm_powder_peaks_project_scan_v8a",
        "materials": list(HM_PAIR),
        "source_energies_kev": source_energies,
        "source_wavelengths_a": [wavelength_from_energy(value) for value in source_energies],
        "thicknesses_mm": thicknesses,
        "detector_sectors": sector_names,
        "sample_count": int(len(feature_frame)),
        "sidecar_row_count": int(len(long_frame)),
        "q_bin_count": int(len(q_axis)),
        "q_min_a_inv": args.q_min,
        "q_max_a_inv": args.q_max,
        "q_bin_width_a_inv": args.q_bin_width,
        "feature_q_stride_a_inv": args.feature_q_stride,
        "detector_resolution_deg": args.detector_resolution_deg,
        "angular_bin_width_deg": args.angular_bin_width_deg,
        "background_level": args.background_level,
        "background_slope_sigma": args.background_slope_sigma,
        "throughput_model": "sector_lognormal_response_with_per_sector_background",
        "absorption_strength": args.absorption_strength,
        "source_bandwidth_fraction": args.source_bandwidth_fraction,
        "counts_scale": args.counts_scale,
        "schema_fields": {
            "long_sidecar": [
                "material",
                "split",
                "random_seed",
                "thickness_mm",
                "pose_index",
                "source_energy_kev",
                "source_wavelength_a",
                "peak_table_id",
                "bin_axis",
                "q_bin_center_a_inv",
                "d_bin_center_a",
                "detector_sector",
                "background_level_effective",
                "detector_resolution_deg",
                "angular_bin_width_deg",
                "throughput",
                "absorption_factor",
                "sidecar_intensity_norm",
            ],
            "sample_features": [
                "q_or_d_bin_x_detector_sector_features",
                "unique_peak_family_area_sums",
                "overlap_guard_features",
                "control_count_background_thickness_features",
            ],
        },
        "powder_peaks_two_theta_deg_cu_k_alpha": {
            material: [{"two_theta_deg": c, "relative_weight": w, "q_a_inv": q_from_two_theta(c), "d_a": d_from_q(q_from_two_theta(c))} for c, w in peaks]
            for material, peaks in POWDER_PEAKS.items()
        },
    }
    return long_frame, feature_frame, manifest


def pair_recalls(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    recalls = {}
    for material in HM_PAIR:
        mask = y_true == material
        recalls[material] = float(np.mean(predictions[mask] == material)) if mask.any() else 0.0
    return recalls


def evaluate_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    method_name: str,
    estimator,
    *,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 8813,
) -> tuple[dict, pd.DataFrame]:
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("validation").to_numpy()
    x_train = frame.loc[train, feature_cols].to_numpy(dtype=np.float64)
    x_validation = frame.loc[validation, feature_cols].to_numpy(dtype=np.float64)
    y_train = frame.loc[train, "material"].astype(str).to_numpy()
    y_validation = frame.loc[validation, "material"].astype(str).to_numpy()
    if shuffle_train_labels:
        rng = np.random.default_rng(shuffle_seed)
        y_train = rng.permutation(y_train)
    estimator.fit(x_train, y_train)
    predictions = np.asarray(estimator.predict(x_validation)).astype(str)
    recalls = pair_recalls(y_validation, predictions)
    decisions = frame.loc[validation, ["sample_id", "material", "split", "random_seed", "thickness_mm", "pose_index", "source_id"]].copy()
    decisions["method"] = method_name
    decisions["prediction"] = predictions
    decisions["is_correct"] = decisions["material"].astype(str).to_numpy() == predictions
    by_thickness = []
    for _, group in decisions.groupby("thickness_mm", sort=True):
        thickness_recalls = pair_recalls(group["material"].astype(str).to_numpy(), group["prediction"].astype(str).to_numpy())
        by_thickness.append(min(thickness_recalls.values()))
    return (
        {
            "method": method_name,
            "feature_count": int(len(feature_cols)),
            "validation_samples": int(len(y_validation)),
            "hematite_recall": recalls["Hematite"],
            "magnetite_recall": recalls["Magnetite"],
            "hm_min_recall": float(min(recalls.values())),
            "pairwise_hm_min_recall": float(min(recalls.values())),
            "worst_thickness_hm_min_recall": float(min(by_thickness)) if by_thickness else 0.0,
        },
        decisions,
    )


def observability_metrics(frame: pd.DataFrame, feature_cols: list[str], roc_auc_score) -> pd.DataFrame:
    validation = frame["split"].eq("validation")
    y = frame.loc[validation, "material"].astype(str).to_numpy()
    y_binary = (y == "Magnetite").astype(int)
    rows = []
    for col in feature_cols:
        values = frame.loc[validation, col].to_numpy(dtype=np.float64)
        h = values[y == "Hematite"]
        m = values[y == "Magnetite"]
        pooled = math.sqrt(0.5 * (float(np.var(h)) + float(np.var(m))) + EPS)
        d_prime = abs(float(np.mean(h) - np.mean(m))) / pooled
        try:
            auc = float(roc_auc_score(y_binary, values))
            oriented_auc = max(auc, 1.0 - auc)
        except ValueError:
            auc = 0.5
            oriented_auc = 0.5
        rows.append(
            {
                "feature": col,
                "oriented_auc": oriented_auc,
                "raw_auc_magnetite_positive": auc,
                "d_prime_abs": d_prime,
                "hematite_mean": float(np.mean(h)),
                "magnetite_mean": float(np.mean(m)),
            }
        )
    return pd.DataFrame(rows).sort_values(["oriented_auc", "d_prime_abs"], ascending=[False, False])


def is_control_feature(column: str) -> bool:
    tokens = (
        "thickness_mm",
        "observed_total_counts",
        "expected_total_counts",
        "mean_background_level",
        "mean_throughput",
        "mean_absorption_factor",
        "source_energy_kev",
        "source_wavelength_a",
        "detector_resolution_deg",
        "angular_bin_width_deg",
        "source_bandwidth_fraction",
    )
    return column in tokens


def is_overlap_feature(column: str) -> bool:
    return "overlap" in column


def is_main_feature(column: str) -> bool:
    metadata = {
        "protocol",
        "development_only",
        "shadow_or_final_used",
        "reads_existing_xrt_cubes",
        "sample_id",
        "split",
        "material",
        "random_seed",
        "pose_index",
        "source_id",
    }
    if column in metadata or is_control_feature(column) or is_overlap_feature(column):
        return False
    return column.endswith("_norm") or "hematite_unique" in column or "magnetite_unique" in column or column in {
        "signature_h_over_m_log_ratio",
        "signature_h_minus_m_area",
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(output_dir: Path, gate: dict, model_selection: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# v8A transport-sidecar smoke report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only transport-like detector smoke. This is not a Geant4 C++ integration, not a hardware result, and not a full material matrix.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Best feature AUC: `{gate['best_feature_oriented_auc']:.4f}`",
        f"- Best feature d-prime: `{gate['best_feature_d_prime']:.4f}`",
        f"- Main H/M min recall: `{gate['best_main_hm_min_recall']:.4f}`",
        f"- Main worst-thickness H/M min recall: `{gate['best_main_worst_thickness_hm_min_recall']:.4f}`",
        f"- Control-only H/M min recall: `{gate['control_hm_min_recall']:.4f}`",
        f"- Total-count-only H/M min recall: `{gate['total_count_hm_min_recall']:.4f}`",
        f"- Shuffled-label H/M min recall: `{gate['shuffled_label_hm_min_recall']:.4f}`",
        f"- Overlap-only H/M min recall: `{gate['overlap_only_hm_min_recall']:.4f}`",
        "",
        "## Model Selection",
        "",
        markdown_table(
            model_selection.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False),
            ["method", "hm_min_recall", "hematite_recall", "magnetite_recall", "worst_thickness_hm_min_recall"],
        ),
        "",
        "## Top Features",
        "",
        markdown_table(metrics.head(8), ["feature", "oriented_auc", "d_prime_abs"]),
        "",
        "## Interpretation",
        "",
        "A pass only supports moving to a real Geant4/custom diffraction integration design. It does not allow an H/M accuracy claim, and it does not turn ordinary Rayleigh scattering or attenuation into powder diffraction.",
        "",
    ]
    (output_dir / "v8a_transport_gate_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v8A transport-like diffraction sidecar smoke gate.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_transport_sidecar_smoke")
    parser.add_argument("--random-seed", type=int, default=8901)
    parser.add_argument("--train-seeds", default="6301,6302,6303,6304,6305,6306,6307,6308,6309,6310,6311,6312")
    parser.add_argument("--validation-seeds", default="6401,6402,6403,6404,6405,6406")
    parser.add_argument("--thickness-list", default="3,5,8,10,15,20,30,40")
    parser.add_argument("--poses-per-condition", type=int, default=3)
    parser.add_argument("--source-energies-kev", default="35")
    parser.add_argument("--detector-sector-count", type=int, default=4)
    parser.add_argument("--q-min", type=float, default=0.80)
    parser.add_argument("--q-max", type=float, default=5.80)
    parser.add_argument("--q-bin-width", type=float, default=0.025)
    parser.add_argument("--feature-q-stride", type=float, default=0.050)
    parser.add_argument("--feature-q-half-width", type=float, default=0.080)
    parser.add_argument("--detector-resolution-deg", type=float, default=0.16)
    parser.add_argument("--angular-bin-width-deg", type=float, default=0.20)
    parser.add_argument("--source-bandwidth-fraction", type=float, default=0.006)
    parser.add_argument("--intrinsic-q-sigma", type=float, default=0.040)
    parser.add_argument("--q-calibration-jitter", type=float, default=0.020)
    parser.add_argument("--orientation-sigma", type=float, default=0.80)
    parser.add_argument("--absorption-strength", type=float, default=0.018)
    parser.add_argument("--background-level", type=float, default=0.085)
    parser.add_argument("--background-slope-sigma", type=float, default=0.35)
    parser.add_argument("--counts-scale", type=float, default=900.0)
    parser.add_argument("--read-noise-sigma", type=float, default=0.006)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sk = require_sklearn()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    long_frame, feature_frame, manifest = build_sidecar(args)
    main_feature_cols = [col for col in feature_frame.columns if is_main_feature(col)]
    control_cols = [col for col in feature_frame.columns if is_control_feature(col)]
    total_count_cols = [col for col in ["observed_total_counts", "expected_total_counts", "thickness_mm"] if col in feature_frame.columns]
    overlap_cols = [col for col in feature_frame.columns if is_overlap_feature(col)]

    models = [
        (
            "ExtraTreesTransportMain",
            sk["ExtraTreesClassifier"](n_estimators=600, random_state=8902, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_feature_cols,
            False,
        ),
        (
            "LogisticTransportMain",
            sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=2000, class_weight="balanced", random_state=8903),
            ),
            main_feature_cols,
            False,
        ),
        (
            "ExtraTreesControlOnly",
            sk["ExtraTreesClassifier"](n_estimators=400, random_state=8904, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            control_cols,
            False,
        ),
        (
            "ExtraTreesTotalCountOnly",
            sk["ExtraTreesClassifier"](n_estimators=400, random_state=8905, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            total_count_cols,
            False,
        ),
        (
            "ExtraTreesShuffledTrainLabels",
            sk["ExtraTreesClassifier"](n_estimators=400, random_state=8906, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_feature_cols,
            True,
        ),
        (
            "ExtraTreesOverlapOnly",
            sk["ExtraTreesClassifier"](n_estimators=400, random_state=8907, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            overlap_cols,
            False,
        ),
    ]

    selection_rows = []
    decision_frames = []
    for method_name, estimator, cols, shuffle_labels in models:
        if not cols:
            raise SystemExit(f"No features available for {method_name}.")
        row, decisions = evaluate_model(feature_frame, cols, method_name, estimator, shuffle_train_labels=shuffle_labels)
        selection_rows.append(row)
        decision_frames.append(decisions)
    model_selection = pd.DataFrame(selection_rows)
    validation_decisions = pd.concat(decision_frames, ignore_index=True)
    metrics = observability_metrics(feature_frame, main_feature_cols, sk["roc_auc_score"])

    main_models = model_selection[model_selection["method"].isin(["ExtraTreesTransportMain", "LogisticTransportMain"])]
    best_main = main_models.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False).iloc[0].to_dict()
    control_hm = float(model_selection.loc[model_selection["method"].eq("ExtraTreesControlOnly"), "hm_min_recall"].iloc[0])
    total_count_hm = float(model_selection.loc[model_selection["method"].eq("ExtraTreesTotalCountOnly"), "hm_min_recall"].iloc[0])
    shuffled_hm = float(model_selection.loc[model_selection["method"].eq("ExtraTreesShuffledTrainLabels"), "hm_min_recall"].iloc[0])
    overlap_hm = float(model_selection.loc[model_selection["method"].eq("ExtraTreesOverlapOnly"), "hm_min_recall"].iloc[0])
    best_feature = metrics.iloc[0].to_dict()

    physical_observability_pass = bool(best_feature["oriented_auc"] >= PASS_FEATURE_AUC or best_feature["d_prime_abs"] >= PASS_FEATURE_D_PRIME)
    ml_pass = bool(
        best_main["hm_min_recall"] >= PASS_HM_MIN_RECALL
        and best_main["pairwise_hm_min_recall"] >= PASS_HM_MIN_RECALL
        and best_main["worst_thickness_hm_min_recall"] >= PASS_WORST_THICKNESS_HM_MIN_RECALL
    )
    control_guard_pass = bool(control_hm < MAX_CONTROL_HM_MIN_RECALL)
    total_count_guard_pass = bool(total_count_hm < MAX_TOTAL_COUNT_HM_MIN_RECALL)
    shuffled_guard_pass = bool(shuffled_hm < MAX_SHUFFLED_LABEL_HM_MIN_RECALL)
    overlap_guard_pass = bool(overlap_hm < MAX_OVERLAP_ONLY_HM_MIN_RECALL)
    manifest_guard_pass = bool(
        manifest["development_only"]
        and not manifest["shadow_or_final_used"]
        and not manifest["reads_existing_xrt_cubes"]
        and manifest["bin_axis"] == "q_a_inv"
    )
    gate_passed = bool(
        physical_observability_pass
        and ml_pass
        and control_guard_pass
        and total_count_guard_pass
        and shuffled_guard_pass
        and overlap_guard_pass
        and manifest_guard_pass
    )
    if gate_passed:
        decision = "proceed_to_geant4_custom_diffraction_integration_design"
    elif best_main["hm_min_recall"] < NO_GO_HM_MIN_RECALL:
        decision = "stop_transport_sidecar_write_limitation"
    else:
        decision = "gray_zone_strengthen_detector_background_stress"

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/v8a_transport_sidecar_smoke.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_transport_sidecar_smoke",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "input_data": "hardcoded_reference_powder_peak_table_only",
        "claim_scope": "transport_like_detector_smoke_only_not_geant4_not_hardware_validation_not_full_matrix",
        "gate_passed": gate_passed,
        "decision": decision,
        "physical_observability_pass": physical_observability_pass,
        "ml_pass": ml_pass,
        "control_guard_pass": control_guard_pass,
        "total_count_guard_pass": total_count_guard_pass,
        "shuffled_label_guard_pass": shuffled_guard_pass,
        "overlap_only_guard_pass": overlap_guard_pass,
        "manifest_guard_pass": manifest_guard_pass,
        "uses_q_or_d_axis": True,
        "main_feature_set_excludes_overlap_windows": True,
        "ordinary_geant4_rayleigh_used_as_powder_xrd": False,
        "g4_xray_reflection_used_as_bulk_powder_xrd": False,
        "best_feature": str(best_feature["feature"]),
        "best_feature_oriented_auc": float(best_feature["oriented_auc"]),
        "best_feature_d_prime": float(best_feature["d_prime_abs"]),
        "best_main_method": str(best_main["method"]),
        "best_main_hm_min_recall": float(best_main["hm_min_recall"]),
        "best_main_pairwise_hm_min_recall": float(best_main["pairwise_hm_min_recall"]),
        "best_main_hematite_recall": float(best_main["hematite_recall"]),
        "best_main_magnetite_recall": float(best_main["magnetite_recall"]),
        "best_main_worst_thickness_hm_min_recall": float(best_main["worst_thickness_hm_min_recall"]),
        "control_hm_min_recall": control_hm,
        "total_count_hm_min_recall": total_count_hm,
        "shuffled_label_hm_min_recall": shuffled_hm,
        "overlap_only_hm_min_recall": overlap_hm,
        "thresholds": {
            "pass_feature_auc": PASS_FEATURE_AUC,
            "pass_feature_d_prime": PASS_FEATURE_D_PRIME,
            "pass_hm_min_recall": PASS_HM_MIN_RECALL,
            "pass_worst_thickness_hm_min_recall": PASS_WORST_THICKNESS_HM_MIN_RECALL,
            "max_control_hm_min_recall": MAX_CONTROL_HM_MIN_RECALL,
            "max_total_count_hm_min_recall": MAX_TOTAL_COUNT_HM_MIN_RECALL,
            "max_shuffled_label_hm_min_recall": MAX_SHUFFLED_LABEL_HM_MIN_RECALL,
            "max_overlap_only_hm_min_recall": MAX_OVERLAP_ONLY_HM_MIN_RECALL,
            "no_go_hm_min_recall": NO_GO_HM_MIN_RECALL,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest.update(
        {
            "generated_at_utc": generated_at,
            "feature_count": int(len(feature_frame.columns)),
            "main_feature_count": int(len(main_feature_cols)),
            "control_feature_count": int(len(control_cols)),
            "overlap_only_feature_count": int(len(overlap_cols)),
            "gate_file": "v8a_transport_gate.json",
        }
    )

    long_frame.to_csv(output_dir / "v8a_transport_sidecar_long.csv", index=False, lineterminator="\n")
    feature_frame.to_csv(output_dir / "v8a_transport_sidecar_features.csv", index=False, lineterminator="\n")
    metrics.to_csv(output_dir / "v8a_transport_observability_metrics.csv", index=False, lineterminator="\n")
    model_selection.to_csv(output_dir / "v8a_transport_model_selection.csv", index=False, lineterminator="\n")
    validation_decisions.to_csv(output_dir / "v8a_transport_validation_decisions.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_transport_sidecar_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "v8a_transport_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    write_report(output_dir, gate, model_selection, metrics)

    print(
        "decision={decision} gate_passed={passed} main_hm={hm:.4f} worst_thickness={worst:.4f} control={control:.4f} total_count={total:.4f}".format(
            decision=decision,
            passed=str(gate_passed).lower(),
            hm=gate["best_main_hm_min_recall"],
            worst=gate["best_main_worst_thickness_hm_min_recall"],
            control=control_hm,
            total=total_count_hm,
        )
    )


if __name__ == "__main__":
    main()
