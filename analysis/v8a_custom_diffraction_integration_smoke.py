from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v8a_transport_sidecar_smoke import (
    EPS,
    HM_PAIR,
    MAX_CONTROL_HM_MIN_RECALL,
    MAX_OVERLAP_ONLY_HM_MIN_RECALL,
    MAX_SHUFFLED_LABEL_HM_MIN_RECALL,
    MAX_TOTAL_COUNT_HM_MIN_RECALL,
    NO_GO_HM_MIN_RECALL,
    PASS_FEATURE_AUC,
    PASS_FEATURE_D_PRIME,
    PASS_HM_MIN_RECALL,
    PASS_WORST_THICKNESS_HM_MIN_RECALL,
    POWDER_PEAKS,
    baseline_correct_and_normalize,
    d_from_q,
    ensure_output_dir,
    evaluate_model,
    gaussian,
    is_control_feature,
    is_main_feature,
    is_overlap_feature,
    make_q_axis,
    markdown_table,
    observability_metrics,
    parse_csv_floats,
    parse_csv_ints,
    peak_q_windows,
    q_from_two_theta,
    q_resolution_sigma,
    require_sklearn,
    sector_response,
    two_theta_from_q,
    wavelength_from_energy,
)


MAX_LEAKAGE_OFF_HM_MIN_RECALL = 0.75


def stable_condition_seed(random_seed: int, split_seed: int, thickness: float, pose_index: int, source_index: int, sector_index: int) -> int:
    thickness_offset = int(round(thickness * 100.0))
    return random_seed + split_seed * 97 + thickness_offset * 31 + pose_index * 1009 + source_index * 7919 + sector_index * 211


def stable_diffraction_seed(random_seed: int, material: str, split_seed: int, thickness: float, pose_index: int, source_index: int, sector_index: int) -> int:
    material_offset = 300000 if material == "Magnetite" else 0
    return material_offset + stable_condition_seed(random_seed, split_seed, thickness, pose_index, source_index, sector_index)


def continuum_transport(
    *,
    q_axis: np.ndarray,
    rng: np.random.Generator,
    background_level: float,
    background_slope_sigma: float,
    sector_background_factor: float,
) -> tuple[np.ndarray, float, float]:
    centered_q = (q_axis - float(q_axis.mean())) / max(float(np.ptp(q_axis)), EPS)
    background_slope = rng.normal(0.0, background_slope_sigma)
    continuum = background_level * sector_background_factor * (1.0 + background_slope * centered_q)
    continuum += 0.018 * np.exp(-0.22 * (q_axis - float(q_axis.min()))) * rng.lognormal(mean=0.0, sigma=0.10)
    continuum = np.clip(continuum, background_level * 0.10, None)
    return continuum, float(np.median(continuum)), float(background_slope)


def diffraction_table_signal(
    *,
    q_axis: np.ndarray,
    material: str,
    source_wavelength_a: float,
    sector_index: int,
    sector_orientation_factor: float,
    rng: np.random.Generator,
    detector_resolution_deg: float,
    angular_bin_width_deg: float,
    source_bandwidth_fraction: float,
    intrinsic_q_sigma: float,
    q_calibration_jitter: float,
    orientation_sigma: float,
) -> tuple[np.ndarray, float]:
    q_shift = rng.normal(0.0, q_calibration_jitter)
    signal = np.zeros_like(q_axis, dtype=np.float64)
    for peak_index, (two_theta_deg, relative_intensity) in enumerate(POWDER_PEAKS[material]):
        q_center = q_from_two_theta(two_theta_deg)
        if two_theta_from_q(q_center, source_wavelength_a) is None:
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
        signal += (
            relative_intensity
            * sector_orientation_factor
            * orientation
            * (1.0 + texture_phase)
            * gaussian(q_axis, q_center + q_shift, peak_sigma)
        )
    return signal, float(q_shift)


def simulate_integrated_sector(
    *,
    q_axis: np.ndarray,
    material: str,
    thickness_mm: float,
    source_energy_kev: float,
    source_index: int,
    sector_index: int,
    sector_count: int,
    random_seed: int,
    split_seed: int,
    pose_index: int,
    diffraction_source_enabled: bool,
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
    condition_seed = stable_condition_seed(random_seed, split_seed, thickness_mm, pose_index, source_index, sector_index)
    transport_rng = np.random.default_rng(condition_seed)
    sector = sector_response(sector_index, transport_rng, sector_count)
    continuum, background_effective, background_slope = continuum_transport(
        q_axis=q_axis,
        rng=transport_rng,
        background_level=background_level,
        background_slope_sigma=background_slope_sigma,
        sector_background_factor=sector["sector_background_factor"],
    )
    attenuation = math.exp(-absorption_strength * thickness_mm)
    self_absorption = math.exp(-0.50 * absorption_strength * thickness_mm)

    signal = np.zeros_like(q_axis, dtype=np.float64)
    q_shift = 0.0
    if diffraction_source_enabled:
        diffraction_rng = np.random.default_rng(
            stable_diffraction_seed(random_seed, material, split_seed, thickness_mm, pose_index, source_index, sector_index)
        )
        signal, q_shift = diffraction_table_signal(
            q_axis=q_axis,
            material=material,
            source_wavelength_a=source_wavelength_a,
            sector_index=sector_index,
            sector_orientation_factor=sector["sector_orientation_factor"],
            rng=diffraction_rng,
            detector_resolution_deg=detector_resolution_deg,
            angular_bin_width_deg=angular_bin_width_deg,
            source_bandwidth_fraction=source_bandwidth_fraction,
            intrinsic_q_sigma=intrinsic_q_sigma,
            q_calibration_jitter=q_calibration_jitter,
            orientation_sigma=orientation_sigma,
        )
        signal *= attenuation * self_absorption * sector["sector_throughput"]

    expected_counts = np.clip((continuum + signal) * counts_scale, 0.0, None)
    count_rng = np.random.default_rng(condition_seed + 17)
    observed_counts = count_rng.poisson(expected_counts).astype(np.float64)
    intensity = observed_counts / max(counts_scale, EPS)
    intensity += count_rng.normal(0.0, read_noise_sigma, size=intensity.shape)
    intensity = np.clip(intensity, 0.0, None)
    controls = {
        **sector,
        "source_wavelength_a": float(source_wavelength_a),
        "source_energy_kev": float(source_energy_kev),
        "absorption_factor": float(attenuation),
        "self_absorption_factor": float(self_absorption),
        "background_level_effective": background_effective,
        "expected_total_counts": float(expected_counts.sum()),
        "observed_total_counts": float(observed_counts.sum()),
        "q_calibration_shift_a_inv": float(q_shift),
        "background_slope": background_slope,
        "diffraction_signal_sum": float(signal.sum()),
        "continuum_signal_sum": float(continuum.sum()),
    }
    return intensity.astype(np.float64), controls


def build_integration_sidecar(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    q_axis = make_q_axis(args.q_min, args.q_max, args.q_bin_width)
    source_energies = parse_csv_floats(args.source_energies_kev)
    thicknesses = parse_csv_floats(args.thickness_list)
    split_seed_map = {"train": parse_csv_ints(args.train_seeds), "validation": parse_csv_ints(args.validation_seeds)}
    sector_names = [f"sector_{idx:02d}" for idx in range(args.detector_sector_count)]
    q_windows = peak_q_windows()
    source_enabled = args.diffraction_source == "on"
    protocol = "v8A_custom_diffraction_integration_smoke"
    long_rows: list[dict] = []
    feature_rows: list[dict] = []

    for split, seeds in split_seed_map.items():
        for split_seed in seeds:
            for material in HM_PAIR:
                for thickness in thicknesses:
                    for pose_index in range(args.poses_per_condition):
                        for source_index, source_energy in enumerate(source_energies):
                            sample_id = f"v8a_integration_{args.diffraction_source}_{split}_{material}_{split_seed}_{thickness:g}_{pose_index}_src{source_index}"
                            sector_arrays: dict[str, np.ndarray] = {}
                            sample_controls: list[dict[str, float]] = []
                            for sector_index, sector_name in enumerate(sector_names):
                                intensity, controls = simulate_integrated_sector(
                                    q_axis=q_axis,
                                    material=material,
                                    thickness_mm=thickness,
                                    source_energy_kev=source_energy,
                                    source_index=source_index,
                                    sector_index=sector_index,
                                    sector_count=args.detector_sector_count,
                                    random_seed=args.random_seed,
                                    split_seed=split_seed,
                                    pose_index=pose_index,
                                    diffraction_source_enabled=source_enabled,
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
                                sector_arrays[sector_name] = normalized
                                sample_controls.append(controls)
                                for q_index, (q_value, raw_value, normalized_value) in enumerate(zip(q_axis, intensity, normalized)):
                                    long_rows.append(
                                        {
                                            "protocol": protocol,
                                            "integration_mode": "custom_diffraction_table_source",
                                            "development_only": True,
                                            "shadow_or_final_used": False,
                                            "reads_existing_xrt_cubes": False,
                                            "diffraction_source_enabled": source_enabled,
                                            "custom_diffraction_table_source_enabled": source_enabled,
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
                                            "ordinary_rayleigh_contribution": 0.0,
                                            "g4_xray_reflection_contribution": 0.0,
                                            "diffraction_signal_sum": controls["diffraction_signal_sum"],
                                            "continuum_signal_sum": controls["continuum_signal_sum"],
                                            "raw_intensity": float(raw_value),
                                            "sidecar_intensity_norm": float(normalized_value),
                                        }
                                    )

                            feature_row: dict[str, float | int | str | bool] = {
                                "protocol": protocol,
                                "integration_mode": "custom_diffraction_table_source",
                                "development_only": True,
                                "shadow_or_final_used": False,
                                "reads_existing_xrt_cubes": False,
                                "diffraction_source_enabled": source_enabled,
                                "custom_diffraction_table_source_enabled": source_enabled,
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
                                "mean_diffraction_signal_sum": float(np.mean([item["diffraction_signal_sum"] for item in sample_controls])),
                                "mean_continuum_signal_sum": float(np.mean([item["continuum_signal_sum"] for item in sample_controls])),
                            }
                            stride = max(1, int(round(args.feature_q_stride / args.q_bin_width)))
                            for sector_name, normalized in sector_arrays.items():
                                for q_index in range(0, len(q_axis), stride):
                                    q_token = f"{q_axis[q_index]:.2f}".replace(".", "p")
                                    feature_row[f"{sector_name}_q_{q_token}_norm"] = float(normalized[q_index])
                                for family, centers in q_windows.items():
                                    values = []
                                    for center_q in centers:
                                        mask = np.abs(q_axis - center_q) <= args.feature_q_half_width
                                        values.append(float(normalized[mask].sum()) if mask.any() else 0.0)
                                    feature_row[f"{sector_name}_{family}_area_sum"] = float(np.sum(values))
                                    feature_row[f"{sector_name}_{family}_area_max"] = float(np.max(values) if values else 0.0)

                            h_values = [float(feature_row[f"{sector_name}_hematite_unique_area_sum"]) for sector_name in sector_names]
                            m_values = [float(feature_row[f"{sector_name}_magnetite_unique_area_sum"]) for sector_name in sector_names]
                            overlap_values = [float(feature_row[f"{sector_name}_overlap_area_sum"]) for sector_name in sector_names]
                            feature_row["signature_hematite_unique_area_sum"] = float(np.sum(h_values))
                            feature_row["signature_magnetite_unique_area_sum"] = float(np.sum(m_values))
                            feature_row["signature_overlap_area_sum"] = float(np.sum(overlap_values))
                            feature_row["signature_h_over_m_log_ratio"] = float(math.log((np.sum(h_values) + EPS) / (np.sum(m_values) + EPS)))
                            feature_row["signature_h_minus_m_area"] = float(np.sum(h_values) - np.sum(m_values))
                            feature_rows.append(feature_row)

    long_frame = pd.DataFrame(long_rows)
    feature_frame = pd.DataFrame(feature_rows)
    manifest = {
        "protocol_name": protocol,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_shadow_or_final": False,
        "reads_existing_xrt_cubes": False,
        "input_data": "hardcoded_reference_powder_peak_table_only",
        "integration_mode": "custom_diffraction_table_source",
        "diffraction_source_enabled": source_enabled,
        "custom_diffraction_table_source_enabled": source_enabled,
        "leakage_control_mode": not source_enabled,
        "geant4_role": "geometry_transport_absorption_background_detector_response_only",
        "ordinary_geant4_rayleigh_used_as_powder_xrd": False,
        "g4_xray_reflection_used_as_bulk_powder_xrd": False,
        "ordinary_rayleigh_contribution": 0.0,
        "g4_xray_reflection_contribution": 0.0,
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
        "transport_noise_seed_rule": "material_independent_when_diffraction_source_off",
        "throughput_model": "sector_lognormal_response_with_per_sector_background",
        "absorption_strength": args.absorption_strength,
        "source_bandwidth_fraction": args.source_bandwidth_fraction,
        "counts_scale": args.counts_scale,
        "schema_matches_v8a_transport_sidecar_smoke": True,
        "powder_peaks_two_theta_deg_cu_k_alpha": {
            material: [{"two_theta_deg": c, "relative_weight": w, "q_a_inv": q_from_two_theta(c), "d_a": d_from_q(q_from_two_theta(c))} for c, w in peaks]
            for material, peaks in POWDER_PEAKS.items()
        },
    }
    return long_frame, feature_frame, manifest


def feature_sets(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    main_cols = [col for col in frame.columns if is_main_feature(col)]
    control_cols = [col for col in frame.columns if is_control_feature(col) or col == "mean_continuum_signal_sum"]
    total_count_cols = [col for col in ["observed_total_counts", "expected_total_counts", "thickness_mm"] if col in frame.columns]
    overlap_cols = [col for col in frame.columns if is_overlap_feature(col)]
    return main_cols, control_cols, total_count_cols, overlap_cols


def evaluate_gate(frame: pd.DataFrame, manifest: dict, sk: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_cols, control_cols, total_count_cols, overlap_cols = feature_sets(frame)
    source_enabled = bool(manifest["diffraction_source_enabled"])
    models = [
        (
            "ExtraTreesIntegrationMain",
            sk["ExtraTreesClassifier"](n_estimators=600, random_state=9002, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            False,
        ),
        (
            "LogisticIntegrationMain",
            sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=2000, class_weight="balanced", random_state=9003),
            ),
            main_cols,
            False,
        ),
        (
            "ExtraTreesControlOnly",
            sk["ExtraTreesClassifier"](n_estimators=400, random_state=9004, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            control_cols,
            False,
        ),
        (
            "ExtraTreesTotalCountOnly",
            sk["ExtraTreesClassifier"](n_estimators=400, random_state=9005, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            total_count_cols,
            False,
        ),
        (
            "ExtraTreesShuffledTrainLabels",
            sk["ExtraTreesClassifier"](n_estimators=400, random_state=9006, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            True,
        ),
        (
            "ExtraTreesOverlapOnly",
            sk["ExtraTreesClassifier"](n_estimators=400, random_state=9007, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            overlap_cols,
            False,
        ),
    ]

    rows = []
    decisions = []
    for method_name, estimator, cols, shuffle_labels in models:
        if not cols:
            raise SystemExit(f"No features available for {method_name}.")
        row, method_decisions = evaluate_model(frame, cols, method_name, estimator, shuffle_train_labels=shuffle_labels)
        rows.append(row)
        decisions.append(method_decisions)
    model_selection = pd.DataFrame(rows)
    validation_decisions = pd.concat(decisions, ignore_index=True)
    metrics = observability_metrics(frame, main_cols, sk["roc_auc_score"])
    main_models = model_selection[model_selection["method"].isin(["ExtraTreesIntegrationMain", "LogisticIntegrationMain"])]
    best_main = main_models.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False).iloc[0].to_dict()
    best_feature = metrics.iloc[0].to_dict()
    control_hm = float(model_selection.loc[model_selection["method"].eq("ExtraTreesControlOnly"), "hm_min_recall"].iloc[0])
    total_count_hm = float(model_selection.loc[model_selection["method"].eq("ExtraTreesTotalCountOnly"), "hm_min_recall"].iloc[0])
    shuffled_hm = float(model_selection.loc[model_selection["method"].eq("ExtraTreesShuffledTrainLabels"), "hm_min_recall"].iloc[0])
    overlap_hm = float(model_selection.loc[model_selection["method"].eq("ExtraTreesOverlapOnly"), "hm_min_recall"].iloc[0])

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
        and not manifest["ordinary_geant4_rayleigh_used_as_powder_xrd"]
        and not manifest["g4_xray_reflection_used_as_bulk_powder_xrd"]
    )
    leakage_control_passed = bool((not source_enabled) and best_main["hm_min_recall"] < MAX_LEAKAGE_OFF_HM_MIN_RECALL)
    integration_gate_passed = bool(
        source_enabled
        and physical_observability_pass
        and ml_pass
        and control_guard_pass
        and total_count_guard_pass
        and shuffled_guard_pass
        and overlap_guard_pass
        and manifest_guard_pass
    )
    if source_enabled:
        gate_passed = integration_gate_passed
        if gate_passed:
            decision = "proceed_to_small_development_sidecar_training"
        elif best_main["hm_min_recall"] < NO_GO_HM_MIN_RECALL:
            decision = "stop_custom_diffraction_integration_write_limitation"
        else:
            decision = "gray_zone_strengthen_integration_stress"
    else:
        gate_passed = leakage_control_passed and manifest_guard_pass
        decision = "leakage_control_passed_no_diffraction_signal" if gate_passed else "leakage_control_failed_possible_transport_leakage"

    gate = {
        "generated_by": "analysis/v8a_custom_diffraction_integration_smoke.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": manifest["protocol_name"],
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "diffraction_source_enabled": source_enabled,
        "custom_diffraction_table_source_enabled": source_enabled,
        "claim_scope": "custom_diffraction_table_integration_smoke_only_not_full_matrix_not_shadow_final",
        "gate_passed": gate_passed,
        "integration_gate_passed": integration_gate_passed,
        "leakage_control_passed": leakage_control_passed,
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
            "max_leakage_off_hm_min_recall": MAX_LEAKAGE_OFF_HM_MIN_RECALL,
            "no_go_hm_min_recall": NO_GO_HM_MIN_RECALL,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    return gate, model_selection, validation_decisions, metrics


def write_report(output_dir: Path, gate: dict, model_selection: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# v8A custom diffraction integration smoke report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only custom diffraction/table sidecar integration smoke. This is not a full Geant4 material matrix and not shadow/final evidence.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Diffraction source enabled: `{str(gate['diffraction_source_enabled']).lower()}`",
        f"- Integration gate passed: `{str(gate['integration_gate_passed']).lower()}`",
        f"- Leakage control passed: `{str(gate['leakage_control_passed']).lower()}`",
        f"- Main H/M min recall: `{gate['best_main_hm_min_recall']:.4f}`",
        f"- Worst-thickness H/M min recall: `{gate['best_main_worst_thickness_hm_min_recall']:.4f}`",
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
    ]
    (output_dir / "v8a_integration_gate_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v8A custom diffraction/table sidecar integration smoke.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_custom_diffraction_integration_smoke")
    parser.add_argument("--diffraction-source", choices=["on", "off"], default="on")
    parser.add_argument("--random-seed", type=int, default=9001)
    parser.add_argument("--train-seeds", default="6501,6502,6503,6504,6505,6506,6507,6508,6509,6510,6511,6512")
    parser.add_argument("--validation-seeds", default="6601,6602,6603,6604,6605,6606")
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

    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    sk = require_sklearn()

    long_frame, feature_frame, manifest = build_integration_sidecar(args)
    gate, model_selection, validation_decisions, metrics = evaluate_gate(feature_frame, manifest, sk)
    manifest.update(
        {
            "generated_at_utc": gate["generated_at_utc"],
            "feature_count": int(len(feature_frame.columns)),
            "main_feature_count": int(len(feature_sets(feature_frame)[0])),
            "control_feature_count": int(len(feature_sets(feature_frame)[1])),
            "overlap_only_feature_count": int(len(feature_sets(feature_frame)[3])),
            "gate_file": "v8a_integration_gate.json",
        }
    )

    long_frame.to_csv(output_dir / "v8a_integration_sidecar_long.csv", index=False, lineterminator="\n")
    feature_frame.to_csv(output_dir / "v8a_integration_sidecar_features.csv", index=False, lineterminator="\n")
    metrics.to_csv(output_dir / "v8a_integration_observability_metrics.csv", index=False, lineterminator="\n")
    model_selection.to_csv(output_dir / "v8a_integration_model_selection.csv", index=False, lineterminator="\n")
    validation_decisions.to_csv(output_dir / "v8a_integration_validation_decisions.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_integration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "v8a_integration_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(output_dir, gate, model_selection, metrics)
    print(
        "decision={decision} gate_passed={passed} source={source} main_hm={hm:.4f} leakage_control={leakage}".format(
            decision=gate["decision"],
            passed=str(gate["gate_passed"]).lower(),
            source=args.diffraction_source,
            hm=gate["best_main_hm_min_recall"],
            leakage=str(gate["leakage_control_passed"]).lower(),
        )
    )


if __name__ == "__main__":
    main()
