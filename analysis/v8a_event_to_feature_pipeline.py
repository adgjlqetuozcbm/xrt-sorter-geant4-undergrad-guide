from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


HC_KEV_A = 12.398419843320026
DEFAULT_PROFILE = "v8a_custom_diffraction_g4_smoke"
DEFAULT_SCHEMA_CONTRACT = "analysis/configs/v8a_diffraction_output_schema_contract.json"
DEFAULT_PEAK_MANIFEST = "source_models/config/diffraction_peak_tables/hm_powder_peaks_cif_or_literature_v8a_manifest.json"
DEFAULT_OUTPUT_DIR = "results/accuracy_v3/v8a_event_to_feature_smoke"
DEFAULT_BIN_AXIS = "q_a_inv"
DEFAULT_Q_BIN_WIDTH = 0.05
DEFAULT_D_BIN_WIDTH = 0.05
DEFAULT_PEAK_WINDOW_A_INV = 0.075
OVERLAP_Q_TOLERANCE_A_INV = 0.08
SOURCE_OFF_SIGNAL_MAX = 0.01


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_key_value(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip().lower()] = value.strip()
    return values


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def safe_int(value: Any, fallback: int = 0) -> int:
    try:
        if value is None or value == "":
            return fallback
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def stable_sample_id(row: dict[str, str]) -> str:
    row_index = safe_int(row.get("row_index"), 0)
    parts = [
        row.get("profile", DEFAULT_PROFILE),
        str(row_index),
        row.get("split", "unknown_split"),
        row.get("run_role", "material"),
        f"t{safe_float(row.get('thickness_mm'), 0.0):g}",
        f"seed{safe_int(row.get('random_seed'), 0)}",
        row.get("source_id", "unknown_source"),
        row.get("stress_label", "unknown_stress"),
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"v8a_evt_{row_index:04d}_{digest}"


def normalize_source_mode(value: str) -> str:
    raw = value.strip().lower()
    if raw in {"on", "custom_diffraction_on", "diffraction_on"}:
        return "custom_diffraction_on"
    if raw in {"off", "custom_diffraction_off", "diffraction_off", "leakage_off"}:
        return "custom_diffraction_off"
    return raw or "unknown"


def safe_feature_id(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", value.strip()).strip("_").lower()
    return value or "unnamed"


def wavelength_from_energy(energy_kev: float) -> float:
    if energy_kev <= 0.0:
        return 0.0
    return HC_KEV_A / energy_kev


def q_from_recorded_scatter_angle(theta_deg: float, wavelength_a: float) -> float:
    if theta_deg < 0.0 or wavelength_a <= 0.0:
        return -1.0
    return 4.0 * math.pi * math.sin(math.radians(theta_deg) / 2.0) / wavelength_a


def d_from_q(q_a_inv: float) -> float:
    if q_a_inv <= 0.0:
        return 0.0
    return 2.0 * math.pi / q_a_inv


def q_bin_center(q_a_inv: float, width: float) -> float:
    if q_a_inv < 0.0:
        return -1.0
    return round((math.floor(q_a_inv / width) * width) + 0.5 * width, 6)


def d_bin_center(d_a: float, width: float) -> float:
    if d_a <= 0.0:
        return 0.0
    return round((math.floor(d_a / width) * width) + 0.5 * width, 6)


def bin_centers(q_a_inv: float, bin_axis: str, q_bin_width: float, d_bin_width: float) -> tuple[float, float]:
    if q_a_inv <= 0.0:
        return -1.0, 0.0
    if bin_axis == "d_a":
        d_center = d_bin_center(d_from_q(q_a_inv), d_bin_width)
        q_center = round(2.0 * math.pi / d_center, 6) if d_center > 0.0 else -1.0
        return q_center, d_center
    q_center = q_bin_center(q_a_inv, q_bin_width)
    return q_center, d_from_q(q_center)


def detector_sector(hit: dict[str, str]) -> str:
    detector_id = hit.get("detector_id", "unknown") or "unknown"
    x = safe_float(hit.get("x_mm"), 0.0)
    y = safe_float(hit.get("y_mm"), 0.0)
    z = safe_float(hit.get("z_mm"), 0.0)
    if detector_id == "transmission":
        y_side = "yp" if y >= 0.0 else "yn"
        z_side = "zp" if z >= 0.0 else "zn"
        return f"transmission_{y_side}_{z_side}"
    if detector_id == "side_scatter":
        x_side = "xp" if x >= 0.0 else "xn"
        z_side = "zp" if z >= 0.0 else "zn"
        return f"side_scatter_{x_side}_{z_side}"
    return f"{safe_feature_id(detector_id)}_all"


def output_paths_from_config(project_root: Path, config: dict[str, str]) -> tuple[Path, Path, Path]:
    output_dir = Path(config.get("output_dir", "."))
    if not output_dir.is_absolute():
        output_dir = project_root / "build" / output_dir
    output_prefix = config["output_prefix"]
    return (
        output_dir / f"{output_prefix}_events.csv",
        output_dir / f"{output_prefix}_hits.csv",
        output_dir / f"{output_prefix}_metadata.json",
    )


def completed_smoke_rows(project_root: Path, profile: str) -> list[dict[str, str]]:
    matrix_path = project_root / "source_models" / "config" / "material_sorting_matrix" / profile / "material_sorting_matrix.csv"
    status_path = project_root / "results" / "material_sorting" / f"run_status_{profile}.csv"
    matrix_rows = read_csv(matrix_path)
    status_rows = read_csv(status_path)
    matrix_by_index = {row.get("row_index", ""): row for row in matrix_rows}
    completed: list[dict[str, str]] = []
    for status in status_rows:
        if status.get("returncode") != "0":
            continue
        merged = dict(matrix_by_index.get(status.get("row_index", ""), {}))
        merged.update({key: value for key, value in status.items() if value != ""})
        if not merged.get("split"):
            merged["split"] = "unknown"
        completed.append(merged)
    return sorted(completed, key=lambda row: safe_int(row.get("row_index"), 0))


def load_peaks(peak_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    peaks: list[dict[str, Any]] = []
    for material_block in peak_manifest.get("materials", []):
        material = str(material_block.get("material", "unknown"))
        for peak in material_block.get("peaks", []):
            peak_id = str(peak.get("peak_id", "unknown_peak"))
            peaks.append(
                {
                    "material": material,
                    "peak_id": peak_id,
                    "safe_peak_id": safe_feature_id(peak_id),
                    "q_a_inv": safe_float(peak.get("q_a_inv")),
                    "d_a": safe_float(peak.get("d_a")),
                    "relative_intensity": safe_float(peak.get("relative_intensity"), 0.0),
                }
            )
    return peaks


def overlap_peak_ids(peaks: list[dict[str, Any]], tolerance: float) -> set[str]:
    overlap: set[str] = set()
    for left in peaks:
        for right in peaks:
            if left["material"] == right["material"]:
                continue
            if abs(float(left["q_a_inv"]) - float(right["q_a_inv"])) <= tolerance:
                overlap.add(str(left["peak_id"]))
                overlap.add(str(right["peak_id"]))
    return overlap


def classify_peak_windows(q_value: float, peaks: list[dict[str, Any]], peak_window: float) -> list[dict[str, Any]]:
    return [peak for peak in peaks if abs(q_value - float(peak["q_a_inv"])) <= peak_window]


def class_counts(rows: list[dict[str, Any]], subset_key: str | None = None, subset_value: str | None = None) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if subset_key is not None and row.get(subset_key) != subset_value:
            continue
        counts[str(row.get("material", ""))] += 1
    return dict(counts)


def effect_size_by_material(rows: list[dict[str, Any]], feature: str, *, source_mode: str | None = None) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if source_mode is not None and row.get("source_mode") != source_mode:
            continue
        grouped[str(row.get("material", ""))].append(safe_float(row.get(feature), 0.0))
    if len(grouped) != 2 or any(len(values) < 2 for values in grouped.values()):
        return {
            "feature": feature,
            "source_mode": source_mode or "all",
            "status": "not_evaluable_insufficient_balanced_support",
            "metric": "effect_size_abs",
            "value": "",
            "effect_size_abs": "",
        }
    materials = sorted(grouped)
    left_values = grouped[materials[0]]
    right_values = grouped[materials[1]]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_var = sum((value - left_mean) ** 2 for value in left_values) / max(len(left_values) - 1, 1)
    right_var = sum((value - right_mean) ** 2 for value in right_values) / max(len(right_values) - 1, 1)
    pooled = math.sqrt(0.5 * (left_var + right_var))
    effect = abs(left_mean - right_mean) / max(pooled, 1e-12)
    left_key = f"{safe_feature_id(materials[0])}_mean"
    right_key = f"{safe_feature_id(materials[1])}_mean"
    left_rounded = round(left_mean, 8)
    right_rounded = round(right_mean, 8)
    return {
        "feature": feature,
        "source_mode": source_mode or "all",
        "status": "audit_only_not_training_metric",
        "metric": "effect_size_abs",
        "value": round(effect, 6),
        "effect_size_abs": round(effect, 6),
        left_key: left_rounded,
        right_key: right_rounded,
        "details": f"{left_key}={left_rounded}; {right_key}={right_rounded}",
    }


def build_sidecar_tables(
    *,
    project_root: Path,
    profile: str,
    schema_contract: dict[str, Any],
    peak_manifest: dict[str, Any],
    bin_axis: str,
    q_bin_width: float,
    d_bin_width: float,
    peak_window: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    required_hit_fields = set(schema_contract["geant4_hit_csv_fields"]["current_required"])
    required_event_fields = set(schema_contract["geant4_event_csv_fields"]["current_required"])
    peaks = load_peaks(peak_manifest)
    overlap_ids = overlap_peak_ids(peaks, OVERLAP_Q_TOLERANCE_A_INV)
    rows = completed_smoke_rows(project_root, profile)
    long_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    input_checks: list[dict[str, Any]] = []

    for row in rows:
        config_path = as_project_path(project_root, row["config_path"])
        config = read_key_value(config_path)
        events_path, hits_path, metadata_path = output_paths_from_config(project_root, config)
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            events_path = Path(metadata.get("event_file", events_path))
            hits_path = Path(metadata.get("hit_file", hits_path))
        else:
            metadata = {}

        hit_header = set(read_csv_header(hits_path))
        event_header = set(read_csv_header(events_path))
        missing_hit = sorted(required_hit_fields - hit_header)
        missing_event = sorted(required_event_fields - event_header)
        if missing_hit or missing_event:
            raise RuntimeError(
                f"Schema mismatch for row {row.get('row_index')}: missing_hit={missing_hit} missing_event={missing_event}"
            )

        hits = read_csv(hits_path)
        events = read_csv(events_path)
        n_events = safe_int(metadata.get("n_events"), len(events))
        material = row.get("material", metadata.get("ore_primary_material", "unknown"))
        split = row.get("split", "unknown")
        source_mode = normalize_source_mode(row.get("source_mode", ""))
        source_mode_raw = row.get("source_mode", "")
        source_energy_kev = safe_float(row.get("source_energy_kev"), safe_float(metadata.get("mono_energy_keV"), 0.0))
        source_wavelength_a = wavelength_from_energy(source_energy_kev)
        sample_id = stable_sample_id(row)
        analysis_peak_table_id = str(peak_manifest.get("peak_table_id", "unknown"))
        source_peak_table_id = str(row.get("peak_table_id", analysis_peak_table_id))
        thickness_mm = safe_float(row.get("thickness_mm"), safe_float(metadata.get("ore_thickness_mm"), 0.0))
        random_seed = safe_int(row.get("random_seed"), safe_int(metadata.get("random_seed"), 0))
        detector_resolution_deg = 0.0
        angular_bin_width_deg = 0.0
        pose_index = safe_int(row.get("pose_index"), safe_int(metadata.get("pose_index"), 0))

        bin_agg: dict[tuple[str, str, float], dict[str, Any]] = {}
        total_hit_count = 0
        primary_hit_count = 0
        direct_primary_count = 0
        scattered_primary_count = 0
        high_angle_primary_count = 0
        peak_window_union_intensity = 0.0
        overlap_window_union_intensity = 0.0

        for hit in hits:
            total_hit_count += 1
            is_primary = safe_int(hit.get("is_primary"), 0) == 1
            is_direct = safe_int(hit.get("is_direct_primary"), 0) == 1
            is_scattered = safe_int(hit.get("is_scattered_primary"), 0) == 1
            theta_deg = safe_float(hit.get("theta_deg"), -1.0)
            energy_kev = safe_float(hit.get("photon_energy_keV"), source_energy_kev)
            wavelength_a = wavelength_from_energy(energy_kev)
            q_value = q_from_recorded_scatter_angle(theta_deg, wavelength_a)
            if is_primary:
                primary_hit_count += 1
            if is_direct:
                direct_primary_count += 1
            if is_scattered:
                scattered_primary_count += 1
            if is_primary and theta_deg >= 2.0:
                high_angle_primary_count += 1
            if q_value <= 0.0:
                continue

            q_center, d_center = bin_centers(q_value, bin_axis, q_bin_width, d_bin_width)
            sector = detector_sector(hit)
            detector_id = hit.get("detector_id", "unknown") or "unknown"
            key = (detector_id, sector, q_center)
            if key not in bin_agg:
                bin_agg[key] = {
                    "sample_id": sample_id,
                    "split": split,
                    "material": material,
                    "random_seed": random_seed,
                    "thickness_mm": thickness_mm,
                    "pose_index": pose_index,
                    "source_id": row.get("source_id", ""),
                    "source_mode": source_mode,
                    "source_energy_kev": source_energy_kev,
                    "source_wavelength_a": source_wavelength_a,
                    "peak_table_id": analysis_peak_table_id,
                    "source_peak_table_id": source_peak_table_id,
                    "bin_axis": bin_axis,
                    "q_bin_center_a_inv": q_center,
                    "d_bin_center_a": d_center,
                    "detector_sector": sector,
                    "detector_id_source": detector_id,
                    "hit_count": 0,
                    "primary_hit_count": 0,
                    "sidecar_intensity_raw": 0.0,
                    "sidecar_intensity_norm": 0.0,
                    "background_level_effective": 0.0,
                    "throughput": 1.0,
                    "detector_resolution_deg": detector_resolution_deg,
                    "angular_bin_width_deg": angular_bin_width_deg,
                    "absorption_factor": 1.0,
                }
            bin_agg[key]["hit_count"] += 1
            bin_agg[key]["primary_hit_count"] += 1 if is_primary else 0
            bin_agg[key]["sidecar_intensity_raw"] += 1.0

        for agg_row in bin_agg.values():
            agg_row["sidecar_intensity_norm"] = round(
                safe_float(agg_row["primary_hit_count"]) / max(n_events, 1),
                10,
            )
            long_rows.append(agg_row)

        feature_row: dict[str, Any] = {
            "sample_id": sample_id,
            "split": split,
            "material": material,
            "random_seed": random_seed,
            "thickness_mm": thickness_mm,
            "pose_index": pose_index,
            "source_id": row.get("source_id", ""),
            "source_mode": source_mode,
            "source_mode_raw": source_mode_raw,
            "source_energy_kev": source_energy_kev,
            "source_wavelength_a": round(source_wavelength_a, 10),
            "peak_table_id": analysis_peak_table_id,
            "source_peak_table_id": source_peak_table_id,
            "bin_axis": bin_axis,
            "stress_label": row.get("stress_label", ""),
            "development_only": boolish(row.get("development_only", True)),
            "shadow_or_final_used": boolish(row.get("shadow_or_final_used", False)),
            "row_index": row.get("row_index", ""),
            "events_path": relpath(events_path, project_root),
            "hits_path": relpath(hits_path, project_root),
            "metadata_path": relpath(metadata_path, project_root),
            "control_total_count_hit_count": total_hit_count,
            "control_total_count_primary_hit_count": primary_hit_count,
            "control_total_count_norm": round(total_hit_count / max(n_events, 1), 10),
            "control_high_angle_primary_norm": round(high_angle_primary_count / max(n_events, 1), 10),
            "control_direct_primary_norm": round(direct_primary_count / max(n_events, 1), 10),
            "control_scattered_primary_norm": round(scattered_primary_count / max(n_events, 1), 10),
            "control_thickness_pose_thickness_mm": thickness_mm,
            "control_thickness_pose_pose_index": pose_index,
            "control_source_off_flag": 1 if source_mode == "custom_diffraction_off" else 0,
        }
        for peak in peaks:
            feature_row[f"diffraction_peak_{peak['safe_peak_id']}_norm"] = 0.0

        for agg_row in bin_agg.values():
            q_center = safe_float(agg_row["q_bin_center_a_inv"])
            intensity = safe_float(agg_row["sidecar_intensity_norm"])
            matched_peaks = classify_peak_windows(q_center, peaks, peak_window)
            if matched_peaks:
                peak_window_union_intensity += intensity
            if any(str(peak["peak_id"]) in overlap_ids for peak in matched_peaks):
                overlap_window_union_intensity += intensity
            for peak in matched_peaks:
                feature_row[f"diffraction_peak_{peak['safe_peak_id']}_norm"] += intensity

        for key, value in list(feature_row.items()):
            if key.startswith("diffraction_peak_"):
                feature_row[key] = round(safe_float(value), 10)

        hematite_unique_sum = 0.0
        magnetite_unique_sum = 0.0
        for peak in peaks:
            value = safe_float(feature_row[f"diffraction_peak_{peak['safe_peak_id']}_norm"])
            if str(peak["peak_id"]) in overlap_ids:
                continue
            if peak["material"] == "Hematite":
                hematite_unique_sum += value
            elif peak["material"] == "Magnetite":
                magnetite_unique_sum += value
        all_unique_sum = hematite_unique_sum + magnetite_unique_sum
        feature_row["diffraction_window_hematite_unique_sum"] = round(hematite_unique_sum, 10)
        feature_row["diffraction_window_magnetite_unique_sum"] = round(magnetite_unique_sum, 10)
        feature_row["diffraction_window_all_peaks_sum"] = round(peak_window_union_intensity, 10)
        feature_row["diffraction_ratio_hm_unique_balance"] = round(
            (hematite_unique_sum - magnetite_unique_sum) / max(all_unique_sum, 1e-12),
            10,
        )
        feature_row["control_overlap_only_peak_norm"] = round(overlap_window_union_intensity, 10)
        feature_rows.append(feature_row)

        input_checks.append(
            {
                "row_index": row.get("row_index", ""),
                "sample_id": sample_id,
                "events_path": relpath(events_path, project_root),
                "hits_path": relpath(hits_path, project_root),
                "metadata_path": relpath(metadata_path, project_root),
                "hit_rows": len(hits),
                "event_rows": len(events),
                "n_events_metadata": n_events,
                "hit_schema_ok": not missing_hit,
                "event_schema_ok": not missing_event,
                "source_mode": source_mode,
                "peak_table_id": analysis_peak_table_id,
                "source_peak_table_id": source_peak_table_id,
            }
        )

    return long_rows, feature_rows, input_checks, {"peak_count": len(peaks), "overlap_peak_count": len(overlap_ids)}


def build_control_audit(feature_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    on_rows = [row for row in feature_rows if row.get("source_mode") == "custom_diffraction_on"]
    off_rows = [row for row in feature_rows if row.get("source_mode") == "custom_diffraction_off"]
    on_signal = [safe_float(row.get("diffraction_window_all_peaks_sum"), 0.0) for row in on_rows]
    off_signal = [safe_float(row.get("diffraction_window_all_peaks_sum"), 0.0) for row in off_rows]
    on_median = median(on_signal) if on_signal else 0.0
    off_median = median(off_signal) if off_signal else 0.0
    off_max = max(off_signal) if off_signal else 0.0
    source_on_signal_gt_source_off = bool(on_signal and off_signal and on_median > off_median)
    source_off_low = bool(off_signal and off_max <= SOURCE_OFF_SIGNAL_MAX)
    split_counts: dict[str, int] = defaultdict(int)
    for row in feature_rows:
        split_counts[str(row.get("split", ""))] += 1
    source_on_counts = class_counts(feature_rows, "source_mode", "custom_diffraction_on")
    source_off_counts = class_counts(feature_rows, "source_mode", "custom_diffraction_off")
    has_validation_split = any(split not in {"train", "unknown", ""} for split in split_counts)
    balanced_training_support = (
        len(source_on_counts) == 2
        and len(source_off_counts) == 2
        and min(source_on_counts.values(), default=0) >= 3
        and min(source_off_counts.values(), default=0) >= 2
        and has_validation_split
    )

    rng = random.Random(8801)
    labels = [row.get("material", "") for row in feature_rows]
    shuffled = labels[:]
    rng.shuffle(shuffled)
    shuffled_matches = sum(1 for left, right in zip(labels, shuffled) if left == right)

    total_count_audit = effect_size_by_material(feature_rows, "control_total_count_norm", source_mode="custom_diffraction_on")
    overlap_audit = effect_size_by_material(feature_rows, "control_overlap_only_peak_norm", source_mode="custom_diffraction_on")
    if not balanced_training_support:
        total_count_audit["status"] = "audit_only_training_blocked_by_tiny_support"
        overlap_audit["status"] = "audit_only_training_blocked_by_tiny_support"

    rows = [
        {
            "control_group": "source_on_vs_source_off",
            "status": "passed" if source_on_signal_gt_source_off else "failed_or_not_evaluable",
            "metric": "median_diffraction_window_all_peaks_sum",
            "value": round(on_median - off_median, 10),
            "details": f"source_on_median={on_median:.10f}; source_off_median={off_median:.10f}",
        },
        {
            "control_group": "source_off",
            "status": "passed_low_signal" if source_off_low else "failed_or_not_evaluable",
            "metric": "max_diffraction_window_all_peaks_sum",
            "value": round(off_max, 10),
            "details": f"threshold={SOURCE_OFF_SIGNAL_MAX}",
        },
        {
            "control_group": "total_count_only",
            **total_count_audit,
        },
        {
            "control_group": "overlap_only",
            **overlap_audit,
        },
        {
            "control_group": "thickness_pose_lineage",
            "status": "recorded_for_audit_not_model_input",
            "metric": "unique_thicknesses",
            "value": len({row.get("thickness_mm") for row in feature_rows}),
            "details": f"split_counts={dict(split_counts)}",
        },
        {
            "control_group": "shuffled_label",
            "status": "not_evaluable_without_training_split",
            "metric": "deterministic_shuffle_label_match_fraction",
            "value": round(shuffled_matches / max(len(labels), 1), 6),
            "details": "diagnostic only; no model trained from this 12-row boundary smoke",
        },
        {
            "control_group": "feature_input_leakage",
            "status": "passed",
            "metric": "main_feature_prefix",
            "value": "diffraction_",
            "details": "lineage columns material/source_id/sample_id/random_seed/thickness/pose are excluded from declared main feature inputs",
        },
    ]
    summary = {
        "source_on_rows": len(on_rows),
        "source_off_rows": len(off_rows),
        "source_on_counts_by_material": source_on_counts,
        "source_off_counts_by_material": source_off_counts,
        "split_counts": dict(split_counts),
        "source_on_signal_median": on_median,
        "source_off_signal_median": off_median,
        "source_off_signal_max": off_max,
        "source_on_signal_gt_source_off": source_on_signal_gt_source_off,
        "source_off_low": source_off_low,
        "balanced_training_support": balanced_training_support,
        "has_validation_split": has_validation_split,
    }
    return rows, summary


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path: Path, gate: dict[str, Any], audit_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# v8A event-to-feature schema gate report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only event/hit-to-sidecar conversion from the completed v8A Geant4 boundary smoke. This report is not H/M accuracy, hardware validation, final sorter performance, or manuscript-grade powder XRD evidence.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Schema/control gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Tiny training gate allowed: `{str(gate['tiny_training_gate_allowed']).lower()}`",
        f"- Development only: `{str(gate['development_only']).lower()}`",
        f"- Shadow/final used: `{str(gate['shadow_or_final_used']).lower()}`",
        f"- Bin axis: `{gate['bin_axis']}`",
        f"- Peak table: `{gate['peak_table_id']}`",
        "",
        "## Counts",
        "",
        f"- Samples: `{manifest['sample_count']}`",
        f"- Long rows: `{manifest['long_row_count']}`",
        f"- Feature columns: `{manifest['feature_column_count']}`",
        f"- Source-on rows: `{gate['control_summary']['source_on_rows']}`",
        f"- Source-off rows: `{gate['control_summary']['source_off_rows']}`",
        "",
        "## Control Audit",
        "",
        markdown_table(audit_rows, ["control_group", "status", "metric", "value", "details"]),
        "",
        "## Stop Conditions",
        "",
    ]
    if gate["stop_reasons"]:
        for reason in gate["stop_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- None for the schema/control layer.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "The output may be used to inspect whether recorded Geant4 hits can be mapped into q/d sidecar features under the v8A contract. It must not be cited as product accuracy, ordinary XRT H/M sorting, final validation, or publishable powder-XRD simulation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert completed v8A Geant4 boundary-smoke event/hit outputs into tiny sidecar features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--schema-contract", default=DEFAULT_SCHEMA_CONTRACT)
    parser.add_argument("--peak-manifest", default=DEFAULT_PEAK_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bin-axis", choices=["q_a_inv", "d_a"], default=DEFAULT_BIN_AXIS)
    parser.add_argument("--q-bin-width", type=float, default=DEFAULT_Q_BIN_WIDTH)
    parser.add_argument("--d-bin-width", type=float, default=DEFAULT_D_BIN_WIDTH)
    parser.add_argument("--peak-window-a-inv", type=float, default=DEFAULT_PEAK_WINDOW_A_INV)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = as_project_path(project_root, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty; pass --overwrite to replace deterministic smoke outputs: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_path = as_project_path(project_root, args.schema_contract)
    peak_manifest_path = as_project_path(project_root, args.peak_manifest)
    schema_contract = read_json(schema_path)
    peak_manifest = read_json(peak_manifest_path)
    allowed_axes = set(schema_contract["canonical_axis"]["allowed"])
    if args.bin_axis not in allowed_axes:
        raise SystemExit(f"bin_axis={args.bin_axis} is not allowed by schema contract: {sorted(allowed_axes)}")

    long_rows, feature_rows, input_checks, peak_summary = build_sidecar_tables(
        project_root=project_root,
        profile=args.profile,
        schema_contract=schema_contract,
        peak_manifest=peak_manifest,
        bin_axis=args.bin_axis,
        q_bin_width=args.q_bin_width,
        d_bin_width=args.d_bin_width,
        peak_window=args.peak_window_a_inv,
    )
    if not feature_rows:
        raise SystemExit("No completed boundary-smoke rows were converted.")

    required_long_fields = schema_contract["sidecar_long_table_fields"]["required"]
    long_fieldnames = required_long_fields
    feature_lineage = schema_contract["feature_table_groups"]["lineage"]
    feature_lineage_fields = feature_lineage.get("required_fields", feature_lineage.get("required", []))
    feature_fieldnames: list[str] = []
    for field in feature_lineage_fields:
        if field not in feature_fieldnames:
            feature_fieldnames.append(field)
    for row in feature_rows:
        for key in row:
            if key not in feature_fieldnames:
                feature_fieldnames.append(key)

    write_csv(output_dir / "v8a_event_sidecar_long.csv", long_rows, long_fieldnames)
    write_csv(output_dir / "v8a_event_sidecar_features.csv", feature_rows, feature_fieldnames)
    write_csv(output_dir / "v8a_event_input_schema_audit.csv", input_checks)
    control_audit, control_summary = build_control_audit(feature_rows)
    write_csv(output_dir / "v8a_event_control_audit.csv", control_audit)

    shadow_or_final_used = any(boolish(row.get("shadow_or_final_used", False)) for row in feature_rows)
    development_only = all(boolish(row.get("development_only", True)) for row in feature_rows)
    peak_table_id = str(peak_manifest.get("peak_table_id", "unknown"))
    peak_id_matches = all(row.get("peak_table_id") == peak_table_id for row in feature_rows)
    source_peak_table_ids = sorted({str(row.get("source_peak_table_id", "")) for row in feature_rows})
    source_peak_table_matches_analysis = source_peak_table_ids == [peak_table_id]
    long_schema_ok = set(required_long_fields).issubset(long_rows[0].keys()) if long_rows else False
    source_lineage_ok = control_summary["source_on_rows"] > 0 and control_summary["source_off_rows"] > 0
    schema_gate_passed = bool(
        long_schema_ok
        and peak_id_matches
        and args.bin_axis in allowed_axes
        and development_only
        and not shadow_or_final_used
        and source_lineage_ok
        and control_summary["source_on_signal_gt_source_off"]
        and control_summary["source_off_low"]
    )
    tiny_training_gate_allowed = bool(schema_gate_passed and control_summary["balanced_training_support"])
    stop_reasons: list[str] = []
    if not control_summary["balanced_training_support"]:
        stop_reasons.append(
            "Tiny training gate is blocked: completed boundary smoke lacks a validation split and balanced H/M source-off/source-on support."
        )
    if not control_summary["source_off_low"]:
        stop_reasons.append("Source-off peak-window signal is not low enough for leakage control.")
    if shadow_or_final_used:
        stop_reasons.append("Shadow/final usage was detected.")
    if not peak_id_matches:
        stop_reasons.append("Analysis peak table id does not match the requested peak manifest.")

    decision = (
        "schema_control_gate_passed_ready_for_tiny_training_gate"
        if tiny_training_gate_allowed
        else "schema_control_gate_passed_training_gate_blocked_by_tiny_boundary_support"
        if schema_gate_passed
        else "stop_or_rework_event_to_feature_schema_control_gate"
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    main_feature_columns = [field for field in feature_fieldnames if field.startswith("diffraction_")]
    control_feature_columns = [field for field in feature_fieldnames if field.startswith("control_")]
    manifest = {
        "generated_by": "analysis/v8a_event_to_feature_pipeline.py",
        "generated_at_utc": generated_at,
        "profile": args.profile,
        "development_only": development_only,
        "shadow_or_final_used": shadow_or_final_used,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "bin_axis": args.bin_axis,
        "q_bin_width_a_inv": args.q_bin_width,
        "d_bin_width_a": args.d_bin_width,
        "peak_window_a_inv": args.peak_window_a_inv,
        "peak_table_id": peak_table_id,
        "source_peak_table_ids": source_peak_table_ids,
        "source_peak_table_matches_analysis": source_peak_table_matches_analysis,
        "lineage_note": (
            "source_peak_table_ids record the manifest used to generate the existing Geant4 phase-space/source rows; "
            "peak_table_id records the manifest used for this event-to-feature re-windowing pass."
        ),
        "peak_manifest_path": relpath(peak_manifest_path, project_root),
        "schema_contract_path": relpath(schema_path, project_root),
        "sample_count": len(feature_rows),
        "long_row_count": len(long_rows),
        "feature_column_count": len(feature_fieldnames),
        "main_feature_columns": main_feature_columns,
        "control_feature_columns": control_feature_columns,
        "lineage_columns_excluded_from_main_features": [
            "sample_id",
            "material",
            "source_id",
            "source_mode",
            "source_peak_table_id",
            "random_seed",
            "thickness_mm",
            "pose_index",
            "row_index",
            "events_path",
            "hits_path",
            "metadata_path",
        ],
        "peak_summary": peak_summary,
        "claim_scope": "development-only schema/control conversion from existing v8A boundary smoke",
    }
    gate = {
        "generated_by": "analysis/v8a_event_to_feature_pipeline.py",
        "generated_at_utc": generated_at,
        "gate_passed": schema_gate_passed,
        "tiny_training_gate_allowed": tiny_training_gate_allowed,
        "decision": decision,
        "development_only": development_only,
        "shadow_or_final_used": shadow_or_final_used,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "bin_axis": args.bin_axis,
        "peak_table_id": peak_table_id,
        "source_peak_table_ids": source_peak_table_ids,
        "source_peak_table_matches_analysis": source_peak_table_matches_analysis,
        "lineage_note": (
            "Existing development hits may have been generated from an earlier peak table; medium matrix generation must use the successor manifest directly."
        ),
        "schema_contract_ok": True,
        "long_schema_ok": long_schema_ok,
        "peak_table_id_matches": peak_id_matches,
        "source_lineage_ok": source_lineage_ok,
        "control_summary": control_summary,
        "stop_reasons": stop_reasons,
        "claim_boundary": "development-only event-to-feature schema/control gate; not H/M accuracy, hardware validation, final sorter performance, or publishable powder XRD evidence",
    }
    write_json(output_dir / "v8a_event_feature_manifest.json", manifest)
    write_json(output_dir / "v8a_event_schema_gate.json", gate)
    write_report(output_dir / "v8a_event_schema_gate_report.md", gate, control_audit, manifest)
    print(f"wrote {len(feature_rows)} samples, {len(long_rows)} sidecar rows to {output_dir}")
    print(f"decision={decision} gate_passed={str(schema_gate_passed).lower()} tiny_training_gate_allowed={str(tiny_training_gate_allowed).lower()}")


if __name__ == "__main__":
    main()
