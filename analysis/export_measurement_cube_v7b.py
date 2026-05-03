from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import material_sorting_v2 as v2


TARGET_MATERIALS = list(v2.TARGET_MATERIALS)
DEFAULT_TRAIN_SEEDS = list(range(4101, 4113))
DEFAULT_VALIDATION_SEEDS = list(range(4201, 4207))
DEFAULT_SHADOW_SEEDS = list(range(4301, 4307))
DETECTORS = ["transmission", "side_scatter"]
CHANNELS = [
    "hit_rate",
    "calibrated_hit_ratio",
    "attenuation",
    "energy_mean_keV",
    "tail120_rate",
    "tail120_fraction",
    "primary_rate",
    "direct_primary_rate",
    "scattered_primary_rate",
    "theta_mean_deg",
    "radius_mean_mm",
    "detector_total_rate",
]
VARIANT_RANK = {
    "normal_narrow": 0,
    "normal_wide": 1,
    "oblique_10deg": 2,
    "oblique_20deg": 3,
    "oblique_30deg": 4,
    "oblique_40deg": 5,
}
EPS = 1e-6


def log_progress(message: str, *, start_time: float | None = None) -> None:
    elapsed = ""
    if start_time is not None:
        elapsed = f" elapsed={perf_counter() - start_time:.1f}s"
    print(f"[v7b-export] {datetime.now().isoformat(timespec='seconds')} {message}{elapsed}", flush=True)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_raw_dirs(project_root: Path, raw_dir: str, raw_dirs: str) -> list[Path]:
    values = parse_str_list(raw_dirs) if raw_dirs.strip() else [raw_dir]
    # Do not call resolve() here: on Windows it strips \\?\UNC prefixes and can
    # reintroduce MAX_PATH failures for long WSL output filenames.
    return [project_root / value for value in values]


def rel_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def source_sort_key(source_id: str) -> tuple[float, int, str]:
    energy = math.inf
    variant = ""
    if source_id.startswith("mono_") and "kev" in source_id:
        raw = source_id.removeprefix("mono_")
        energy_text, _, variant = raw.partition("kev")
        variant = variant.removeprefix("_")
        try:
            energy = float(energy_text.replace("p", "."))
        except ValueError:
            energy = math.inf
    return energy, VARIANT_RANK.get(variant, 99), source_id


def split_for_seed(seed: int, train_seeds: set[int], validation_seeds: set[int]) -> str:
    if seed in train_seeds:
        return "train"
    if seed in validation_seeds:
        return "validation"
    return "unused"


def detector_axes(detector_id: str, grid_bins: int) -> tuple[str, str, np.ndarray, np.ndarray]:
    if detector_id == "side_scatter":
        return "x_mm", "z_mm", np.linspace(-120.0, 120.0, grid_bins + 1), np.linspace(-100.0, 100.0, grid_bins + 1)
    return "y_mm", "z_mm", np.linspace(-100.0, 100.0, grid_bins + 1), np.linspace(-100.0, 100.0, grid_bins + 1)


def material_catalog(project_root: Path) -> dict[str, dict[str, str]]:
    table = pd.read_csv(project_root / v2.MATERIALS_FILE)
    result = {}
    for row in table.to_dict(orient="records"):
        name = str(row.get("material_name", "")).strip()
        if name:
            result[name] = {
                "group_label": str(row.get("group_label", "")),
                "category": str(row.get("category", "")),
                "formula": str(row.get("formula", "")),
            }
    return result


def discover_records(project_root: Path, raw_dirs: list[Path], materials: set[str]) -> tuple[list[v2.RunRecord], list[v2.RunRecord]]:
    material_records: list[v2.RunRecord] = []
    calibration_records: list[v2.RunRecord] = []
    for raw_dir in raw_dirs:
        material_part, calibration_part = v2.discover_records(project_root, raw_dir)
        material_records.extend(record for record in material_part if record.material in materials)
        calibration_records.extend(calibration_part)
    if not material_records:
        raise ValueError(f"No target material records found in raw dirs: {[path.as_posix() for path in raw_dirs]}")
    return material_records, calibration_records


def filter_material_records(
    records: list[v2.RunRecord],
    train_seeds: set[int],
    validation_seeds: set[int],
    shadow_seeds: set[int],
    source_ids: set[str],
    thicknesses: set[float],
    include_shadow: bool,
) -> list[v2.RunRecord]:
    allowed_seeds = set(train_seeds) | set(validation_seeds)
    if include_shadow:
        allowed_seeds |= set(shadow_seeds)
    selected = []
    for record in records:
        seed = int(record.random_seed)
        if seed not in allowed_seeds:
            continue
        if seed in shadow_seeds and not include_shadow:
            continue
        if source_ids and record.source_id not in source_ids:
            continue
        if thicknesses and float(record.thickness_mm) not in thicknesses:
            continue
        if not record.hit_file.exists() or not record.metadata_file.exists():
            continue
        selected.append(record)
    if not selected:
        raise ValueError("No material records remain after v7B filters.")
    return selected


def filter_calibration_records(records: list[v2.RunRecord], material_records: list[v2.RunRecord]) -> list[v2.RunRecord]:
    needed = {(record.source_id, int(record.random_seed)) for record in material_records}
    return [record for record in records if (record.source_id, int(record.random_seed)) in needed and record.hit_file.exists()]


def read_complete_samples(record: v2.RunRecord, photon_budget: int) -> int:
    meta = v2.read_metadata(record.metadata_file)
    n_events = int(meta.get("n_events", 0))
    if n_events <= 0 and record.event_file.exists():
        n_events = int(max(pd.read_csv(record.event_file, usecols=["event_id"])["event_id"]) + 1)
    return n_events // photon_budget


def build_sample_index(
    records: list[v2.RunRecord],
    photon_budget: int,
    train_seeds: set[int],
    validation_seeds: set[int],
    catalog: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, dict[tuple[str, float, int, int], int]]:
    keys: set[tuple[str, float, int, int]] = set()
    for record in records:
        for sample_id in range(read_complete_samples(record, photon_budget)):
            keys.add((record.material, float(record.thickness_mm), int(record.random_seed), int(sample_id)))
    rows = []
    for material, thickness, seed, sample_id in sorted(keys, key=lambda item: (split_for_seed(item[2], train_seeds, validation_seeds), item[0], item[1], item[2], item[3])):
        meta = catalog.get(material, {})
        rows.append(
            {
                "sample_index": len(rows),
                "material": material,
                "group_label": meta.get("group_label", ""),
                "category": meta.get("category", ""),
                "thickness_mm": thickness,
                "random_seed": seed,
                "sample_id": sample_id,
                "split": split_for_seed(seed, train_seeds, validation_seeds),
                "legacy_source": "v7b",
                "sample_weight_base": 1.0,
            }
        )
    metadata = pd.DataFrame(rows)
    index = {
        (row.material, float(row.thickness_mm), int(row.random_seed), int(row.sample_id)): int(row.sample_index)
        for row in metadata.itertuples(index=False)
    }
    return metadata, index


def ensure_hit_columns(hits: pd.DataFrame) -> pd.DataFrame:
    if "detector_id" not in hits.columns:
        hits["detector_id"] = "transmission"
    if "x_mm" not in hits.columns:
        hits["x_mm"] = 0.0
    for col in ["is_primary", "is_direct_primary", "is_scattered_primary"]:
        if col not in hits.columns:
            hits[col] = 0
    if "theta_deg" not in hits.columns:
        hits["theta_deg"] = -1.0
    hits["detector_id"] = hits["detector_id"].fillna("transmission").astype(str)
    return hits


def read_hits(record: v2.RunRecord, photon_budget: int) -> pd.DataFrame:
    usecols = [
        "event_id",
        "detector_id",
        "x_mm",
        "y_mm",
        "z_mm",
        "photon_energy_keV",
        "is_primary",
        "theta_deg",
        "is_direct_primary",
        "is_scattered_primary",
    ]
    hits = pd.read_csv(record.hit_file)
    hits = ensure_hit_columns(hits)
    missing = [col for col in usecols if col not in hits.columns]
    if missing:
        raise ValueError(f"Missing hit columns in {record.hit_file}: {missing}")
    hits = hits[usecols].copy()
    hits["sample_id"] = (hits["event_id"].astype(int) // photon_budget).astype(int)
    return hits


def calibration_rates(
    records: list[v2.RunRecord],
    photon_budget: int,
    progress_every: int = 25,
    start_time: float | None = None,
) -> dict[tuple[str, int, str], float]:
    rates: dict[tuple[str, int, str], float] = {}
    for index, record in enumerate(records, start=1):
        if record.hit_file.stat().st_size <= 0:
            continue
        hits = read_hits(record, photon_budget)
        grouped = hits.groupby(["sample_id", "detector_id"]).size().reset_index(name="count")
        for detector_id, part in grouped.groupby("detector_id"):
            rates[(record.source_id, int(record.random_seed), str(detector_id))] = float(part["count"].mean()) / float(photon_budget)
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(records)):
            log_progress(f"calibration_records={index}/{len(records)} calibration_rates={len(rates)}", start_time=start_time)
    return rates


def write_record_into_cube(
    cube: np.ndarray,
    record: v2.RunRecord,
    source_index: int,
    key_to_index: dict[tuple[str, float, int, int], int],
    calib_rates: dict[tuple[str, int, str], float],
    photon_budget: int,
    grid_bins: int,
) -> None:
    if record.hit_file.stat().st_size <= 0:
        return
    hits = read_hits(record, photon_budget)
    total_by_sample_detector = hits.groupby(["sample_id", "detector_id"]).size().to_dict()
    for detector_index, detector_id in enumerate(DETECTORS):
        part = hits[hits["detector_id"].eq(detector_id)].copy()
        if part.empty:
            continue
        axis_a, axis_b, edges_a, edges_b = detector_axes(detector_id, grid_bins)
        part["grid_a"] = np.searchsorted(edges_a, part[axis_a].to_numpy(dtype=float), side="right") - 1
        part["grid_b"] = np.searchsorted(edges_b, part[axis_b].to_numpy(dtype=float), side="right") - 1
        part = part[(part["grid_a"] >= 0) & (part["grid_a"] < grid_bins) & (part["grid_b"] >= 0) & (part["grid_b"] < grid_bins)]
        if part.empty:
            continue
        part["tail120"] = (part["photon_energy_keV"].astype(float) >= 120.0).astype(float)
        part["theta_valid"] = part["theta_deg"].where(part["theta_deg"].astype(float) >= 0.0, np.nan)
        if detector_id == "side_scatter":
            part["radius_mm"] = np.sqrt(part["x_mm"].astype(float) ** 2 + part["z_mm"].astype(float) ** 2)
        else:
            part["radius_mm"] = np.sqrt(part["y_mm"].astype(float) ** 2 + part["z_mm"].astype(float) ** 2)
        grouped = part.groupby(["sample_id", "grid_a", "grid_b"], sort=False)
        for (sample_id, grid_a, grid_b), group in grouped:
            row_index = key_to_index.get((record.material, float(record.thickness_mm), int(record.random_seed), int(sample_id)))
            if row_index is None:
                continue
            count = float(len(group))
            hit_rate = count / float(photon_budget)
            calib = max(calib_rates.get((record.source_id, int(record.random_seed), detector_id), 1.0), EPS)
            ratio = hit_rate / calib
            theta_mean = group["theta_valid"].mean()
            detector_total_rate = float(total_by_sample_detector.get((sample_id, detector_id), 0)) / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 0] = hit_rate
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 1] = ratio
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 2] = -math.log(max(min(ratio, 1e6), EPS))
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 3] = float(group["photon_energy_keV"].mean()) if count else 0.0
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 4] = float(group["tail120"].sum()) / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 5] = float(group["tail120"].mean()) if count else 0.0
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 6] = float(group["is_primary"].sum()) / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 7] = float(group["is_direct_primary"].sum()) / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 8] = float(group["is_scattered_primary"].sum()) / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 9] = 0.0 if pd.isna(theta_mean) else float(theta_mean)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 10] = float(group["radius_mm"].mean()) if count else 0.0
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 11] = detector_total_rate


def feature_names(source_ids: list[str], grid_bins: int) -> list[str]:
    names = []
    for source_id in source_ids:
        for detector_id in DETECTORS:
            for grid_a in range(grid_bins):
                for grid_b in range(grid_bins):
                    for channel in CHANNELS:
                        names.append(f"{source_id}__{detector_id}__g{grid_a}_{grid_b}__{channel}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v7B ten-material measurement cubes.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/v7b_hard_negative_dev")
    parser.add_argument("--raw-dirs", default="")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v7b_hard_negative_dev")
    parser.add_argument("--photon-budget", type=int, default=5000)
    parser.add_argument("--grid-bins", type=int, default=8)
    parser.add_argument("--train-seeds", default=",".join(str(seed) for seed in DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--validation-seeds", default=",".join(str(seed) for seed in DEFAULT_VALIDATION_SEEDS))
    parser.add_argument("--shadow-seeds", default=",".join(str(seed) for seed in DEFAULT_SHADOW_SEEDS))
    parser.add_argument("--materials", default=",".join(TARGET_MATERIALS))
    parser.add_argument("--source-ids", default="", help="Optional comma-separated source filter for smoke runs.")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed filter applied after split rules.")
    parser.add_argument("--thicknesses", default="", help="Optional comma-separated thickness filter.")
    parser.add_argument("--progress-every", type=int, default=250, help="Print progress after this many material records; <=0 disables per-record progress.")
    parser.add_argument("--include-shadow", action="store_true")
    parser.add_argument("--write-feature-csv", action="store_true")
    args = parser.parse_args()

    start_time = perf_counter()
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = set(parse_int_list(args.train_seeds))
    validation_seeds = set(parse_int_list(args.validation_seeds))
    shadow_seeds = set(parse_int_list(args.shadow_seeds))
    explicit_seeds = set(parse_int_list(args.seeds)) if args.seeds.strip() else set()
    source_ids_filter = set(parse_str_list(args.source_ids))
    thickness_filter = set(parse_float_list(args.thicknesses)) if args.thicknesses.strip() else set()
    materials = set(parse_str_list(args.materials))
    catalog = material_catalog(project_root)
    log_progress(f"start project_root={project_root} output_dir={output_dir}", start_time=start_time)

    material_records, calibration_records = discover_records(project_root, parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs), materials)
    log_progress(
        f"discovered material_records={len(material_records)} calibration_records={len(calibration_records)}",
        start_time=start_time,
    )
    material_records = filter_material_records(
        material_records,
        train_seeds,
        validation_seeds,
        shadow_seeds,
        source_ids_filter,
        thickness_filter,
        args.include_shadow,
    )
    if explicit_seeds:
        material_records = [record for record in material_records if int(record.random_seed) in explicit_seeds]
    if not material_records:
        raise ValueError("No records remain after explicit seed filter.")
    forbidden_shadow = sorted({int(record.random_seed) for record in material_records} & shadow_seeds)
    if forbidden_shadow and not args.include_shadow:
        raise RuntimeError(f"Shadow seeds leaked into v7B export: {forbidden_shadow}")

    calibration_records = filter_calibration_records(calibration_records, material_records)
    source_ids = sorted({record.source_id for record in material_records}, key=source_sort_key)
    source_to_index = {source_id: index for index, source_id in enumerate(source_ids)}
    metadata, key_to_index = build_sample_index(material_records, args.photon_budget, train_seeds, validation_seeds, catalog)
    log_progress(
        f"filtered material_records={len(material_records)} calibration_records={len(calibration_records)} samples={len(metadata)} sources={len(source_ids)}",
        start_time=start_time,
    )
    calib_rates = calibration_rates(
        calibration_records,
        args.photon_budget,
        progress_every=max(1, min(args.progress_every, 50)) if args.progress_every > 0 else 0,
        start_time=start_time,
    )
    cube = np.zeros((len(metadata), len(source_ids), len(DETECTORS), args.grid_bins, args.grid_bins, len(CHANNELS)), dtype=np.float32)
    log_progress(f"allocated cube_shape={cube.shape} cube_bytes={cube.nbytes}", start_time=start_time)

    for index, record in enumerate(material_records, start=1):
        write_record_into_cube(
            cube,
            record,
            source_to_index[record.source_id],
            key_to_index,
            calib_rates,
            args.photon_budget,
            args.grid_bins,
        )
        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(material_records)):
            log_progress(f"material_records={index}/{len(material_records)}", start_time=start_time)

    names = np.array(feature_names(source_ids, args.grid_bins), dtype=object)
    log_progress("writing compressed npz", start_time=start_time)
    np.savez_compressed(
        output_dir / "measurement_cube.npz",
        X=cube,
        feature_names=names,
        source_ids=np.array(source_ids, dtype=object),
        detector_ids=np.array(DETECTORS, dtype=object),
        channels=np.array(CHANNELS, dtype=object),
    )
    log_progress("writing metadata files", start_time=start_time)
    metadata.to_csv(output_dir / "sample_metadata.csv", index=False, lineterminator="\n")
    (output_dir / "feature_columns.txt").write_bytes(("\n".join(str(name) for name in names) + "\n").encode("utf-8"))
    split_audit = (
        metadata.groupby(["split", "random_seed", "material"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(["split", "random_seed", "material"])
    )
    split_audit.to_csv(output_dir / "split_audit.csv", index=False, lineterminator="\n")
    if args.write_feature_csv:
        flat = cube.reshape((cube.shape[0], -1))
        feature_table = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(flat, columns=[str(name) for name in names])], axis=1)
        feature_table.to_csv(output_dir / "feature_table.csv", index=False, lineterminator="\n")

    manifest = {
        "generated_by": "analysis/export_measurement_cube_v7b.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v7b_hard_negative_dev",
        "development_only": not args.include_shadow,
        "shadow_or_final_used": bool(args.include_shadow),
        "shadow_seeds_excluded": not args.include_shadow,
        "raw_dirs": [rel_path(path, project_root) for path in parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs)],
        "output_dir": args.output_dir,
        "photon_budget": args.photon_budget,
        "grid_bins": args.grid_bins,
        "materials": sorted(materials),
        "train_seeds": sorted(train_seeds),
        "validation_seeds": sorted(validation_seeds),
        "shadow_seeds": sorted(shadow_seeds),
        "source_ids": source_ids,
        "detectors": DETECTORS,
        "channels": CHANNELS,
        "records_used": len(material_records),
        "calibration_records_used": len(calibration_records),
        "calibration_rates": len(calib_rates),
        "samples": int(len(metadata)),
        "tensor_shape": list(cube.shape),
        "feature_count": int(len(names)),
        "split_counts": metadata["split"].value_counts().sort_index().to_dict(),
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    (output_dir / "measurement_cube_manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    log_progress(f"Wrote v7B measurement cube to {output_dir}", start_time=start_time)
    log_progress(f"tensor_shape={tuple(cube.shape)} samples={len(metadata)} feature_count={len(names)}", start_time=start_time)


if __name__ == "__main__":
    main()
