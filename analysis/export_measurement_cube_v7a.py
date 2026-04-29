from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import material_sorting_v2 as v2


HM_PAIR = ["Hematite", "Magnetite"]
DEFAULT_TRAIN_SEEDS = list(range(2101, 2113))
DEFAULT_VALIDATION_SEEDS = list(range(2201, 2207))
DEFAULT_SHADOW_SEEDS = list(range(2301, 2307))
DETECTORS = ["transmission", "side_scatter"]
CHANNELS = [
    "hit_rate",
    "energy_mean_keV",
    "tail120_rate",
    "primary_rate",
    "direct_primary_rate",
    "scattered_primary_rate",
    "theta_mean_deg",
]
VARIANT_RANK = {"normal_narrow": 0, "normal_wide": 1, "oblique_10deg": 2}


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_raw_dirs(project_root: Path, raw_dir: str, raw_dirs: str) -> list[Path]:
    values = parse_str_list(raw_dirs) if raw_dirs.strip() else [raw_dir]
    return [(project_root / value).resolve() for value in values]


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


def discover_hm_records(project_root: Path, raw_dirs: list[Path]) -> list[v2.RunRecord]:
    records: list[v2.RunRecord] = []
    for raw_dir in raw_dirs:
        material_records, _ = v2.discover_records(project_root, raw_dir)
        records.extend(record for record in material_records if record.material in HM_PAIR)
    if not records:
        raise ValueError(f"No H/M material records found in raw dirs: {[path.as_posix() for path in raw_dirs]}")
    return records


def filter_records(
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
        raise ValueError("No records remain after v7A filters.")
    return selected


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
) -> tuple[pd.DataFrame, dict[tuple[str, float, int, int], int]]:
    keys: set[tuple[str, float, int, int]] = set()
    for record in records:
        complete = read_complete_samples(record, photon_budget)
        for sample_id in range(complete):
            keys.add((record.material, float(record.thickness_mm), int(record.random_seed), int(sample_id)))
    rows = []
    for material, thickness, seed, sample_id in sorted(keys, key=lambda item: (split_for_seed(item[2], train_seeds, validation_seeds), item[0], item[1], item[2], item[3])):
        rows.append(
            {
                "sample_index": len(rows),
                "material": material,
                "thickness_mm": thickness,
                "random_seed": seed,
                "sample_id": sample_id,
                "split": split_for_seed(seed, train_seeds, validation_seeds),
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


def write_record_into_cube(
    cube: np.ndarray,
    record: v2.RunRecord,
    source_index: int,
    key_to_index: dict[tuple[str, float, int, int], int],
    photon_budget: int,
    grid_bins: int,
) -> None:
    if record.hit_file.stat().st_size <= 0:
        return
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
        grouped = part.groupby(["sample_id", "grid_a", "grid_b"], sort=False)
        for (sample_id, grid_a, grid_b), group in grouped:
            row_index = key_to_index.get((record.material, float(record.thickness_mm), int(record.random_seed), int(sample_id)))
            if row_index is None:
                continue
            count = float(len(group))
            theta_mean = group["theta_valid"].mean()
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 0] = count / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 1] = float(group["photon_energy_keV"].mean()) if count else 0.0
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 2] = float(group["tail120"].sum()) / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 3] = float(group["is_primary"].sum()) / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 4] = float(group["is_direct_primary"].sum()) / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 5] = float(group["is_scattered_primary"].sum()) / float(photon_budget)
            cube[row_index, source_index, detector_index, int(grid_a), int(grid_b), 6] = 0.0 if pd.isna(theta_mean) else float(theta_mean)


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
    parser = argparse.ArgumentParser(description="Export v7A H/M measurement cubes from v6c raw hits.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/v6c_hm_source_design")
    parser.add_argument("--raw-dirs", default="")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v7a_hm_measurement_cube")
    parser.add_argument("--photon-budget", type=int, default=5000)
    parser.add_argument("--grid-bins", type=int, default=8)
    parser.add_argument("--train-seeds", default=",".join(str(seed) for seed in DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--validation-seeds", default=",".join(str(seed) for seed in DEFAULT_VALIDATION_SEEDS))
    parser.add_argument("--shadow-seeds", default=",".join(str(seed) for seed in DEFAULT_SHADOW_SEEDS))
    parser.add_argument("--source-ids", default="", help="Optional comma-separated source filter for smoke runs.")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed filter applied after split rules.")
    parser.add_argument("--thicknesses", default="", help="Optional comma-separated thickness filter.")
    parser.add_argument("--include-shadow", action="store_true")
    parser.add_argument("--write-feature-csv", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = set(parse_int_list(args.train_seeds))
    validation_seeds = set(parse_int_list(args.validation_seeds))
    shadow_seeds = set(parse_int_list(args.shadow_seeds))
    explicit_seeds = set(parse_int_list(args.seeds)) if args.seeds.strip() else set()
    source_ids_filter = set(parse_str_list(args.source_ids))
    thickness_filter = set(parse_float_list(args.thicknesses)) if args.thicknesses.strip() else set()

    records = discover_hm_records(project_root, parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs))
    records = filter_records(
        records,
        train_seeds,
        validation_seeds,
        shadow_seeds,
        source_ids_filter,
        thickness_filter,
        args.include_shadow,
    )
    if explicit_seeds:
        records = [record for record in records if int(record.random_seed) in explicit_seeds]
    if not records:
        raise ValueError("No records remain after explicit seed filter.")
    forbidden_shadow = sorted({int(record.random_seed) for record in records} & shadow_seeds)
    if forbidden_shadow and not args.include_shadow:
        raise RuntimeError(f"Shadow seeds leaked into v7A export: {forbidden_shadow}")

    source_ids = sorted({record.source_id for record in records}, key=source_sort_key)
    source_to_index = {source_id: index for index, source_id in enumerate(source_ids)}
    metadata, key_to_index = build_sample_index(records, args.photon_budget, train_seeds, validation_seeds)
    cube = np.zeros((len(metadata), len(source_ids), len(DETECTORS), args.grid_bins, args.grid_bins, len(CHANNELS)), dtype=np.float32)

    for record in records:
        write_record_into_cube(
            cube,
            record,
            source_to_index[record.source_id],
            key_to_index,
            args.photon_budget,
            args.grid_bins,
        )

    names = np.array(feature_names(source_ids, args.grid_bins), dtype=object)
    np.savez_compressed(
        output_dir / "measurement_cube.npz",
        X=cube,
        feature_names=names,
        source_ids=np.array(source_ids, dtype=object),
        detector_ids=np.array(DETECTORS, dtype=object),
        channels=np.array(CHANNELS, dtype=object),
    )
    metadata.to_csv(output_dir / "sample_metadata.csv", index=False, lineterminator="\n")
    (output_dir / "feature_columns.txt").write_bytes(
        ("\n".join(str(name) for name in names) + "\n").encode("utf-8")
    )
    split_audit = (
        metadata.groupby(["split", "random_seed", "material"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(["split", "random_seed", "material"])
    )
    split_audit.to_csv(output_dir / "split_audit.csv", index=False, lineterminator="\n")

    if args.write_feature_csv:
        flat = cube.reshape((cube.shape[0], -1))
        feature_table = pd.concat(
            [metadata.reset_index(drop=True), pd.DataFrame(flat, columns=[str(name) for name in names])],
            axis=1,
        )
        feature_table.to_csv(output_dir / "feature_table.csv", index=False, lineterminator="\n")

    manifest = {
        "generated_by": "analysis/export_measurement_cube_v7a.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v7a_hm_measurement_cube",
        "development_only": not args.include_shadow,
        "shadow_or_final_used": bool(args.include_shadow),
        "shadow_seeds_excluded": not args.include_shadow,
        "project_root": project_root.as_posix(),
        "raw_dirs": [rel_path(path, project_root) for path in parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs)],
        "output_dir": args.output_dir,
        "photon_budget": args.photon_budget,
        "grid_bins": args.grid_bins,
        "materials": HM_PAIR,
        "train_seeds": sorted(train_seeds),
        "validation_seeds": sorted(validation_seeds),
        "shadow_seeds": sorted(shadow_seeds),
        "source_ids": source_ids,
        "detectors": DETECTORS,
        "channels": CHANNELS,
        "records_used": len(records),
        "samples": int(len(metadata)),
        "tensor_shape": list(cube.shape),
        "feature_count": int(len(names)),
        "split_counts": metadata["split"].value_counts().sort_index().to_dict(),
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    (output_dir / "measurement_cube_manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    print(f"Wrote v7A measurement cube to {output_dir}")
    print(f"tensor_shape={tuple(cube.shape)} samples={len(metadata)} feature_count={len(names)}")


if __name__ == "__main__":
    main()
