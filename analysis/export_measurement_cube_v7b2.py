from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import export_measurement_cube_v7b as base


HM_PAIR = ("Hematite", "Magnetite")
DEFAULT_TRAIN_SEEDS = list(range(5101, 5105))
DEFAULT_VALIDATION_SEEDS = list(range(5201, 5203))
DEFAULT_SHADOW_SEEDS = list(range(5301, 5303))
FULL_SOURCE_LIMIT = 18


def require_pilot_gate(project_root: Path, pilot_gate_path: str) -> dict:
    gate_path = project_root / pilot_gate_path
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing v7B2 Pilot gate for full export: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if bool(gate.get("shadow_or_final_used", False)):
        raise RuntimeError("Refusing v7B2 export because Pilot gate reports shadow/final use.")
    if not bool(gate.get("gate_passed", False)):
        raise RuntimeError(f"v7B2 Pilot gate did not pass; decision={gate.get('decision')}")
    return gate


def full_source_filter(explicit_source_ids: set[str], pilot_gate: dict | None) -> set[str]:
    if pilot_gate is None:
        return explicit_source_ids
    selected = set(str(source_id) for source_id in pilot_gate.get("selected_full_v7b2_source_ids", []))
    if not selected:
        raise RuntimeError("Passing v7B2 Pilot gate did not record selected_full_v7b2_source_ids.")
    if len(selected) > FULL_SOURCE_LIMIT:
        raise RuntimeError(f"v7B2 Pilot selected too many full sources: {len(selected)} > {FULL_SOURCE_LIMIT}.")
    if not explicit_source_ids:
        return selected
    unknown = sorted(explicit_source_ids - selected)
    if unknown:
        raise RuntimeError(f"Full v7B2 source filter must be a subset of Pilot-selected sources; unknown={unknown}")
    return explicit_source_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v7B2 H/M physical-matrix measurement cubes.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/v7b2_hm_physics_dev")
    parser.add_argument("--raw-dirs", default="")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v7b2_hm_physics_dev")
    parser.add_argument("--photon-budget", type=int, default=5000)
    parser.add_argument("--grid-bins", type=int, default=8)
    parser.add_argument("--train-seeds", default=",".join(str(seed) for seed in DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--validation-seeds", default=",".join(str(seed) for seed in DEFAULT_VALIDATION_SEEDS))
    parser.add_argument("--shadow-seeds", default=",".join(str(seed) for seed in DEFAULT_SHADOW_SEEDS))
    parser.add_argument("--materials", default=",".join(HM_PAIR))
    parser.add_argument("--pilot-gate", default="results/accuracy_v3/v7b2_hm_physics_dev/v7b2_pilot_gate.json")
    parser.add_argument("--full-materials", action="store_true", help="Export ten-material full v7B2 only after a passing Pilot gate.")
    parser.add_argument("--source-ids", default="", help="Optional comma-separated source filter for smoke runs.")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed filter applied after split rules.")
    parser.add_argument("--thicknesses", default="", help="Optional comma-separated thickness filter.")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--include-shadow", action="store_true")
    parser.add_argument("--write-feature-csv", action="store_true")
    args = parser.parse_args()

    start_time = perf_counter()
    # Preserve extended UNC prefixes such as \\?\UNC\...; Path.resolve() drops
    # them on Windows and can reintroduce MAX_PATH failures on long WSL names.
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = set(base.parse_int_list(args.train_seeds))
    validation_seeds = set(base.parse_int_list(args.validation_seeds))
    shadow_seeds = set(base.parse_int_list(args.shadow_seeds))
    explicit_seeds = set(base.parse_int_list(args.seeds)) if args.seeds.strip() else set()
    source_ids_filter = set(base.parse_str_list(args.source_ids))
    thickness_filter = set(base.parse_float_list(args.thicknesses)) if args.thicknesses.strip() else set()
    materials = set(base.parse_str_list(args.materials))
    pilot_gate = None
    if args.full_materials:
        pilot_gate = require_pilot_gate(project_root, args.pilot_gate)
        if materials == set(HM_PAIR):
            materials = set(base.TARGET_MATERIALS)
        source_ids_filter = full_source_filter(source_ids_filter, pilot_gate)
    elif materials != set(HM_PAIR):
        raise RuntimeError(f"v7B2 Pilot export is preregistered as H/M-only; use --full-materials after Pilot pass for {sorted(materials)}")
    catalog = base.material_catalog(project_root)
    base.log_progress(f"start v7b2 project_root={project_root} output_dir={output_dir}", start_time=start_time)

    material_records, calibration_records = base.discover_records(project_root, base.parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs), materials)
    material_records = base.filter_material_records(
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
        raise RuntimeError(f"Shadow seeds leaked into v7B2 export: {forbidden_shadow}")

    calibration_records = base.filter_calibration_records(calibration_records, material_records)
    source_ids = sorted({record.source_id for record in material_records}, key=base.source_sort_key)
    source_to_index = {source_id: index for index, source_id in enumerate(source_ids)}
    metadata, key_to_index = base.build_sample_index(material_records, args.photon_budget, train_seeds, validation_seeds, catalog)
    base.log_progress(
        f"filtered material_records={len(material_records)} calibration_records={len(calibration_records)} samples={len(metadata)} sources={len(source_ids)}",
        start_time=start_time,
    )
    calib_rates = base.calibration_rates(
        calibration_records,
        args.photon_budget,
        progress_every=max(1, min(args.progress_every, 50)) if args.progress_every > 0 else 0,
        start_time=start_time,
    )
    cube = np.zeros((len(metadata), len(source_ids), len(base.DETECTORS), args.grid_bins, args.grid_bins, len(base.CHANNELS)), dtype=np.float32)
    base.log_progress(f"allocated cube_shape={cube.shape} cube_bytes={cube.nbytes}", start_time=start_time)

    for index, record in enumerate(material_records, start=1):
        base.write_record_into_cube(
            cube,
            record,
            source_to_index[record.source_id],
            key_to_index,
            calib_rates,
            args.photon_budget,
            args.grid_bins,
        )
        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(material_records)):
            base.log_progress(f"material_records={index}/{len(material_records)}", start_time=start_time)

    names = np.array(base.feature_names(source_ids, args.grid_bins), dtype=object)
    base.log_progress("writing compressed npz", start_time=start_time)
    np.savez_compressed(
        output_dir / "measurement_cube.npz",
        X=cube,
        feature_names=names,
        source_ids=np.array(source_ids, dtype=object),
        detector_ids=np.array(base.DETECTORS, dtype=object),
        channels=np.array(base.CHANNELS, dtype=object),
    )
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
        "generated_by": "analysis/export_measurement_cube_v7b2.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v7b2_hm_physics_dev",
        "development_only": not args.include_shadow,
        "shadow_or_final_used": bool(args.include_shadow),
        "shadow_seeds_excluded": not args.include_shadow,
        "raw_dirs": [base.rel_path(path, project_root) for path in base.parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs)],
        "output_dir": args.output_dir,
        "photon_budget": args.photon_budget,
        "grid_bins": args.grid_bins,
        "materials": sorted(materials),
        "full_materials": bool(args.full_materials),
        "pilot_gate": pilot_gate,
        "train_seeds": sorted(train_seeds),
        "validation_seeds": sorted(validation_seeds),
        "shadow_seeds": sorted(shadow_seeds),
        "source_ids": source_ids,
        "detectors": base.DETECTORS,
        "channels": base.CHANNELS,
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
    base.log_progress(f"Wrote v7B2 measurement cube to {output_dir}", start_time=start_time)
    base.log_progress(f"tensor_shape={tuple(cube.shape)} samples={len(metadata)} feature_count={len(names)}", start_time=start_time)


if __name__ == "__main__":
    main()
