from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from pathlib import Path

from run_v8a_multiclass_context_v4_count_overlap_after_geant4 import EXPECTED_ROWS, PROFILE, postprocess, status, timestamp


STATUS_FIELDS = [
    "row_index",
    "profile",
    "run_role",
    "material",
    "source_id",
    "thickness_mm",
    "random_seed",
    "returncode",
    "elapsed_seconds",
    "config_path",
    "output_prefix",
]


def read_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def status_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("run_role", "material"),
        row.get("material", ""),
        row.get("source_id", ""),
        row.get("thickness_mm", ""),
        row.get("random_seed", ""),
    )


def write_status(path: Path, rows: list[dict[str, str]]) -> None:
    latest: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    order: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        normalized = {field: str(row.get(field, "")) for field in STATUS_FIELDS}
        key = status_key(normalized)
        if key not in latest:
            order.append(key)
        latest[key] = normalized
    ordered_rows = sorted((latest[key] for key in order), key=lambda item: int(item.get("row_index", "0") or 0))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered_rows)
    tmp.replace(path)


def init_shard_statuses(status_dir: Path, shard_count: int) -> None:
    default_status = status_dir / f"run_status_{PROFILE}.csv"
    existing_rows = read_status(default_status)
    for index in range(shard_count):
        shard_path = status_dir / f"run_status_{PROFILE}_shard{index:02d}.csv"
        shard_rows = read_status(shard_path)
        write_status(shard_path, existing_rows + shard_rows)


def merge_statuses(status_dir: Path, shard_count: int) -> None:
    default_status = status_dir / f"run_status_{PROFILE}.csv"
    rows = read_status(default_status)
    for index in range(shard_count):
        rows.extend(read_status(status_dir / f"run_status_{PROFILE}_shard{index:02d}.csv"))
    write_status(default_status, rows)


def run_shards(project_root: Path, python_exe: str, shard_count: int) -> None:
    chunk = math.ceil(EXPECTED_ROWS / shard_count)
    processes: list[tuple[int, int, subprocess.Popen[str]]] = []
    for index in range(shard_count):
        start = index * chunk
        limit = max(0, min(chunk, EXPECTED_ROWS - start))
        if limit <= 0:
            continue
        cmd = [
            python_exe,
            "analysis/run_material_sorting_matrix.py",
            "--profile",
            PROFILE,
            "--start",
            str(start),
            "--limit",
            str(limit),
            "--status-suffix",
            f"_shard{index:02d}",
        ]
        log_path = project_root / "results" / "material_sorting" / f"{PROFILE}_shard{index:02d}.log"
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(cmd, cwd=project_root, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        processes.append((index, start, proc))
        print(f"[{timestamp()}] shard{index:02d} pid={proc.pid} start={start} limit={limit}", flush=True)

    remaining = set(range(len(processes)))
    while remaining:
        for position, (index, _, proc) in enumerate(processes):
            if position not in remaining:
                continue
            rc = proc.poll()
            if rc is None:
                continue
            remaining.remove(position)
            print(f"[{timestamp()}] shard{index:02d} exited rc={rc}", flush=True)
            if rc != 0:
                raise RuntimeError(f"Shard {index:02d} failed with returncode={rc}")
        print(f"[{timestamp()}] waiting for {len(remaining)} active shards", flush=True)
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v8A v4 count-overlap Geant4 in independent status shards, merge, then postprocess.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--overwrite-stage-outputs", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if args.log_file:
        log_path = project_root / args.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log_handle
        sys.stderr = log_handle
    shard_count = max(1, int(args.shards))
    status_dir = project_root / "results" / "material_sorting"
    status_dir.mkdir(parents=True, exist_ok=True)
    init_shard_statuses(status_dir, shard_count)
    run_shards(project_root, args.python_exe, shard_count)
    merge_statuses(status_dir, shard_count)
    current = status(project_root, args.python_exe)
    if current["failed"] > 0 or current["completed"] < EXPECTED_ROWS or current["pending"] > 0:
        raise RuntimeError(f"Merged status is not complete: {current}")
    postprocess(project_root, args.python_exe, args.overwrite_stage_outputs)
    print(f"[{timestamp()}] v8A v4 parallel count-overlap pipeline complete", flush=True)


if __name__ == "__main__":
    main()
