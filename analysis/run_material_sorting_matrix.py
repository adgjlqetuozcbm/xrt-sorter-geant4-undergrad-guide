from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path


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


def infer_macro_profile(profile: str) -> str:
    if profile.startswith("selected_rebuild"):
        return "selected_rebuild"
    if profile.startswith("energy_scan"):
        return "energy_scan"
    if profile.startswith("v6c_"):
        return "v6c"
    if profile.startswith("v7b2_"):
        return "v7b2"
    if profile.startswith("v8a_"):
        return "v8a_smoke"
    if profile.startswith("v7b_"):
        return "v7b"
    if profile.startswith("accuracy_v3") or profile.startswith("v3_hm") or profile.startswith("v5_"):
        return "selected_rebuild"
    return profile


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_status_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {field: str(row.get(field, "")) for field in STATUS_FIELDS}
    if not normalized["run_role"]:
        normalized["run_role"] = "material"
    return normalized


def load_status_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [normalize_status_row(row) for row in csv.DictReader(f)]


def load_completed_status(rows: list[dict[str, str]]) -> set[tuple[str, str, str, str, str]]:
    completed = set()
    for row in rows:
        if str(row.get("returncode", "")) != "0":
            continue
        completed.add(
            (
                row.get("run_role", "material"),
                row.get("material", ""),
                row.get("source_id", ""),
                row.get("thickness_mm", ""),
                row.get("random_seed", ""),
            )
        )
    return completed


def status_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("run_role", "material"),
        row.get("material", ""),
        row.get("source_id", ""),
        row.get("thickness_mm", ""),
        row.get("random_seed", ""),
    )


def write_status_rows(path: Path, rows: list[dict[str, str]]) -> None:
    latest: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    key_order: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        normalized = normalize_status_row(row)
        key = status_key(normalized)
        if key not in latest:
            key_order.append(key)
        latest[key] = normalized
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([latest[key] for key in key_order])
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generated Geant4 material sorting configs.")
    parser.add_argument("--profile", default="pilot", help="Matrix profile directory to run.")
    parser.add_argument(
        "--macro-profile",
        default="",
        help="Macro profile to reuse for profile aliases. Defaults to selected_rebuild for selected_rebuild_* aliases.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional number of matrix rows to run.")
    parser.add_argument("--start", type=int, default=0, help="Start row offset for resumable batches.")
    parser.add_argument("--role", choices=["all", "calibration", "material"], default="all")
    parser.add_argument("--rerun-existing", action="store_true", help="Run rows even when status CSV already has returncode 0.")
    parser.add_argument("--status-only", action="store_true", help="Only summarize completed/pending/failed rows.")
    parser.add_argument("--status-suffix", default="", help="Optional suffix for independent shard status CSVs.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    matrix_path = project_root / "source_models" / "config" / "material_sorting_matrix" / args.profile / "material_sorting_matrix.csv"
    rows = []
    for matrix_index, row in enumerate(load_rows(matrix_path)):
        indexed_row = dict(row)
        indexed_row["row_index"] = str(matrix_index)
        rows.append(indexed_row)
    if args.role != "all":
        rows = [row for row in rows if row.get("run_role", "material") == args.role]
    if args.start:
        rows = rows[args.start :]
    if args.limit:
        rows = rows[: args.limit]

    build_dir = project_root / "build"
    exe = build_dir / "xrt_sorter"
    macro_profile = args.macro_profile.strip() or infer_macro_profile(args.profile)
    macro = project_root / "analysis" / "configs" / f"run_material_sorting_{macro_profile}.mac"
    if not exe.exists():
        raise FileNotFoundError(f"Missing executable: {exe}. Run cmake --build build first.")
    if not macro.exists():
        raise FileNotFoundError(f"Missing macro: {macro}")

    status_dir = project_root / "results" / "material_sorting"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_suffix = str(args.status_suffix).strip()
    if status_suffix and not re.fullmatch(r"[_A-Za-z0-9.-]+", status_suffix):
        raise ValueError(f"Unsafe status suffix: {status_suffix}")
    status_path = status_dir / f"run_status_{args.profile}{status_suffix}.csv"
    existing_status_rows = load_status_rows(status_path)
    completed = set() if args.rerun_existing else load_completed_status(existing_status_rows)
    selected_keys = {
        (
            row.get("run_role", "material"),
            row["material"],
            row["source_id"],
            row["thickness_mm"],
            row["random_seed"],
        )
        for row in rows
    }
    if args.status_only:
        failed = {
            (
                row.get("run_role", "material"),
                row.get("material", ""),
                row.get("source_id", ""),
                row.get("thickness_mm", ""),
                row.get("random_seed", ""),
            )
            for row in existing_status_rows
            if row.get("returncode", "") not in {"", "0"}
        }
        done = selected_keys & completed
        failed_selected = (selected_keys & failed) - done
        pending = selected_keys - done - failed_selected
        print(f"profile={args.profile} role={args.role}")
        print(f"selected_rows={len(selected_keys)} completed={len(done)} failed={len(failed_selected)} pending={len(pending)}")
        return
    status_rows = []

    for index, row in enumerate(rows, start=args.start):
        key = (
            row.get("run_role", "material"),
            row["material"],
            row["source_id"],
            row["thickness_mm"],
            row["random_seed"],
        )
        if key in completed:
            print(
                f"[{row.get('row_index', str(index))}] skip existing {row.get('run_role', 'material')} "
                f"{row['material']} {row['source_id']} {row['thickness_mm']}mm seed={row['random_seed']}"
            )
            continue
        config_path = project_root / row["config_path"]
        env = os.environ.copy()
        env["XRT_EXPERIMENT_CONFIG"] = str(config_path)
        started = time.time()
        proc = subprocess.run(
            [str(exe), str(macro)],
            cwd=build_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.time() - started
        status_rows.append(
            {
                "row_index": row.get("row_index", str(index)),
                "profile": args.profile,
                "run_role": row.get("run_role", "material"),
                "material": row["material"],
                "source_id": row["source_id"],
                "thickness_mm": row["thickness_mm"],
                "random_seed": row["random_seed"],
                "returncode": proc.returncode,
                "elapsed_seconds": f"{elapsed:.3f}",
                "config_path": row["config_path"],
                "output_prefix": row["output_prefix"],
            }
        )
        write_status_rows(status_path, existing_status_rows + status_rows)
        print(
            f"[{row.get('row_index', str(index))}] {row.get('run_role', 'material')} {row['material']} {row['source_id']} "
            f"{row['thickness_mm']}mm seed={row['random_seed']} rc={proc.returncode}"
        )
        if proc.returncode != 0:
            log_path = status_dir / f"failed_{args.profile}_{index}.log"
            log_path.write_text(proc.stdout, encoding="utf-8")
            raise RuntimeError(f"Run failed for {row['config_path']}; see {log_path}")

    if status_rows:
        write_status_rows(status_path, existing_status_rows + status_rows)
        print(f"Wrote status to {status_path}")
    else:
        print("No rows selected.")


if __name__ == "__main__":
    main()
