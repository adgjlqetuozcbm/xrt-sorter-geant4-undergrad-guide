from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight-audit the v8A full-10 count-overlap matrix.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_multiclass_context_v4_count_overlap_config.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v4_count_overlap_preflight")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(as_project_path(project_root, args.config))
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    profile = str(config["profile"])
    matrix_path = project_root / "source_models" / "config" / "material_sorting_matrix" / profile / "material_sorting_matrix.csv"
    manifest_path = matrix_path.parent / "matrix_manifest.json"
    rows = read_csv(matrix_path)
    manifest = load_json(manifest_path)

    stop_reasons: list[str] = []
    warnings: list[str] = []
    materials = set(map(str, config["target_materials"]))
    if config.get("status") != "development_training_preregistration":
        stop_reasons.append("config_status_not_development_training_preregistration")
    if len(rows) != int(config["expected_rows"]["total"]):
        stop_reasons.append(f"row_count_mismatch:{len(rows)}!={config['expected_rows']['total']}")
    if bool(manifest.get("runs_geant4")) or bool(manifest.get("training_unlocked")):
        stop_reasons.append("manifest_reports_geant4_or_training")
    if set(row.get("material", "") for row in rows) != materials:
        stop_reasons.append("material_set_mismatch")
    observed_energies = sorted({float(row.get("source_energy_kev", 0.0)) for row in rows})
    if observed_energies != sorted(float(item) for item in config["source_energies_kev"]):
        stop_reasons.append(f"source_energy_grid_mismatch:{observed_energies}")
    observed_thickness = sorted({float(row.get("thickness_mm", 0.0)) for row in rows})
    if observed_thickness != sorted(float(item) for item in config["thickness_mm"]):
        stop_reasons.append(f"thickness_grid_mismatch:{observed_thickness}")
    if set(row.get("source_mode", "") for row in rows) != {"on"}:
        stop_reasons.append("source_mode_not_on_only")
    if set(row.get("stress_label", "") for row in rows) != {"default"}:
        stop_reasons.append("stress_label_not_default_only")
    if any(bool(str(row.get("shadow_or_final_used", "")).lower() == "true") for row in rows):
        stop_reasons.append("row_reports_shadow_or_final_used")

    if observed_energies != [50.0] or not set(observed_thickness).issubset({60.0, 120.0}):
        stop_reasons.append("matrix_not_limited_to_v3_scout_selected_count_overlap_regions")

    cell_keys = list(config["strict_match_cell_keys"])
    cells: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    random_seeds_by_cell: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in cell_keys)
        cells[key][str(row.get("material", ""))] += 1
        random_seeds_by_cell[key].add(str(row.get("random_seed", "")))

    bad_cells = 0
    multi_seed_cells = 0
    cells_by_split: Counter[str] = Counter()
    rows_by_split: Counter[str] = Counter()
    for row in rows:
        rows_by_split[str(row.get("split", ""))] += 1
    for key, counts in cells.items():
        split = key[0]
        cells_by_split[split] += 1
        if set(counts) != materials or any(int(value) != 1 for value in counts.values()):
            bad_cells += 1
        if len(random_seeds_by_cell[key]) != 1:
            multi_seed_cells += 1
    if bad_cells:
        stop_reasons.append(f"context_cells_not_full10_exactly_once:{bad_cells}")
    if multi_seed_cells:
        stop_reasons.append(f"context_cells_do_not_share_random_seed:{multi_seed_cells}")
    for split, expected in config["expected_context_cells"].items():
        if split == "total":
            continue
        if int(cells_by_split.get(split, 0)) != int(expected):
            stop_reasons.append(f"context_cell_count_mismatch:{split}:{cells_by_split.get(split, 0)}!={expected}")
    for split, expected in config["expected_rows"].items():
        if split == "total":
            continue
        if int(rows_by_split.get(split, 0)) != int(expected):
            stop_reasons.append(f"split_row_count_mismatch:{split}:{rows_by_split.get(split, 0)}!={expected}")

    target = config["count_control_target"]["minimum_supported_context_cells_after_geant4"]
    for split, expected in target.items():
        if int(config["expected_context_cells"].get(split, 0)) < int(expected):
            stop_reasons.append(f"not_enough_context_cells_for_count_control_target:{split}")
    warnings.append("post_geant4_sliding_window_count_balance_still_required_before_training_promotion")

    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_v4_count_overlap_preflight.py",
        "generated_at_utc": generated_at,
        "development_only": True,
        "shadow_or_final_used": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "gate_passed": gate_passed,
        "decision": "count_overlap_preflight_passed_ready_for_development_geant4" if gate_passed else "stop_count_overlap_preflight",
        "profile": profile,
        "row_count": len(rows),
        "context_cells_by_split": {str(k): int(v) for k, v in sorted(cells_by_split.items())},
        "rows_by_split": {str(k): int(v) for k, v in sorted(rows_by_split.items())},
        "source_energies_kev": observed_energies,
        "thickness_mm": observed_thickness,
        "count_control_target": config["count_control_target"],
        "stop_reasons": stop_reasons,
        "warnings": warnings,
        "software": {"python": platform.python_version()},
    }
    write_json(output_dir / "v8a_multiclass_context_v4_count_overlap_preflight_gate.json", gate)
    print(f"decision={gate['decision']} gate_passed={str(gate_passed).lower()} rows={len(rows)}")


if __name__ == "__main__":
    main()
