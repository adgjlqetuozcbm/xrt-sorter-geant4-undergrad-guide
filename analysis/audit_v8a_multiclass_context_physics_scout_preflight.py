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
    parser = argparse.ArgumentParser(description="Preflight-audit the v8A full-10 physics scout matrix.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_multiclass_context_v3_physics_scout_config.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v3_physics_scout_preflight")
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
    if config.get("status") != "development_scout_preregistration":
        stop_reasons.append("config_status_not_development_scout_preregistration")
    if len(rows) != int(config["expected_rows"]["total"]):
        stop_reasons.append(f"row_count_mismatch:{len(rows)}!={config['expected_rows']['total']}")
    if bool(manifest.get("runs_geant4")) or bool(manifest.get("training_unlocked")):
        stop_reasons.append("manifest_reports_geant4_or_training")
    if set(row.get("material", "") for row in rows) != materials:
        stop_reasons.append("material_set_mismatch")
    observed_energies = sorted({float(row.get("source_energy_kev", 0.0)) for row in rows})
    if observed_energies != sorted(float(item) for item in config["source_energies_kev"]):
        stop_reasons.append(f"source_energy_grid_mismatch:{observed_energies}")
    if set(row.get("source_mode", "") for row in rows) != {"on"}:
        stop_reasons.append("source_mode_not_on_only")
    if set(row.get("stress_label", "") for row in rows) != {"default"}:
        stop_reasons.append("stress_label_not_default_only")
    if any(bool(str(row.get("shadow_or_final_used", "")).lower() == "true") for row in rows):
        stop_reasons.append("row_reports_shadow_or_final_used")

    cell_keys = list(config["strict_match_cell_keys"])
    cells: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        cells[tuple(str(row.get(key, "")) for key in cell_keys)][str(row.get("material", ""))] += 1
    bad_cells = 0
    cells_by_split: Counter[str] = Counter()
    for key, counts in cells.items():
        cells_by_split[key[0]] += 1
        if set(counts) != materials or any(int(value) != 1 for value in counts.values()):
            bad_cells += 1
    if bad_cells:
        stop_reasons.append(f"context_cells_not_full10:{bad_cells}")
    for split, expected in config["expected_context_cells"].items():
        if split == "total":
            continue
        if int(cells_by_split.get(split, 0)) != int(expected):
            stop_reasons.append(f"context_cell_count_mismatch:{split}:{cells_by_split.get(split, 0)}!={expected}")
    split_counts = Counter(str(row.get("split", "")) for row in rows)
    for split, expected in config["expected_rows"].items():
        if split == "total":
            continue
        if int(split_counts.get(split, 0)) != int(expected):
            stop_reasons.append(f"split_row_count_mismatch:{split}:{split_counts.get(split, 0)}!={expected}")

    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_physics_scout_preflight.py",
        "generated_at_utc": generated_at,
        "development_only": True,
        "shadow_or_final_used": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "gate_passed": gate_passed,
        "decision": "physics_scout_preflight_passed_ready_for_development_run" if gate_passed else "stop_physics_scout_preflight",
        "profile": profile,
        "row_count": len(rows),
        "context_cells_by_split": {str(k): int(v) for k, v in sorted(cells_by_split.items())},
        "source_energies_kev": observed_energies,
        "stop_reasons": stop_reasons,
        "warnings": warnings,
        "software": {"python": platform.python_version()},
    }
    write_json(output_dir / "v8a_multiclass_context_physics_scout_preflight_gate.json", gate)
    print(f"decision={gate['decision']} gate_passed={str(gate_passed).lower()} rows={len(rows)}")


if __name__ == "__main__":
    main()
