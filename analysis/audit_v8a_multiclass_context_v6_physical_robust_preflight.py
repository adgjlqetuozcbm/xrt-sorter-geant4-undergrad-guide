from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_COLUMNS = [
    "peak_q_shift_fraction",
    "peak_theta_sigma_deg",
    "continuum_fraction",
    "peak_intensity_jitter_sigma",
    "source_energy_scale",
]


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


def context_cell_audit(rows: list[dict[str, str]], materials: set[str]) -> tuple[dict[str, int], int, list[str]]:
    cells: dict[str, Counter[str]] = defaultdict(Counter)
    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cell_id = str(row.get("clean_context_cell_id", ""))
        cells[cell_id][str(row.get("material", ""))] += 1
        splits[cell_id].add(str(row.get("split", "")))
    by_split: Counter[str] = Counter()
    bad_count = 0
    examples: list[str] = []
    for cell_id, counts in cells.items():
        split_values = splits[cell_id]
        if len(split_values) == 1:
            by_split[next(iter(split_values))] += 1
        else:
            bad_count += 1
            examples.append(cell_id)
            continue
        if set(counts) != materials or any(int(count) != 1 for count in counts.values()):
            bad_count += 1
            if len(examples) < 8:
                examples.append(cell_id)
    return {str(key): int(value) for key, value in sorted(by_split.items())}, bad_count, examples[:8]


def cross_split_counts(rows: list[dict[str, str]], columns: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in columns:
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = str(row.get(column, ""))
            if value:
                grouped[value].add(str(row.get("split", "")))
        result[column] = sum(1 for splits in grouped.values() if len(splits) > 1)
    return result


def profile_signature(profile: dict[str, Any]) -> tuple[float, ...]:
    return tuple(round(float(profile[column]), 8) for column in PROFILE_COLUMNS)


def signature_split_overlap(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = {str(item["physical_perturbation_profile"]): item for item in config["perturbation_profiles"]}
    signatures_by_split: dict[str, dict[tuple[float, ...], set[str]]] = {}
    for split, design in config["split_design"].items():
        signatures_by_split[split] = defaultdict(set)
        for name in design["profiles"]:
            profile_name = str(name)
            signatures_by_split[split][profile_signature(profiles[profile_name])].add(profile_name)
    result: dict[str, dict[str, Any]] = {}
    pairs = [("train", "validation"), ("train", "stress_holdout"), ("validation", "stress_holdout")]
    for left, right in pairs:
        shared = sorted(set(signatures_by_split[left]) & set(signatures_by_split[right]))
        shared_names = []
        for signature in shared:
            shared_names.append(
                {
                    "signature": list(signature),
                    left: sorted(signatures_by_split[left][signature]),
                    right: sorted(signatures_by_split[right][signature]),
                }
            )
        result[f"{left}|{right}"] = {"count": len(shared), "shared": shared_names}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight v8A v6 physical robustness matrix.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_multiclass_context_v6_physical_robust_config.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_preflight")
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
    configured_profiles = {str(item["physical_perturbation_profile"]) for item in config["perturbation_profiles"]}
    if config.get("status") != "development_physical_robust_preregistration":
        stop_reasons.append("config_status_not_development_physical_robust_preregistration")
    if len(rows) != int(config["expected_rows"]["total"]):
        stop_reasons.append(f"row_count_mismatch:{len(rows)}!={config['expected_rows']['total']}")
    if bool(manifest.get("runs_geant4")) or bool(manifest.get("training_unlocked")):
        stop_reasons.append("manifest_reports_geant4_or_training")
    if bool(manifest.get("shadow_or_final_used")):
        stop_reasons.append("manifest_reports_shadow_or_final")
    if set(row.get("material", "") for row in rows) != materials:
        stop_reasons.append("material_set_mismatch")
    if set(row.get("source_mode", "") for row in rows) != {"on"}:
        stop_reasons.append("source_mode_not_on_only")
    if set(row.get("physical_perturbation_profile", "") for row in rows) != configured_profiles:
        stop_reasons.append("perturbation_profile_set_mismatch")

    rows_by_split = Counter(str(row.get("split", "")) for row in rows)
    profiles_by_split = {
        split: sorted({str(row.get("physical_perturbation_profile", "")) for row in rows if row.get("split") == split})
        for split in config["split_design"]
    }
    for split, expected_rows in config["expected_rows"].items():
        if split == "total":
            continue
        if int(rows_by_split.get(split, 0)) != int(expected_rows):
            stop_reasons.append(f"split_row_count_mismatch:{split}:{rows_by_split.get(split, 0)}!={expected_rows}")
    for split, design in config["split_design"].items():
        expected_profiles = sorted(map(str, design["profiles"]))
        if profiles_by_split.get(split, []) != expected_profiles:
            stop_reasons.append(f"profile_coverage_mismatch:{split}:{profiles_by_split.get(split, [])}!={expected_profiles}")
    if "nominal" not in profiles_by_split.get("train", []):
        stop_reasons.append("train_split_missing_nominal")
    if len(profiles_by_split.get("train", [])) < 6:
        stop_reasons.append("train_split_not_perturbation_augmented_enough")
    if "combined_stress_high" not in profiles_by_split.get("stress_holdout", []):
        stop_reasons.append("stress_holdout_missing_combined_stress_high")

    context_cells, bad_cells, bad_examples = context_cell_audit(rows, materials)
    if bad_cells:
        stop_reasons.append(f"context_cells_not_full10_or_cross_split:{bad_cells}")
    for split, expected in config["expected_context_cells"].items():
        if split == "total":
            continue
        if int(context_cells.get(split, 0)) != int(expected):
            stop_reasons.append(f"context_cell_count_mismatch:{split}:{context_cells.get(split, 0)}!={expected}")

    cross_split = cross_split_counts(rows, ["clean_context_cell_id", "nuisance_cell_id", "random_seed", "seed_block"])
    for column, count in cross_split.items():
        if count:
            stop_reasons.append(f"{column}_crosses_split_boundaries:{count}")

    profile_signature_overlap = signature_split_overlap(config)
    for pair, payload in profile_signature_overlap.items():
        count = int(payload["count"])
        shared_names = payload["shared"]
        if count != 1:
            stop_reasons.append(f"profile_parameter_signature_overlap_not_nominal_only:{pair}:{count}")
            continue
        left, right = pair.split("|", 1)
        if shared_names[0][left] != ["nominal"] or shared_names[0][right] != ["nominal"]:
            stop_reasons.append(f"profile_parameter_signature_overlap_not_nominal_only:{pair}:{shared_names}")

    profile_counts = Counter(str(row.get("physical_perturbation_profile", "")) for row in rows)
    profile_split_counts = Counter(
        f"{row.get('split','')}|{row.get('physical_perturbation_profile','')}" for row in rows
    )
    warnings.append("v6_train_is_perturbation_augmented_validation_and_stress_use_independent_seed_blocks_and_shifted_profile_parameters")

    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_v6_physical_robust_preflight.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "shadow_or_final_used": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "claim_scope": config["claim_scope"],
        "gate_passed": gate_passed,
        "decision": "v6_physical_robust_preflight_passed_ready_for_development_geant4" if gate_passed else "stop_v6_physical_robust_preflight",
        "profile": profile,
        "row_count": len(rows),
        "rows_by_split": {str(key): int(value) for key, value in sorted(rows_by_split.items())},
        "context_cells_by_split": context_cells,
        "profiles_by_split": profiles_by_split,
        "profile_counts": {str(key): int(value) for key, value in sorted(profile_counts.items())},
        "profile_split_counts": {str(key): int(value) for key, value in sorted(profile_split_counts.items())},
        "profile_signature_overlap": profile_signature_overlap,
        "cross_split_counts": cross_split,
        "bad_cell_examples": bad_examples,
        "stop_reasons": stop_reasons,
        "warnings": warnings,
        "software": {"python": platform.python_version()},
    }
    write_json(output_dir / "v8a_multiclass_context_v6_physical_robust_preflight_gate.json", gate)
    print(f"decision={gate['decision']} gate_passed={str(gate_passed).lower()} rows={len(rows)}")


if __name__ == "__main__":
    main()
