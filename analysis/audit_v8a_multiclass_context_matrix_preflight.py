from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLAIM_SCOPE = (
    "development-only v8A ten-material context matrix preflight; source-on default sidecar sampling design only; "
    "not a Geant4 run, not training evidence, not product accuracy, not hardware validation, "
    "not shadow/final validation, and not manuscript-grade powder XRD"
)

REQUIRED_LINEAGE_FIELDS = [
    "clean_matrix_origin",
    "source_family",
    "seed_block",
    "seed_block_seed",
    "count_target_bin",
    "count_target_photons",
    "clean_context_cell_id",
    "nuisance_cell_id",
    "context_replicate_index",
    "context_material_count",
]

MAIN_FEATURE_FORBIDDEN_TOKENS = [
    "material",
    "source_id",
    "source_family",
    "sample_id",
    "seed",
    "thickness",
    "pose",
    "split",
    "origin",
    "path",
    "row_index",
    "context_cell",
    "nuisance_cell",
    "count_target_bin",
    "count_target_photons",
]


def load_json(path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
    if not path.exists():
        if missing_ok:
            return {"missing": True, "path": path.as_posix()}
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_output_dir(path: Path, overwrite: bool, project_root: Path) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace preflight artifacts.")
    if path.exists() and any(path.iterdir()) and overwrite:
        resolved_output = path.resolve()
        resolved_results = (project_root / "results" / "accuracy_v3").resolve()
        if not resolved_output.is_relative_to(resolved_results):
            raise RuntimeError(f"Refusing to clean unexpected output path: {resolved_output}")
        shutil.rmtree(resolved_output)
    path.mkdir(parents=True, exist_ok=True)


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_sets(rows: list[dict[str, str]], key: str) -> dict[str, list[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values[str(row.get(key, ""))].add(str(row.get("split", "")))
    return {value: sorted(splits) for value, splits in sorted(values.items()) if value}


def count_key(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight-audit the v8A ten-material context matrix before any Geant4 run.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_multiclass_context_v1_config.json")
    parser.add_argument("--readiness-gate", default="results/accuracy_v3/v8a_multiclass_context_v1_readiness/v8a_multiclass_context_readiness_gate.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v1_matrix_preflight")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(as_project_path(project_root, args.config))
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite, project_root)

    profile = str(config["profile"])
    matrix_path = project_root / "source_models" / "config" / "material_sorting_matrix" / profile / "material_sorting_matrix.csv"
    manifest_path = matrix_path.parent / "matrix_manifest.json"
    rows = read_csv(matrix_path)
    manifest = load_json(manifest_path)
    readiness_gate = load_json(as_project_path(project_root, args.readiness_gate), missing_ok=True)
    required_inputs = config["required_inputs"]
    peak_gate = load_json(as_project_path(project_root, required_inputs["ten_material_peak_provenance_gate"]), missing_ok=True)

    stop_reasons: list[str] = []
    warnings: list[str] = []
    if config.get("status") != "development_readiness_review":
        stop_reasons.append("config_status_not_development_readiness_review")
    if not bool(readiness_gate.get("gate_passed", False)) or not bool(readiness_gate.get("matrix_generation_unlocked", False)):
        stop_reasons.append("readiness_gate_did_not_unlock_matrix_generation")
    if not bool(peak_gate.get("gate_passed", False)):
        stop_reasons.append("ten_material_peak_provenance_gate_missing_or_failed")

    expected_total = int(config["expected_rows"]["total"])
    if len(rows) != expected_total:
        stop_reasons.append(f"matrix_row_count_mismatch:{len(rows)}!={expected_total}")
    if bool(manifest.get("shadow_or_final_used")):
        stop_reasons.append("matrix_manifest_reports_shadow_or_final_use")
    if bool(manifest.get("reads_existing_xrt_cubes")):
        stop_reasons.append("matrix_manifest_reports_existing_xrt_cube_reads")
    if bool(manifest.get("runs_geant4")):
        stop_reasons.append("matrix_manifest_reports_geant4_execution")
    if bool(manifest.get("training_unlocked")):
        stop_reasons.append("matrix_manifest_unlocked_training")
    if not bool(manifest.get("full_ten_material_context_matrix", False)):
        stop_reasons.append("matrix_manifest_does_not_report_full_ten_material_context")

    configured_materials = set(map(str, config["target_materials"]))
    material_set = set(row.get("material", "") for row in rows)
    if material_set != configured_materials:
        stop_reasons.append(f"material_set_mismatch:observed={sorted(material_set)} configured={sorted(configured_materials)}")
    split_set = set(row.get("split", "") for row in rows)
    configured_splits = set(config["splits"])
    if split_set != configured_splits:
        stop_reasons.append(f"split_set_mismatch:observed={sorted(split_set)} configured={sorted(configured_splits)}")
    if set(row.get("source_mode", "") for row in rows) != {"on"}:
        stop_reasons.append("source_mode_not_on_only")
    if set(row.get("stress_label", "") for row in rows) != {"default"}:
        stop_reasons.append("stress_label_not_default_only")
    if any("off" in row.get("source_id", "").lower() or "leakage" in row.get("source_id", "").lower() for row in rows):
        stop_reasons.append("source_id_contains_off_or_leakage_lineage")

    for field in REQUIRED_LINEAGE_FIELDS:
        missing_count = sum(1 for row in rows if not str(row.get(field, "")).strip())
        if missing_count:
            stop_reasons.append(f"required_lineage_field_missing:{field}:{missing_count}")
    origin_values = set(row.get("clean_matrix_origin", "") for row in rows)
    source_family_values = set(row.get("source_family", "") for row in rows)
    if origin_values != {str(config["clean_matrix_origin"])}:
        stop_reasons.append(f"clean_matrix_origin_mismatch:{sorted(origin_values)}")
    if source_family_values != {str(config["source_family"])}:
        stop_reasons.append(f"source_family_mismatch:{sorted(source_family_values)}")

    count_bin_photons = {str(item["count_target_bin"]): int(item["photons_per_row"]) for item in config["planned_count_target_bins"]}
    for row in rows:
        count_bin = str(row.get("count_target_bin", ""))
        if count_bin not in count_bin_photons:
            stop_reasons.append(f"unexpected_count_target_bin:{count_bin}")
            break
        if int(float(row.get("count_target_photons", -1))) != count_bin_photons[count_bin]:
            stop_reasons.append(f"count_target_photons_mismatch_for_bin:{count_bin}")
            break
    expected_bins = sorted(count_bin_photons)
    count_bins_by_split = {split: sorted({row.get("count_target_bin", "") for row in rows if row.get("split") == split}) for split in configured_splits}
    for split, bins in count_bins_by_split.items():
        if bins != expected_bins:
            stop_reasons.append(f"split_count_bin_coverage_mismatch:{split}:{bins}!={expected_bins}")

    for key in ["seed_block", "random_seed", "seed_block_seed", "clean_context_cell_id"]:
        overlapping = {value: splits for value, splits in split_sets(rows, key).items() if len(splits) > 1}
        if overlapping:
            sample = dict(list(overlapping.items())[:5])
            stop_reasons.append(f"{key}_crosses_split_boundaries:{sample}")

    split_counts = count_key(rows, "split")
    material_counts = count_key(rows, "material")
    source_mode_counts = count_key(rows, "source_mode")
    stress_label_counts = count_key(rows, "stress_label")
    for split, expected in config["expected_rows"].items():
        if split == "total":
            continue
        observed = split_counts.get(split, 0)
        if observed != int(expected):
            stop_reasons.append(f"split_row_count_mismatch:{split}:{observed}!={expected}")

    cell_keys = list(config["strict_match_cell_keys"])
    cells: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in cell_keys)
        cells[key][str(row.get("material", ""))] += 1
    bad_cells: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    context_cells_by_split: Counter[str] = Counter()
    hm_pairs_by_split: Counter[str] = Counter()
    expected_material_count = len(configured_materials)
    for key, counts in sorted(cells.items()):
        split = key[0]
        context_cells_by_split[split] += 1
        hematite_rows = int(counts.get("Hematite", 0))
        magnetite_rows = int(counts.get("Magnetite", 0))
        if hematite_rows == 1 and magnetite_rows == 1:
            hm_pairs_by_split[split] += 1
        missing_materials = sorted(configured_materials - set(counts))
        duplicated_materials = sorted(material for material, count in counts.items() if count != 1)
        cell_row = {col: value for col, value in zip(cell_keys, key)}
        cell_row.update(
            {
                "material_rows": sum(counts.values()),
                "distinct_materials": len(counts),
                "hematite_rows": hematite_rows,
                "magnetite_rows": magnetite_rows,
                "hm_pair": int(hematite_rows == 1 and magnetite_rows == 1),
                "missing_materials": ";".join(missing_materials),
                "duplicated_materials": ";".join(duplicated_materials),
            }
        )
        cell_rows.append(cell_row)
        if sum(counts.values()) != expected_material_count or missing_materials or duplicated_materials:
            bad_cells.append(cell_row)
    if bad_cells:
        stop_reasons.append(f"context_cells_not_exactly_one_row_per_material:{len(bad_cells)}")

    context_cells = {split: int(context_cells_by_split.get(split, 0)) for split in sorted(config["splits"])}
    hm_pairs = {split: int(hm_pairs_by_split.get(split, 0)) for split in sorted(config["splits"])}
    for split, expected in config["expected_context_cells"].items():
        if split == "total":
            continue
        if context_cells.get(split, 0) != int(expected):
            stop_reasons.append(f"context_cell_count_mismatch:{split}:{context_cells.get(split, 0)}!={expected}")
    for split, expected in config["expected_hm_pairs"].items():
        if split == "total":
            continue
        if hm_pairs.get(split, 0) != int(expected):
            stop_reasons.append(f"hm_pair_count_mismatch:{split}:{hm_pairs.get(split, 0)}!={expected}")

    if any(boolish(row.get("shadow_or_final_used", False)) for row in rows):
        stop_reasons.append("row_reports_shadow_or_final_used")
    if any(not boolish(row.get("development_only", True)) for row in rows):
        stop_reasons.append("row_not_development_only")
    if sorted(set(row.get("peak_table_id", "") for row in rows)) != [str(config["required_peak_table_id"])]:
        stop_reasons.append("row_peak_table_id_mismatch")
    if len(rows) != len({row.get("output_prefix", "") for row in rows}):
        stop_reasons.append("output_prefix_not_unique")
    if len(rows) != len({row.get("phase_space_file", "") for row in rows}):
        stop_reasons.append("phase_space_file_not_unique")
    if any("_off_" in row.get("output_prefix", "").lower() or "leakage" in row.get("output_prefix", "").lower() for row in rows):
        stop_reasons.append("output_prefix_contains_off_or_leakage_lineage")

    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_matrix_preflight.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_multiclass_context_matrix_preflight",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "development_run_prereg_unlocked": gate_passed,
        "claim_scope": CLAIM_SCOPE,
        "gate_passed": gate_passed,
        "decision": (
            "multiclass_context_matrix_preflight_passed_ready_for_development_run_prereg_only"
            if gate_passed
            else "stop_multiclass_context_matrix_preflight"
        ),
        "profile": profile,
        "row_count": len(rows),
        "split_counts": split_counts,
        "material_counts": material_counts,
        "source_mode_counts": source_mode_counts,
        "stress_label_counts": stress_label_counts,
        "clean_matrix_origin_values": sorted(origin_values),
        "source_family_values": sorted(source_family_values),
        "count_bins_by_split": count_bins_by_split,
        "strict_match_cell_keys": cell_keys,
        "context_cells_by_split": context_cells,
        "hm_pairs_by_split": hm_pairs,
        "bad_cell_count": len(bad_cells),
        "required_lineage_fields": REQUIRED_LINEAGE_FIELDS,
        "main_feature_forbidden_tokens": MAIN_FEATURE_FORBIDDEN_TOKENS,
        "context_training_intent": config["context_training_intent"],
        "stop_reasons": stop_reasons,
        "warnings": warnings,
        "software": {"python": platform.python_version()},
    }
    write_json(output_dir / "v8a_multiclass_context_matrix_preflight_gate.json", gate)
    write_csv(
        output_dir / "v8a_multiclass_context_matrix_cell_balance.csv",
        cell_rows,
        cell_keys
        + [
            "material_rows",
            "distinct_materials",
            "hematite_rows",
            "magnetite_rows",
            "hm_pair",
            "missing_materials",
            "duplicated_materials",
        ],
    )

    lines = [
        "# v8A ten-material context matrix preflight report",
        "",
        f"Generated: {generated_at}",
        "",
        "Scope: development-only matrix preflight. This report does not run Geant4, does not unlock training, and does not touch shadow/final.",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate_passed).lower()}`",
        f"- Profile: `{profile}`",
        f"- Rows: `{len(rows)}`",
        f"- Context cells: `{context_cells}`",
        f"- H/M pairs inside context cells: `{hm_pairs}`",
        f"- Development-run prereg unlocked: `{str(gate['development_run_prereg_unlocked']).lower()}`",
        f"- Training unlocked: `{str(gate['training_unlocked']).lower()}`",
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in stop_reasons) if stop_reasons else lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This is ten-material context sampling design evidence only.",
            "- It is not model performance evidence.",
            "- It is not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD.",
        ]
    )
    (output_dir / "v8a_multiclass_context_matrix_preflight_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "decision={decision} gate_passed={passed} rows={rows} context_cells={cells} hm_pairs={pairs} "
        "development_run_prereg_unlocked={run_unlocked} training_unlocked=false".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            rows=len(rows),
            cells=context_cells,
            pairs=hm_pairs,
            run_unlocked=str(gate["development_run_prereg_unlocked"]).lower(),
        )
    )


if __name__ == "__main__":
    main()
