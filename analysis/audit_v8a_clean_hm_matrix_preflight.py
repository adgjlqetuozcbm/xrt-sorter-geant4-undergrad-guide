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
    "development-only clean H/M matrix preflight; source-on default sidecar sampling design only; "
    "not a Geant4 run, not training evidence, not product accuracy, not hardware validation, "
    "not shadow/final validation, not full ten-material matrix, and not manuscript-grade powder XRD"
)

REQUIRED_LINEAGE_FIELDS = [
    "clean_matrix_origin",
    "source_family",
    "seed_block",
    "seed_block_seed",
    "count_target_bin",
    "count_target_photons",
    "clean_pair_id",
    "nuisance_cell_id",
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
    "pair_id",
    "count_target_bin",
    "count_target_photons",
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
    parser = argparse.ArgumentParser(description="Preflight-audit the v8A clean H/M source-on/default development matrix before any Geant4 run.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_clean_hm_development_matrix_config.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_clean_hm_development_matrix_preflight")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(project_root / args.config)
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite, project_root)

    profile = str(config["profile"])
    matrix_path = project_root / "source_models" / "config" / "material_sorting_matrix" / profile / "material_sorting_matrix.csv"
    manifest_path = matrix_path.parent / "matrix_manifest.json"
    rows = read_csv(matrix_path)
    manifest = load_json(manifest_path)
    peak_gate_path = project_root / "results" / "accuracy_v3" / "v8a_peak_provenance_audit" / "v8a_peak_provenance_gate.json"
    peak_gate = load_json(peak_gate_path) if peak_gate_path.exists() else {"gate_passed": False, "missing": True}

    stop_reasons: list[str] = []
    warnings: list[str] = []
    if config.get("status") != "development_preregistration":
        stop_reasons.append("Config status is not development_preregistration.")
    if not bool(peak_gate.get("gate_passed")):
        stop_reasons.append("Peak provenance gate is missing or did not pass.")
    expected_total = int(config["expected_rows"]["total"])
    if len(rows) != expected_total:
        stop_reasons.append(f"Matrix row count mismatch: {len(rows)} != expected {expected_total}.")
    if bool(manifest.get("shadow_or_final_used")) or bool(manifest.get("full_ten_material_matrix")):
        stop_reasons.append("Matrix manifest reports shadow/final or full ten-material matrix use.")
    if bool(manifest.get("reads_existing_xrt_cubes")):
        stop_reasons.append("Matrix manifest reports existing XRT cube reads.")
    if bool(manifest.get("runs_geant4")):
        stop_reasons.append("Matrix manifest reports Geant4 execution; preflight must not run Geant4.")
    if bool(manifest.get("training_unlocked")):
        stop_reasons.append("Matrix manifest unlocked training; preflight must keep training locked.")

    configured_materials = set(map(str, config["materials"]))
    material_set = set(row.get("material", "") for row in rows)
    if material_set != configured_materials or configured_materials != {"Hematite", "Magnetite"}:
        stop_reasons.append(f"Material set is not exactly H/M: observed={sorted(material_set)} config={sorted(configured_materials)}.")
    split_set = set(row.get("split", "") for row in rows)
    configured_splits = set(config["splits"])
    if split_set != configured_splits:
        stop_reasons.append(f"Split set mismatch: observed={sorted(split_set)} config={sorted(configured_splits)}.")
    source_modes = set(row.get("source_mode", "") for row in rows)
    stress_labels = set(row.get("stress_label", "") for row in rows)
    if source_modes != {"on"}:
        stop_reasons.append(f"Clean matrix must contain only source_mode=on, observed={sorted(source_modes)}.")
    if stress_labels != {"default"}:
        stop_reasons.append(f"Clean matrix must contain only stress_label=default, observed={sorted(stress_labels)}.")
    if any(str(row.get("source_id", "")).lower().find("off") >= 0 or "leakage" in str(row.get("source_id", "")).lower() for row in rows):
        stop_reasons.append("At least one source_id looks like source-off/leakage lineage.")

    for field in REQUIRED_LINEAGE_FIELDS:
        missing_count = sum(1 for row in rows if not str(row.get(field, "")).strip())
        if missing_count:
            stop_reasons.append(f"Required clean lineage field {field} is missing/blank in {missing_count} rows.")
    origin_values = set(row.get("clean_matrix_origin", "") for row in rows)
    source_family_values = set(row.get("source_family", "") for row in rows)
    if origin_values != {str(config["clean_matrix_origin"])}:
        stop_reasons.append(f"Clean matrix origin is mixed or mismatched: {sorted(origin_values)}.")
    if source_family_values != {str(config["source_family"])}:
        stop_reasons.append(f"Source family is mixed or mismatched: {sorted(source_family_values)}.")

    count_bin_photons = {str(item["count_target_bin"]): int(item["photons_per_row"]) for item in config["count_target_bins"]}
    for row in rows:
        count_bin = str(row.get("count_target_bin", ""))
        if count_bin not in count_bin_photons:
            stop_reasons.append(f"Unexpected count_target_bin {count_bin}.")
            break
        if int(float(row.get("count_target_photons", -1))) != count_bin_photons[count_bin]:
            stop_reasons.append(f"count_target_photons does not match config for bin {count_bin}.")
            break
    count_bins_by_split = {split: sorted({row.get("count_target_bin", "") for row in rows if row.get("split") == split}) for split in configured_splits}
    expected_bins = sorted(count_bin_photons)
    for split, bins in count_bins_by_split.items():
        if bins != expected_bins:
            stop_reasons.append(f"Split {split} count-target bin coverage mismatch: {bins} != {expected_bins}.")

    for key in ["seed_block", "random_seed", "seed_block_seed", "clean_pair_id"]:
        overlapping = {value: splits for value, splits in split_sets(rows, key).items() if len(splits) > 1}
        if overlapping:
            sample = dict(list(overlapping.items())[:5])
            stop_reasons.append(f"{key} crosses split boundaries; examples={sample}.")

    split_counts = count_key(rows, "split")
    material_counts = count_key(rows, "material")
    source_mode_counts = count_key(rows, "source_mode")
    stress_label_counts = count_key(rows, "stress_label")
    for split, expected in config["expected_rows"].items():
        if split == "total":
            continue
        observed = split_counts.get(split, 0)
        if observed != int(expected):
            stop_reasons.append(f"Split {split} row count mismatch: {observed} != expected {expected}.")

    cell_keys = list(config["strict_match_cell_keys"])
    cells: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in cell_keys)
        cells[key][str(row.get("material", ""))] += 1
    bad_cells: list[dict[str, Any]] = []
    strict_pairs_by_split: Counter[str] = Counter()
    cell_rows: list[dict[str, Any]] = []
    for key, counts in sorted(cells.items()):
        split = key[0]
        h_count = int(counts.get("Hematite", 0))
        m_count = int(counts.get("Magnetite", 0))
        strict_pairs = min(h_count, m_count)
        strict_pairs_by_split[split] += strict_pairs
        cell_row = {col: value for col, value in zip(cell_keys, key)}
        cell_row.update({"hematite_rows": h_count, "magnetite_rows": m_count, "strict_pairs": strict_pairs})
        cell_rows.append(cell_row)
        if h_count != m_count or h_count < 1:
            bad_cells.append(cell_row)
    if bad_cells:
        stop_reasons.append(f"{len(bad_cells)} strict nuisance cells are not H/M paired and balanced.")

    expected_pairs = {split: int(value) for split, value in config["expected_strict_pairs"].items() if split != "total"}
    minimum_pairs = {split: int(value) for split, value in config["minimum_strict_pairs"].items()}
    strict_pairs = {split: int(strict_pairs_by_split.get(split, 0)) for split in sorted(config["splits"])}
    for split, expected in expected_pairs.items():
        observed = strict_pairs.get(split, 0)
        if observed != expected:
            stop_reasons.append(f"Split {split} strict pair count mismatch: {observed} != expected {expected}.")
    for split, minimum in minimum_pairs.items():
        observed = strict_pairs.get(split, 0)
        if observed < minimum:
            stop_reasons.append(f"Split {split} strict pair support too low: {observed} < {minimum}.")

    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pair_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_id = str(row.get("clean_pair_id", ""))
        pair_counts[pair_id][str(row.get("material", ""))] += 1
        pair_splits[pair_id].add(str(row.get("split", "")))
    malformed_pairs = []
    for pair_id, counts in sorted(pair_counts.items()):
        if counts.get("Hematite", 0) != 1 or counts.get("Magnetite", 0) != 1 or len(pair_splits[pair_id]) != 1:
            malformed_pairs.append({"clean_pair_id": pair_id, "counts": dict(counts), "splits": sorted(pair_splits[pair_id])})
    if malformed_pairs:
        stop_reasons.append(f"{len(malformed_pairs)} clean_pair_id groups are not exactly one H/M pair within one split.")

    if any(boolish(row.get("shadow_or_final_used", False)) for row in rows):
        stop_reasons.append("At least one row reports shadow_or_final_used=true.")
    if any(not boolish(row.get("development_only", True)) for row in rows):
        stop_reasons.append("At least one row is not development_only=true.")
    if sorted(set(row.get("peak_table_id", "") for row in rows)) != [str(config["required_peak_table_id"])]:
        stop_reasons.append("Rows do not all use the required peak_table_id.")
    if len(rows) != len({row.get("output_prefix", "") for row in rows}):
        stop_reasons.append("output_prefix is not unique by row.")
    if len(rows) != len({row.get("phase_space_file", "") for row in rows}):
        stop_reasons.append("phase_space_file is not unique by row.")
    if any("_off_" in row.get("output_prefix", "").lower() or "leakage" in row.get("output_prefix", "").lower() for row in rows):
        stop_reasons.append("At least one output_prefix looks like source-off/leakage lineage.")
    if any(row.get("stress_label") != "default" for row in rows):
        warnings.append("A non-default stress label was observed; this should already be a stop reason.")

    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_clean_hm_matrix_preflight.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_clean_HM_matrix_preflight",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "development_run_prereg_unlocked": gate_passed,
        "claim_scope": CLAIM_SCOPE,
        "gate_passed": gate_passed,
        "decision": (
            "clean_matrix_preflight_passed_ready_for_development_run_prereg_only"
            if gate_passed
            else "stop_clean_matrix_preflight"
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
        "strict_pairs_by_split": strict_pairs,
        "minimum_strict_pairs": minimum_pairs,
        "bad_cell_count": len(bad_cells),
        "malformed_pair_id_count": len(malformed_pairs),
        "required_lineage_fields": REQUIRED_LINEAGE_FIELDS,
        "main_feature_forbidden_tokens": MAIN_FEATURE_FORBIDDEN_TOKENS,
        "development_run_unlock_conditions": config["development_run_unlock_conditions"],
        "training_unlock_conditions": config["training_unlock_conditions"],
        "stop_reasons": stop_reasons,
        "warnings": warnings,
        "software": {"python": platform.python_version()},
    }
    write_json(output_dir / "v8a_clean_hm_matrix_preflight_gate.json", gate)
    write_csv(
        output_dir / "v8a_clean_hm_matrix_strict_cell_balance.csv",
        cell_rows,
        cell_keys + ["hematite_rows", "magnetite_rows", "strict_pairs"],
    )
    if malformed_pairs:
        write_csv(
            output_dir / "v8a_clean_hm_matrix_malformed_pairs.csv",
            malformed_pairs,
            ["clean_pair_id", "counts", "splits"],
        )

    lines = [
        "# v8A clean H/M matrix preflight report",
        "",
        f"Generated: {generated_at}",
        "",
        "Scope: development-only matrix preflight. This report does not run Geant4, does not unlock training, and does not touch shadow/final.",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate_passed).lower()}`",
        f"- Profile: `{profile}`",
        f"- Rows: `{len(rows)}`",
        f"- Strict pairs: `{strict_pairs}`",
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
            "- This is clean sampling design evidence only.",
            "- It is not model performance evidence.",
            "- It is not product accuracy, hardware validation, shadow/final validation, full ten-material matrix evidence, or manuscript-grade powder XRD.",
        ]
    )
    (output_dir / "v8a_clean_hm_matrix_preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(
        "decision={decision} gate_passed={passed} rows={rows} strict_pairs={pairs} "
        "development_run_prereg_unlocked={run_unlocked} training_unlocked=false".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            rows=len(rows),
            pairs=strict_pairs,
            run_unlocked=str(gate["development_run_prereg_unlocked"]).lower(),
        )
    )


if __name__ == "__main__":
    main()
