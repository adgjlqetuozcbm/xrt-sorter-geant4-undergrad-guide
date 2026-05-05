from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
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


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def count_key(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def count_pair(rows: list[dict[str, str]], left: str, right: str) -> dict[str, int]:
    return dict(sorted(Counter(f"{row.get(left, '')}|{row.get(right, '')}" for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the v8A count-overlap extension preregistration artifact.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_count_overlap_extension_config.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_count_overlap_extension_prereg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(project_root / args.config)
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    peak_gate = load_json(project_root / "results/accuracy_v3/v8a_peak_provenance_audit/v8a_peak_provenance_gate.json")
    medium_stress_gate = load_json(project_root / "results/accuracy_v3/v8a_medium_event_feature_stress_gate/v8a_event_feature_stress_gate.json")
    count_balance_gate = load_json(project_root / "results/accuracy_v3/v8a_count_balance_sensitivity/v8a_count_balance_sensitivity_gate.json")
    profile = str(config["profile"])
    matrix_path = project_root / "source_models" / "config" / "material_sorting_matrix" / profile / "material_sorting_matrix.csv"
    manifest_path = matrix_path.parent / "matrix_manifest.json"
    rows = read_csv(matrix_path)
    matrix_manifest = load_json(manifest_path)

    stop_reasons: list[str] = []
    hard_guards = config.get("hard_guards", {})
    expected_guards = {
        "development_only": True,
        "shadow_or_final_used": False,
        "full_ten_material_matrix": False,
        "source_on_extension_only": True,
        "requires_peak_provenance_gate_passed": True,
        "requires_medium_stress_gate_passed": True,
        "requires_count_balance_sensitivity_stop": True,
        "requires_successor_source_peak_table": True,
        "requires_no_lineage_leakage": True,
    }
    for key, expected in expected_guards.items():
        if hard_guards.get(key) is not expected:
            stop_reasons.append(f"Config hard guard {key} is {hard_guards.get(key)!r}; expected {expected!r}.")
    expected_total = int(config["expected_rows"]["total"])
    if len(rows) != expected_total:
        stop_reasons.append(f"Matrix row count mismatch: {len(rows)} != expected {expected_total}.")
    if not bool(peak_gate.get("gate_passed")):
        stop_reasons.append("Peak provenance gate did not pass.")
    if not bool(medium_stress_gate.get("gate_passed")):
        stop_reasons.append("Medium stress gate did not pass.")
    if str(count_balance_gate.get("decision")) != "existing_medium_outputs_need_count_overlap_extension":
        stop_reasons.append(f"Count-balance sensitivity did not request extension: {count_balance_gate.get('decision')}.")
    if bool(count_balance_gate.get("gate_passed", True)):
        stop_reasons.append("Count-balance sensitivity unexpectedly passed; extension should not be preregistered.")
    peak_table_id = str(config["required_peak_table_id"])
    if any(str(row.get("peak_table_id")) != peak_table_id for row in rows):
        stop_reasons.append("At least one extension row does not use the successor peak table id.")
    if any(str(row.get("source_mode")) != "on" for row in rows):
        stop_reasons.append("Extension contains source-off rows; this preregistration is source-on only.")
    if any(str(row.get("development_only")).lower() != "true" for row in rows):
        stop_reasons.append("At least one extension row is not development_only.")
    if any(str(row.get("shadow_or_final_used")).lower() == "true" for row in rows):
        stop_reasons.append("At least one extension row reports shadow/final usage.")
    if bool(matrix_manifest.get("shadow_or_final_used")) or bool(matrix_manifest.get("full_ten_material_matrix")):
        stop_reasons.append("Extension manifest reports shadow/final or full ten-material matrix.")
    if not bool(matrix_manifest.get("source_on_extension_only")):
        stop_reasons.append("Extension manifest does not record source_on_extension_only=true.")

    split_counts = count_key(rows, "split")
    material_counts = count_key(rows, "material")
    source_mode_counts = count_key(rows, "source_mode")
    stress_label_counts = count_key(rows, "stress_label")
    split_material_counts = count_pair(rows, "split", "material")
    for split, expected in config["expected_rows"].items():
        if split == "total":
            continue
        observed = split_counts.get(split, 0)
        if observed != int(expected):
            stop_reasons.append(f"Split {split} row count mismatch: {observed} != expected {expected}.")
    for split in config["splits"]:
        per_material = [split_material_counts.get(f"{split}|{material}", 0) for material in config["materials"]]
        if len(set(per_material)) > 1:
            stop_reasons.append(f"Split {split} material balance is not uniform: {per_material}.")

    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/audit_v8a_count_overlap_extension_prereg.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": config["claim_scope"],
        "gate_passed": gate_passed,
        "decision": "count_overlap_extension_preregistered_not_run" if gate_passed else "stop_count_overlap_extension_prereg",
        "profile": profile,
        "base_profile": config["base_profile"],
        "peak_table_id": peak_table_id,
        "hard_guards": hard_guards,
        "row_count": len(rows),
        "split_counts": split_counts,
        "material_counts": material_counts,
        "source_mode_counts": source_mode_counts,
        "stress_label_counts": stress_label_counts,
        "split_material_counts": split_material_counts,
        "training_unlocked": False,
        "post_run_conditions": config["post_run_conditions"],
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version()},
    }
    (output_dir / "v8a_count_overlap_extension_prereg_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# v8A count-overlap extension preregistration report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only extension preregistration. The matrix is not run here, training is not unlocked here, and shadow/final remain sealed.",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Profile: `{profile}`",
        f"- Rows: `{len(rows)}`",
        f"- Peak table: `{peak_table_id}`",
        f"- Training unlocked: `{str(gate['training_unlocked']).lower()}`",
        "",
        "## Stop Reasons",
        "",
    ]
    if stop_reasons:
        lines.extend(f"- {reason}" for reason in stop_reasons)
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_count_overlap_extension_prereg_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"decision={gate['decision']} gate_passed={str(gate_passed).lower()} rows={len(rows)} training_unlocked=false")


if __name__ == "__main__":
    main()
