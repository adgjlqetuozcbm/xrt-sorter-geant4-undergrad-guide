from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only admission audit for a crystal-clean v8A H/M feature view; "
    "this unlocks only renewed development-only diagnostics, not product accuracy, shadow/final, "
    "hardware validation, or manuscript-grade powder XRD"
)

THRESHOLDS = {
    "train_pairs_min": 30,
    "validation_pairs_min": 20,
    "stress_holdout_pairs_min": 20,
    "nonmaterial_balanced_accuracy_max": 0.75,
    "fixed_threshold_null_hm_max": 0.55,
    "selected_threshold_null_hm_max": 0.55,
    "within_strata_fixed_threshold_null_hm_max": 0.55,
}


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def pair_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "clean_match_pair_id" not in frame.columns:
        return {"train": 0, "validation": 0, "stress_holdout": 0}
    return {
        split: int(frame[frame["split"].astype(str).eq(split)]["clean_match_pair_id"].nunique())
        for split in ["train", "validation", "stress_holdout"]
    }


def leakage_like_main_features(main_cols: list[str]) -> list[str]:
    tokens = ["material", "source_id", "sample_id", "path", "seed", "thickness", "pose", "split", "row_index", "stress", "origin", "count_bin"]
    return [col for col in main_cols if any(token in col.lower() for token in tokens)]


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A crystal-clean admission audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Training unlocked: `{str(gate['training_unlocked']).lower()}`",
        f"- Matched pairs: train `{gate['matched_pair_counts']['train']}`, validation `{gate['matched_pair_counts']['validation']}`, stress-holdout `{gate['matched_pair_counts']['stress_holdout']}`",
        f"- Max non-material balanced accuracy: `{gate['max_nonmaterial_balanced_accuracy']:.4f}`",
        f"- Fixed-threshold null max: `{gate['fixed_threshold_null_hm_max']:.4f}`",
        f"- Selected-threshold null max: `{gate['selected_threshold_null_hm_max']:.4f}`",
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Passing this audit would only allow renewed development-only diagnostics on the clean view. It never unlocks shadow/final, a large development matrix, full ten-material v8A, product accuracy, hardware validation, or manuscript-grade powder-XRD claims.",
            "",
        ]
    )
    (output_dir / "v8a_crystal_clean_admission_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether a v8A crystal-clean feature view is eligible for renewed training diagnostics.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--null-gate", required=True)
    parser.add_argument("--shortcut-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    null_gate = load_json(project_root / args.null_gate)
    shortcut_gate = load_json(project_root / args.shortcut_gate)
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    counts = pair_counts(frame)
    lineage_like = leakage_like_main_features(main_cols)
    max_nonmaterial = float(shortcut_gate.get("max_nonmaterial_balanced_accuracy", 1.0))
    fixed_null = float(null_gate.get("fixed_threshold_hm_min_recall_max", 1.0))
    selected_null = float(null_gate.get("selected_threshold_hm_min_recall_max", 1.0))
    within_null = float(null_gate.get("within_strata_fixed_threshold_hm_min_recall_max", 1.0))
    pass_items = {
        "view_schema_gate_passed": bool(schema_gate.get("gate_passed", False)),
        "view_training_not_preunlocked": not bool(schema_gate.get("training_unlocked", False)),
        "manifest_training_not_preunlocked": not bool(manifest.get("training_unlocked", False)),
        "development_only": bool(schema_gate.get("development_only", False)) and bool(manifest.get("development_only", False)),
        "no_shadow_final": not bool(schema_gate.get("shadow_or_final_used", False)) and not bool(manifest.get("shadow_or_final_used", False)),
        "no_existing_xrt_cube_reads": not bool(schema_gate.get("reads_existing_xrt_cubes", False)) and not bool(manifest.get("reads_existing_xrt_cubes", False)),
        "train_pair_support": counts["train"] >= THRESHOLDS["train_pairs_min"],
        "validation_pair_support": counts["validation"] >= THRESHOLDS["validation_pairs_min"],
        "stress_holdout_pair_support": counts["stress_holdout"] >= THRESHOLDS["stress_holdout_pairs_min"],
        "no_lineage_like_main_features": not lineage_like,
        "shortcut_gate_passed": bool(shortcut_gate.get("gate_passed", False)),
        "nonmaterial_predictability_below_ceiling": max_nonmaterial <= THRESHOLDS["nonmaterial_balanced_accuracy_max"],
        "null_gate_passed": bool(null_gate.get("gate_passed", False)),
        "fixed_threshold_null_below_ceiling": fixed_null <= THRESHOLDS["fixed_threshold_null_hm_max"],
        "selected_threshold_null_below_ceiling": selected_null <= THRESHOLDS["selected_threshold_null_hm_max"],
        "within_strata_null_below_ceiling": within_null <= THRESHOLDS["within_strata_fixed_threshold_null_hm_max"],
    }
    failure_labels = {
        "view_schema_gate_passed": "view_schema_gate_failed",
        "view_training_not_preunlocked": "view_training_was_preunlocked",
        "manifest_training_not_preunlocked": "manifest_training_was_preunlocked",
        "development_only": "development_only_false",
        "no_shadow_final": "shadow_or_final_detected",
        "no_existing_xrt_cube_reads": "existing_xrt_cube_reads_detected",
        "train_pair_support": "train_pair_support_below_minimum",
        "validation_pair_support": "validation_pair_support_below_minimum",
        "stress_holdout_pair_support": "stress_holdout_pair_support_below_minimum",
        "no_lineage_like_main_features": "lineage_like_main_features_detected",
        "shortcut_gate_passed": "shortcut_gate_failed",
        "nonmaterial_predictability_below_ceiling": "nonmaterial_predictability_exceeded_ceiling",
        "null_gate_passed": "null_gate_failed",
        "fixed_threshold_null_below_ceiling": "fixed_threshold_null_exceeded_ceiling",
        "selected_threshold_null_below_ceiling": "selected_threshold_null_exceeded_ceiling",
        "within_strata_null_below_ceiling": "within_strata_null_exceeded_ceiling",
    }
    stop_reasons = [failure_labels[name] for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_crystal_clean_admission.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_crystal_clean_admission_audit",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "null_gate": args.null_gate,
        "shortcut_gate": args.shortcut_gate,
        "gate_passed": gate_passed,
        "training_unlocked": gate_passed,
        "tiny_training_gate_allowed": gate_passed,
        "decision": "crystal_clean_view_training_diagnostics_unlocked" if gate_passed else "stop_crystal_clean_view_before_training",
        "matched_pair_counts": counts,
        "main_feature_count": int(len(main_cols)),
        "lineage_like_main_features": lineage_like,
        "max_nonmaterial_balanced_accuracy": max_nonmaterial,
        "fixed_threshold_null_hm_max": fixed_null,
        "selected_threshold_null_hm_max": selected_null,
        "within_strata_fixed_threshold_null_hm_max": within_null,
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(output_dir / "v8a_crystal_clean_admission_gate.json", json_clean(gate))
    write_report(output_dir, gate)
    print(
        "decision={decision} gate_passed={passed} training_unlocked={unlocked}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            unlocked=str(gate["training_unlocked"]).lower(),
        )
    )


if __name__ == "__main__":
    main()
