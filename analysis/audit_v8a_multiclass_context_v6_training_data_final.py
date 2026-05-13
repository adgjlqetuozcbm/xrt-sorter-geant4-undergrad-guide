from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json


TARGET_MATERIALS = [
    "Quartz",
    "Calcite",
    "Orthoclase",
    "Albite",
    "Dolomite",
    "Pyrite",
    "Hematite",
    "Magnetite",
    "Chalcopyrite",
    "Galena",
]
EXPECTED_SPLIT_COUNTS = {"stress_holdout": 400, "train": 600, "validation": 400}
EXPECTED_PROFILE_COUNTS = {
    "nominal": 260,
    "calibration_shift_train_pos": 100,
    "calibration_shift_train_neg": 100,
    "resolution_blur_train_moderate": 100,
    "source_energy_scale_train_high": 100,
    "combined_train_moderate": 100,
    "calibration_shift_validation_mid": 80,
    "resolution_blur_validation_mid": 80,
    "source_energy_scale_validation_mid": 80,
    "combined_validation_mid": 80,
    "calibration_shift_stress_neg": 80,
    "resolution_blur_stress_high": 80,
    "source_energy_scale_stress_low": 80,
    "combined_stress_high": 80,
}
LEAK_TOKENS = [
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
    "count_target",
    "perturbation",
    "profile",
    "stress",
]


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().to_dict().items()}


def context_cell_integrity(frame: pd.DataFrame) -> dict[str, Any]:
    target_set = set(TARGET_MATERIALS)
    bad_examples: list[str] = []
    bad_count = 0
    cells_by_split: Counter[str] = Counter()
    cells_by_profile: Counter[str] = Counter()
    cross_split = 0
    for cell_id, group in frame.groupby("clean_context_cell_id", sort=True):
        splits = set(group["split"].astype(str))
        if len(splits) == 1:
            cells_by_split[next(iter(splits))] += 1
        else:
            cross_split += 1
        profiles = set(group["physical_perturbation_profile"].astype(str))
        if len(profiles) == 1:
            cells_by_profile[next(iter(profiles))] += 1
        material_counts = group["material"].astype(str).value_counts().to_dict()
        if set(material_counts) != target_set or any(int(count) != 1 for count in material_counts.values()):
            bad_count += 1
            if len(bad_examples) < 8:
                bad_examples.append(str(cell_id))
    return {
        "cell_count": int(frame["clean_context_cell_id"].nunique()),
        "bad_cell_count": int(bad_count),
        "cross_split_cell_count": int(cross_split),
        "cells_by_split": {str(key): int(value) for key, value in sorted(cells_by_split.items())},
        "cells_by_profile": {str(key): int(value) for key, value in sorted(cells_by_profile.items())},
        "bad_cell_examples": bad_examples,
    }


def cross_split_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in columns:
        cross = 0
        for _, group in frame.groupby(column, sort=False):
            if group["split"].astype(str).nunique() > 1:
                cross += 1
        result[column] = int(cross)
    return result


def main_feature_integrity(frame: pd.DataFrame, main_cols: list[str]) -> dict[str, Any]:
    numeric = frame[main_cols].apply(pd.to_numeric, errors="coerce") if main_cols else pd.DataFrame()
    return {
        "main_feature_count": int(len(main_cols)),
        "lineage_like_main_features": [col for col in main_cols if any(token in col.lower() for token in LEAK_TOKENS)],
        "nan_count": int(numeric.isna().sum().sum()) if not numeric.empty else 0,
        "inf_count": int(np.isinf(numeric.to_numpy(dtype=np.float64)).sum()) if not numeric.empty else 0,
    }


def profiles_by_split(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        split: sorted(group["physical_perturbation_profile"].astype(str).unique().tolist())
        for split, group in frame.groupby("split", sort=True)
    }


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A v6 physical robustness training-data audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Training unlocked: `{str(gate['training_unlocked']).lower()}`",
        f"- Samples: `{gate['sample_count']}`",
        f"- Train profiles: `{gate['profiles_by_split'].get('train', [])}`",
        f"- Validation profiles: `{gate['profiles_by_split'].get('validation', [])}`",
        f"- Stress profiles: `{gate['profiles_by_split'].get('stress_holdout', [])}`",
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in gate["warnings"]) if gate["warnings"] else lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiclass_context_v6_training_data_final_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-audit v8A v6 physical robustness feature table before model diagnostics.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--matrix-preflight-gate", default="results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_preflight/v8a_multiclass_context_v6_physical_robust_preflight_gate.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    feature_manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    matrix_preflight = load_json(as_project_path(project_root, args.matrix_preflight_gate))
    main_cols, control_cols, total_count_cols, _, _ = feature_sets(frame)

    split_counts = value_counts(frame, "split")
    material_counts = value_counts(frame, "material")
    profile_counts = value_counts(frame, "physical_perturbation_profile")
    split_profile_counts = (
        frame.groupby(["split", "physical_perturbation_profile"], sort=True)
        .size()
        .reset_index(name="count")
    )
    split_profile_counts.to_csv(output_dir / "v8a_multiclass_context_v6_split_profile_counts.csv", index=False)
    split_profiles = profiles_by_split(frame)
    cell_integrity = context_cell_integrity(frame)
    cross_split = cross_split_counts(frame, ["clean_context_cell_id", "nuisance_cell_id", "random_seed", "seed_block"])
    feature_integrity = main_feature_integrity(frame, main_cols)

    stop_reasons: list[str] = []
    warnings: list[str] = []
    if not bool(matrix_preflight.get("gate_passed", False)):
        stop_reasons.append("matrix_preflight_gate_not_passed")
    if bool(feature_manifest.get("shadow_or_final_used", False)):
        stop_reasons.append("shadow_or_final_used_detected")
    if set(frame["material"].astype(str).unique()) != set(TARGET_MATERIALS):
        stop_reasons.append("target_material_set_mismatch")
    if split_counts != EXPECTED_SPLIT_COUNTS:
        stop_reasons.append(f"split_count_mismatch:{split_counts}")
    if any(count != 140 for count in material_counts.values()):
        stop_reasons.append(f"material_count_not_balanced:{material_counts}")
    if profile_counts != EXPECTED_PROFILE_COUNTS:
        stop_reasons.append(f"profile_count_mismatch:{profile_counts}")
    if len(split_profiles.get("train", [])) < 6 or "combined_train_moderate" not in split_profiles.get("train", []):
        stop_reasons.append("train_split_not_perturbation_augmented")
    if "combined_validation_mid" not in split_profiles.get("validation", []):
        stop_reasons.append("validation_missing_combined_validation_mid")
    if "combined_stress_high" not in split_profiles.get("stress_holdout", []):
        stop_reasons.append("stress_missing_combined_stress_high")
    if cell_integrity["bad_cell_count"]:
        stop_reasons.append("context_cells_not_exactly_one_row_per_material")
    if cell_integrity["cross_split_cell_count"]:
        stop_reasons.append("context_cells_cross_split_boundaries")
    for col, count in cross_split.items():
        if count:
            stop_reasons.append(f"{col}_crosses_split_boundaries:{count}")
    if feature_integrity["lineage_like_main_features"]:
        stop_reasons.append("lineage_like_main_features_detected")
    if feature_integrity["nan_count"] or feature_integrity["inf_count"]:
        stop_reasons.append("nan_or_inf_in_main_features")
    warnings.append("source_off_control_not_required_for_v6_physical_robust_but_total_count_and_lineage_controls_are_required_in_model_gate")

    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_v6_training_data_final.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "shadow_or_final_used": False,
        "claim_scope": "development-only v6 perturbation-augmented physical robustness data audit; not hardware validation or product accuracy",
        "gate_passed": gate_passed,
        "training_unlocked": gate_passed,
        "decision": "v6_physical_robust_diagnostics_unlocked" if gate_passed else "stop_v6_training_data",
        "input_dir": args.input_dir,
        "sample_count": int(len(frame)),
        "split_counts": split_counts,
        "material_counts": material_counts,
        "profile_counts": profile_counts,
        "profiles_by_split": split_profiles,
        "context_cell_integrity": cell_integrity,
        "cross_split_counts": cross_split,
        "main_feature_integrity": feature_integrity,
        "main_feature_count": int(len(main_cols)),
        "control_feature_count": int(len(control_cols)),
        "total_count_feature_count": int(len(total_count_cols)),
        "stop_reasons": stop_reasons,
        "warnings": warnings,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(output_dir / "v8a_multiclass_context_v6_training_data_final_gate.json", json_clean(gate))
    write_report(output_dir, gate)
    print(f"decision={gate['decision']} gate_passed={str(gate_passed).lower()} training_unlocked={str(gate['training_unlocked']).lower()} samples={len(frame)}")


if __name__ == "__main__":
    main()
