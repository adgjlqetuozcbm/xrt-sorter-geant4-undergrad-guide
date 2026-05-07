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
    "development-only final data audit for the admitted v8A H/M full-cell sidecar view; "
    "not product accuracy, hardware validation, shadow/final validation, full ten-material evidence, "
    "or manuscript-grade powder XRD"
)

DEFAULT_INPUT_DIR = (
    "results/accuracy_v3/"
    "v8a_fullcell_residualization_protocol_rework_v1_source_scaled_no_residualization_event_to_feature"
)
DEFAULT_ADMISSION_DIR = (
    "results/accuracy_v3/"
    "v8a_fullcell_residualization_protocol_rework_v1_source_scaled_no_residualization_admission"
)
DEFAULT_OUTPUT_DIR = "results/accuracy_v3/v8a_fullcell_training_data_final_audit"

HM_PAIR = ("Hematite", "Magnetite")
SPLITS = ("train", "validation", "stress_holdout")
COUNT_GAP_REVIEW_THRESHOLD = 0.25
COUNT_FEATURE_CORR_REVIEW_THRESHOLD = 0.50


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
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().to_dict().items()}


def split_material_balance(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, group in frame.groupby("split", sort=True):
        counts = group["material"].astype(str).value_counts().to_dict()
        rows.append(
            {
                "split": str(split),
                "hematite": int(counts.get("Hematite", 0)),
                "magnetite": int(counts.get("Magnetite", 0)),
                "balanced": int(counts.get("Hematite", 0)) == int(counts.get("Magnetite", 0)),
            }
        )
    return rows


def count_gap_by_split(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "control_total_count_norm" not in frame.columns:
        return rows
    for split in SPLITS:
        subset = frame[frame["split"].astype(str).eq(split)]
        h = subset[subset["material"].astype(str).eq("Hematite")]["control_total_count_norm"].to_numpy(dtype=np.float64)
        m = subset[subset["material"].astype(str).eq("Magnetite")]["control_total_count_norm"].to_numpy(dtype=np.float64)
        if len(h) == 0 or len(m) == 0:
            gap = 0.0
            pooled = 0.0
        else:
            pooled = float(np.sqrt(0.5 * (np.var(h) + np.var(m)) + 1e-12))
            gap = float(abs(float(np.mean(m) - np.mean(h))) / pooled) if pooled > 0 else 0.0
        rows.append(
            {
                "split": split,
                "hematite_n": int(len(h)),
                "magnetite_n": int(len(m)),
                "hematite_mean": float(np.mean(h)) if len(h) else 0.0,
                "magnetite_mean": float(np.mean(m)) if len(m) else 0.0,
                "hematite_std": float(np.std(h)) if len(h) else 0.0,
                "magnetite_std": float(np.std(m)) if len(m) else 0.0,
                "pooled_std": pooled,
                "standardized_gap": gap,
            }
        )
    return rows


def feature_count_correlations(frame: pd.DataFrame, main_cols: list[str]) -> list[dict[str, Any]]:
    if "control_total_count_norm" not in frame.columns:
        return []
    total = frame["control_total_count_norm"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for col in main_cols:
        values = frame[col].fillna(0.0).to_numpy(dtype=np.float64)
        if np.std(values) <= 1e-12 or np.std(total) <= 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(values, total)[0, 1])
        rows.append({"feature": col, "pearson_r_total_count_norm": corr, "abs_r": abs(corr)})
    return sorted(rows, key=lambda row: row["abs_r"], reverse=True)


def pair_integrity(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pair_column_present": "clean_match_pair_id" in frame.columns,
        "pair_count": 0,
        "pair_size_counts": {},
        "bad_pair_material_count": 0,
        "cross_split_pair_count": 0,
        "cross_seed_block_pair_count": 0,
        "fullcell_orientation_review": "delegated_to_admission_paired_null_gate",
        "bad_pair_examples": [],
    }
    if "clean_match_pair_id" not in frame.columns:
        return result
    group = frame.groupby("clean_match_pair_id", sort=True)
    result["pair_count"] = int(group.ngroups)
    result["pair_size_counts"] = {str(k): int(v) for k, v in group.size().value_counts().sort_index().to_dict().items()}
    bad_examples: list[str] = []
    bad_material = 0
    cross_split = 0
    cross_seed = 0
    for pair_id, pair_frame in group:
        materials = sorted(pair_frame["material"].astype(str).tolist())
        if len(pair_frame) != 2 or materials != ["Hematite", "Magnetite"]:
            bad_material += 1
            if len(bad_examples) < 8:
                bad_examples.append(str(pair_id))
        if pair_frame["split"].astype(str).nunique() > 1:
            cross_split += 1
        if "seed_block" in pair_frame.columns and pair_frame["seed_block"].astype(str).nunique() > 1:
            cross_seed += 1
    result["bad_pair_material_count"] = bad_material
    result["cross_split_pair_count"] = cross_split
    result["cross_seed_block_pair_count"] = cross_seed
    result["bad_pair_examples"] = bad_examples
    return result


def cross_split_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for col in columns:
        if col not in frame.columns:
            result[col] = 0
            continue
        cross = 0
        for _, group in frame.groupby(col, sort=False):
            if group["split"].astype(str).nunique() > 1:
                cross += 1
        result[col] = int(cross)
    return result


def main_feature_integrity(frame: pd.DataFrame, main_cols: list[str]) -> dict[str, Any]:
    leak_tokens = [
        "material",
        "source_id",
        "sample_id",
        "seed",
        "thickness",
        "pose",
        "split",
        "origin",
        "path",
        "row_index",
        "raw",
        "count_bin",
    ]
    numeric = frame[main_cols].apply(pd.to_numeric, errors="coerce") if main_cols else pd.DataFrame()
    nan_count = int(numeric.isna().sum().sum()) if not numeric.empty else 0
    inf_count = int(np.isinf(numeric.to_numpy(dtype=np.float64)).sum()) if not numeric.empty else 0
    return {
        "main_feature_count": int(len(main_cols)),
        "lineage_like_main_features": [col for col in main_cols if any(token in col.lower() for token in leak_tokens)],
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def write_report(output_dir: Path, gate: dict[str, Any], count_gaps: pd.DataFrame, correlations: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        if frame.empty:
            return ""
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for _, row in frame[columns].iterrows():
            values = []
            for col in columns:
                value = row[col]
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# v8A full-cell training data final audit",
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
        f"- Count controls required: `{str(gate['count_controls_required']).lower()}`",
        f"- Sample count: `{gate['sample_count']}`",
        f"- Main feature count: `{gate['main_feature_integrity']['main_feature_count']}`",
        "",
        "## Count Risk",
        "",
        table(count_gaps, ["split", "hematite_mean", "magnetite_mean", "standardized_gap"]),
        "",
        "## Top Count-Correlated Main Features",
        "",
        table(correlations.head(10), ["feature", "pearson_r_total_count_norm", "abs_r"]),
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
            "This audit admits only development training diagnostics. Any real-label model result must pass total-count-only and count-balanced controls before it can be interpreted as crystal-difference evidence.",
            "",
        ]
    )
    (output_dir / "v8a_fullcell_training_data_final_audit_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Final data audit for the admitted v8A full-cell H/M training view.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--admission-dir", default=DEFAULT_ADMISSION_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    admission_dir = as_project_path(project_root, args.admission_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    feature_manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    admission_gate = load_json(admission_dir / "v8a_crystal_clean_admission_gate.json")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, control_cols, total_count_cols, overlap_cols, thickness_pose_cols = feature_sets(frame)

    pairs = pair_integrity(frame)
    split_balance = split_material_balance(frame)
    seed_cross = cross_split_counts(frame, ["seed_block", "random_seed", "clean_pair_id", "clean_match_pair_id"])
    feature_integrity = main_feature_integrity(frame, main_cols)
    count_gaps = pd.DataFrame(count_gap_by_split(frame))
    correlations = pd.DataFrame(feature_count_correlations(frame, main_cols))

    expected_split_counts = {split: int(value) for split, value in frame["split"].value_counts().to_dict().items()}
    max_count_gap = float(count_gaps["standardized_gap"].max()) if not count_gaps.empty else 0.0
    max_count_feature_corr = float(correlations["abs_r"].max()) if not correlations.empty else 0.0
    count_controls_required = bool(
        max_count_gap > COUNT_GAP_REVIEW_THRESHOLD
        or max_count_feature_corr > COUNT_FEATURE_CORR_REVIEW_THRESHOLD
    )

    pass_items = {
        "schema_gate_passed": bool(schema_gate.get("gate_passed", False)),
        "admission_gate_passed": bool(admission_gate.get("gate_passed", False)),
        "admission_training_unlocked": bool(admission_gate.get("training_unlocked", False)),
        "feature_manifest_no_shadow_final": not bool(feature_manifest.get("shadow_or_final_used", False)),
        "feature_manifest_no_existing_xrt_cube_reads": not bool(feature_manifest.get("reads_existing_xrt_cubes", False)),
        "frame_development_only": bool(frame["development_only"].all()) if "development_only" in frame.columns else False,
        "frame_no_shadow_final": not bool(frame["shadow_or_final_used"].astype(bool).any()) if "shadow_or_final_used" in frame.columns else False,
        "no_shadow_final_splits": not any(str(item).lower() in {"shadow", "final"} for item in frame["split"].dropna().unique()),
        "source_on_default_only": value_counts(frame, "source_mode") == {"custom_diffraction_on": int(len(frame))}
        and value_counts(frame, "stress_label") == {"default": int(len(frame))},
        "hm_balanced_per_split": all(bool(row["balanced"]) for row in split_balance),
        "pair_column_present": bool(pairs["pair_column_present"]),
        "exact_two_rows_per_pair": pairs["pair_size_counts"] == {"2": int(pairs["pair_count"])},
        "exact_hm_per_pair": int(pairs["bad_pair_material_count"]) == 0,
        "no_cross_split_pairs": int(pairs["cross_split_pair_count"]) == 0,
        "no_cross_seed_block_pairs": int(pairs["cross_seed_block_pair_count"]) == 0,
        "no_cross_split_seed_block": int(seed_cross.get("seed_block", 0)) == 0,
        "no_cross_split_random_seed": int(seed_cross.get("random_seed", 0)) == 0,
        "no_cross_split_clean_pair_id": int(seed_cross.get("clean_pair_id", 0)) == 0,
        "no_cross_split_clean_match_pair_id": int(seed_cross.get("clean_match_pair_id", 0)) == 0,
        "main_features_present": len(main_cols) > 0,
        "no_lineage_like_main_features": not feature_integrity["lineage_like_main_features"],
        "no_main_feature_nan": int(feature_integrity["nan_count"]) == 0,
        "no_main_feature_inf": int(feature_integrity["inf_count"]) == 0,
        "no_duplicate_sample_id": int(frame["sample_id"].duplicated().sum()) == 0 if "sample_id" in frame.columns else False,
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    training_unlocked = bool(gate_passed and bool(admission_gate.get("training_unlocked", False)))
    decision = (
        "training_unlocked_with_count_controls_required"
        if training_unlocked and count_controls_required
        else ("training_unlocked_without_count_risk_flag" if training_unlocked else "stop_fullcell_training_data_final_audit")
    )
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_fullcell_training_data_final.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_fullcell_training_data_final_audit",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "admission_dir": args.admission_dir,
        "gate_passed": gate_passed,
        "training_unlocked": training_unlocked,
        "count_controls_required": count_controls_required,
        "decision": decision,
        "sample_count": int(len(frame)),
        "split_counts": expected_split_counts,
        "material_counts": value_counts(frame, "material"),
        "source_mode_counts": value_counts(frame, "source_mode"),
        "stress_label_counts": value_counts(frame, "stress_label"),
        "split_material_balance": split_balance,
        "pair_integrity": pairs,
        "seed_split_isolation": seed_cross,
        "main_feature_integrity": feature_integrity,
        "control_feature_counts": {
            "total_count": int(len(total_count_cols)),
            "overlap": int(len(overlap_cols)),
            "thickness_pose": int(len(thickness_pose_cols)),
            "all_controls": int(len(control_cols)),
        },
        "count_risk": {
            "review_thresholds": {
                "standardized_gap": COUNT_GAP_REVIEW_THRESHOLD,
                "feature_abs_correlation": COUNT_FEATURE_CORR_REVIEW_THRESHOLD,
            },
            "max_standardized_gap": max_count_gap,
            "max_main_feature_abs_corr_total_count_norm": max_count_feature_corr,
            "count_controls_required": count_controls_required,
        },
        "input_gate_decisions": {
            "schema_gate": schema_gate.get("decision"),
            "feature_manifest_candidate_view": feature_manifest.get("candidate_view"),
            "admission_gate": admission_gate.get("decision"),
            "admission_null_gate_protocol": admission_gate.get("null_gate_protocol"),
            "admission_paired_null_protocol": admission_gate.get("paired_null_protocol"),
        },
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest = {
        "generated_by": gate["generated_by"],
        "generated_at_utc": generated_at,
        "protocol_name": gate["protocol_name"],
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "input_dir": args.input_dir,
        "admission_dir": args.admission_dir,
        "output_dir": args.output_dir,
        "gate_file": "v8a_fullcell_training_data_final_audit_gate.json",
        "count_risk_file": "v8a_fullcell_training_data_count_risk.csv",
        "feature_count_correlation_file": "v8a_fullcell_training_data_feature_count_correlations.csv",
    }

    pd.DataFrame(split_balance).to_csv(output_dir / "v8a_fullcell_training_data_split_material_balance.csv", index=False, lineterminator="\n")
    count_gaps.to_csv(output_dir / "v8a_fullcell_training_data_count_risk.csv", index=False, lineterminator="\n")
    correlations.to_csv(output_dir / "v8a_fullcell_training_data_feature_count_correlations.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_fullcell_training_data_final_audit_manifest.json", manifest)
    write_json(output_dir / "v8a_fullcell_training_data_final_audit_gate.json", json_clean(gate))
    write_report(output_dir, json_clean(gate), count_gaps, correlations)
    print(
        "decision={decision} gate_passed={passed} training_unlocked={unlocked} count_controls_required={count} max_count_gap={gap:.4f} max_count_corr={corr:.4f}".format(
            decision=decision,
            passed=str(gate_passed).lower(),
            unlocked=str(training_unlocked).lower(),
            count=str(count_controls_required).lower(),
            gap=max_count_gap,
            corr=max_count_feature_corr,
        )
    )


if __name__ == "__main__":
    main()
