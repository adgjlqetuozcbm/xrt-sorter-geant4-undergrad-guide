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
    "development-only low-freedom v8A H/M feature rebuild candidates after paired-clean null-tail anatomy; "
    "not training evidence, product accuracy, hardware validation, shadow/final validation, full ten-material "
    "matrix, or manuscript-grade powder XRD"
)

LINEAGE_AND_CONTROL_PREFIXES = ("control_",)
BASE_LINEAGE_COLUMNS = [
    "sample_id",
    "split",
    "material",
    "random_seed",
    "source_id",
    "source_mode",
    "peak_table_id",
    "thickness_mm",
    "pose_index",
    "source_mode_raw",
    "source_energy_kev",
    "source_wavelength_a",
    "source_peak_table_id",
    "bin_axis",
    "stress_label",
    "clean_matrix_origin",
    "source_family",
    "seed_block",
    "seed_block_seed",
    "count_target_bin",
    "count_target_photons",
    "clean_pair_id",
    "nuisance_cell_id",
    "development_only",
    "shadow_or_final_used",
    "row_index",
    "events_path",
    "hits_path",
    "metadata_path",
    "clean_count_bin",
    "clean_match_pair_id",
    "clean_match_delta_total_count_norm",
]


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


def feature_family(col: str) -> str:
    if "peak_hematite" in col:
        return "peak_hematite"
    if "peak_magnetite" in col:
        return "peak_magnetite"
    if "window_hematite_unique" in col:
        return "window_hematite_unique"
    if "window_magnetite_unique" in col:
        return "window_magnetite_unique"
    if "window_all_peaks" in col:
        return "window_all_peaks"
    if "ratio" in col or "balance" in col:
        return "ratio_balance"
    return "other"


def zscore_train_apply(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    train_mask = frame["split"].astype(str).eq("train")
    result = pd.DataFrame(index=frame.index)
    for col in columns:
        train_values = frame.loc[train_mask, col].fillna(0.0).to_numpy(dtype=np.float64)
        center = float(np.mean(train_values)) if len(train_values) else 0.0
        scale = float(np.std(train_values)) if len(train_values) else 1.0
        scale = scale if scale > 1e-12 else 1.0
        result[col] = (frame[col].fillna(0.0).to_numpy(dtype=np.float64) - center) / scale
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def mean_or_zero(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.zeros(len(frame), dtype=np.float64)
    return frame[columns].fillna(0.0).to_numpy(dtype=np.float64).mean(axis=1)


def build_aggregate_features(frame: pd.DataFrame, main_cols: list[str]) -> pd.DataFrame:
    standardized = zscore_train_apply(frame, main_cols)
    families = {name: [col for col in main_cols if feature_family(col) == name] for name in sorted({feature_family(col) for col in main_cols})}
    output = pd.DataFrame(index=frame.index)
    h_peak = mean_or_zero(standardized, families.get("peak_hematite", []))
    m_peak = mean_or_zero(standardized, families.get("peak_magnetite", []))
    h_window = mean_or_zero(standardized, families.get("window_hematite_unique", []))
    m_window = mean_or_zero(standardized, families.get("window_magnetite_unique", []))
    all_window = mean_or_zero(standardized, families.get("window_all_peaks", []))
    ratio = mean_or_zero(standardized, families.get("ratio_balance", []))
    output["diffraction_tail_rebuild_peak_hematite_family_mean"] = h_peak
    output["diffraction_tail_rebuild_peak_magnetite_family_mean"] = m_peak
    output["diffraction_tail_rebuild_peak_family_difference_m_minus_h"] = m_peak - h_peak
    output["diffraction_tail_rebuild_peak_family_sum"] = m_peak + h_peak
    output["diffraction_tail_rebuild_window_hematite_unique"] = h_window
    output["diffraction_tail_rebuild_window_magnetite_unique"] = m_window
    output["diffraction_tail_rebuild_window_unique_difference_m_minus_h"] = m_window - h_window
    output["diffraction_tail_rebuild_window_all_peaks"] = all_window
    output["diffraction_tail_rebuild_ratio_hm_unique_balance"] = ratio
    return output.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def candidate_features(frame: pd.DataFrame, main_cols: list[str], view_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    standardized = zscore_train_apply(frame, main_cols)
    aggregate = build_aggregate_features(frame, main_cols)
    families = {name: [col for col in main_cols if feature_family(col) == name] for name in sorted({feature_family(col) for col in main_cols})}
    if view_id == "peak_family_balanced":
        selected = aggregate.copy()
        policy = "family_aggregates_plus_windows"
    elif view_id == "window_ratio_only":
        selected_cols = [
            "diffraction_tail_rebuild_window_hematite_unique",
            "diffraction_tail_rebuild_window_magnetite_unique",
            "diffraction_tail_rebuild_window_unique_difference_m_minus_h",
            "diffraction_tail_rebuild_window_all_peaks",
            "diffraction_tail_rebuild_ratio_hm_unique_balance",
        ]
        selected = aggregate[selected_cols].copy()
        policy = "windows_and_ratio_only"
    elif view_id == "leave_out_peak_hematite":
        keep = families.get("peak_magnetite", []) + families.get("window_hematite_unique", []) + families.get("window_magnetite_unique", []) + families.get("window_all_peaks", []) + families.get("ratio_balance", [])
        selected = standardized[keep].add_prefix("diffraction_tail_rebuild_keep_").copy()
        policy = "drop_peak_hematite"
    elif view_id == "leave_out_individual_peaks":
        selected_cols = [
            "diffraction_tail_rebuild_peak_hematite_family_mean",
            "diffraction_tail_rebuild_peak_magnetite_family_mean",
            "diffraction_tail_rebuild_peak_family_difference_m_minus_h",
            "diffraction_tail_rebuild_peak_family_sum",
            "diffraction_tail_rebuild_window_hematite_unique",
            "diffraction_tail_rebuild_window_magnetite_unique",
            "diffraction_tail_rebuild_window_unique_difference_m_minus_h",
            "diffraction_tail_rebuild_window_all_peaks",
            "diffraction_tail_rebuild_ratio_hm_unique_balance",
        ]
        selected = aggregate[selected_cols].copy()
        policy = "drop_individual_peaks"
    else:
        raise ValueError(f"Unknown tail rebuild candidate view: {view_id}")
    selected = selected.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    metadata = {
        "view_id": view_id,
        "feature_policy": policy,
        "source_main_feature_count": int(len(main_cols)),
        "source_family_counts": {family: int(len(cols)) for family, cols in families.items()},
        "output_feature_count": int(len(selected.columns)),
        "output_feature_columns": list(selected.columns),
    }
    return selected, metadata


def pair_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        split: int(frame[frame["split"].astype(str).eq(split)]["clean_match_pair_id"].nunique())
        for split in ["train", "validation", "stress_holdout"]
    }


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A tail rebuild candidate view",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- View: `{gate['view_id']}`",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Training unlocked: `{str(gate['training_unlocked']).lower()}`",
        f"- Main feature count: `{gate['main_feature_count']}`",
        f"- Matched pairs: train `{gate['matched_pair_counts']['train']}`, validation `{gate['matched_pair_counts']['validation']}`, stress-holdout `{gate['matched_pair_counts']['stress_holdout']}`",
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This builder only creates candidate feature views for null/admission auditing. It does not unlock training by itself.",
            "",
        ]
    )
    (output_dir / "v8a_tail_rebuild_feature_view_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build low-freedom v8A tail-rebuild feature candidates.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = project_root / args.config
    config = load_json(config_path)
    input_dir = project_root / str(config["input_feature_dir"])
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest_in = load_json(input_dir / "v8a_event_feature_manifest.json")
    tail_gate = load_json(project_root / str(config["input_tail_feature_family_gate"]))
    for name, payload in {"schema_gate": schema_gate, "manifest": manifest_in, "tail_gate": tail_gate}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing tail rebuild because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing tail rebuild because {name} reports existing XRT cube reads.")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features available.")
    features, feature_meta = candidate_features(frame, main_cols, args.view_id)
    lineage_cols = [col for col in BASE_LINEAGE_COLUMNS if col in frame.columns]
    control_cols = [col for col in frame.columns if col.startswith(LINEAGE_AND_CONTROL_PREFIXES)]
    output = pd.concat([frame[lineage_cols].copy(), frame[control_cols].copy(), features], axis=1)
    output = output.loc[:, ~output.columns.duplicated()].copy()
    main_feature_columns = list(features.columns)
    leak_tokens = ["material", "source_id", "sample_id", "path", "seed", "thickness", "pose", "split", "row_index", "stress", "origin", "count_bin"]
    lineage_like = [col for col in main_feature_columns if any(token in col.lower() for token in leak_tokens)]
    counts = pair_counts(output)
    pass_items = {
        "input_schema_gate_passed": bool(schema_gate.get("gate_passed", False)),
        "tail_feature_family_gate_detected_rebuild_need": str(tail_gate.get("decision")) == "feature_family_rebuild_prereg_needed",
        "development_only": bool(schema_gate.get("development_only", False)) and bool(manifest_in.get("development_only", False)),
        "no_shadow_final": not bool(schema_gate.get("shadow_or_final_used", False)) and not bool(manifest_in.get("shadow_or_final_used", False)),
        "no_existing_xrt_cube_reads": not bool(schema_gate.get("reads_existing_xrt_cubes", False)) and not bool(manifest_in.get("reads_existing_xrt_cubes", False)),
        "no_lineage_like_main_features": not lineage_like,
        "main_feature_count_reduced": len(main_feature_columns) <= len(main_cols),
        "train_pair_support_preserved": counts["train"] >= 100,
        "validation_pair_support_preserved": counts["validation"] >= 50,
        "stress_holdout_pair_support_preserved": counts["stress_holdout"] >= 50,
    }
    failure_labels = {
        "input_schema_gate_passed": "input_schema_gate_failed",
        "tail_feature_family_gate_detected_rebuild_need": "tail_feature_family_rebuild_need_not_confirmed",
        "development_only": "development_only_false",
        "no_shadow_final": "shadow_or_final_detected",
        "no_existing_xrt_cube_reads": "existing_xrt_cube_reads_detected",
        "no_lineage_like_main_features": "lineage_like_main_features_detected",
        "main_feature_count_reduced": "main_feature_count_not_reduced",
        "train_pair_support_preserved": "train_pair_support_not_preserved",
        "validation_pair_support_preserved": "validation_pair_support_not_preserved",
        "stress_holdout_pair_support_preserved": "stress_holdout_pair_support_not_preserved",
    }
    stop_reasons = [failure_labels[name] for name, passed in pass_items.items() if not passed]
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/build_v8a_tail_rebuild_features.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_tail_rebuild_candidate_feature_view",
        "view_id": args.view_id,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "config": args.config,
        "input_dir": str(config["input_feature_dir"]),
        "output_dir": args.output_dir,
        "sample_count": int(len(output)),
        "matched_pair_counts": counts,
        "main_feature_count": int(len(main_feature_columns)),
        "main_feature_columns": main_feature_columns,
        "lineage_like_main_features": lineage_like,
        "feature_metadata": feature_meta,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "gate_passed": gate_passed,
        "training_unlocked": False,
        "tiny_training_gate_allowed": False,
        "decision": "tail_rebuild_candidate_ready_for_null_audit" if gate_passed else "stop_tail_rebuild_candidate_view",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest = dict(manifest_in)
    manifest.update(
        {
            "generated_by": "analysis/build_v8a_tail_rebuild_features.py",
            "generated_at_utc": generated_at,
            "protocol_name": "v8A_tail_rebuild_candidate_feature_view",
            "transform_id": f"v8a_tail_rebuild_v1_{args.view_id}",
            "view_id": args.view_id,
            "development_only": True,
            "shadow_or_final_used": False,
            "reads_existing_xrt_cubes": False,
            "runs_geant4": False,
            "claim_scope": CLAIM_SCOPE,
            "input_dir": str(config["input_feature_dir"]),
            "output_dir": args.output_dir,
            "sample_count": int(len(output)),
            "main_feature_count": int(len(main_feature_columns)),
            "main_feature_columns": main_feature_columns,
            "source_main_feature_columns": main_cols,
            "feature_metadata": feature_meta,
            "training_unlocked": False,
        }
    )
    output.to_csv(output_dir / "v8a_event_sidecar_features.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_event_schema_gate.json", json_clean(gate))
    write_json(output_dir / "v8a_event_feature_manifest.json", json_clean(manifest))
    write_report(output_dir, gate)
    print(
        "decision={decision} view={view} gate_passed={passed} features={features}".format(
            decision=gate["decision"],
            view=args.view_id,
            passed=str(gate_passed).lower(),
            features=len(main_feature_columns),
        )
    )


if __name__ == "__main__":
    main()
