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
    "development-only count-robust feature transform for v8A H/M sidecar observability; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)


LINEAGE_COLUMNS = [
    "combined_feature_origin",
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
    "development_only",
    "shadow_or_final_used",
    "row_index",
    "events_path",
    "hits_path",
    "metadata_path",
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


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.fillna(0.0).astype(float) / denominator.fillna(0.0).astype(float).replace(0.0, np.nan).fillna(1e-12)


def peak_cols(main_cols: list[str]) -> list[str]:
    return [col for col in main_cols if col.startswith("diffraction_peak_") and col.endswith("_norm")]


def window_cols(main_cols: list[str]) -> list[str]:
    return [col for col in main_cols if col.startswith("diffraction_window_")]


def residualize_train_only(frame: pd.DataFrame, cols: list[str], total_col: str) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    result = pd.DataFrame(index=frame.index)
    train_mask = frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq("custom_diffraction_on")
    total_train = frame.loc[train_mask, total_col].astype(float).to_numpy(dtype=np.float64)
    total_all = frame[total_col].astype(float).to_numpy(dtype=np.float64)
    params: dict[str, dict[str, float]] = {}
    for col in cols:
        y_train = frame.loc[train_mask, col].astype(float).to_numpy(dtype=np.float64)
        if len(total_train) < 2 or np.var(total_train) <= 1e-12:
            slope = 0.0
            intercept = float(np.mean(y_train)) if len(y_train) else 0.0
        else:
            slope, intercept = np.polyfit(total_train, y_train, 1)
            slope = float(slope)
            intercept = float(intercept)
        residual = frame[col].astype(float).to_numpy(dtype=np.float64) - (intercept + slope * total_all)
        center = float(np.mean(y_train - (intercept + slope * total_train))) if len(y_train) else 0.0
        scale = float(np.std(y_train - (intercept + slope * total_train))) if len(y_train) else 1.0
        scale = scale if scale > 1e-12 else 1.0
        new_col = f"diffraction_residual_{col.removeprefix('diffraction_')}"
        result[new_col] = (residual - center) / scale
        params[new_col] = {
            "source_column": col,
            "fit_split": "train",
            "fit_source_mode": "custom_diffraction_on",
            "total_count_column": total_col,
            "intercept": intercept,
            "slope": slope,
            "center": center,
            "scale": scale,
        }
    return result, params


def build_features(frame: pd.DataFrame, total_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    main_cols, control_cols, _, overlap_cols, thickness_pose_cols = feature_sets(frame)
    p_cols = peak_cols(main_cols)
    w_cols = window_cols(main_cols)
    if total_col not in frame.columns:
        raise RuntimeError(f"Missing total-count column: {total_col}")
    if not p_cols:
        raise RuntimeError("No peak-level diffraction features available for count-robust transform.")

    robust = pd.DataFrame(index=frame.index)
    total_peak_sum = frame[p_cols].fillna(0.0).sum(axis=1).replace(0.0, np.nan).fillna(1e-12)
    total_window_sum = frame[w_cols].fillna(0.0).sum(axis=1).replace(0.0, np.nan).fillna(1e-12) if w_cols else total_peak_sum
    hematite_peaks = [col for col in p_cols if "_hematite_" in col]
    magnetite_peaks = [col for col in p_cols if "_magnetite_" in col]
    hematite_sum = frame[hematite_peaks].fillna(0.0).sum(axis=1) if hematite_peaks else pd.Series(0.0, index=frame.index)
    magnetite_sum = frame[magnetite_peaks].fillna(0.0).sum(axis=1) if magnetite_peaks else pd.Series(0.0, index=frame.index)

    for col in p_cols:
        suffix = col.removeprefix("diffraction_peak_").removesuffix("_norm")
        robust[f"diffraction_prop_peak_{suffix}"] = safe_divide(frame[col], total_peak_sum)
    for col in w_cols:
        suffix = col.removeprefix("diffraction_window_")
        robust[f"diffraction_prop_window_{suffix}"] = safe_divide(frame[col], total_window_sum)

    robust["diffraction_ratio_hm_peak_sum_balance"] = safe_divide(magnetite_sum - hematite_sum, magnetite_sum + hematite_sum + 1e-12)
    robust["diffraction_ratio_magnetite_to_hematite_peak_sum_log1p"] = np.log1p(magnetite_sum.clip(lower=0.0)) - np.log1p(hematite_sum.clip(lower=0.0))
    if "diffraction_window_hematite_unique_sum" in frame.columns and "diffraction_window_magnetite_unique_sum" in frame.columns:
        h_unique = frame["diffraction_window_hematite_unique_sum"].astype(float)
        m_unique = frame["diffraction_window_magnetite_unique_sum"].astype(float)
        robust["diffraction_ratio_unique_window_balance"] = safe_divide(m_unique - h_unique, m_unique + h_unique + 1e-12)
        robust["diffraction_ratio_unique_window_log1p"] = np.log1p(m_unique.clip(lower=0.0)) - np.log1p(h_unique.clip(lower=0.0))

    residual_source_cols = p_cols + [col for col in w_cols if col != "diffraction_window_all_peaks_sum"]
    residuals, residual_params = residualize_train_only(frame, residual_source_cols, total_col)
    robust = pd.concat([robust, residuals], axis=1)
    robust = robust.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    metadata = {
        "source_main_feature_columns": main_cols,
        "source_peak_feature_columns": p_cols,
        "source_window_feature_columns": w_cols,
        "robust_main_feature_columns": list(robust.columns),
        "control_feature_columns": control_cols,
        "overlap_feature_columns": overlap_cols,
        "thickness_pose_feature_columns": thickness_pose_cols,
        "residualization": residual_params,
        "removed_absolute_count_like_features": main_cols,
    }
    return robust, metadata


def copy_schema_gate(input_gate: dict[str, Any], output_dir_arg: str, sample_count: int, main_cols: list[str]) -> dict[str, Any]:
    gate = dict(input_gate)
    gate.update(
        {
            "generated_by": "analysis/build_v8a_count_robust_features.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol_name": "v8A_count_robust_feature_schema_gate",
            "claim_scope": CLAIM_SCOPE,
            "output_dir": output_dir_arg,
            "sample_count": sample_count,
            "feature_column_count": len(main_cols),
            "count_robust_feature_transform": "v8a_count_robust_v1",
            "gate_passed": bool(input_gate.get("gate_passed", False)),
            "tiny_training_gate_allowed": bool(input_gate.get("tiny_training_gate_allowed", False)),
            "decision": "count_robust_feature_schema_gate_passed_ready_for_reworked_training"
            if bool(input_gate.get("gate_passed", False)) and bool(input_gate.get("tiny_training_gate_allowed", False))
            else "stop_count_robust_feature_schema_gate",
        }
    )
    return gate


def write_report(output_dir: Path, manifest: dict[str, Any], gate: dict[str, Any]) -> None:
    lines = [
        "# v8A count-robust feature transform report",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Transform",
        "",
        f"- Transform id: `{manifest['transform_id']}`",
        f"- Input dir: `{manifest['input_dir']}`",
        f"- Output dir: `{manifest['output_dir']}`",
        f"- Samples: `{manifest['sample_count']}`",
        f"- Main feature count: `{manifest['main_feature_count']}`",
        f"- Removed absolute count-like source features: `{len(manifest['removed_absolute_count_like_features'])}`",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Training allowed: `{str(gate.get('tiny_training_gate_allowed', False)).lower()}`",
        "",
        "## Claim Boundary",
        "",
        "This transform creates development-only candidate features for renewed controls. It does not unlock shadow/final or any product, hardware, or manuscript-grade claim.",
        "",
    ]
    (output_dir / "v8a_count_robust_feature_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build split-safe count-robust v8A H/M feature table.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_event_to_feature")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_event_to_feature")
    parser.add_argument("--total-count-column", default="control_total_count_norm")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    input_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    input_manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    if bool(input_gate.get("shadow_or_final_used")) or bool(input_manifest.get("shadow_or_final_used")):
        raise RuntimeError("Refusing count-robust transform because input reports shadow/final use.")
    if bool(input_gate.get("reads_existing_xrt_cubes")) or bool(input_manifest.get("reads_existing_xrt_cubes")):
        raise RuntimeError("Refusing count-robust transform because input reports existing XRT cube reads.")

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    robust, metadata = build_features(frame, args.total_count_column)
    lineage_cols = [col for col in LINEAGE_COLUMNS if col in frame.columns]
    control_cols = [col for col in frame.columns if col.startswith("control_")]
    output = pd.concat([frame[lineage_cols].copy(), frame[control_cols].copy(), robust], axis=1)
    output = output.loc[:, ~output.columns.duplicated()].copy()

    main_cols = list(robust.columns)
    leak_tokens = ["material", "source_id", "sample_id", "path", "seed", "thickness", "pose", "split", "row_index"]
    lineage_like = [col for col in main_cols if any(token in col.lower() for token in leak_tokens)]
    if lineage_like:
        raise RuntimeError(f"Count-robust main feature names look lineage-like: {lineage_like}")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = dict(input_manifest)
    manifest.update(
        {
            "generated_by": "analysis/build_v8a_count_robust_features.py",
            "generated_at_utc": generated_at,
            "protocol_name": "v8A_count_robust_feature_transform",
            "transform_id": "v8a_count_robust_v1",
            "development_only": True,
            "shadow_or_final_used": False,
            "reads_existing_xrt_cubes": False,
            "runs_geant4": False,
            "claim_scope": CLAIM_SCOPE,
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "sample_count": int(len(output)),
            "main_feature_count": int(len(main_cols)),
            "control_feature_count": int(len([col for col in output.columns if col.startswith("control_")])),
            "main_feature_columns": main_cols,
            "control_feature_columns": [col for col in output.columns if col.startswith("control_")],
            "lineage_columns_excluded_from_main_features": lineage_cols,
            "removed_absolute_count_like_features": metadata["removed_absolute_count_like_features"],
            "source_main_feature_columns": metadata["source_main_feature_columns"],
            "residualization_fit_policy": "fit only on train split and custom_diffraction_on rows; apply frozen parameters to validation and stress_holdout",
            "residualization": metadata["residualization"],
        }
    )
    gate = copy_schema_gate(input_gate, args.output_dir, int(len(output)), main_cols)
    gate["lineage_like_main_features"] = lineage_like
    gate["no_lineage_like_main_features"] = not lineage_like
    output.to_csv(output_dir / "v8a_event_sidecar_features.csv", index=False, lineterminator="\n")
    if (input_dir / "v8a_event_sidecar_long.csv").exists():
        # Keep the original long table only as provenance for downstream audits; transformed features are in the feature table.
        (output_dir / "v8a_event_sidecar_long.csv").write_text(
            (input_dir / "v8a_event_sidecar_long.csv").read_text(encoding="utf-8"),
            encoding="utf-8",
            newline="\n",
        )
    if (input_dir / "v8a_event_control_audit.csv").exists():
        (output_dir / "v8a_event_control_audit.csv").write_text(
            (input_dir / "v8a_event_control_audit.csv").read_text(encoding="utf-8"),
            encoding="utf-8",
            newline="\n",
        )
    write_json(output_dir / "v8a_event_feature_manifest.json", json_clean(manifest))
    write_json(output_dir / "v8a_event_schema_gate.json", json_clean(gate))
    write_report(output_dir, manifest, gate)
    print(
        "decision={decision} samples={samples} main_features={features} removed_absolute={removed}".format(
            decision=gate["decision"],
            samples=len(output),
            features=len(main_cols),
            removed=len(metadata["removed_absolute_count_like_features"]),
        )
    )


if __name__ == "__main__":
    main()
