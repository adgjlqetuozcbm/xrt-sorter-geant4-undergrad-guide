from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_v8a_count_robust_features import (
    LINEAGE_COLUMNS,
    ensure_output_dir,
    json_clean,
    peak_cols,
    residualize_train_only,
    safe_divide,
    window_cols,
    write_json,
)
from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only v8A count-robust v2 feature candidates for H/M sidecar observability; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)

VARIANTS = {
    "v2_proportion_only",
    "v2_residualized_no_absolute_windows",
}


def family_peak_columns(columns: list[str], family: str) -> list[str]:
    return [col for col in columns if f"_{family}_" in col]


def unique_window_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if col in {"diffraction_window_hematite_unique_sum", "diffraction_window_magnetite_unique_sum"}]


def build_v2_features(frame: pd.DataFrame, total_col: str, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown v2 variant: {variant}")
    main_cols, control_cols, _, overlap_cols, thickness_pose_cols = feature_sets(frame)
    p_cols = peak_cols(main_cols)
    w_cols = window_cols(main_cols)
    u_cols = unique_window_columns(w_cols)
    if total_col not in frame.columns:
        raise RuntimeError(f"Missing total-count column: {total_col}")
    if not p_cols:
        raise RuntimeError("No peak-level diffraction features available for v2 transform.")

    robust = pd.DataFrame(index=frame.index)
    peak_sum = frame[p_cols].fillna(0.0).sum(axis=1).replace(0.0, np.nan).fillna(1e-12)
    hematite_peaks = family_peak_columns(p_cols, "hematite")
    magnetite_peaks = family_peak_columns(p_cols, "magnetite")
    hematite_sum = frame[hematite_peaks].fillna(0.0).sum(axis=1) if hematite_peaks else pd.Series(0.0, index=frame.index)
    magnetite_sum = frame[magnetite_peaks].fillna(0.0).sum(axis=1) if magnetite_peaks else pd.Series(0.0, index=frame.index)

    for col in p_cols:
        suffix = col.removeprefix("diffraction_peak_").removesuffix("_norm")
        robust[f"diffraction_v2_prop_peak_{suffix}"] = safe_divide(frame[col], peak_sum)

    robust["diffraction_v2_prop_peak_hematite_family_share"] = safe_divide(hematite_sum, peak_sum)
    robust["diffraction_v2_prop_peak_magnetite_family_share"] = safe_divide(magnetite_sum, peak_sum)
    robust["diffraction_v2_ratio_hm_peak_family_balance"] = safe_divide(magnetite_sum - hematite_sum, magnetite_sum + hematite_sum + 1e-12)
    robust["diffraction_v2_ratio_magnetite_to_hematite_peak_family_log1p"] = (
        np.log1p(magnetite_sum.clip(lower=0.0)) - np.log1p(hematite_sum.clip(lower=0.0))
    )

    if len(u_cols) == 2:
        h_unique = frame["diffraction_window_hematite_unique_sum"].astype(float)
        m_unique = frame["diffraction_window_magnetite_unique_sum"].astype(float)
        unique_sum = (h_unique + m_unique).replace(0.0, np.nan).fillna(1e-12)
        robust["diffraction_v2_prop_window_hematite_unique_share"] = safe_divide(h_unique, unique_sum)
        robust["diffraction_v2_prop_window_magnetite_unique_share"] = safe_divide(m_unique, unique_sum)
        robust["diffraction_v2_ratio_unique_window_balance"] = safe_divide(m_unique - h_unique, m_unique + h_unique + 1e-12)
        robust["diffraction_v2_ratio_unique_window_log1p"] = np.log1p(m_unique.clip(lower=0.0)) - np.log1p(h_unique.clip(lower=0.0))

    residual_params: dict[str, dict[str, float]] = {}
    if variant == "v2_residualized_no_absolute_windows":
        residuals, residual_params = residualize_train_only(frame, p_cols, total_col)
        residuals = residuals.rename(columns=lambda col: col.replace("diffraction_residual_", "diffraction_v2_residual_"))
        residual_params = {
            key.replace("diffraction_residual_", "diffraction_v2_residual_"): value
            for key, value in residual_params.items()
        }
        robust = pd.concat([robust, residuals], axis=1)

    robust = robust.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    metadata = {
        "source_main_feature_columns": main_cols,
        "source_peak_feature_columns": p_cols,
        "source_window_feature_columns": w_cols,
        "source_unique_window_feature_columns": u_cols,
        "robust_main_feature_columns": list(robust.columns),
        "control_feature_columns": control_cols,
        "overlap_feature_columns": overlap_cols,
        "thickness_pose_feature_columns": thickness_pose_cols,
        "residualization": residual_params,
        "removed_absolute_count_like_features": main_cols,
        "removed_absolute_window_features": w_cols,
        "variant": variant,
    }
    return robust, metadata


def copy_schema_gate(input_gate: dict[str, Any], output_dir_arg: str, sample_count: int, main_cols: list[str], variant: str) -> dict[str, Any]:
    gate = dict(input_gate)
    gate.update(
        {
            "generated_by": "analysis/build_v8a_count_robust_v2_features.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol_name": "v8A_count_robust_v2_feature_schema_gate",
            "claim_scope": CLAIM_SCOPE,
            "output_dir": output_dir_arg,
            "sample_count": sample_count,
            "feature_column_count": len(main_cols),
            "count_robust_feature_transform": variant,
            "gate_passed": bool(input_gate.get("gate_passed", False)),
            "tiny_training_gate_allowed": bool(input_gate.get("tiny_training_gate_allowed", False)),
            "decision": "count_robust_v2_feature_schema_gate_passed_ready_for_reworked_training"
            if bool(input_gate.get("gate_passed", False)) and bool(input_gate.get("tiny_training_gate_allowed", False))
            else "stop_count_robust_v2_feature_schema_gate",
        }
    )
    return gate


def write_report(output_dir: Path, manifest: dict[str, Any], gate: dict[str, Any]) -> None:
    lines = [
        "# v8A count-robust v2 feature transform report",
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
        f"- Removed absolute window source features: `{len(manifest['removed_absolute_window_features'])}`",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Training allowed: `{str(gate.get('tiny_training_gate_allowed', False)).lower()}`",
        "",
        "## Claim Boundary",
        "",
        "This transform creates development-only candidate features for renewed controls. It does not unlock shadow/final, a large development matrix, or any product, hardware, or manuscript-grade claim.",
        "",
    ]
    (output_dir / "v8a_count_robust_v2_feature_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build split-safe count-robust v2 v8A H/M feature candidates.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_event_to_feature")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
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
        raise RuntimeError("Refusing v2 transform because input reports shadow/final use.")
    if bool(input_gate.get("reads_existing_xrt_cubes")) or bool(input_manifest.get("reads_existing_xrt_cubes")):
        raise RuntimeError("Refusing v2 transform because input reports existing XRT cube reads.")

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    robust, metadata = build_v2_features(frame, args.total_count_column, args.variant)
    lineage_cols = [col for col in LINEAGE_COLUMNS if col in frame.columns]
    control_cols = [col for col in frame.columns if col.startswith("control_")]
    output = pd.concat([frame[lineage_cols].copy(), frame[control_cols].copy(), robust], axis=1)
    output = output.loc[:, ~output.columns.duplicated()].copy()

    main_cols = list(robust.columns)
    leak_tokens = ["material", "source_id", "sample_id", "path", "seed", "thickness", "pose", "split", "row_index"]
    lineage_like = [col for col in main_cols if any(token in col.lower() for token in leak_tokens)]
    if lineage_like:
        raise RuntimeError(f"v2 main feature names look lineage-like: {lineage_like}")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = dict(input_manifest)
    manifest.update(
        {
            "generated_by": "analysis/build_v8a_count_robust_v2_features.py",
            "generated_at_utc": generated_at,
            "protocol_name": "v8A_count_robust_v2_feature_transform",
            "transform_id": args.variant,
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
            "removed_absolute_window_features": metadata["removed_absolute_window_features"],
            "source_main_feature_columns": metadata["source_main_feature_columns"],
            "residualization_fit_policy": "fit only on train split and custom_diffraction_on rows; apply frozen parameters to validation and stress_holdout"
            if metadata["residualization"]
            else "not_used_for_this_variant",
            "residualization": metadata["residualization"],
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
    )
    gate = copy_schema_gate(input_gate, args.output_dir, int(len(output)), main_cols, args.variant)
    gate["lineage_like_main_features"] = lineage_like
    gate["no_lineage_like_main_features"] = not lineage_like

    output.to_csv(output_dir / "v8a_event_sidecar_features.csv", index=False, lineterminator="\n")
    for filename in ["v8a_event_sidecar_long.csv", "v8a_event_control_audit.csv"]:
        source_path = input_dir / filename
        if source_path.exists():
            (output_dir / filename).write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    write_json(output_dir / "v8a_event_feature_manifest.json", json_clean(manifest))
    write_json(output_dir / "v8a_event_schema_gate.json", json_clean(gate))
    write_report(output_dir, manifest, gate)
    print(
        "decision={decision} variant={variant} samples={samples} main_features={features} residualized={residualized}".format(
            decision=gate["decision"],
            variant=args.variant,
            samples=len(output),
            features=len(main_cols),
            residualized=str(bool(metadata["residualization"])).lower(),
        )
    )


if __name__ == "__main__":
    main()
