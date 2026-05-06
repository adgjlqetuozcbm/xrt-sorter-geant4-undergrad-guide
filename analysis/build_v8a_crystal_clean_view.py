from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_v8a_count_robust_features import LINEAGE_COLUMNS, ensure_output_dir, json_clean, write_json
from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only crystal-clean v8A H/M feature view; source/stress/origin/count/thickness/pose "
    "are nuisance controls, not model inputs; not product accuracy, hardware validation, shadow/final "
    "validation, or manuscript-grade powder XRD"
)

AUDIT_SUPPORT_THRESHOLDS = {
    "train_pairs_min": 30,
    "validation_pairs_min": 20,
    "stress_holdout_pairs_min": 20,
}

MATCH_STRATEGIES = {
    "strict_source_origin_thickness_pose_count": [
        "split",
        "stress_label",
        "combined_feature_origin",
        "source_id",
        "thickness_mm",
        "pose_index",
        "clean_count_bin",
    ],
    "clean_design_cell": [
        "split",
        "clean_matrix_origin",
        "source_family",
        "thickness_mm",
        "pose_index",
        "count_target_bin",
        "seed_block",
    ],
    "origin_thickness_pose_count": [
        "split",
        "stress_label",
        "combined_feature_origin",
        "thickness_mm",
        "pose_index",
        "clean_count_bin",
    ],
    "origin_thickness_pose": [
        "split",
        "stress_label",
        "combined_feature_origin",
        "thickness_mm",
        "pose_index",
    ],
    "thickness_pose": [
        "split",
        "stress_label",
        "thickness_mm",
        "pose_index",
    ],
    "thickness_pose_count": [
        "split",
        "stress_label",
        "thickness_mm",
        "pose_index",
        "clean_count_bin",
    ],
    "origin_count": [
        "split",
        "stress_label",
        "combined_feature_origin",
        "clean_count_bin",
    ],
    "count_only": [
        "split",
        "stress_label",
        "clean_count_bin",
    ],
}

BASE_NUISANCE_CATEGORICAL_COLUMNS = [
    "stress_label",
    "combined_feature_origin",
    "source_id",
    "thickness_mm",
    "pose_index",
    "clean_count_bin",
]

OPTIONAL_CLEAN_NUISANCE_CATEGORICAL_COLUMNS = [
    "clean_matrix_origin",
    "source_family",
    "seed_block",
    "count_target_bin",
]


def stable_str(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def add_count_bin(frame: pd.DataFrame, total_col: str, width: float) -> pd.Series:
    values = frame[total_col].fillna(0.0).to_numpy(dtype=np.float64)
    return np.floor(values / float(width)).astype(int).astype(str)


def exact_match_pairs(frame: pd.DataFrame, group_columns: list[str], total_col: str, strategy_name: str) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for keys, group in frame.groupby(group_columns, sort=True, observed=True):
        hematite = group[group["material"].astype(str).eq("Hematite")].sort_values([total_col, "sample_id"])
        magnetite = group[group["material"].astype(str).eq("Magnetite")].sort_values([total_col, "sample_id"])
        pair_count = min(len(hematite), len(magnetite))
        if pair_count <= 0:
            continue
        key_text = "|".join(stable_str(item) for item in (keys if isinstance(keys, tuple) else (keys,)))
        for pair_index in range(pair_count):
            h_row = hematite.iloc[pair_index].copy()
            m_row = magnetite.iloc[pair_index].copy()
            pair_id = f"{strategy_name}|{key_text}|pair{pair_index + 1:03d}"
            delta = abs(float(h_row[total_col]) - float(m_row[total_col]))
            h_row["clean_match_pair_id"] = pair_id
            m_row["clean_match_pair_id"] = pair_id
            h_row["clean_match_delta_total_count_norm"] = delta
            m_row["clean_match_delta_total_count_norm"] = delta
            rows.extend([h_row, m_row])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def pair_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {"train": 0, "validation": 0, "stress_holdout": 0}
    return {
        split: int(frame[frame["split"].astype(str).eq(split)]["clean_match_pair_id"].nunique())
        for split in ["train", "validation", "stress_holdout"]
    }


def design_matrix(train: pd.DataFrame, apply: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_parts = [np.ones((len(train), 1), dtype=np.float64)]
    apply_parts = [np.ones((len(apply), 1), dtype=np.float64)]
    metadata: dict[str, Any] = {"categorical_levels": {}, "numeric_scaling": {}}
    for col in numeric_cols:
        train_values = train[col].fillna(0.0).to_numpy(dtype=np.float64)
        mean = float(np.mean(train_values)) if len(train_values) else 0.0
        scale = float(np.std(train_values)) if len(train_values) else 1.0
        scale = scale if scale > 1e-12 else 1.0
        train_parts.append(((train_values - mean) / scale).reshape(-1, 1))
        apply_values = apply[col].fillna(0.0).to_numpy(dtype=np.float64)
        apply_parts.append(((apply_values - mean) / scale).reshape(-1, 1))
        metadata["numeric_scaling"][col] = {"mean": mean, "scale": scale}
    for col in categorical_cols:
        levels = sorted(train[col].astype(str).fillna("missing").unique().tolist())
        metadata["categorical_levels"][col] = levels
        for level in levels:
            train_parts.append(train[col].astype(str).fillna("missing").eq(level).astype(float).to_numpy().reshape(-1, 1))
            apply_parts.append(apply[col].astype(str).fillna("missing").eq(level).astype(float).to_numpy().reshape(-1, 1))
    return np.hstack(train_parts), np.hstack(apply_parts), metadata


def residualize_against_nuisance(
    frame: pd.DataFrame,
    main_cols: list[str],
    categorical_cols: list[str],
    numeric_cols: list[str],
    ridge_alpha: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = frame[frame["split"].astype(str).eq("train")].copy()
    if train.empty:
        raise RuntimeError("Crystal-clean residualization requires train rows.")
    x_train, x_all, design_meta = design_matrix(train, frame, categorical_cols, numeric_cols)
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    xtx = x_train.T @ x_train + penalty
    xtx_inv_xt = np.linalg.pinv(xtx) @ x_train.T
    result = pd.DataFrame(index=frame.index)
    params: dict[str, Any] = {
        "fit_split": "train",
        "fit_policy": "fit nuisance regression on train rows only; apply frozen nuisance coefficients to validation/stress_holdout",
        "categorical_nuisance_columns": categorical_cols,
        "numeric_nuisance_columns": numeric_cols,
        "ridge_alpha": float(ridge_alpha),
        "design": design_meta,
        "features": {},
    }
    for col in main_cols:
        y_train = train[col].fillna(0.0).to_numpy(dtype=np.float64)
        beta = xtx_inv_xt @ y_train
        train_residual = y_train - x_train @ beta
        center = float(np.mean(train_residual)) if len(train_residual) else 0.0
        scale = float(np.std(train_residual)) if len(train_residual) else 1.0
        scale = scale if scale > 1e-12 else 1.0
        residual = frame[col].fillna(0.0).to_numpy(dtype=np.float64) - x_all @ beta
        suffix = col.removeprefix("diffraction_")
        new_col = f"diffraction_crystal_clean_{suffix}"
        result[new_col] = (residual - center) / scale
        params["features"][new_col] = {
            "source_column": col,
            "center": center,
            "scale": scale,
            "coefficient_count": int(len(beta)),
        }
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0), params


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A crystal-clean feature view report",
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
        f"- Filtered source mode: `{gate['filtered_source_mode']}`",
        f"- Allowed stress labels: `{', '.join(gate['allowed_stress_labels'])}`",
        f"- Matched pairs: train `{gate['matched_pair_counts']['train']}`, validation `{gate['matched_pair_counts']['validation']}`, stress-holdout `{gate['matched_pair_counts']['stress_holdout']}`",
        f"- Main feature count: `{gate['main_feature_count']}`",
        "",
        "## What Was Removed From Learning",
        "",
        "- Main features are only `diffraction_crystal_clean_*` residuals.",
        "- Source/stress/origin/count/thickness/pose fields are retained only as audit lineage, not as model inputs.",
        "- Residualization is fit on train split only and then frozen for validation/stress-holdout.",
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_crystal_clean_view_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a crystal-clean v8A feature view before any renewed H/M training.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-mode", default="custom_diffraction_on")
    parser.add_argument("--allowed-stress-labels", default="default")
    parser.add_argument("--match-strategy", choices=sorted(MATCH_STRATEGIES), default="strict_source_origin_thickness_pose_count")
    parser.add_argument("--total-count-column", default="control_total_count_norm")
    parser.add_argument("--count-bin-width", type=float, default=0.003)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest_in = load_json(input_dir / "v8a_event_feature_manifest.json")
    for name, payload in {"schema_gate": schema_gate, "manifest": manifest_in}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing crystal-clean view because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing crystal-clean view because {name} reports existing XRT cube reads.")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features available.")
    allowed_stress = [item.strip() for item in args.allowed_stress_labels.split(",") if item.strip()]
    filtered = frame[
        frame["source_mode"].astype(str).eq(args.source_mode)
        & frame["stress_label"].astype(str).isin(allowed_stress)
    ].copy()
    if filtered.empty:
        raise RuntimeError("Crystal-clean filter produced no rows.")
    filtered["clean_count_bin"] = add_count_bin(filtered, args.total_count_column, float(args.count_bin_width))
    match_columns = MATCH_STRATEGIES[args.match_strategy]
    matched = exact_match_pairs(filtered, match_columns, args.total_count_column, f"crystal_clean_{args.match_strategy}")
    if matched.empty:
        raise RuntimeError("Crystal-clean exact matching produced no H/M pairs.")
    nuisance_categorical = [
        col
        for col in BASE_NUISANCE_CATEGORICAL_COLUMNS + OPTIONAL_CLEAN_NUISANCE_CATEGORICAL_COLUMNS
        if col in matched.columns
    ]
    nuisance_numeric = [args.total_count_column]
    clean_features, residualization = residualize_against_nuisance(
        matched,
        main_cols,
        nuisance_categorical,
        nuisance_numeric,
        float(args.ridge_alpha),
    )
    lineage_cols = [col for col in LINEAGE_COLUMNS if col in matched.columns]
    extra_lineage = ["clean_count_bin", "clean_match_pair_id", "clean_match_delta_total_count_norm"]
    control_cols = [col for col in matched.columns if col.startswith("control_")]
    output = pd.concat([matched[lineage_cols + extra_lineage].copy(), matched[control_cols].copy(), clean_features], axis=1)
    output = output.loc[:, ~output.columns.duplicated()].copy()
    clean_main_cols = list(clean_features.columns)
    leak_tokens = ["material", "source_id", "sample_id", "path", "seed", "thickness", "pose", "split", "row_index", "stress", "origin", "count_bin"]
    lineage_like = [col for col in clean_main_cols if any(token in col.lower() for token in leak_tokens)]
    counts = pair_counts(output)
    pass_items = {
        "input_schema_gate_passed": bool(schema_gate.get("gate_passed", False)),
        "train_audit_support": counts["train"] >= AUDIT_SUPPORT_THRESHOLDS["train_pairs_min"],
        "validation_audit_support": counts["validation"] >= AUDIT_SUPPORT_THRESHOLDS["validation_pairs_min"],
        "stress_holdout_audit_support": counts["stress_holdout"] >= AUDIT_SUPPORT_THRESHOLDS["stress_holdout_pairs_min"],
        "no_lineage_like_main_features": not lineage_like,
        "single_source_mode_for_learning": output["source_mode"].astype(str).nunique() == 1,
        "stress_restricted_for_learning": set(output["stress_label"].astype(str).unique()).issubset(set(allowed_stress)),
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/build_v8a_crystal_clean_view.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_crystal_clean_feature_view",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "filtered_source_mode": args.source_mode,
        "allowed_stress_labels": allowed_stress,
        "match_strategy": args.match_strategy,
        "source_row_count": int(len(frame)),
        "filtered_row_count": int(len(filtered)),
        "sample_count": int(len(output)),
        "matched_pair_counts": counts,
        "match_columns": match_columns,
        "nuisance_categorical_columns": nuisance_categorical,
        "nuisance_numeric_columns": nuisance_numeric,
        "main_feature_count": int(len(clean_main_cols)),
        "main_feature_columns": clean_main_cols,
        "lineage_like_main_features": lineage_like,
        "support_thresholds": AUDIT_SUPPORT_THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "gate_passed": gate_passed,
        "training_unlocked": False,
        "tiny_training_gate_allowed": False,
        "decision": "crystal_clean_view_ready_for_null_shortcut_admission_audit" if gate_passed else "stop_crystal_clean_view_support_or_lineage",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest = dict(manifest_in)
    manifest.update(
        {
            "generated_by": "analysis/build_v8a_crystal_clean_view.py",
            "generated_at_utc": generated_at,
            "protocol_name": "v8A_crystal_clean_feature_view",
            "transform_id": "v8a_crystal_clean_exact_match_residual_v1",
            "match_strategy": args.match_strategy,
            "development_only": True,
            "shadow_or_final_used": False,
            "reads_existing_xrt_cubes": False,
            "runs_geant4": False,
            "claim_scope": CLAIM_SCOPE,
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "sample_count": int(len(output)),
            "main_feature_count": int(len(clean_main_cols)),
            "main_feature_columns": clean_main_cols,
            "lineage_columns_excluded_from_main_features": lineage_cols + extra_lineage,
            "source_main_feature_columns": main_cols,
            "residualization": json_clean(residualization),
            "training_unlocked": False,
        }
    )
    output.to_csv(output_dir / "v8a_event_sidecar_features.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_event_schema_gate.json", json_clean(gate))
    write_json(output_dir / "v8a_event_feature_manifest.json", json_clean(manifest))
    write_report(output_dir, gate)
    print(
        "decision={decision} gate_passed={passed} samples={samples} pairs={pairs}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            samples=len(output),
            pairs=counts,
        )
    )


if __name__ == "__main__":
    main()
