from __future__ import annotations

import argparse
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only final data audit for the v8A ten-material context sidecar view; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)

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
EXPECTED_SPLIT_COUNTS = {"train": 2160, "validation": 1440, "stress_holdout": 1440}
EXPECTED_PER_MATERIAL = {"train": 216, "validation": 144, "stress_holdout": 144}
SPLITS = ("train", "validation", "stress_holdout")
LEAK_TOKENS = [
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
    "context_cell",
    "nuisance_cell",
    "count_target",
]
COUNT_GAP_REVIEW_THRESHOLD = 0.50


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


def source_on(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().to_dict().items()}


def expected_counts_from_manifest(manifest: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    expected_rows = manifest.get("expected_rows", {})
    materials = manifest.get("materials", TARGET_MATERIALS)
    material_count = max(len(materials), 1)
    split_counts = {
        split: int(expected_rows.get(split, EXPECTED_SPLIT_COUNTS[split]))
        for split in SPLITS
    }
    per_material = {
        split: int(split_counts[split] // material_count)
        for split in SPLITS
    }
    return split_counts, per_material


def infer_matrix_manifest(project_root: Path, feature_manifest: dict[str, Any], explicit_path: str) -> dict[str, Any]:
    if explicit_path:
        return load_json(as_project_path(project_root, explicit_path))
    profile = str(feature_manifest.get("profile", "")).strip()
    if profile:
        path = project_root / "source_models" / "config" / "material_sorting_matrix" / profile / "matrix_manifest.json"
        if path.exists():
            return load_json(path)
    return {"expected_rows": EXPECTED_SPLIT_COUNTS, "materials": TARGET_MATERIALS}


def split_material_counts(frame: pd.DataFrame, expected_per_material: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, group in frame.groupby("split", sort=True):
        counts = group["material"].astype(str).value_counts().to_dict()
        for material in TARGET_MATERIALS:
            rows.append(
                {
                    "split": str(split),
                    "material": material,
                    "count": int(counts.get(material, 0)),
                    "expected": expected_per_material.get(str(split), 0),
                    "matches_expected": int(counts.get(material, 0)) == expected_per_material.get(str(split), 0),
                }
            )
    return rows


def context_cell_integrity(frame: pd.DataFrame) -> dict[str, Any]:
    result = {
        "cell_column_present": "clean_context_cell_id" in frame.columns,
        "cell_count": 0,
        "bad_cell_count": 0,
        "cross_split_cell_count": 0,
        "bad_cell_examples": [],
        "cells_by_split": {},
    }
    if "clean_context_cell_id" not in frame.columns:
        return result
    target_set = set(TARGET_MATERIALS)
    bad_examples: list[str] = []
    bad_count = 0
    cross_split = 0
    cells_by_split: Counter[str] = Counter()
    for cell_id, group in frame.groupby("clean_context_cell_id", sort=True):
        splits = set(group["split"].astype(str))
        if len(splits) == 1:
            cells_by_split[next(iter(splits))] += 1
        else:
            cross_split += 1
        material_counts = group["material"].astype(str).value_counts().to_dict()
        if set(material_counts) != target_set or any(int(count) != 1 for count in material_counts.values()):
            bad_count += 1
            if len(bad_examples) < 8:
                bad_examples.append(str(cell_id))
    result.update(
        {
            "cell_count": int(frame["clean_context_cell_id"].nunique()),
            "bad_cell_count": int(bad_count),
            "cross_split_cell_count": int(cross_split),
            "bad_cell_examples": bad_examples,
            "cells_by_split": {str(key): int(value) for key, value in sorted(cells_by_split.items())},
        }
    )
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
    numeric = frame[main_cols].apply(pd.to_numeric, errors="coerce") if main_cols else pd.DataFrame()
    return {
        "main_feature_count": int(len(main_cols)),
        "lineage_like_main_features": [col for col in main_cols if any(token in col.lower() for token in LEAK_TOKENS)],
        "nan_count": int(numeric.isna().sum().sum()) if not numeric.empty else 0,
        "inf_count": int(np.isinf(numeric.to_numpy(dtype=np.float64)).sum()) if not numeric.empty else 0,
    }


def count_gap_review(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "control_total_count_norm" not in frame.columns:
        return rows
    for split, split_frame in frame.groupby("split", sort=True):
        values_by_material = {
            material: split_frame[split_frame["material"].astype(str).eq(material)]["control_total_count_norm"].to_numpy(dtype=np.float64)
            for material in TARGET_MATERIALS
        }
        for left_index, left in enumerate(TARGET_MATERIALS):
            for right in TARGET_MATERIALS[left_index + 1 :]:
                a = values_by_material[left]
                b = values_by_material[right]
                if len(a) == 0 or len(b) == 0:
                    gap = 0.0
                else:
                    pooled = float(np.sqrt(0.5 * (np.var(a) + np.var(b)) + 1e-12))
                    gap = float(abs(float(np.mean(a) - np.mean(b))) / pooled) if pooled > 0 else 0.0
                rows.append(
                    {
                        "split": str(split),
                        "material_a": left,
                        "material_b": right,
                        "standardized_total_count_gap": gap,
                    }
                )
    return sorted(rows, key=lambda row: row["standardized_total_count_gap"], reverse=True)


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A ten-material context final data audit",
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
        f"- Samples: `{gate['sample_count']}`",
        f"- Context cells: `{gate['context_cell_integrity']['cell_count']}`",
        f"- Max total-count gap: `{gate['max_total_count_gap']:.4f}`",
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in gate["warnings"]) if gate["warnings"] else lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiclass_context_training_data_final_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-audit the v8A ten-material context feature table before development training.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--matrix-preflight-gate", default="results/accuracy_v3/v8a_multiclass_context_v1_matrix_preflight/v8a_multiclass_context_matrix_preflight_gate.json")
    parser.add_argument("--matrix-manifest", default="")
    parser.add_argument("--required-peak-table-id", default="ten_material_powder_peaks_cif_or_literature_v8a")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    feature_manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    matrix_preflight = load_json(as_project_path(project_root, args.matrix_preflight_gate))
    matrix_manifest = infer_matrix_manifest(project_root, feature_manifest, args.matrix_manifest)
    expected_split_counts, expected_per_material = expected_counts_from_manifest(matrix_manifest)
    source = source_on(frame)
    main_cols, control_cols, total_count_cols, _, _ = feature_sets(source)

    split_counts = value_counts(source, "split")
    material_counts = value_counts(source, "material")
    split_material = pd.DataFrame(split_material_counts(source, expected_per_material))
    cell_integrity = context_cell_integrity(source)
    cross_split = cross_split_counts(source, ["clean_context_cell_id", "nuisance_cell_id", "random_seed", "seed_block"])
    feature_integrity = main_feature_integrity(source, main_cols)
    count_gaps = pd.DataFrame(count_gap_review(source))
    max_count_gap = float(count_gaps["standardized_total_count_gap"].max()) if not count_gaps.empty else 0.0
    count_gaps.head(80).to_csv(output_dir / "v8a_multiclass_context_total_count_gap_review.csv", index=False)
    split_material.to_csv(output_dir / "v8a_multiclass_context_split_material_counts.csv", index=False)

    stop_reasons: list[str] = []
    warnings: list[str] = []
    if not bool(schema_gate.get("gate_passed", False)):
        stop_reasons.append("event_schema_gate_not_passed")
    if not bool(matrix_preflight.get("gate_passed", False)):
        stop_reasons.append("matrix_preflight_gate_not_passed")
    if bool(feature_manifest.get("shadow_or_final_used", False)):
        stop_reasons.append("shadow_or_final_used_detected")
    if not bool(feature_manifest.get("development_only", False)):
        stop_reasons.append("feature_manifest_not_development_only")
    if feature_manifest.get("peak_table_id") != args.required_peak_table_id:
        stop_reasons.append("peak_table_id_mismatch")
    if set(source["material"].astype(str).unique()) != set(TARGET_MATERIALS):
        stop_reasons.append("target_material_set_mismatch")
    if len(source) != sum(expected_split_counts.values()):
        stop_reasons.append(f"sample_count_mismatch:{len(source)}")
    for split, expected in expected_split_counts.items():
        if int(split_counts.get(split, 0)) != expected:
            stop_reasons.append(f"split_count_mismatch:{split}:{split_counts.get(split, 0)}!={expected}")
    if not split_material.empty and not bool(split_material["matches_expected"].all()):
        stop_reasons.append("split_material_counts_not_balanced")
    if not cell_integrity["cell_column_present"]:
        stop_reasons.append("clean_context_cell_id_missing")
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
    count_controls_required = max_count_gap > COUNT_GAP_REVIEW_THRESHOLD
    if count_controls_required:
        warnings.append("total_count_gap_review_exceeds_threshold_count_controls_required")

    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_training_data_final.py",
        "generated_at_utc": generated_at,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "claim_scope": CLAIM_SCOPE,
        "gate_passed": gate_passed,
        "training_unlocked": gate_passed,
        "decision": "multiclass_context_training_diagnostics_unlocked" if gate_passed else "stop_multiclass_context_training_data",
        "input_dir": args.input_dir,
        "sample_count": int(len(source)),
        "expected_split_counts": expected_split_counts,
        "expected_per_material": expected_per_material,
        "split_counts": split_counts,
        "material_counts": material_counts,
        "context_cell_integrity": cell_integrity,
        "cross_split_counts": cross_split,
        "main_feature_integrity": feature_integrity,
        "main_feature_count": int(len(main_cols)),
        "control_feature_count": int(len(control_cols)),
        "total_count_feature_count": int(len(total_count_cols)),
        "max_total_count_gap": max_count_gap,
        "count_gap_review_threshold": COUNT_GAP_REVIEW_THRESHOLD,
        "count_controls_required": count_controls_required,
        "stop_reasons": stop_reasons,
        "warnings": warnings,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(output_dir / "v8a_multiclass_context_training_data_final_gate.json", json_clean(gate))
    write_report(output_dir, gate)
    print(
        "decision={decision} gate_passed={passed} training_unlocked={training} samples={samples} "
        "count_controls_required={count_controls}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            training=str(gate["training_unlocked"]).lower(),
            samples=len(source),
            count_controls=str(count_controls_required).lower(),
        )
    )


if __name__ == "__main__":
    main()
