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
MATERIAL_GROUPS = {
    "full_10": TARGET_MATERIALS,
    "hm_pair": ["Hematite", "Magnetite"],
    "hm_count_overlap_core_5": ["Magnetite", "Hematite", "Pyrite", "Galena", "Chalcopyrite"],
    "oxide_sulfide_6_with_quartz_boundary": ["Magnetite", "Hematite", "Pyrite", "Galena", "Chalcopyrite", "Quartz"],
}
COUNT_BIN_WIDTHS = [0.003, 0.005, 0.010, 0.015, 0.020, 0.040, 0.050, 0.100, 0.200, 0.500, 1.000]
SLIDING_WINDOW_WIDTHS = [0.020, 0.040, 0.050, 0.075, 0.100, 0.200]
STRICT_CELL_KEYS = ["split", "clean_context_cell_id"]
FALLBACK_STRICT_CELL_KEYS = [
    "split",
    "source_id",
    "source_family",
    "stress_label",
    "thickness_mm",
    "pose_index",
    "seed_block",
    "count_target_bin",
]


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def strict_cell_keys(frame: pd.DataFrame) -> list[str]:
    if all(column in frame.columns for column in STRICT_CELL_KEYS):
        return STRICT_CELL_KEYS
    return [column for column in FALLBACK_STRICT_CELL_KEYS if column in frame.columns]


def count_balanced_support(frame: pd.DataFrame, materials: list[str], width: float, cell_keys: list[str]) -> dict[str, Any]:
    subset = frame[frame["material"].astype(str).isin(materials)].copy()
    subset["count_balance_bin"] = np.floor(subset["control_total_count_norm"].astype(float) / width).astype(int).astype(str)
    group_cols = cell_keys + ["count_balance_bin"]
    cells_by_split: Counter[str] = Counter()
    row_support_by_split: Counter[str] = Counter()
    for keys, group in subset.groupby(group_cols, sort=True, observed=True):
        material_counts = group["material"].astype(str).value_counts().to_dict()
        if set(material_counts) == set(materials) and all(int(material_counts[item]) >= 1 for item in materials):
            split = str(keys[0])
            cells_by_split[split] += 1
            row_support_by_split[split] += len(materials)
    return {
        "count_bin_width": width,
        "cells_train": int(cells_by_split.get("train", 0)),
        "cells_validation": int(cells_by_split.get("validation", 0)),
        "cells_stress_holdout": int(cells_by_split.get("stress_holdout", 0)),
        "rows_train": int(row_support_by_split.get("train", 0)),
        "rows_validation": int(row_support_by_split.get("validation", 0)),
        "rows_stress_holdout": int(row_support_by_split.get("stress_holdout", 0)),
    }


def sliding_window_support(
    frame: pd.DataFrame, materials: list[str], width: float, cell_keys: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subset = frame[frame["material"].astype(str).isin(materials)].copy()
    cells_by_split: Counter[str] = Counter()
    row_support_by_split: Counter[str] = Counter()
    cell_rows: list[dict[str, Any]] = []
    for keys, group in subset.groupby(cell_keys, sort=True, observed=True):
        material_counts = group["material"].astype(str).value_counts().to_dict()
        if not (set(material_counts) == set(materials) and all(int(material_counts[item]) >= 1 for item in materials)):
            continue
        values = group["control_total_count_norm"].astype(float)
        span = float(values.max() - values.min())
        if span > width:
            continue
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_map = dict(zip(cell_keys, key_values))
        split = str(key_map.get("split", group["split"].iloc[0]))
        cells_by_split[split] += 1
        row_support_by_split[split] += len(materials)
        row = {
            "sliding_window_width": width,
            "material_group": "",
            "material_count": len(materials),
            "split": split,
            "cell_count_min": int(min(material_counts.values())),
            "rows_selected": len(materials),
            "count_min": float(values.min()),
            "count_max": float(values.max()),
            "count_span": span,
        }
        for column in [
            "clean_context_cell_id",
            "source_id",
            "source_energy_kev",
            "thickness_mm",
            "pose_index",
            "seed_block",
            "count_target_bin",
        ]:
            if column in group.columns:
                row[column] = group[column].iloc[0]
        cell_rows.append(row)
    summary = {
        "sliding_window_width": width,
        "cells_train": int(cells_by_split.get("train", 0)),
        "cells_validation": int(cells_by_split.get("validation", 0)),
        "cells_stress_holdout": int(cells_by_split.get("stress_holdout", 0)),
        "rows_train": int(row_support_by_split.get("train", 0)),
        "rows_validation": int(row_support_by_split.get("validation", 0)),
        "rows_stress_holdout": int(row_support_by_split.get("stress_holdout", 0)),
    }
    return summary, cell_rows


def range_overlap(frame: pd.DataFrame, materials: list[str]) -> dict[str, Any]:
    rows = []
    low = -float("inf")
    high = float("inf")
    for material in materials:
        values = frame.loc[frame["material"].astype(str).eq(material), "control_total_count_norm"].astype(float)
        mat_min = float(values.min())
        mat_max = float(values.max())
        rows.append({"material": material, "min": mat_min, "max": mat_max, "mean": float(values.mean()), "std": float(values.std())})
        low = max(low, mat_min)
        high = min(high, mat_max)
    return {
        "overlap_low": low,
        "overlap_high": high,
        "overlap_width": max(0.0, high - low),
        "has_global_overlap": high >= low,
        "material_ranges": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Design-audit cleanliness constraints for the v8A ten-material context data.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_multiclass_context_v1_event_to_feature")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v2_cleanliness_design")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    frame = frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    cell_keys = strict_cell_keys(frame)

    range_rows = (
        frame.groupby(["material", "thickness_mm"], sort=True)["control_total_count_norm"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    range_rows.to_csv(output_dir / "v8a_multiclass_context_v1_count_ranges_by_material_thickness.csv", index=False)

    support_rows: list[dict[str, Any]] = []
    sliding_rows: list[dict[str, Any]] = []
    sliding_cell_rows: list[dict[str, Any]] = []
    overlap_summary: dict[str, Any] = {}
    for group_name, materials in MATERIAL_GROUPS.items():
        overlap_summary[group_name] = range_overlap(frame, materials)
        for width in COUNT_BIN_WIDTHS:
            support = count_balanced_support(frame, materials, width, cell_keys)
            support["material_group"] = group_name
            support["material_count"] = len(materials)
            support_rows.append(support)
        for width in SLIDING_WINDOW_WIDTHS:
            sliding_support, cells = sliding_window_support(frame, materials, width, cell_keys)
            sliding_support["material_group"] = group_name
            sliding_support["material_count"] = len(materials)
            sliding_rows.append(sliding_support)
            for cell in cells:
                cell["material_group"] = group_name
            sliding_cell_rows.extend(cells)
    support = pd.DataFrame(support_rows)
    support.to_csv(output_dir / "v8a_multiclass_context_v1_count_balanced_support_scan.csv", index=False)
    sliding_support = pd.DataFrame(sliding_rows)
    sliding_support.to_csv(output_dir / "v8a_multiclass_context_sliding_window_support_scan.csv", index=False)
    pd.DataFrame(sliding_cell_rows).to_csv(output_dir / "v8a_multiclass_context_sliding_window_cells.csv", index=False)

    full10_reasonable = support[support["material_group"].eq("full_10") & support["count_bin_width"].le(0.05)]
    full10_strict_support = full10_reasonable[["cells_train", "cells_validation", "cells_stress_holdout"]].sum().sum()
    full10_split_support = {
        "train": int(full10_reasonable["cells_train"].sum()),
        "validation": int(full10_reasonable["cells_validation"].sum()),
        "stress_holdout": int(full10_reasonable["cells_stress_holdout"].sum()),
    }
    full10_has_all_splits = all(value > 0 for value in full10_split_support.values())
    full10_sliding_reasonable = sliding_support[
        sliding_support["material_group"].eq("full_10") & sliding_support["sliding_window_width"].le(0.05)
    ]
    full10_sliding_split_support = {
        "train": int(full10_sliding_reasonable["cells_train"].sum()),
        "validation": int(full10_sliding_reasonable["cells_validation"].sum()),
        "stress_holdout": int(full10_sliding_reasonable["cells_stress_holdout"].sum()),
    }
    full10_sliding_has_all_splits = all(value > 0 for value in full10_sliding_split_support.values())
    full10_sliding_best = (
        sliding_support[sliding_support["material_group"].eq("full_10")]
        .assign(total_cells=lambda x: x["cells_train"] + x["cells_validation"] + x["cells_stress_holdout"])
        .sort_values(["total_cells", "sliding_window_width"], ascending=[False, True])
        .head(1)
        .to_dict(orient="records")
    )
    core5_best = (
        support[support["material_group"].eq("hm_count_overlap_core_5")]
        .assign(total_cells=lambda x: x["cells_train"] + x["cells_validation"] + x["cells_stress_holdout"])
        .sort_values("total_cells", ascending=False)
        .head(1)
        .to_dict(orient="records")
    )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_cleanliness_design.py",
        "generated_at_utc": generated_at,
        "development_only": True,
        "claim_scope": "development-only cleanliness design audit; not training evidence or hardware validation",
        "input_dir": args.input_dir,
        "sample_count": int(len(frame)),
        "strict_cell_keys": cell_keys,
        "material_groups": MATERIAL_GROUPS,
        "count_bin_widths": COUNT_BIN_WIDTHS,
        "range_overlap_summary": overlap_summary,
        "full10_reasonable_bin_support_cells": int(full10_strict_support),
        "full10_reasonable_bin_support_by_split": full10_split_support,
        "sliding_window_widths": SLIDING_WINDOW_WIDTHS,
        "full10_sliding_reasonable_support_by_split": full10_sliding_split_support,
        "full10_sliding_best_support": full10_sliding_best[0] if full10_sliding_best else {},
        "hm_core5_best_support": core5_best[0] if core5_best else {},
        "decision": (
            "full10_count_balanced_support_available"
            if full10_has_all_splits
            else "full10_sliding_window_overlap_found_targeted_matrix_needed"
            if full10_sliding_has_all_splits
            else "full10_strict_support_insufficient_but_physics_overlap_found"
            if int(full10_strict_support) > 0 or overlap_summary["full_10"]["has_global_overlap"]
            else "full10_requires_count_invariant_or_new_physics_scout_design"
        ),
        "recommended_tracks": [
            {
                "track": "count_invariant_feature_view",
                "reason": "Use existing completed Geant4 data but remove absolute peak-count scale from main features.",
                "next": "Build compositional diffraction features and rerun controls without treating full-10 count-balanced support as mandatory.",
            },
            {
                "track": "hm_count_overlap_core_context",
                "reason": "Magnetite/Hematite/Pyrite/Galena/Chalcopyrite occupy the same output-count band and can support stricter count controls.",
                "next": "Use as a smaller clean H/M-centered context if the goal is defensible H/M separation.",
            },
            {
                "track": "full10_physics_scout",
                "reason": "Full 10 materials do not overlap in normalized hit count under the current 35 keV fixed-thickness grid.",
                "next": "Before another full matrix, run a small energy/thickness/detector-window scout to find output-overlap regions.",
            },
        ],
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(output_dir / "v8a_multiclass_context_cleanliness_design_gate.json", gate)

    lines = [
        "# v8A multiclass context cleanliness design audit",
        "",
        f"Generated: {generated_at}",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Full-10 count-balanced support at bin <=0.05: `{gate['full10_reasonable_bin_support_cells']}` cells",
        f"- Full-10 support by split at bin <=0.05: `{gate['full10_reasonable_bin_support_by_split']}`",
        f"- Full-10 sliding-window support by split at width <=0.05: `{gate['full10_sliding_reasonable_support_by_split']}`",
        "",
        "## Interpretation",
        "",
        "- The v1 matrix controlled input-side nuisance variables, but the output normalized hit-count distributions still separate by material.",
        "- Re-running the same full-10 design with more rows will not create strict count-balanced support unless the physics grid or feature representation changes.",
        "- The next clean tracks are count-invariant feature representation, H/M-centered overlap context, or a small full-10 physics scout before any large rerun.",
        "",
    ]
    (output_dir / "v8a_multiclass_context_cleanliness_design_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"decision={gate['decision']} full10_reasonable_bin_support_cells={gate['full10_reasonable_bin_support_cells']}"
    )


if __name__ == "__main__":
    main()
