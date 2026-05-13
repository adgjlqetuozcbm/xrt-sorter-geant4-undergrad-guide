from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json
from train_v8a_multiclass_context_model import ensure_output_dir, json_clean, write_json


FOCUS_MATERIALS = ["Hematite", "Magnetite", "Ilmenite", "Goethite", "Siderite", "Rutile"]
FOCUS_PEAK_GROUPS = ["hematite", "magnetite", "ilmenite", "goethite", "siderite", "rutile"]
PEAK_RE = re.compile(r"^diffraction_peak_([^_]+)_.*_norm$")


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def peak_group_columns(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for column in columns:
        match = PEAK_RE.match(column)
        if not match:
            continue
        groups.setdefault(match.group(1), []).append(column)
    return groups


def add_peak_group_sums(frame: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    result = frame.copy()
    for material, cols in groups.items():
        result[f"peak_sum_{material}"] = result[cols].fillna(0.0).sum(axis=1)
    return result


def condition_label(row: pd.Series) -> str:
    return f"{row['split']}|{row['physical_perturbation_profile']}"


def peak_sum_summary(frame: pd.DataFrame) -> pd.DataFrame:
    peak_sum_cols = [f"peak_sum_{name}" for name in FOCUS_PEAK_GROUPS if f"peak_sum_{name}" in frame.columns]
    value_cols = peak_sum_cols + [
        col
        for col in ["diffraction_window_hematite_unique_sum", "diffraction_window_magnetite_unique_sum", "diffraction_window_all_peaks_sum"]
        if col in frame.columns
    ]
    summary = (
        frame[frame["material"].astype(str).isin(FOCUS_MATERIALS)]
        .groupby(["split", "physical_perturbation_profile", "seed_block", "material"], sort=True)[value_cols]
        .mean()
        .reset_index()
    )
    if "peak_sum_hematite" in summary.columns and "peak_sum_ilmenite" in summary.columns:
        summary["hematite_minus_ilmenite_peak_sum"] = summary["peak_sum_hematite"] - summary["peak_sum_ilmenite"]
    if "peak_sum_magnetite" in summary.columns and "peak_sum_ilmenite" in summary.columns:
        summary["magnetite_minus_ilmenite_peak_sum"] = summary["peak_sum_magnetite"] - summary["peak_sum_ilmenite"]
    return summary


def zscore(frame: pd.DataFrame, columns: list[str], train_mask: pd.Series) -> pd.DataFrame:
    values = frame[columns].fillna(0.0).to_numpy(dtype=np.float64)
    train_values = frame.loc[train_mask, columns].fillna(0.0).to_numpy(dtype=np.float64)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std < 1e-12] = 1.0
    scaled = (values - mean) / std
    return pd.DataFrame(scaled, columns=columns, index=frame.index)


def centroid_table(frame: pd.DataFrame, scaled: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    meta = frame[["split", "physical_perturbation_profile", "seed_block", "material"]].copy()
    data = pd.concat([meta, scaled[columns]], axis=1)
    for keys, group in data.groupby(["split", "physical_perturbation_profile", "material"], sort=True):
        split, profile, material = keys
        rows.append(
            {
                "split": split,
                "physical_perturbation_profile": profile,
                "material": material,
                "n": int(len(group)),
                "centroid": group[columns].mean(axis=0).to_numpy(dtype=np.float64),
            }
        )
    return pd.DataFrame(rows)


def nearest_train_centroids(
    centroids: pd.DataFrame,
    *,
    query_split: str,
    query_profile: str,
    query_material: str,
    train_profile: str | None,
    limit: int,
) -> pd.DataFrame:
    query = centroids[
        centroids["split"].astype(str).eq(query_split)
        & centroids["physical_perturbation_profile"].astype(str).eq(query_profile)
        & centroids["material"].astype(str).eq(query_material)
    ]
    if query.empty:
        return pd.DataFrame()
    query_vec = query.iloc[0]["centroid"]
    candidates = centroids[centroids["split"].astype(str).eq("train")].copy()
    if train_profile is not None:
        candidates = candidates[candidates["physical_perturbation_profile"].astype(str).eq(train_profile)].copy()
    rows = []
    for _, row in candidates.iterrows():
        vec = row["centroid"]
        dist = float(np.linalg.norm(query_vec - vec))
        rows.append(
            {
                "query_split": query_split,
                "query_profile": query_profile,
                "query_material": query_material,
                "train_profile_filter": train_profile or "__all_train_profiles__",
                "candidate_split": row["split"],
                "candidate_profile": row["physical_perturbation_profile"],
                "candidate_material": row["material"],
                "candidate_n": int(row["n"]),
                "euclidean_distance_zscore": dist,
            }
        )
    return pd.DataFrame(rows).sort_values("euclidean_distance_zscore").head(limit)


def focus_pair_distances(centroids: pd.DataFrame, profile: str) -> pd.DataFrame:
    rows = []
    subset = centroids[
        centroids["split"].astype(str).eq("stress_holdout")
        & centroids["physical_perturbation_profile"].astype(str).eq(profile)
        & centroids["material"].astype(str).isin(FOCUS_MATERIALS)
    ].copy()
    for i, left in subset.iterrows():
        for j, right in subset.iterrows():
            if str(left["material"]) >= str(right["material"]):
                continue
            rows.append(
                {
                    "split": "stress_holdout",
                    "physical_perturbation_profile": profile,
                    "left_material": left["material"],
                    "right_material": right["material"],
                    "left_n": int(left["n"]),
                    "right_n": int(right["n"]),
                    "euclidean_distance_zscore": float(np.linalg.norm(left["centroid"] - right["centroid"])),
                }
            )
    return pd.DataFrame(rows).sort_values("euclidean_distance_zscore")


def error_rows(model_dir: Path) -> pd.DataFrame:
    decisions_path = model_dir / "v8a_multiclass_context_v7_twenty_material_scalability_scout_decisions.csv"
    if not decisions_path.exists():
        return pd.DataFrame()
    decisions = pd.read_csv(decisions_path)
    return decisions[
        decisions["split"].astype(str).eq("stress_holdout")
        & decisions["physical_perturbation_profile"].astype(str).eq("combined_stress_high")
        & (
            decisions["material"].astype(str).isin(["Hematite", "Magnetite", "Ilmenite"])
            | decisions["prediction"].astype(str).isin(["Hematite", "Magnetite", "Ilmenite"])
        )
        & (decisions["material"].astype(str) != decisions["prediction"].astype(str))
    ].copy()


def compact_table(frame: pd.DataFrame, limit: int = 16) -> str:
    if frame.empty:
        return "No rows."
    return frame.head(limit).to_csv(index=False).strip()


def write_report(output_dir: Path, gate: dict[str, Any], nearest: pd.DataFrame, pair_distances: pd.DataFrame, peak_summary: pd.DataFrame) -> None:
    lines = [
        "# v8A v7 hard-negative diagnostics",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Diagnostic completed: `{str(gate['diagnostic_completed']).lower()}`",
        f"- Main finding: `{gate['main_finding']}`",
        "",
        "## Nearest Train Centroids",
        "",
        compact_table(nearest, limit=16),
        "",
        "## Stress Pair Distances",
        "",
        compact_table(pair_distances, limit=16),
        "",
        "## Peak Sum Focus",
        "",
        compact_table(
            peak_summary[
                peak_summary["physical_perturbation_profile"].astype(str).isin(
                    ["combined_train_moderate", "combined_validation_mid", "combined_stress_high"]
                )
                & peak_summary["material"].astype(str).isin(["Hematite", "Ilmenite", "Magnetite"])
            ],
            limit=36,
        )
        if not peak_summary.empty
        else "No rows.",
        "",
    ]
    (output_dir / "v8a_multiclass_context_v7_hard_negative_diagnostics_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v7 Hematite/Magnetite/Ilmenite hard-negative behavior.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_event_to_feature")
    parser.add_argument("--model-dir", default="results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_model")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_hard_negative_diagnostics")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    model_dir = as_project_path(project_root, args.model_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    groups = peak_group_columns(list(frame.columns))
    frame = add_peak_group_sums(frame, groups)
    peak_summary = peak_sum_summary(frame)
    peak_summary.to_csv(output_dir / "v8a_multiclass_context_v7_peak_group_sums_by_profile_material.csv", index=False)

    train_mask = frame["split"].astype(str).eq("train")
    scaled = zscore(frame, main_cols, train_mask)
    centroids = centroid_table(frame, scaled, main_cols)
    nearest_parts = [
        nearest_train_centroids(
            centroids,
            query_split="stress_holdout",
            query_profile="combined_stress_high",
            query_material="Hematite",
            train_profile=None,
            limit=12,
        ),
        nearest_train_centroids(
            centroids,
            query_split="stress_holdout",
            query_profile="combined_stress_high",
            query_material="Hematite",
            train_profile="combined_train_moderate",
            limit=12,
        ),
    ]
    nearest = pd.concat([part for part in nearest_parts if not part.empty], ignore_index=True)
    nearest.to_csv(output_dir / "v8a_multiclass_context_v7_hematite_combined_stress_nearest_train_centroids.csv", index=False)

    pair_distances = focus_pair_distances(centroids, "combined_stress_high")
    pair_distances.to_csv(output_dir / "v8a_multiclass_context_v7_combined_stress_focus_pair_distances.csv", index=False)

    errors = error_rows(model_dir)
    errors.to_csv(output_dir / "v8a_multiclass_context_v7_hm_ilmenite_error_rows.csv", index=False)

    model_gate_path = model_dir / "v8a_multiclass_context_v7_twenty_material_scalability_scout_gate.json"
    model_gate = load_json(model_gate_path) if model_gate_path.exists() else {}
    hematite_ilmenite_error_count = int(
        len(errors[
            errors["material"].astype(str).eq("Hematite")
            & errors["prediction"].astype(str).eq("Ilmenite")
        ])
    ) if not errors.empty else 0
    nearest_top = nearest.head(3)[["candidate_profile", "candidate_material", "euclidean_distance_zscore"]].to_dict("records") if not nearest.empty else []
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_v7_hard_negative_diagnostics.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "diagnostic_completed": True,
        "decision": "v7_hard_negative_diagnostic_completed_next_step_targeted_hm_ilmenite_robustness_design",
        "main_finding": "combined_stress_high creates a Hematite-Ilmenite hard-negative failure in the 20-material scout",
        "sample_count": int(len(frame)),
        "main_feature_count": int(len(main_cols)),
        "model_gate_passed": bool(model_gate.get("gate_passed", False)),
        "model_gate_stop_reasons": model_gate.get("stop_reasons", []),
        "hematite_to_ilmenite_error_rows": hematite_ilmenite_error_count,
        "nearest_train_centroids_top3_for_hematite_combined_stress": nearest_top,
        "outputs": {
            "peak_group_sums": "v8a_multiclass_context_v7_peak_group_sums_by_profile_material.csv",
            "nearest_train_centroids": "v8a_multiclass_context_v7_hematite_combined_stress_nearest_train_centroids.csv",
            "focus_pair_distances": "v8a_multiclass_context_v7_combined_stress_focus_pair_distances.csv",
            "hm_ilmenite_error_rows": "v8a_multiclass_context_v7_hm_ilmenite_error_rows.csv",
        },
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(output_dir / "v8a_multiclass_context_v7_hard_negative_diagnostics_gate.json", json_clean(gate))
    write_report(output_dir, gate, nearest, pair_distances, peak_summary)
    print(
        "decision={decision} hematite_to_ilmenite_errors={errors} nearest_top={nearest}".format(
            decision=gate["decision"],
            errors=hematite_ilmenite_error_count,
            nearest=nearest_top,
        )
    )


if __name__ == "__main__":
    main()
