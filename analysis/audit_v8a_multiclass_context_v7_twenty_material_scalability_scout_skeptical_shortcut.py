from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json
from train_v8a_multiclass_context_model import ensure_output_dir, json_clean, write_json


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
THRESHOLDS = {
    "cross_split_exact_main_feature_hash_overlap_max": 0,
    "hm_off_diagonal_main_errors_max": 0,
    "total_count_only_worst_profile_hm_max": 0.65,
    "lineage_only_worst_profile_hm_max": 0.65,
    "shuffled_label_worst_profile_hm_p95_max": 0.55,
}


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def feature_hashes(frame: pd.DataFrame, main_cols: list[str]) -> pd.Series:
    numeric = frame[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    hashes = []
    for row in numeric:
        rounded = np.round(row, 12)
        hashes.append(hashlib.sha1(rounded.tobytes()).hexdigest())
    return pd.Series(hashes, index=frame.index, name="main_feature_hash")


def cross_split_hash_overlap(frame: pd.DataFrame, main_cols: list[str]) -> dict[str, Any]:
    data = frame[["split", "material", "sample_id"]].copy()
    data["main_feature_hash"] = feature_hashes(frame, main_cols)
    rows = []
    for hash_value, group in data.groupby("main_feature_hash", sort=True):
        splits = sorted(group["split"].astype(str).unique().tolist())
        if len(splits) <= 1:
            continue
        rows.append(
            {
                "main_feature_hash": hash_value,
                "splits": "|".join(splits),
                "materials": "|".join(sorted(group["material"].astype(str).unique().tolist())),
                "sample_count": int(len(group)),
                "sample_ids": "|".join(group["sample_id"].astype(str).head(6).tolist()),
            }
        )
    overlap = pd.DataFrame(rows)
    return {
        "overlap": overlap,
        "overlap_count": int(len(overlap)),
        "conflicting_material_overlap_count": int(
            sum(1 for _, row in overlap.iterrows() if len(str(row["materials"]).split("|")) > 1)
        )
        if not overlap.empty
        else 0,
    }


def selected_best_confusions(confusions: pd.DataFrame, best_main: pd.DataFrame) -> pd.DataFrame:
    if confusions.empty or best_main.empty:
        return pd.DataFrame()
    selected = best_main[best_main["physical_perturbation_profile"].astype(str) != "__overall__"][
        ["eval_split", "physical_perturbation_profile", "method"]
    ].drop_duplicates()
    selected = selected.rename(columns={"eval_split": "eval_split_best", "method": "method_best"})
    merged = confusions.merge(
        selected,
        left_on=["eval_split", "physical_perturbation_profile", "method"],
        right_on=["eval_split_best", "physical_perturbation_profile", "method_best"],
        how="inner",
    )
    return merged[merged["actual"].astype(str) != merged["predicted"].astype(str)].copy()


def write_report(output_dir: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# v8A v7 twenty-material scalability scout skeptical shortcut audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Cross-split exact main-feature hash overlap: `{gate['cross_split_exact_main_feature_hash_overlap_count']}`",
        f"- Best-main off-diagonal errors, all materials: `{gate['off_diagonal_main_errors_count']}`",
        f"- Best-main H/M off-diagonal errors: `{gate['hm_off_diagonal_main_errors_count']}`",
        f"- Total-count-only worst profile H/M: `{gate['total_count_only_worst_profile_hm']:.4f}`",
        f"- Lineage-only worst profile H/M: `{gate['lineage_only_worst_profile_hm']:.4f}`",
        f"- Shuffled-label worst profile H/M p95: `{gate['shuffled_label_worst_profile_hm_p95']:.4f}`",
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiclass_context_v7_twenty_material_scalability_scout_skeptical_shortcut_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Skeptical shortcut audit for v8A v7 twenty-material scalability scout results.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    model_dir = as_project_path(project_root, args.model_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    model_gate = load_json(model_dir / "v8a_multiclass_context_v7_twenty_material_scalability_scout_gate.json")
    best_main = pd.read_csv(model_dir / "v8a_multiclass_context_v7_twenty_material_scalability_scout_best_main.csv")
    confusions_path = model_dir / "v8a_multiclass_context_v7_twenty_material_scalability_scout_confusions.csv"
    confusions = pd.read_csv(confusions_path) if confusions_path.exists() else pd.DataFrame()

    overlap = cross_split_hash_overlap(frame, main_cols)
    if not overlap["overlap"].empty:
        overlap["overlap"].to_csv(output_dir / "v8a_multiclass_context_v7_cross_split_main_feature_hash_overlap.csv", index=False)

    lineage_like_main = [col for col in main_cols if any(token in col.lower() for token in LEAK_TOKENS)]
    off_diagonal = selected_best_confusions(confusions, best_main)
    off_diagonal_count = int(off_diagonal["count"].sum()) if "count" in off_diagonal.columns else int(len(off_diagonal))
    hm_off_diagonal = off_diagonal[
        off_diagonal["actual"].astype(str).isin(["Hematite", "Magnetite"])
        | off_diagonal["predicted"].astype(str).isin(["Hematite", "Magnetite"])
    ].copy() if not off_diagonal.empty else pd.DataFrame()
    hm_off_diagonal_count = int(hm_off_diagonal["count"].sum()) if "count" in hm_off_diagonal.columns else int(len(hm_off_diagonal))
    if not off_diagonal.empty:
        off_diagonal.to_csv(output_dir / "v8a_multiclass_context_v7_selected_best_main_off_diagonal_errors.csv", index=False)

    total_count_worst = float(model_gate.get("total_count_only_worst_profile_hm", 0.0))
    lineage_worst = float(model_gate.get("lineage_only_worst_profile_hm", 0.0))
    shuffled_p95 = float(model_gate.get("shuffled_label_worst_profile_hm_p95", 0.0))

    stop_reasons: list[str] = []
    if not bool(model_gate.get("gate_passed", False)):
        stop_reasons.append("model_gate_not_passed")
    if lineage_like_main:
        stop_reasons.append("lineage_like_main_features_detected")
    if overlap["overlap_count"] > THRESHOLDS["cross_split_exact_main_feature_hash_overlap_max"]:
        stop_reasons.append("cross_split_exact_main_feature_hash_overlap_detected")
    if hm_off_diagonal_count > THRESHOLDS["hm_off_diagonal_main_errors_max"]:
        stop_reasons.append("hm_off_diagonal_main_model_errors_detected")
    if total_count_worst > THRESHOLDS["total_count_only_worst_profile_hm_max"]:
        stop_reasons.append("total_count_only_control_above_threshold")
    if lineage_worst > THRESHOLDS["lineage_only_worst_profile_hm_max"]:
        stop_reasons.append("lineage_only_control_above_threshold")
    if shuffled_p95 > THRESHOLDS["shuffled_label_worst_profile_hm_p95_max"]:
        stop_reasons.append("shuffled_label_control_above_threshold")

    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_v7_twenty_material_scalability_scout_skeptical_shortcut.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "claim_scope": "skeptical development audit for v7 twenty-material scalability scout; not hardware validation or product accuracy",
        "gate_passed": gate_passed,
        "decision": "v7_twenty_material_scalability_scout_skeptical_shortcut_audit_passed" if gate_passed else "stop_v7_twenty_material_scalability_scout_skeptical_shortcut_audit",
        "sample_count": int(len(frame)),
        "main_feature_count": int(len(main_cols)),
        "lineage_like_main_features": lineage_like_main,
        "cross_split_exact_main_feature_hash_overlap_count": int(overlap["overlap_count"]),
        "cross_split_exact_main_feature_hash_conflicting_material_count": int(overlap["conflicting_material_overlap_count"]),
        "off_diagonal_main_errors_count": off_diagonal_count,
        "hm_off_diagonal_main_errors_count": hm_off_diagonal_count,
        "total_count_only_worst_profile_hm": total_count_worst,
        "lineage_only_worst_profile_hm": lineage_worst,
        "shuffled_label_worst_profile_hm_p95": shuffled_p95,
        "thresholds": THRESHOLDS,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(output_dir / "v8a_multiclass_context_v7_twenty_material_scalability_scout_skeptical_shortcut_gate.json", json_clean(gate))
    write_report(output_dir, gate)
    print(
        "decision={decision} gate_passed={passed} overlap={overlap_count} hm_offdiag={hm_offdiag} all_offdiag={all_offdiag}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            overlap_count=gate["cross_split_exact_main_feature_hash_overlap_count"],
            hm_offdiag=gate["hm_off_diagonal_main_errors_count"],
            all_offdiag=gate["off_diagonal_main_errors_count"],
        )
    )


if __name__ == "__main__":
    main()
