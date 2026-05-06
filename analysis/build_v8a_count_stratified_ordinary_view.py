from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_v8a_count_balance_sensitivity import STRATEGIES, build_balanced_subset
from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only count-stratified ordinary-gate feature view for v8A H/M sidecar observability; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)

SUPPORT_THRESHOLDS = {
    "train_pairs_min": 100,
    "validation_pairs_min": 50,
    "stress_holdout_pairs_min": 50,
    "count_gap_standardized_max": 0.15,
}


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


def strategy_by_name(name: str) -> dict[str, Any]:
    for strategy in STRATEGIES:
        if str(strategy["strategy"]) == name:
            return strategy
    raise ValueError(f"Unknown count-balance strategy: {name}")


def pair_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "match_pair_id" not in frame.columns:
        return {"train": 0, "validation": 0, "stress_holdout": 0}
    values = frame.groupby("split")["match_pair_id"].nunique().to_dict()
    return {split: int(values.get(split, 0)) for split in ["train", "validation", "stress_holdout"]}


def standardized_gap(values_left: pd.Series, values_right: pd.Series) -> float:
    left = values_left.fillna(0.0).to_numpy(dtype=np.float64)
    right = values_right.fillna(0.0).to_numpy(dtype=np.float64)
    pooled = np.sqrt(0.5 * (np.var(left) + np.var(right)) + 1e-12)
    return float(abs(float(np.mean(right) - np.mean(left))) / pooled)


def count_gap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_on = frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")]
    for split, group in source_on.groupby("split", sort=True):
        hematite = group[group["material"].astype(str).eq("Hematite")]
        magnetite = group[group["material"].astype(str).eq("Magnetite")]
        rows.append(
            {
                "split": split,
                "samples": int(len(group)),
                "matched_pairs": int(group["match_pair_id"].nunique()) if "match_pair_id" in group.columns else 0,
                "hematite_mean_total_count_norm": float(hematite["control_total_count_norm"].mean()) if not hematite.empty else 0.0,
                "magnetite_mean_total_count_norm": float(magnetite["control_total_count_norm"].mean()) if not magnetite.empty else 0.0,
                "standardized_count_gap_abs": standardized_gap(hematite["control_total_count_norm"], magnetite["control_total_count_norm"])
                if not hematite.empty and not magnetite.empty
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], gaps: pd.DataFrame) -> None:
    lines = [
        "# v8A count-stratified ordinary-gate view report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Strategy: `{gate['count_balance_strategy']}`",
        f"- Matched pairs: train `{gate['matched_pair_counts']['train']}`, validation `{gate['matched_pair_counts']['validation']}`, stress-holdout `{gate['matched_pair_counts']['stress_holdout']}`",
        f"- Max standardized count gap: `{gate['max_standardized_count_gap_abs']:.4f}`",
        "",
        "## Count Gaps",
        "",
        markdown_table(
            gaps,
            [
                "split",
                "samples",
                "matched_pairs",
                "hematite_mean_total_count_norm",
                "magnetite_mean_total_count_norm",
                "standardized_count_gap_abs",
            ],
        ),
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_count_stratified_ordinary_view_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a count-stratified ordinary-gate view from v8A event features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strategy", default="fixed_bin_width_0p003")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    if not bool(schema_gate.get("gate_passed", False)) or not bool(schema_gate.get("tiny_training_gate_allowed", False)):
        raise RuntimeError(f"Input feature gate is not training-allowed: {schema_gate.get('decision')}")
    for name, payload in {"schema_gate": schema_gate, "manifest": manifest}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing count-stratified view because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing count-stratified view because {name} reports existing XRT cube reads.")

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    strategy = strategy_by_name(args.strategy)
    balanced_source_on = build_balanced_subset(frame, strategy)
    if balanced_source_on.empty:
        raise RuntimeError(f"Strategy produced no balanced source-on rows: {args.strategy}")
    source_off = frame[frame["source_mode"].astype(str).eq("custom_diffraction_off")].copy()
    source_off["count_balance_bin"] = "source_off_control"
    source_off["match_pair_id"] = "source_off_control"
    source_off["match_delta_total_count_norm"] = 0.0
    output = pd.concat([balanced_source_on, source_off], ignore_index=True, sort=False)
    main_cols, control_cols, _, _, _ = feature_sets(output)
    leak_tokens = ["material", "source_id", "sample_id", "path", "seed", "thickness", "pose", "split", "row_index"]
    lineage_like = [col for col in main_cols if any(token in col.lower() for token in leak_tokens)]
    if lineage_like:
        raise RuntimeError(f"Main feature names look lineage-like: {lineage_like}")

    gaps = count_gap_summary(output)
    pairs = pair_counts(balanced_source_on)
    max_gap = float(gaps["standardized_count_gap_abs"].max()) if not gaps.empty else 1.0
    pass_items = {
        "train_pair_support": pairs["train"] >= SUPPORT_THRESHOLDS["train_pairs_min"],
        "validation_pair_support": pairs["validation"] >= SUPPORT_THRESHOLDS["validation_pairs_min"],
        "stress_holdout_pair_support": pairs["stress_holdout"] >= SUPPORT_THRESHOLDS["stress_holdout_pairs_min"],
        "count_gap_close_to_zero": max_gap <= SUPPORT_THRESHOLDS["count_gap_standardized_max"],
        "no_lineage_like_main_features": not lineage_like,
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    view_gate = dict(schema_gate)
    view_gate.update(
        {
            "generated_by": "analysis/build_v8a_count_stratified_ordinary_view.py",
            "generated_at_utc": generated_at,
            "protocol_name": "v8A_count_stratified_ordinary_feature_view_gate",
            "development_only": True,
            "shadow_or_final_used": False,
            "reads_existing_xrt_cubes": False,
            "runs_geant4": False,
            "claim_scope": CLAIM_SCOPE,
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "count_balance_strategy": args.strategy,
            "sample_count": int(len(output)),
            "source_on_sample_count": int(len(balanced_source_on)),
            "source_off_control_sample_count": int(len(source_off)),
            "feature_column_count": int(len(main_cols)),
            "control_feature_count": int(len(control_cols)),
            "matched_pair_counts": pairs,
            "max_standardized_count_gap_abs": max_gap,
            "support_thresholds": SUPPORT_THRESHOLDS,
            "pass_items": pass_items,
            "stop_reasons": stop_reasons,
            "gate_passed": gate_passed,
            "tiny_training_gate_allowed": gate_passed,
            "decision": "count_stratified_ordinary_view_passed_ready_for_reworked_training"
            if gate_passed
            else "stop_count_stratified_ordinary_view",
            "lineage_like_main_features": lineage_like,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
    )
    view_manifest = dict(manifest)
    view_manifest.update(
        {
            "generated_by": "analysis/build_v8a_count_stratified_ordinary_view.py",
            "generated_at_utc": generated_at,
            "protocol_name": "v8A_count_stratified_ordinary_feature_view",
            "development_only": True,
            "shadow_or_final_used": False,
            "reads_existing_xrt_cubes": False,
            "runs_geant4": False,
            "claim_scope": CLAIM_SCOPE,
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "count_balance_strategy": args.strategy,
            "sample_count": int(len(output)),
            "source_on_sample_count": int(len(balanced_source_on)),
            "source_off_control_sample_count": int(len(source_off)),
            "main_feature_count": int(len(main_cols)),
            "main_feature_columns": main_cols,
            "control_feature_columns": control_cols,
            "count_stratified_ordinary_view": True,
        }
    )

    output.to_csv(output_dir / "v8a_event_sidecar_features.csv", index=False, lineterminator="\n")
    gaps.to_csv(output_dir / "v8a_count_stratified_ordinary_count_gaps.csv", index=False, lineterminator="\n")
    for filename in ["v8a_event_sidecar_long.csv", "v8a_event_control_audit.csv"]:
        source_path = input_dir / filename
        if source_path.exists():
            (output_dir / filename).write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    write_json(output_dir / "v8a_event_schema_gate.json", json_clean(view_gate))
    write_json(output_dir / "v8a_event_feature_manifest.json", json_clean(view_manifest))
    write_report(output_dir, view_gate, gaps)
    print(
        "decision={decision} strategy={strategy} pairs={train}/{validation}/{holdout} max_gap={gap:.4f}".format(
            decision=view_gate["decision"],
            strategy=args.strategy,
            train=pairs["train"],
            validation=pairs["validation"],
            holdout=pairs["stress_holdout"],
            gap=max_gap,
        )
    )


if __name__ == "__main__":
    main()
