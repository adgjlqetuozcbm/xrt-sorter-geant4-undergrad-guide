from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_v8a_count_balance_sensitivity import STRATEGIES, build_balanced_subset
from train_v8a_event_feature_smoke import feature_sets, load_json
from train_v8a_medium_development_model import (
    MAIN_METHODS,
    THRESHOLDS,
    build_models,
    evaluate_estimator,
    fit_estimator,
    require_sklearn,
    selected_frame,
    value_for,
)


CLAIM_SCOPE = (
    "development-only stability replication audit over count-robust v8A H/M features; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)


MODEL_SEEDS = [9801, 9811, 9821]
COUNT_BALANCE_STRATEGIES = ["fixed_bin_width_0p003", "quantile_bins_12"]


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


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


def set_model_seed(model: dict[str, Any], seed: int) -> dict[str, Any]:
    seeded = dict(model)
    estimator = deepcopy(model["estimator"])
    if hasattr(estimator, "set_params"):
        params = estimator.get_params()
        updates = {}
        for key in params:
            if key.endswith("random_state") or key == "random_state":
                updates[key] = seed
        if updates:
            estimator.set_params(**updates)
    seeded["estimator"] = estimator
    seeded["method"] = f"{model['method']}_seed{seed}"
    return seeded


def evaluate_models(frame: pd.DataFrame, sk: dict[str, Any], seed: int, strategy_name: str, strategy_frame: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    eval_frame = strategy_frame if strategy_frame is not None else frame
    main_cols, _, total_count_cols, overlap_cols, thickness_pose_cols = feature_sets(eval_frame)
    models = [set_model_seed(model, seed) for model in build_models(sk, main_cols, total_count_cols, overlap_cols, thickness_pose_cols)]
    rows: list[dict[str, Any]] = []
    for model in models:
        train = selected_frame(eval_frame, "train", model["train_source_mode"])
        estimator = fit_estimator(model, train)
        validation = selected_frame(eval_frame, "validation", model["eval_source_mode"])
        summary, _, _, _, _ = evaluate_estimator(model, estimator, validation, "validation", sk, selected_threshold=None)
        threshold = float(summary["threshold"])
        holdout = selected_frame(eval_frame, "stress_holdout", model["eval_source_mode"])
        holdout_summary, _, _, _, _ = evaluate_estimator(model, estimator, holdout, "stress_holdout", sk, selected_threshold=threshold)
        for item in [summary, holdout_summary]:
            item["model_seed"] = seed
            item["count_balance_strategy"] = strategy_name
            item["base_method"] = str(model["method"]).rsplit("_seed", 1)[0]
            rows.append(item)
    return rows


def build_strategy_frame(frame: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    strategy = strategy_by_name(strategy_name)
    balanced = build_balanced_subset(frame, strategy)
    source_off = frame[frame["source_mode"].astype(str).eq("custom_diffraction_off")].copy()
    source_off["count_balance_bin"] = "source_off_control"
    source_off["match_pair_id"] = "source_off_control"
    source_off["match_delta_total_count_norm"] = 0.0
    return pd.concat([balanced, source_off], ignore_index=True, sort=False)


def summarize(selection: pd.DataFrame) -> dict[str, float]:
    rows: dict[str, float] = {}
    for strategy in sorted(selection["count_balance_strategy"].unique()):
        for seed in sorted(selection["model_seed"].unique()):
            subset = selection[selection["count_balance_strategy"].eq(strategy) & selection["model_seed"].eq(seed)]
            validation_main = subset[subset["base_method"].isin(MAIN_METHODS) & subset["eval_split"].eq("validation")]
            if validation_main.empty:
                continue
            selected = validation_main.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall", "worst_pose_hm_min_recall", "accuracy"], ascending=False).iloc[0]
            method = str(selected["method"])
            rows[f"{strategy}|seed{seed}|selected_validation_main"] = float(selected["hm_min_recall"])
            rows[f"{strategy}|seed{seed}|selected_stress_main"] = value_for(subset, method, "stress_holdout", "hm_min_recall")
            rows[f"{strategy}|seed{seed}|total_count_max"] = max(
                value_for(subset, f"ExtraTreesTotalCountOnly_seed{seed}", "validation", "hm_min_recall"),
                value_for(subset, f"ExtraTreesTotalCountOnly_seed{seed}", "stress_holdout", "hm_min_recall"),
            )
    return rows


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 18) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.head(limit)[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], selection: pd.DataFrame) -> None:
    lines = [
        "# v8A count-robust stability audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Model seeds: `{','.join(str(item) for item in gate['model_seeds'])}`",
        f"- Count-balance strategies: `{','.join(gate['count_balance_strategies'])}`",
        f"- Worst validation main H/M min recall: `{gate['worst_validation_main_hm_min_recall']:.4f}`",
        f"- Worst stress-holdout main H/M min recall: `{gate['worst_stress_holdout_main_hm_min_recall']:.4f}`",
        f"- Worst total-count-only H/M min recall: `{gate['worst_total_count_only_hm_min_recall']:.4f}`",
        "",
        "## Selection Snapshot",
        "",
        markdown_table(
            selection.sort_values(["count_balance_strategy", "model_seed", "eval_split", "base_method"]),
            ["count_balance_strategy", "model_seed", "base_method", "eval_split", "hm_min_recall", "worst_thickness_hm_min_recall", "worst_pose_hm_min_recall"],
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
    (output_dir / "v8a_count_robust_stability_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit stability replication for count-robust v8A H/M features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_event_to_feature")
    parser.add_argument("--phase4-gate", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_development_model/v8a_medium_development_model_gate.json")
    parser.add_argument("--count-balanced-gate", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_count_balanced_retest/v8a_count_balanced_retest_gate.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_stability")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    phase4_gate = load_json(project_root / args.phase4_gate)
    count_balanced_gate = load_json(project_root / args.count_balanced_gate)
    for name, payload in {"phase4_gate": phase4_gate, "count_balanced_gate": count_balanced_gate}.items():
        if bool(payload.get("shadow_or_final_used")) or bool(payload.get("reads_existing_xrt_cubes")):
            raise RuntimeError(f"Refusing stability audit because {name} reports forbidden input use.")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    sk = require_sklearn()
    rows: list[dict[str, Any]] = []
    for strategy_name in COUNT_BALANCE_STRATEGIES:
        strategy_frame = build_strategy_frame(frame, strategy_name)
        pairs = pair_counts(strategy_frame[strategy_frame["source_mode"].astype(str).eq("custom_diffraction_on")])
        if min(pairs.values()) <= 0:
            raise RuntimeError(f"Strategy {strategy_name} has no support: {pairs}")
        for seed in MODEL_SEEDS:
            rows.extend(evaluate_models(frame, sk, seed, strategy_name, strategy_frame=strategy_frame))
    selection = pd.DataFrame(rows)
    metrics = summarize(selection)
    validation_main_values = [value for key, value in metrics.items() if key.endswith("selected_validation_main")]
    stress_main_values = [value for key, value in metrics.items() if key.endswith("selected_stress_main")]
    total_count_values = [value for key, value in metrics.items() if key.endswith("total_count_max")]
    worst_validation_main = float(min(validation_main_values)) if validation_main_values else 0.0
    worst_stress_main = float(min(stress_main_values)) if stress_main_values else 0.0
    worst_total_count = float(max(total_count_values)) if total_count_values else 1.0
    pass_items = {
        "phase4_gate_passed": bool(phase4_gate.get("gate_passed")),
        "count_balanced_gate_passed": bool(count_balanced_gate.get("gate_passed")),
        "model_seed_count": len(MODEL_SEEDS) >= 3,
        "count_balance_strategy_count": len(COUNT_BALANCE_STRATEGIES) >= 2,
        "validation_main_hm_min_recall": worst_validation_main >= THRESHOLDS["validation_main_hm_min_recall_min"],
        "stress_holdout_main_hm_min_recall": worst_stress_main >= THRESHOLDS["stress_holdout_main_hm_min_recall_min"],
        "total_count_only_below_ceiling": worst_total_count < THRESHOLDS["total_count_only_hm_min_recall_max"],
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/audit_v8a_count_robust_stability.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v8A_count_robust_stability_audit",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "gate_passed": gate_passed,
        "decision": "count_robust_stability_passed_ready_for_development_matrix_prereg" if gate_passed else "stop_or_rework_count_robust_stability",
        "model_seeds": MODEL_SEEDS,
        "count_balance_strategies": COUNT_BALANCE_STRATEGIES,
        "worst_validation_main_hm_min_recall": worst_validation_main,
        "worst_stress_holdout_main_hm_min_recall": worst_stress_main,
        "worst_total_count_only_hm_min_recall": worst_total_count,
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    selection.to_csv(output_dir / "v8a_count_robust_stability_model_selection.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_count_robust_stability_gate.json", gate)
    write_report(output_dir, gate, selection)
    print(
        "decision={decision} gate_passed={passed} worst_main={main:.4f}/{stress:.4f} total_count={total:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            main=worst_validation_main,
            stress=worst_stress_main,
            total=worst_total_count,
        )
    )


if __name__ == "__main__":
    main()
