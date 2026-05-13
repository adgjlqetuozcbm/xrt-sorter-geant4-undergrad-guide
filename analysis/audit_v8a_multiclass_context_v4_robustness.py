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
from train_v8a_multiclass_context_model import (
    DEFAULT_STRATEGY,
    add_lineage_controls,
    build_models,
    choose_count_balanced,
    ensure_output_dir,
    evaluate_estimator,
    fit_estimator,
    json_clean,
    require_sklearn,
    source_on,
    split_frame,
    write_json,
)


CLAIM_SCOPE = (
    "development-only perturbation robustness diagnostic for v8A v4 ten-material context H/M; "
    "not hardware validation, not shadow/final validation, and not a product claim"
)

THRESHOLDS = {
    "clean_replay_hm_min_recall_min": 0.95,
    "required_validation_hm_min_recall_min": 0.80,
    "required_stress_holdout_hm_min_recall_min": 0.75,
    "required_validation_macro_f1_min": 0.70,
    "required_stress_holdout_macro_f1_min": 0.65,
}

SCENARIOS: list[dict[str, Any]] = [
    {"scenario": "clean_replay", "required": True},
    {"scenario": "poisson_resample", "required": True, "poisson": True},
    {"scenario": "mild_intensity_jitter", "required": True, "sample_scale_sigma": 0.05, "feature_scale_sigma": 0.03},
    {"scenario": "moderate_intensity_jitter", "required": True, "sample_scale_sigma": 0.15, "feature_scale_sigma": 0.08},
    {"scenario": "moderate_peak_blur", "required": True, "peak_blur_fraction": 0.20},
    {"scenario": "moderate_peak_dropout", "required": True, "dropout_rate": 0.10},
    {
        "scenario": "combined_moderate",
        "required": True,
        "poisson": True,
        "sample_scale_sigma": 0.10,
        "feature_scale_sigma": 0.05,
        "peak_blur_fraction": 0.20,
        "dropout_rate": 0.10,
        "background_std_scale": 0.02,
    },
    {
        "scenario": "combined_severe_reporting_only",
        "required": False,
        "poisson": True,
        "sample_scale_sigma": 0.25,
        "feature_scale_sigma": 0.15,
        "peak_blur_fraction": 0.40,
        "dropout_rate": 0.25,
        "background_std_scale": 0.06,
    },
]


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def peak_column_angles(columns: list[str]) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    pattern = re.compile(r"^diffraction_peak_.+?_([0-9]+p[0-9]+)_\d+_norm$")
    for column in columns:
        match = pattern.match(column)
        if match:
            result.append((column, float(match.group(1).replace("p", "."))))
    return sorted(result, key=lambda item: item[1])


def blur_peak_columns(frame: pd.DataFrame, peak_cols: list[str], fraction: float) -> None:
    if not peak_cols or fraction <= 0.0:
        return
    original = frame[peak_cols].fillna(0.0).to_numpy(dtype=np.float64)
    blurred = original.copy()
    for index in range(len(peak_cols)):
        left = original[:, max(index - 1, 0)]
        right = original[:, min(index + 1, len(peak_cols) - 1)]
        blurred[:, index] = (1.0 - fraction) * original[:, index] + 0.5 * fraction * (left + right)
    frame.loc[:, peak_cols] = blurred


def perturb_frame(
    clean_eval: pd.DataFrame,
    train_reference: pd.DataFrame,
    main_cols: list[str],
    peak_cols: list[str],
    scenario: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    result = clean_eval.copy()
    if scenario["scenario"] == "clean_replay":
        return result

    rng = np.random.default_rng(seed)
    x = result[main_cols].fillna(0.0).to_numpy(dtype=np.float64)

    if scenario.get("poisson", False):
        if "count_target_photons" in result.columns:
            denominators = pd.to_numeric(result["count_target_photons"], errors="coerce").fillna(4000.0).to_numpy(dtype=np.float64)
        else:
            denominators = np.full(len(result), 4000.0, dtype=np.float64)
        denominators = np.clip(denominators, 1.0, None)
        counts = np.clip(x * denominators[:, None], 0.0, None)
        x = rng.poisson(counts).astype(np.float64) / denominators[:, None]

    sample_sigma = float(scenario.get("sample_scale_sigma", 0.0))
    if sample_sigma > 0.0:
        x *= rng.lognormal(mean=0.0, sigma=sample_sigma, size=(len(result), 1))

    feature_sigma = float(scenario.get("feature_scale_sigma", 0.0))
    if feature_sigma > 0.0:
        x *= rng.lognormal(mean=0.0, sigma=feature_sigma, size=(1, len(main_cols)))

    background_scale = float(scenario.get("background_std_scale", 0.0))
    if background_scale > 0.0:
        reference_std = train_reference[main_cols].fillna(0.0).std(axis=0).replace(0.0, np.nan)
        fallback = float(np.nanmedian(reference_std.to_numpy(dtype=np.float64)))
        if not np.isfinite(fallback) or fallback <= 0.0:
            fallback = 1e-4
        std = reference_std.fillna(fallback).to_numpy(dtype=np.float64)
        x += rng.normal(loc=0.0, scale=background_scale * std, size=x.shape)

    dropout_rate = float(scenario.get("dropout_rate", 0.0))
    if dropout_rate > 0.0:
        x *= rng.random(size=x.shape) >= dropout_rate

    result.loc[:, main_cols] = np.clip(x, 0.0, None)
    blur_peak_columns(result, peak_cols, float(scenario.get("peak_blur_fraction", 0.0)))
    return result


def selected_count_strategy(name: str) -> dict[str, Any]:
    strategy_map = {item["strategy"]: item for item in COUNT_BALANCE_STRATEGIES}
    if name not in strategy_map:
        raise SystemExit(f"Unknown count-balance strategy: {name}")
    return strategy_map[name]


def fit_track_models(
    frame: pd.DataFrame,
    models: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    train = split_frame(frame, "train")
    fitted: dict[str, Any] = {}
    for model in models:
        estimator = fit_estimator(model, train, labels)
        if estimator is not None:
            fitted[str(model["method"])] = estimator
    return fitted


def best_main_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    subset = summary[summary["family"].astype(str).eq("main") & summary["status"].astype(str).eq("evaluated")].copy()
    for _, group in subset.groupby(["track", "scenario", "eval_split"], sort=True):
        rows.append(group.sort_values(["hm_min_recall", "macro_f1", "min_class_recall"], ascending=False).iloc[0])
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 32) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.head(limit)[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], best: pd.DataFrame) -> None:
    lines = [
        "# v8A v4 multiclass-context robustness report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Worst required validation H/M min recall: `{gate['worst_required_validation_hm_min_recall']:.4f}`",
        f"- Worst required stress H/M min recall: `{gate['worst_required_stress_holdout_hm_min_recall']:.4f}`",
        f"- Worst required validation macro-F1: `{gate['worst_required_validation_macro_f1']:.4f}`",
        f"- Worst required stress macro-F1: `{gate['worst_required_stress_holdout_macro_f1']:.4f}`",
        "",
        "## Best Main Model by Scenario",
        "",
        markdown_table(
            best.sort_values(["track", "scenario", "eval_split"]),
            ["track", "scenario", "eval_split", "method", "top1_accuracy", "macro_f1", "min_class_recall", "hematite_recall", "magnetite_recall", "hm_min_recall"],
            limit=64,
        ),
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiclass_context_v4_robustness_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v8A v4 ten-material context robustness under synthetic perturbations.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--input-dir",
        default="results/accuracy_v3/v8a_multiclass_context_v4_count_overlap_event_to_feature",
    )
    parser.add_argument(
        "--model-gate",
        default="results/accuracy_v3/v8a_multiclass_context_v4_count_overlap_development_model/v8a_multiclass_context_model_gate.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/accuracy_v3/v8a_multiclass_context_v4_robustness_audit",
    )
    parser.add_argument("--count-balance-strategy", default="")
    parser.add_argument("--seed", type=int, default=54201)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    model_gate = load_json(as_project_path(project_root, args.model_gate))
    labels = sorted(frame["material"].astype(str).unique().tolist())
    sk = require_sklearn()
    main_cols, _, total_count_cols, _, _ = feature_sets(frame)
    frame, lineage_cols = add_lineage_controls(frame)
    peak_cols = [column for column, _ in peak_column_angles(main_cols)]
    models = [model for model in build_models(sk, main_cols, total_count_cols, lineage_cols) if model["family"] == "main"]

    ordinary = source_on(frame)
    requested_strategy = args.count_balance_strategy or str(model_gate.get("selected_count_balance_strategy", DEFAULT_STRATEGY))
    count_balanced, support_summary, selected_strategy = choose_count_balanced(ordinary, requested_strategy, labels)
    tracks = {"ordinary": ordinary, "count_balanced": count_balanced}
    fitted = {track: fit_track_models(track_frame, models, labels) for track, track_frame in tracks.items()}

    summary_rows: list[dict[str, Any]] = []
    decisions: list[pd.DataFrame] = []
    grouped: list[pd.DataFrame] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for track, track_frame in tracks.items():
            train_reference = split_frame(track_frame, "train")
            for eval_split in ("validation", "stress_holdout"):
                clean_eval = split_frame(track_frame, eval_split)
                perturbed_eval = perturb_frame(
                    clean_eval,
                    train_reference,
                    main_cols,
                    peak_cols,
                    scenario,
                    seed=args.seed + scenario_index * 1000 + (0 if track == "ordinary" else 100) + (0 if eval_split == "validation" else 10),
                )
                for model in models:
                    method = str(model["method"])
                    summary, model_decisions, model_grouped = evaluate_estimator(
                        track=track,
                        model=model,
                        estimator=fitted[track].get(method),
                        eval_frame=perturbed_eval,
                        eval_split=eval_split,
                        labels=labels,
                        sk=sk,
                    )
                    summary["scenario"] = scenario["scenario"]
                    summary["required"] = bool(scenario.get("required", False))
                    summary_rows.append(summary)
                    if not model_decisions.empty:
                        model_decisions["scenario"] = scenario["scenario"]
                        decisions.append(model_decisions)
                    if not model_grouped.empty:
                        model_grouped["scenario"] = scenario["scenario"]
                        grouped.append(model_grouped)

    summary = pd.DataFrame(summary_rows)
    best = best_main_rows(summary)
    summary.to_csv(output_dir / "v8a_multiclass_context_v4_robustness_summary.csv", index=False)
    best.to_csv(output_dir / "v8a_multiclass_context_v4_robustness_best_main.csv", index=False)
    support_summary.to_csv(output_dir / "v8a_multiclass_context_v4_robustness_count_balance_support.csv", index=False)
    if decisions:
        pd.concat(decisions, ignore_index=True).to_csv(output_dir / "v8a_multiclass_context_v4_robustness_decisions.csv", index=False)
    if grouped:
        pd.concat(grouped, ignore_index=True).to_csv(output_dir / "v8a_multiclass_context_v4_robustness_grouped_recall.csv", index=False)

    required = best[best["required"].astype(bool) & ~best["scenario"].astype(str).eq("clean_replay")].copy()
    clean = best[best["scenario"].astype(str).eq("clean_replay")].copy()
    def worst(split: str, column: str, data: pd.DataFrame) -> float:
        subset = data[data["eval_split"].astype(str).eq(split)]
        return float(subset[column].min()) if not subset.empty else 0.0

    stop_reasons: list[str] = []
    if not bool(schema_gate.get("gate_passed", False)):
        stop_reasons.append("event_schema_gate_not_passed")
    if not bool(model_gate.get("gate_passed", False)):
        stop_reasons.append("source_model_gate_not_passed")
    if len(labels) != 10:
        stop_reasons.append(f"expected_10_materials_observed_{len(labels)}")
    selected_support = support_summary[support_summary["selected"].astype(bool)]
    if selected_support.empty or not bool(selected_support["support_pass"].iloc[0]):
        stop_reasons.append("count_balanced_support_below_threshold")
    if worst("validation", "hm_min_recall", clean) < THRESHOLDS["clean_replay_hm_min_recall_min"]:
        stop_reasons.append("clean_replay_validation_hm_min_recall_below_threshold")
    if worst("stress_holdout", "hm_min_recall", clean) < THRESHOLDS["clean_replay_hm_min_recall_min"]:
        stop_reasons.append("clean_replay_stress_hm_min_recall_below_threshold")

    worst_required_validation_hm = worst("validation", "hm_min_recall", required)
    worst_required_stress_hm = worst("stress_holdout", "hm_min_recall", required)
    worst_required_validation_macro = worst("validation", "macro_f1", required)
    worst_required_stress_macro = worst("stress_holdout", "macro_f1", required)
    if worst_required_validation_hm < THRESHOLDS["required_validation_hm_min_recall_min"]:
        stop_reasons.append("required_validation_hm_min_recall_below_threshold")
    if worst_required_stress_hm < THRESHOLDS["required_stress_holdout_hm_min_recall_min"]:
        stop_reasons.append("required_stress_holdout_hm_min_recall_below_threshold")
    if worst_required_validation_macro < THRESHOLDS["required_validation_macro_f1_min"]:
        stop_reasons.append("required_validation_macro_f1_below_threshold")
    if worst_required_stress_macro < THRESHOLDS["required_stress_holdout_macro_f1_min"]:
        stop_reasons.append("required_stress_holdout_macro_f1_below_threshold")

    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_v4_robustness.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "claim_scope": CLAIM_SCOPE,
        "gate_passed": gate_passed,
        "decision": "robustness_development_gate_passed_not_promoted" if gate_passed else "stop_robustness_development_gate",
        "input_dir": args.input_dir,
        "model_gate": args.model_gate,
        "requested_count_balance_strategy": requested_strategy,
        "selected_count_balance_strategy": selected_strategy,
        "sample_count": int(len(frame)),
        "materials": labels,
        "main_feature_count": int(len(main_cols)),
        "peak_feature_count": int(len(peak_cols)),
        "scenarios": SCENARIOS,
        "thresholds": THRESHOLDS,
        "worst_required_validation_hm_min_recall": worst_required_validation_hm,
        "worst_required_stress_holdout_hm_min_recall": worst_required_stress_hm,
        "worst_required_validation_macro_f1": worst_required_validation_macro,
        "worst_required_stress_holdout_macro_f1": worst_required_stress_macro,
        "stop_reasons": stop_reasons,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    write_json(output_dir / "v8a_multiclass_context_v4_robustness_gate.json", json_clean(gate))
    write_report(output_dir, gate, best)
    print(
        "decision={decision} gate_passed={passed} worst_val_hm={val:.4f} worst_stress_hm={stress:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            val=worst_required_validation_hm,
            stress=worst_required_stress_hm,
        )
    )


if __name__ == "__main__":
    main()
