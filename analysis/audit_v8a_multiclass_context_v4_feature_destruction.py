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
    "development-only feature destruction diagnostic for v8A v4 ten-material context H/M; "
    "used to estimate how much idealized diffraction fingerprint redundancy remains"
)

DROPOUT_RATES = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]
REPEATS = 5


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def fit_track_models(frame: pd.DataFrame, models: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    train = split_frame(frame, "train")
    fitted: dict[str, Any] = {}
    for model in models:
        estimator = fit_estimator(model, train, labels)
        if estimator is not None:
            fitted[str(model["method"])] = estimator
    return fitted


def destroy_eval_frame(
    clean_eval: pd.DataFrame,
    main_cols: list[str],
    peak_cols: list[str],
    window_cols: list[str],
    mode: str,
    dropout_rate: float,
    seed: int,
) -> pd.DataFrame:
    result = clean_eval.copy()
    rng = np.random.default_rng(seed)
    if mode == "clean":
        return result
    target_cols = main_cols if mode == "drop_all_main" else peak_cols
    if target_cols and dropout_rate > 0.0:
        values = result[target_cols].fillna(0.0).to_numpy(dtype=np.float64)
        values *= rng.random(size=values.shape) >= dropout_rate
        result.loc[:, target_cols] = values
    if mode in {"drop_all_main_zero_windows", "drop_peaks_zero_windows"} and window_cols:
        result.loc[:, window_cols] = 0.0
    return result


def best_main_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    subset = summary[summary["family"].astype(str).eq("main") & summary["status"].astype(str).eq("evaluated")].copy()
    for _, group in subset.groupby(["track", "mode", "dropout_rate", "repeat", "eval_split"], sort=True):
        rows.append(group.sort_values(["hm_min_recall", "macro_f1", "min_class_recall"], ascending=False).iloc[0])
    return pd.DataFrame(rows)


def aggregate_repeats(best: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["top1_accuracy", "macro_f1", "min_class_recall", "hematite_recall", "magnetite_recall", "hm_min_recall"]
    rows: list[dict[str, Any]] = []
    for keys, group in best.groupby(["track", "mode", "dropout_rate", "eval_split"], sort=True):
        track, mode, dropout_rate, eval_split = keys
        row: dict[str, Any] = {
            "track": track,
            "mode": mode,
            "dropout_rate": float(dropout_rate),
            "eval_split": eval_split,
            "repeats": int(len(group)),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_min"] = float(group[col].min())
        rows.append(row)
    return pd.DataFrame(rows)


def first_failure(summary: pd.DataFrame, track: str, mode: str, split: str, threshold: float) -> float | None:
    subset = summary[
        summary["track"].astype(str).eq(track)
        & summary["mode"].astype(str).eq(mode)
        & summary["eval_split"].astype(str).eq(split)
    ].sort_values("dropout_rate")
    for _, row in subset.iterrows():
        if float(row["hm_min_recall_min"]) < threshold:
            return float(row["dropout_rate"])
    return None


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 60) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.head(limit)[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], aggregate: pd.DataFrame) -> None:
    focused = aggregate[
        aggregate["mode"].isin(["drop_all_main", "drop_all_main_zero_windows"])
        & aggregate["eval_split"].astype(str).eq("validation")
    ].copy()
    lines = [
        "# v8A v4 feature destruction diagnostic",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Summary",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Ordinary validation first H/M failure below 0.80: `{gate['ordinary_validation_first_failure_below_0p80']}`",
        f"- Count-balanced validation first H/M failure below 0.80: `{gate['count_balanced_validation_first_failure_below_0p80']}`",
        "",
        "## Validation Destruction Curve",
        "",
        markdown_table(
            focused.sort_values(["track", "mode", "dropout_rate"]),
            ["track", "mode", "dropout_rate", "hm_min_recall_mean", "hm_min_recall_min", "macro_f1_mean", "min_class_recall_mean"],
            limit=80,
        ),
        "",
    ]
    (output_dir / "v8a_multiclass_context_v4_feature_destruction_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Destroy v8A v4 diffraction features to estimate redundancy and fragility.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_multiclass_context_v4_count_overlap_event_to_feature")
    parser.add_argument(
        "--model-gate",
        default="results/accuracy_v3/v8a_multiclass_context_v4_count_overlap_development_model/v8a_multiclass_context_model_gate.json",
    )
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v4_feature_destruction_audit")
    parser.add_argument("--count-balance-strategy", default="")
    parser.add_argument("--seed", type=int, default=54801)
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
    peak_cols = [col for col in main_cols if col.startswith("diffraction_peak_")]
    window_cols = [col for col in main_cols if col.startswith("diffraction_window_") or col.startswith("diffraction_ratio_")]
    models = [model for model in build_models(sk, main_cols, total_count_cols, lineage_cols) if model["family"] == "main"]

    ordinary = source_on(frame)
    requested_strategy = args.count_balance_strategy or str(model_gate.get("selected_count_balance_strategy", DEFAULT_STRATEGY))
    count_balanced, support_summary, selected_strategy = choose_count_balanced(ordinary, requested_strategy, labels)
    tracks = {"ordinary": ordinary, "count_balanced": count_balanced}
    fitted = {track: fit_track_models(track_frame, models, labels) for track, track_frame in tracks.items()}

    modes = ["clean", "drop_peaks_keep_windows", "drop_peaks_zero_windows", "drop_all_main", "drop_all_main_zero_windows"]
    summary_rows: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(modes):
        rates = [0.0] if mode == "clean" else DROPOUT_RATES
        for rate in rates:
            repeat_count = 1 if rate == 0.0 else REPEATS
            for repeat in range(repeat_count):
                for track, track_frame in tracks.items():
                    for eval_split in ("validation", "stress_holdout"):
                        clean_eval = split_frame(track_frame, eval_split)
                        destroyed_eval = destroy_eval_frame(
                            clean_eval,
                            main_cols,
                            peak_cols,
                            window_cols,
                            mode,
                            rate,
                            seed=args.seed + mode_index * 10000 + repeat * 100 + (0 if track == "ordinary" else 10) + (0 if eval_split == "validation" else 1),
                        )
                        for model in models:
                            method = str(model["method"])
                            summary, _, _ = evaluate_estimator(
                                track=track,
                                model=model,
                                estimator=fitted[track].get(method),
                                eval_frame=destroyed_eval,
                                eval_split=eval_split,
                                labels=labels,
                                sk=sk,
                            )
                            summary["mode"] = mode
                            summary["dropout_rate"] = float(rate)
                            summary["repeat"] = int(repeat)
                            summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    best = best_main_rows(summary)
    aggregate = aggregate_repeats(best)
    summary.to_csv(output_dir / "v8a_multiclass_context_v4_feature_destruction_summary.csv", index=False)
    best.to_csv(output_dir / "v8a_multiclass_context_v4_feature_destruction_best_main.csv", index=False)
    aggregate.to_csv(output_dir / "v8a_multiclass_context_v4_feature_destruction_aggregate.csv", index=False)
    support_summary.to_csv(output_dir / "v8a_multiclass_context_v4_feature_destruction_count_balance_support.csv", index=False)

    gate = {
        "generated_by": "analysis/audit_v8a_multiclass_context_v4_feature_destruction.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "claim_scope": CLAIM_SCOPE,
        "decision": "feature_destruction_diagnostic_completed",
        "schema_gate_passed": bool(schema_gate.get("gate_passed", False)),
        "source_model_gate_passed": bool(model_gate.get("gate_passed", False)),
        "input_dir": args.input_dir,
        "model_gate": args.model_gate,
        "requested_count_balance_strategy": requested_strategy,
        "selected_count_balance_strategy": selected_strategy,
        "sample_count": int(len(frame)),
        "materials": labels,
        "main_feature_count": int(len(main_cols)),
        "peak_feature_count": int(len(peak_cols)),
        "window_feature_count": int(len(window_cols)),
        "dropout_rates": DROPOUT_RATES,
        "repeats": REPEATS,
        "ordinary_validation_first_failure_below_0p80": first_failure(aggregate, "ordinary", "drop_all_main_zero_windows", "validation", 0.80),
        "ordinary_stress_first_failure_below_0p80": first_failure(aggregate, "ordinary", "drop_all_main_zero_windows", "stress_holdout", 0.80),
        "count_balanced_validation_first_failure_below_0p80": first_failure(aggregate, "count_balanced", "drop_all_main_zero_windows", "validation", 0.80),
        "count_balanced_stress_first_failure_below_0p80": first_failure(aggregate, "count_balanced", "drop_all_main_zero_windows", "stress_holdout", 0.80),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    write_json(output_dir / "v8a_multiclass_context_v4_feature_destruction_gate.json", json_clean(gate))
    write_report(output_dir, gate, aggregate)
    print(
        "decision={decision} ordinary_val_failure={ov} count_balanced_val_failure={cv}".format(
            decision=gate["decision"],
            ov=gate["ordinary_validation_first_failure_below_0p80"],
            cv=gate["count_balanced_validation_first_failure_below_0p80"],
        )
    )


if __name__ == "__main__":
    main()
