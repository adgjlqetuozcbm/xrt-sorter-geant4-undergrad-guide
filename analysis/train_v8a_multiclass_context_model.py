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

from train_v8a_event_feature_smoke import feature_sets, load_json
from train_v8a_medium_development_model import expected_calibration_error


CLAIM_SCOPE = (
    "development-only 10-material context model diagnostics for H/M; not product accuracy, "
    "not hardware validation, not shadow/final validation, and not manuscript-grade powder XRD"
)

HM_PAIR = ("Hematite", "Magnetite")
SPLITS = ("train", "validation", "stress_holdout")
MAIN_METHODS = ("LogisticMulticlassMain", "ExtraTreesMulticlassMain")
SHUFFLE_SEEDS = list(range(52001, 52021))

THRESHOLDS = {
    "validation_hm_min_recall_min": 0.70,
    "stress_holdout_hm_min_recall_min": 0.65,
    "validation_macro_f1_min": 0.55,
    "stress_holdout_macro_f1_min": 0.50,
    "validation_min_class_recall_min": 0.30,
    "stress_holdout_min_class_recall_min": 0.25,
    "total_count_only_hm_min_recall_max": 0.60,
    "lineage_only_hm_min_recall_max": 0.60,
    "shuffled_label_hm_p95_max": 0.55,
    "shuffled_label_macro_f1_p95_max": 0.35,
    "real_minus_null_hm_margin_min": 0.15,
    "count_balanced_train_cells_min": 100,
    "count_balanced_validation_cells_min": 60,
    "count_balanced_stress_holdout_cells_min": 60,
}

COUNT_BALANCE_STRATEGIES = [
    {"strategy": "fixed_bin_width_0p003", "mode": "fixed_bin", "bin_width": 0.003},
    {"strategy": "fixed_bin_width_0p005", "mode": "fixed_bin", "bin_width": 0.005},
    {"strategy": "fixed_bin_width_0p010", "mode": "fixed_bin", "bin_width": 0.010},
    {"strategy": "fixed_bin_width_0p015", "mode": "fixed_bin", "bin_width": 0.015},
    {"strategy": "fixed_bin_width_0p020", "mode": "fixed_bin", "bin_width": 0.020},
    {"strategy": "fixed_bin_width_0p040", "mode": "fixed_bin", "bin_width": 0.040},
    {"strategy": "fixed_bin_width_0p050", "mode": "fixed_bin", "bin_width": 0.050},
    {"strategy": "sliding_window_0p040", "mode": "sliding_window", "window_width": 0.040},
    {"strategy": "sliding_window_0p050", "mode": "sliding_window", "window_width": 0.050},
    {"strategy": "sliding_window_0p075", "mode": "sliding_window", "window_width": 0.075},
    {"strategy": "sliding_window_0p100", "mode": "sliding_window", "window_width": 0.100},
]
DEFAULT_STRATEGY = "fixed_bin_width_0p020"


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


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Missing scikit-learn. Run with the project venv, for example "
            "`/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python analysis/train_v8a_multiclass_context_model.py`."
        ) from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
    }


def source_on(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame[frame["split"].astype(str).eq(split)].copy()


def recall_by_class(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for label in labels:
        mask = y_true == label
        recalls[label] = float(np.mean(y_pred[mask] == label)) if mask.any() else 0.0
    return recalls


def metric_row(
    *,
    track: str,
    method: str,
    family: str,
    eval_split: str,
    feature_count: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    sk: dict[str, Any],
) -> dict[str, Any]:
    recalls = recall_by_class(y_true, y_pred, labels)
    hm_recalls = {material: recalls.get(material, 0.0) for material in HM_PAIR}
    return {
        "track": track,
        "method": method,
        "family": family,
        "eval_split": eval_split,
        "samples": int(len(y_true)),
        "feature_count": int(feature_count),
        "top1_accuracy": float(sk["accuracy_score"](y_true, y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": float(sk["balanced_accuracy_score"](y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(sk["f1_score"](y_true, y_pred, labels=labels, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "min_class_recall": float(min(recalls.values())) if recalls else 0.0,
        "hematite_recall": hm_recalls["Hematite"],
        "magnetite_recall": hm_recalls["Magnetite"],
        "hm_min_recall": float(min(hm_recalls.values())),
    }


def decision_rows(eval_frame: pd.DataFrame, track: str, method: str, y_pred: np.ndarray) -> pd.DataFrame:
    cols = [
        col
        for col in [
            "sample_id",
            "split",
            "material",
            "source_mode",
            "stress_label",
            "source_id",
            "random_seed",
            "thickness_mm",
            "pose_index",
            "seed_block",
            "count_target_bin",
            "clean_context_cell_id",
            "nuisance_cell_id",
            "count_balance_bin",
        ]
        if col in eval_frame.columns
    ]
    decisions = eval_frame[cols].copy()
    decisions["track"] = track
    decisions["method"] = method
    decisions["prediction"] = y_pred
    decisions["is_correct"] = decisions["material"].astype(str).to_numpy() == y_pred
    return decisions


def grouped_recall_rows(decisions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name in ["thickness_mm", "pose_index", "seed_block", "count_target_bin"]:
        if group_name not in decisions.columns:
            continue
        for value, group in decisions.groupby(group_name, sort=True):
            y_true = group["material"].astype(str).to_numpy()
            y_pred = group["prediction"].astype(str).to_numpy()
            recalls = recall_by_class(y_true, y_pred, labels)
            hm_values = [recalls.get(material, 0.0) for material in HM_PAIR]
            rows.append(
                {
                    "track": group["track"].iloc[0],
                    "method": group["method"].iloc[0],
                    "eval_split": group["split"].iloc[0],
                    "group": group_name,
                    "value": value,
                    "samples": int(len(group)),
                    "min_class_recall": float(min(recalls.values())) if recalls else 0.0,
                    "hematite_recall": recalls.get("Hematite", 0.0),
                    "magnetite_recall": recalls.get("Magnetite", 0.0),
                    "hm_min_recall": float(min(hm_values)),
                }
            )
    return pd.DataFrame(rows)


def add_lineage_controls(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    cols: list[str] = []
    numeric_controls = ["thickness_mm", "pose_index", "count_target_photons", "seed_block_seed", "random_seed", "context_replicate_index"]
    for col in numeric_controls:
        if col in result.columns:
            name = f"lineage_control_{col}"
            result[name] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
            cols.append(name)
    for col in ["seed_block", "count_target_bin"]:
        if col in result.columns:
            codes, _ = pd.factorize(result[col].astype(str), sort=True)
            name = f"lineage_control_{col}"
            result[name] = codes.astype(float)
            cols.append(name)
    return result, cols


def build_models(sk: dict[str, Any], main_cols: list[str], total_count_cols: list[str], lineage_cols: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "method": "LogisticMulticlassMain",
            "family": "main",
            "feature_cols": main_cols,
            "estimator": sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=4000, class_weight="balanced", random_state=52101),
            ),
        },
        {
            "method": "ExtraTreesMulticlassMain",
            "family": "main",
            "feature_cols": main_cols,
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=500,
                random_state=52102,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
        },
        {
            "method": "ExtraTreesTotalCountOnly",
            "family": "control",
            "feature_cols": total_count_cols,
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=300,
                random_state=52103,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
        },
        {
            "method": "ExtraTreesLineageOnly",
            "family": "control",
            "feature_cols": lineage_cols,
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=300,
                random_state=52104,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
        },
    ]


def fit_estimator(model: dict[str, Any], train: pd.DataFrame, labels: list[str], shuffle_seed: int | None = None) -> Any | None:
    cols = list(model["feature_cols"])
    if not cols or train.empty:
        return None
    y_train = train["material"].astype(str).to_numpy()
    if len(set(y_train)) < 2:
        return None
    if shuffle_seed is not None:
        y_train = np.random.default_rng(shuffle_seed).permutation(y_train)
    estimator = deepcopy(model["estimator"])
    estimator.fit(train[cols].fillna(0.0).to_numpy(dtype=np.float64), y_train)
    return estimator


def evaluate_estimator(
    *,
    track: str,
    model: dict[str, Any],
    estimator: Any | None,
    eval_frame: pd.DataFrame,
    eval_split: str,
    labels: list[str],
    sk: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cols = list(model["feature_cols"])
    method = str(model["method"])
    family = str(model["family"])
    if estimator is None or eval_frame.empty or not cols:
        empty = pd.DataFrame()
        return (
            {
                "track": track,
                "method": method,
                "family": family,
                "eval_split": eval_split,
                "status": "not_evaluable",
                "samples": int(len(eval_frame)),
                "feature_count": int(len(cols)),
                "top1_accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "macro_f1": 0.0,
                "min_class_recall": 0.0,
                "hematite_recall": 0.0,
                "magnetite_recall": 0.0,
                "hm_min_recall": 0.0,
            },
            empty,
            empty,
        )
    y_true = eval_frame["material"].astype(str).to_numpy()
    x_eval = eval_frame[cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_pred = np.asarray(estimator.predict(x_eval)).astype(str)
    summary = metric_row(
        track=track,
        method=method,
        family=family,
        eval_split=eval_split,
        feature_count=len(cols),
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        sk=sk,
    )
    summary["status"] = "evaluated"
    decisions = decision_rows(eval_frame, track, method, y_pred)
    grouped = grouped_recall_rows(decisions, labels)
    return summary, decisions, grouped


def add_count_bins(frame: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    width = float(strategy["bin_width"])
    result["count_balance_bin"] = np.floor(result["control_total_count_norm"].astype(float) / width).astype(int).astype(str)
    return result


def count_context_group_cols(source: pd.DataFrame) -> list[str]:
    if "clean_context_cell_id" in source.columns:
        return ["split", "clean_context_cell_id"]
    group_cols = [
        "split",
        "source_id",
        "source_family",
        "stress_label",
        "thickness_mm",
        "pose_index",
        "seed_block",
        "count_target_bin",
    ]
    return [column for column in group_cols if column in source.columns]


def support_pass(cells: dict[str, int]) -> bool:
    return bool(
        cells.get("train", 0) >= THRESHOLDS["count_balanced_train_cells_min"]
        and cells.get("validation", 0) >= THRESHOLDS["count_balanced_validation_cells_min"]
        and cells.get("stress_holdout", 0) >= THRESHOLDS["count_balanced_stress_holdout_cells_min"]
    )


def context_cell_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "count_balance_context_cell_id" not in frame.columns:
        return {split: 0 for split in SPLITS}
    return {
        split: int(frame[frame["split"].astype(str).eq(split)]["count_balance_context_cell_id"].nunique())
        for split in SPLITS
    }


def build_count_balanced_subset(frame: pd.DataFrame, strategy: dict[str, Any], labels: list[str]) -> pd.DataFrame:
    source = source_on(frame)
    mode = str(strategy.get("mode", "fixed_bin"))
    if mode == "fixed_bin":
        source = add_count_bins(source, strategy)
        group_cols = count_context_group_cols(source) + ["count_balance_bin"]
    elif mode == "sliding_window":
        source = source.copy()
        source["count_balance_bin"] = strategy["strategy"]
        group_cols = count_context_group_cols(source)
    else:
        raise ValueError(f"Unknown count-balance strategy mode: {mode}")
    rows: list[pd.Series] = []
    for keys, group in source.groupby(group_cols, sort=True, observed=True):
        if mode == "sliding_window":
            values = group["control_total_count_norm"].astype(float)
            if float(values.max() - values.min()) > float(strategy["window_width"]):
                continue
        per_material = {
            material: group[group["material"].astype(str).eq(material)].sort_values("control_total_count_norm")
            for material in labels
        }
        cell_count = min(len(material_frame) for material_frame in per_material.values())
        if cell_count <= 0:
            continue
        for cell_index in range(cell_count):
            cell_id = f"{strategy['strategy']}|{keys}|cell{cell_index + 1:03d}"
            for material in labels:
                row = per_material[material].iloc[cell_index].copy()
                row["count_balance_context_cell_id"] = cell_id
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(source.columns) + ["count_balance_context_cell_id"])
    return pd.DataFrame(rows).reset_index(drop=True)


def choose_count_balanced(frame: pd.DataFrame, strategy_name: str, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    strategy_map = {item["strategy"]: item for item in COUNT_BALANCE_STRATEGIES}
    requested = strategy_map[strategy_name]
    summary_rows: list[dict[str, Any]] = []
    selected_name = strategy_name
    selected_frame = build_count_balanced_subset(frame, requested, labels)
    selected_cells = context_cell_counts(selected_frame)
    summary_rows.append(
        {
            "strategy": strategy_name,
            "selected": True,
            **{f"{split}_cells": selected_cells[split] for split in SPLITS},
            "support_pass": support_pass(selected_cells),
        }
    )
    if not support_pass(selected_cells):
        for candidate in COUNT_BALANCE_STRATEGIES:
            if candidate["strategy"] == strategy_name:
                continue
            candidate_frame = build_count_balanced_subset(frame, candidate, labels)
            candidate_cells = context_cell_counts(candidate_frame)
            summary_rows.append(
                {
                    "strategy": candidate["strategy"],
                    "selected": False,
                    **{f"{split}_cells": candidate_cells[split] for split in SPLITS},
                    "support_pass": support_pass(candidate_cells),
                }
            )
            if support_pass(candidate_cells):
                selected_name = str(candidate["strategy"])
                selected_frame = candidate_frame
                summary_rows[0]["selected"] = False
                summary_rows[-1]["selected"] = True
                break
    return selected_frame, pd.DataFrame(summary_rows), selected_name


def run_track(track: str, frame: pd.DataFrame, models: list[dict[str, Any]], labels: list[str], sk: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = split_frame(frame, "train")
    summary_rows: list[dict[str, Any]] = []
    decisions: list[pd.DataFrame] = []
    grouped: list[pd.DataFrame] = []
    for model in models:
        estimator = fit_estimator(model, train, labels)
        for eval_split in ("validation", "stress_holdout"):
            summary, model_decisions, model_grouped = evaluate_estimator(
                track=track,
                model=model,
                estimator=estimator,
                eval_frame=split_frame(frame, eval_split),
                eval_split=eval_split,
                labels=labels,
                sk=sk,
            )
            summary_rows.append(summary)
            if not model_decisions.empty:
                decisions.append(model_decisions)
            if not model_grouped.empty:
                grouped.append(model_grouped)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame(),
        pd.concat(grouped, ignore_index=True) if grouped else pd.DataFrame(),
    )


def shuffled_null(frame: pd.DataFrame, model: dict[str, Any], labels: list[str], sk: dict[str, Any], track: str) -> pd.DataFrame:
    train = split_frame(frame, "train")
    rows: list[dict[str, Any]] = []
    for seed in SHUFFLE_SEEDS:
        estimator = fit_estimator(model, train, labels, shuffle_seed=seed)
        for eval_split in ("validation", "stress_holdout"):
            summary, _, _ = evaluate_estimator(
                track=track,
                model=model,
                estimator=estimator,
                eval_frame=split_frame(frame, eval_split),
                eval_split=eval_split,
                labels=labels,
                sk=sk,
            )
            summary["shuffle_seed"] = seed
            rows.append(summary)
    return pd.DataFrame(rows)


def confusion_rows(decisions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decisions.empty:
        return pd.DataFrame()
    subset = decisions[decisions["method"].isin(MAIN_METHODS)].copy()
    for keys, group in subset.groupby(["track", "method", "split"], sort=True):
        track, method, split = keys
        for actual in labels:
            actual_group = group[group["material"].astype(str).eq(actual)]
            denom = max(len(actual_group), 1)
            for predicted in labels:
                rows.append(
                    {
                        "track": track,
                        "method": method,
                        "eval_split": split,
                        "actual": actual,
                        "predicted": predicted,
                        "count": int((actual_group["prediction"].astype(str) == predicted).sum()),
                        "rate": float((actual_group["prediction"].astype(str) == predicted).sum() / denom),
                    }
                )
    return pd.DataFrame(rows)


def top_main(summary: pd.DataFrame, track: str, split: str) -> pd.Series:
    candidates = summary[
        summary["track"].astype(str).eq(track)
        & summary["eval_split"].astype(str).eq(split)
        & summary["family"].astype(str).eq("main")
        & summary["status"].astype(str).eq("evaluated")
    ].copy()
    if candidates.empty:
        return pd.Series(dtype=object)
    return candidates.sort_values(["hm_min_recall", "macro_f1", "min_class_recall"], ascending=False).iloc[0]


def write_report(output_dir: Path, gate: dict[str, Any], summary: pd.DataFrame, support: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame, columns: list[str], limit: int = 16) -> str:
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

    lines = [
        "# v8A ten-material context model report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Ordinary validation H/M min recall: `{gate['ordinary_validation_hm_min_recall']:.4f}`",
        f"- Ordinary stress H/M min recall: `{gate['ordinary_stress_holdout_hm_min_recall']:.4f}`",
        f"- Count-balanced validation H/M min recall: `{gate['count_balanced_validation_hm_min_recall']:.4f}`",
        f"- Count-balanced stress H/M min recall: `{gate['count_balanced_stress_holdout_hm_min_recall']:.4f}`",
        f"- Total-count-only validation H/M min recall: `{gate['total_count_only_validation_hm_min_recall']:.4f}`",
        f"- Shuffled-label validation H/M p95: `{gate['shuffled_label_validation_hm_p95']:.4f}`",
        "",
        "## Model Summary",
        "",
        table(
            summary.sort_values(["track", "eval_split", "family", "hm_min_recall"], ascending=[True, True, True, False]),
            ["track", "method", "family", "eval_split", "top1_accuracy", "macro_f1", "min_class_recall", "hematite_recall", "magnetite_recall", "hm_min_recall"],
        ),
        "",
        "## Count-Balanced Support",
        "",
        table(support, list(support.columns), limit=12),
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiclass_context_model_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate development-only 10-material context models and H/M controls.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--final-audit-gate", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count-balance-strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    feature_path = input_dir / "v8a_event_sidecar_features.csv"
    manifest_path = input_dir / "v8a_event_feature_manifest.json"
    schema_gate_path = input_dir / "v8a_event_schema_gate.json"
    frame = pd.read_csv(feature_path)
    manifest = load_json(manifest_path)
    schema_gate = load_json(schema_gate_path)
    final_audit_gate = load_json(as_project_path(project_root, args.final_audit_gate)) if args.final_audit_gate else {"gate_passed": True}

    labels = sorted(frame["material"].astype(str).unique().tolist())
    sk = require_sklearn()
    main_cols, control_cols, total_count_cols, _, _ = feature_sets(frame)
    frame, lineage_cols = add_lineage_controls(frame)
    models = build_models(sk, main_cols, total_count_cols, lineage_cols)
    ordinary = source_on(frame)
    count_balanced, support_summary, selected_strategy = choose_count_balanced(ordinary, args.count_balance_strategy, labels)

    ordinary_summary, ordinary_decisions, ordinary_grouped = run_track("ordinary", ordinary, models, labels, sk)
    balanced_summary, balanced_decisions, balanced_grouped = run_track("count_balanced", count_balanced, models, labels, sk)
    summary = pd.concat([ordinary_summary, balanced_summary], ignore_index=True)
    decisions = pd.concat([ordinary_decisions, balanced_decisions], ignore_index=True)
    grouped = pd.concat([ordinary_grouped, balanced_grouped], ignore_index=True)
    shuffle_model = next(model for model in models if model["method"] == "ExtraTreesMulticlassMain")
    ordinary_null = shuffled_null(ordinary, shuffle_model, labels, sk, "ordinary")
    balanced_null = shuffled_null(count_balanced, shuffle_model, labels, sk, "count_balanced")
    null_summary = pd.concat([ordinary_null, balanced_null], ignore_index=True)
    confusion = confusion_rows(decisions, labels)

    summary.to_csv(output_dir / "v8a_multiclass_context_model_summary.csv", index=False)
    decisions.to_csv(output_dir / "v8a_multiclass_context_decisions.csv", index=False)
    grouped.to_csv(output_dir / "v8a_multiclass_context_grouped_recall.csv", index=False)
    support_summary.to_csv(output_dir / "v8a_multiclass_context_count_balance_support.csv", index=False)
    null_summary.to_csv(output_dir / "v8a_multiclass_context_shuffled_label_null.csv", index=False)
    confusion.to_csv(output_dir / "v8a_multiclass_context_confusion.csv", index=False)

    ordinary_val = top_main(summary, "ordinary", "validation")
    ordinary_stress = top_main(summary, "ordinary", "stress_holdout")
    balanced_val = top_main(summary, "count_balanced", "validation")
    balanced_stress = top_main(summary, "count_balanced", "stress_holdout")

    def value(row: pd.Series, key: str) -> float:
        return float(row.get(key, 0.0)) if not row.empty else 0.0

    total_count_val = summary[
        summary["track"].astype(str).eq("ordinary")
        & summary["method"].astype(str).eq("ExtraTreesTotalCountOnly")
        & summary["eval_split"].astype(str).eq("validation")
    ]
    lineage_val = summary[
        summary["track"].astype(str).eq("ordinary")
        & summary["method"].astype(str).eq("ExtraTreesLineageOnly")
        & summary["eval_split"].astype(str).eq("validation")
    ]
    total_count_hm = float(total_count_val["hm_min_recall"].max()) if not total_count_val.empty else 0.0
    lineage_hm = float(lineage_val["hm_min_recall"].max()) if not lineage_val.empty else 0.0
    null_validation = null_summary[
        null_summary["track"].astype(str).eq("ordinary")
        & null_summary["eval_split"].astype(str).eq("validation")
    ]
    null_hm_p95 = float(null_validation["hm_min_recall"].quantile(0.95)) if not null_validation.empty else 0.0
    null_macro_p95 = float(null_validation["macro_f1"].quantile(0.95)) if not null_validation.empty else 0.0
    real_minus_null = value(ordinary_val, "hm_min_recall") - null_hm_p95

    stop_reasons: list[str] = []
    warnings: list[str] = []
    if not bool(schema_gate.get("gate_passed", False)):
        stop_reasons.append("event_schema_gate_not_passed")
    if not bool(final_audit_gate.get("gate_passed", True)):
        stop_reasons.append("final_audit_gate_not_passed")
    if len(labels) != 10:
        stop_reasons.append(f"expected_10_materials_observed_{len(labels)}")
    if value(ordinary_val, "hm_min_recall") < THRESHOLDS["validation_hm_min_recall_min"]:
        stop_reasons.append("ordinary_validation_hm_min_recall_below_threshold")
    if value(ordinary_stress, "hm_min_recall") < THRESHOLDS["stress_holdout_hm_min_recall_min"]:
        stop_reasons.append("ordinary_stress_holdout_hm_min_recall_below_threshold")
    if value(ordinary_val, "macro_f1") < THRESHOLDS["validation_macro_f1_min"]:
        stop_reasons.append("ordinary_validation_macro_f1_below_threshold")
    if value(ordinary_stress, "macro_f1") < THRESHOLDS["stress_holdout_macro_f1_min"]:
        stop_reasons.append("ordinary_stress_holdout_macro_f1_below_threshold")
    if value(ordinary_val, "min_class_recall") < THRESHOLDS["validation_min_class_recall_min"]:
        stop_reasons.append("ordinary_validation_min_class_recall_below_threshold")
    if value(ordinary_stress, "min_class_recall") < THRESHOLDS["stress_holdout_min_class_recall_min"]:
        stop_reasons.append("ordinary_stress_holdout_min_class_recall_below_threshold")
    if total_count_hm > THRESHOLDS["total_count_only_hm_min_recall_max"]:
        stop_reasons.append("total_count_only_hm_min_recall_above_shortcut_threshold")
    if lineage_hm > THRESHOLDS["lineage_only_hm_min_recall_max"]:
        stop_reasons.append("lineage_only_hm_min_recall_above_shortcut_threshold")
    if null_hm_p95 > THRESHOLDS["shuffled_label_hm_p95_max"]:
        stop_reasons.append("shuffled_label_hm_p95_above_null_threshold")
    if null_macro_p95 > THRESHOLDS["shuffled_label_macro_f1_p95_max"]:
        stop_reasons.append("shuffled_label_macro_f1_p95_above_null_threshold")
    if real_minus_null < THRESHOLDS["real_minus_null_hm_margin_min"]:
        stop_reasons.append("real_minus_null_hm_margin_too_small")
    selected_support = support_summary[support_summary["selected"].astype(bool)]
    if selected_support.empty or not bool(selected_support["support_pass"].iloc[0]):
        stop_reasons.append("count_balanced_support_below_threshold")
    if value(balanced_val, "hm_min_recall") <= 0.0 or value(balanced_stress, "hm_min_recall") <= 0.0:
        warnings.append("count_balanced_track_not_evaluable_or_zero_hm_recall")

    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/train_v8a_multiclass_context_model.py",
        "generated_at_utc": generated_at,
        "development_only": True,
        "shadow_or_final_used": bool(manifest.get("shadow_or_final_used", False)),
        "reads_existing_xrt_cubes": False,
        "training_unlocked_by_input_audits": bool(schema_gate.get("gate_passed", False)) and bool(final_audit_gate.get("gate_passed", True)),
        "claim_scope": CLAIM_SCOPE,
        "gate_passed": gate_passed,
        "decision": "multiclass_context_development_gate_passed_not_promoted" if gate_passed else "stop_multiclass_context_development_model",
        "input_dir": args.input_dir,
        "sample_count": int(len(frame)),
        "materials": labels,
        "main_feature_count": int(len(main_cols)),
        "total_count_feature_count": int(len(total_count_cols)),
        "lineage_feature_count": int(len(lineage_cols)),
        "selected_count_balance_strategy": selected_strategy,
        "ordinary_validation_best_method": str(ordinary_val.get("method", "")) if not ordinary_val.empty else "",
        "ordinary_stress_holdout_best_method": str(ordinary_stress.get("method", "")) if not ordinary_stress.empty else "",
        "ordinary_validation_hm_min_recall": value(ordinary_val, "hm_min_recall"),
        "ordinary_stress_holdout_hm_min_recall": value(ordinary_stress, "hm_min_recall"),
        "ordinary_validation_macro_f1": value(ordinary_val, "macro_f1"),
        "ordinary_stress_holdout_macro_f1": value(ordinary_stress, "macro_f1"),
        "ordinary_validation_min_class_recall": value(ordinary_val, "min_class_recall"),
        "ordinary_stress_holdout_min_class_recall": value(ordinary_stress, "min_class_recall"),
        "count_balanced_validation_hm_min_recall": value(balanced_val, "hm_min_recall"),
        "count_balanced_stress_holdout_hm_min_recall": value(balanced_stress, "hm_min_recall"),
        "total_count_only_validation_hm_min_recall": total_count_hm,
        "lineage_only_validation_hm_min_recall": lineage_hm,
        "shuffled_label_validation_hm_p95": null_hm_p95,
        "shuffled_label_validation_macro_f1_p95": null_macro_p95,
        "real_minus_null_validation_hm_margin": real_minus_null,
        "thresholds": THRESHOLDS,
        "stop_reasons": stop_reasons,
        "warnings": warnings,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    write_json(output_dir / "v8a_multiclass_context_model_gate.json", json_clean(gate))
    write_report(output_dir, gate, summary, support_summary)
    print(
        "decision={decision} gate_passed={passed} ordinary_val_hm={val:.4f} ordinary_stress_hm={stress:.4f} "
        "total_count_hm={count:.4f} null_hm_p95={null:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            val=gate["ordinary_validation_hm_min_recall"],
            stress=gate["ordinary_stress_holdout_hm_min_recall"],
            count=gate["total_count_only_validation_hm_min_recall"],
            null=gate["shuffled_label_validation_hm_p95"],
        )
    )


if __name__ == "__main__":
    main()
