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

from train_v8a_event_feature_smoke import feature_sets, load_json, pair_recalls
from train_v8a_medium_development_model import expected_calibration_error


CLAIM_SCOPE = (
    "development-only ordinary and count-balanced baseline model diagnostics for the admitted v8A H/M full-cell "
    "sidecar view; not product accuracy, hardware validation, shadow/final validation, full ten-material evidence, "
    "or manuscript-grade powder XRD"
)

DEFAULT_INPUT_DIR = (
    "results/accuracy_v3/"
    "v8a_fullcell_residualization_protocol_rework_v1_source_scaled_no_residualization_event_to_feature"
)
DEFAULT_FINAL_AUDIT_DIR = "results/accuracy_v3/v8a_fullcell_training_data_final_audit"
DEFAULT_OUTPUT_DIR = "results/accuracy_v3/v8a_fullcell_clean_development_model"

HM_PAIR = ("Hematite", "Magnetite")
SPLITS = ("train", "validation", "stress_holdout")
MAIN_METHODS = ("LogisticEventMain", "ExtraTreesEventMain")
TRACKS = ("ordinary", "count_balanced")

THRESHOLDS = {
    "validation_main_hm_min_recall_min": 0.95,
    "stress_holdout_main_hm_min_recall_min": 0.90,
    "worst_thickness_hm_min_recall_min": 0.90,
    "worst_pose_hm_min_recall_min": 0.90,
    "worst_seed_block_hm_min_recall_min": 0.90,
    "worst_count_bin_hm_min_recall_min": 0.90,
    "validation_ece_max": 0.20,
    "stress_holdout_ece_max": 0.25,
    "total_count_only_hm_min_recall_max": 0.60,
    "lineage_only_hm_min_recall_max": 0.60,
    "shuffled_label_null_p95_max": 0.55,
    "shuffled_label_single_seed_max": 0.65,
    "real_minus_null_margin_min": 0.35,
    "count_balanced_train_pairs_min": 300,
    "count_balanced_validation_pairs_min": 200,
    "count_balanced_stress_holdout_pairs_min": 200,
}

COUNT_BALANCE_STRATEGIES = [
    {"strategy": "fixed_bin_width_0p003", "kind": "fixed_bin", "bin_width": 0.003},
    {"strategy": "fixed_bin_width_0p005", "kind": "fixed_bin", "bin_width": 0.005},
]
DEFAULT_STRATEGY = "fixed_bin_width_0p003"
WEAKER_FALLBACK_STRATEGY = "fixed_bin_width_0p005"
SHUFFLE_SEEDS = list(range(41001, 41031))


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
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Missing scikit-learn. Run with the project venv, for example "
            "`/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python analysis/train_v8a_fullcell_clean_development_model.py`."
        ) from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
        "roc_auc_score": roc_auc_score,
    }


def build_models(sk: dict[str, Any], main_cols: list[str], total_count_cols: list[str], lineage_cols: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "method": "LogisticEventMain",
            "family": "main",
            "estimator": sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=3000, class_weight="balanced", random_state=42001),
            ),
            "feature_cols": main_cols,
            "shuffle_train_labels": False,
        },
        {
            "method": "ExtraTreesEventMain",
            "family": "main",
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=500,
                random_state=42002,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
            "feature_cols": main_cols,
            "shuffle_train_labels": False,
        },
        {
            "method": "ExtraTreesTotalCountOnly",
            "family": "control",
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=300,
                random_state=42003,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
            "feature_cols": total_count_cols,
            "shuffle_train_labels": False,
        },
        {
            "method": "ExtraTreesLineageOnly",
            "family": "control",
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=300,
                random_state=42004,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
            "feature_cols": lineage_cols,
            "shuffle_train_labels": False,
        },
    ]


def add_lineage_controls(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    cols: list[str] = []
    numeric_controls = ["thickness_mm", "pose_index", "count_target_photons", "seed_block_seed", "random_seed"]
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


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame[frame["split"].astype(str).eq(split)].copy()


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = pair_recalls(y_true, y_pred)
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "hematite_recall": recalls["Hematite"],
        "magnetite_recall": recalls["Magnetite"],
        "hm_min_recall": float(min(recalls.values())),
    }


def threshold_sweep(sample_frame: pd.DataFrame, probabilities: np.ndarray, method: str, eval_split: str, track: str) -> pd.DataFrame:
    y_true = sample_frame["material"].astype(str).to_numpy()
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
        predictions = np.where(probabilities >= threshold, "Magnetite", "Hematite")
        rows.append(
            {
                "track": track,
                "method": method,
                "eval_split": eval_split,
                "threshold": float(threshold),
                "samples": int(len(sample_frame)),
                **metrics_from_predictions(y_true, predictions),
            }
        )
    return pd.DataFrame(rows)


def magnetite_probability(estimator: Any, x: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(x)
    classes = [str(item) for item in estimator.classes_]
    if "Magnetite" not in classes:
        return np.zeros(x.shape[0], dtype=np.float64)
    return probabilities[:, classes.index("Magnetite")].astype(np.float64)


def group_recall_rows(decisions: pd.DataFrame, method: str, eval_split: str, track: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, col in [
        ("thickness_mm", "thickness_mm"),
        ("pose_index", "pose_index"),
        ("seed_block", "seed_block"),
        ("count_target_bin", "count_target_bin"),
    ]:
        if col not in decisions.columns:
            continue
        for value, group in decisions.groupby(col, sort=True):
            metrics = metrics_from_predictions(group["material"].astype(str).to_numpy(), group["prediction"].astype(str).to_numpy())
            rows.append(
                {
                    "track": track,
                    "method": method,
                    "eval_split": eval_split,
                    "group": group_name,
                    "value": value,
                    "samples": int(len(group)),
                    **metrics,
                }
            )
    return rows


def fit_estimator(model: dict[str, Any], train: pd.DataFrame, shuffle_seed: int | None = None) -> Any | None:
    cols = list(model["feature_cols"])
    if not cols or train.empty:
        return None
    x_train = train[cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_train = train["material"].astype(str).to_numpy()
    if len(set(y_train)) < 2:
        return None
    if shuffle_seed is not None:
        y_train = np.random.default_rng(shuffle_seed).permutation(y_train)
    estimator = deepcopy(model["estimator"])
    estimator.fit(x_train, y_train)
    return estimator


def evaluate_estimator(
    *,
    track: str,
    model: dict[str, Any],
    estimator: Any | None,
    eval_frame: pd.DataFrame,
    eval_split: str,
    selected_threshold: float | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method = str(model["method"])
    cols = list(model["feature_cols"])
    if estimator is None or eval_frame.empty or not cols:
        empty = pd.DataFrame()
        return (
            {
                "track": track,
                "method": method,
                "family": model["family"],
                "eval_split": eval_split,
                "status": "not_evaluable",
                "samples": int(len(eval_frame)),
                "feature_count": int(len(cols)),
                "threshold": float(selected_threshold) if selected_threshold is not None else 0.5,
                "hm_min_recall": 0.0,
                "worst_thickness_hm_min_recall": 0.0,
                "worst_pose_hm_min_recall": 0.0,
                "worst_seed_block_hm_min_recall": 0.0,
                "worst_count_bin_hm_min_recall": 0.0,
                "expected_calibration_error": 1.0,
            },
            empty,
            empty,
            empty,
            empty,
        )
    x_eval = eval_frame[cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_true = eval_frame["material"].astype(str).to_numpy()
    prob_magnetite = magnetite_probability(estimator, x_eval)
    sweep = threshold_sweep(eval_frame, prob_magnetite, method, eval_split, track)
    if selected_threshold is None:
        ranked = sweep.assign(threshold_distance_to_0p5=(sweep["threshold"] - 0.5).abs())
        selected = ranked.sort_values(["hm_min_recall", "accuracy", "threshold_distance_to_0p5"], ascending=[False, False, True]).iloc[0]
        threshold = float(selected["threshold"])
    else:
        threshold = float(selected_threshold)
    prediction = np.where(prob_magnetite >= threshold, "Magnetite", "Hematite").astype(str)
    y_binary = (y_true == "Magnetite").astype(int)
    ece, calibration = expected_calibration_error(y_binary, prob_magnetite)
    if not calibration.empty:
        calibration.insert(0, "eval_split", eval_split)
        calibration.insert(0, "method", method)
        calibration.insert(0, "track", track)

    decision_cols = [
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
            "clean_match_pair_id",
            "count_balance_pair_id",
            "count_balance_bin",
        ]
        if col in eval_frame.columns
    ]
    decisions = eval_frame[decision_cols].copy()
    decisions["track"] = track
    decisions["method"] = method
    decisions["threshold"] = threshold
    decisions["probability_magnetite"] = prob_magnetite
    decisions["prediction"] = prediction
    decisions["is_correct"] = decisions["material"].astype(str).to_numpy() == prediction
    grouped = pd.DataFrame(group_recall_rows(decisions, method, eval_split, track))
    metrics = metrics_from_predictions(y_true, prediction)

    def worst_group(group_name: str) -> float:
        values = grouped.loc[grouped["group"].eq(group_name), "hm_min_recall"] if not grouped.empty else pd.Series(dtype=float)
        return float(values.min()) if not values.empty else 0.0

    summary = {
        "track": track,
        "method": method,
        "family": model["family"],
        "eval_split": eval_split,
        "status": "evaluated",
        "samples": int(len(eval_frame)),
        "feature_count": int(len(cols)),
        "threshold": threshold,
        **metrics,
        "worst_thickness_hm_min_recall": worst_group("thickness_mm"),
        "worst_pose_hm_min_recall": worst_group("pose_index"),
        "worst_seed_block_hm_min_recall": worst_group("seed_block"),
        "worst_count_bin_hm_min_recall": worst_group("count_target_bin"),
        "expected_calibration_error": float(ece),
    }
    return summary, decisions, grouped, sweep, calibration


def add_count_bins(frame: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    width = float(strategy["bin_width"])
    result["count_balance_bin"] = np.floor(result["control_total_count_norm"].astype(float) / width).astype(int).astype(str)
    return result


def build_count_balanced_subset(frame: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
    source_on = add_count_bins(frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy(), strategy)
    rows: list[pd.Series] = []
    group_cols = ["split", "source_family", "stress_label", "thickness_mm", "pose_index", "seed_block", "count_target_bin", "count_balance_bin"]
    for keys, group in source_on.groupby(group_cols, sort=True, observed=True):
        hematite = group[group["material"].astype(str).eq("Hematite")].sort_values("control_total_count_norm")
        magnetite = group[group["material"].astype(str).eq("Magnetite")].sort_values("control_total_count_norm")
        pair_count = min(len(hematite), len(magnetite))
        if pair_count <= 0:
            continue
        for pair_index, (_, h_row) in enumerate(hematite.head(pair_count).iterrows(), start=1):
            row = h_row.copy()
            row["count_balance_pair_id"] = f"{strategy['strategy']}|{keys}|pair{pair_index:03d}"
            rows.append(row)
        for pair_index, (_, m_row) in enumerate(magnetite.head(pair_count).iterrows(), start=1):
            row = m_row.copy()
            row["count_balance_pair_id"] = f"{strategy['strategy']}|{keys}|pair{pair_index:03d}"
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(source_on.columns) + ["count_balance_pair_id"])
    return pd.DataFrame(rows).reset_index(drop=True)


def pair_counts(frame: pd.DataFrame, pair_col: str) -> dict[str, int]:
    return {
        split: int(frame[frame["split"].astype(str).eq(split)][pair_col].nunique()) if pair_col in frame.columns else 0
        for split in SPLITS
    }


def standardized_count_gap(frame: pd.DataFrame, split: str) -> float:
    subset = split_frame(frame, split)
    h = subset[subset["material"].astype(str).eq("Hematite")]["control_total_count_norm"].to_numpy(dtype=np.float64)
    m = subset[subset["material"].astype(str).eq("Magnetite")]["control_total_count_norm"].to_numpy(dtype=np.float64)
    if len(h) == 0 or len(m) == 0:
        return 0.0
    pooled = np.sqrt(0.5 * (np.var(h) + np.var(m)) + 1e-12)
    return float(abs(float(np.mean(m) - np.mean(h))) / pooled)


def support_pass(counts: dict[str, int]) -> bool:
    return bool(
        counts.get("train", 0) >= THRESHOLDS["count_balanced_train_pairs_min"]
        and counts.get("validation", 0) >= THRESHOLDS["count_balanced_validation_pairs_min"]
        and counts.get("stress_holdout", 0) >= THRESHOLDS["count_balanced_stress_holdout_pairs_min"]
    )


def build_training_tracks(frame: pd.DataFrame, strategy_name: str) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], str, bool]:
    strategy_map = {item["strategy"]: item for item in COUNT_BALANCE_STRATEGIES}
    requested = strategy_map[strategy_name]
    ordinary = frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    summaries: list[dict[str, Any]] = []
    selected_name = strategy_name
    selected_balanced = build_count_balanced_subset(ordinary, requested)
    selected_counts = pair_counts(selected_balanced, "count_balance_pair_id")
    summaries.append(
        {
            "strategy": strategy_name,
            "selected": True,
            "weaker_fallback": False,
            **{f"{split}_pairs": selected_counts[split] for split in SPLITS},
            **{f"{split}_count_gap_standardized": standardized_count_gap(selected_balanced, split) for split in SPLITS},
            "support_pass": support_pass(selected_counts),
        }
    )
    used_weaker_fallback = False
    if not support_pass(selected_counts) and strategy_name == DEFAULT_STRATEGY:
        fallback = strategy_map[WEAKER_FALLBACK_STRATEGY]
        fallback_balanced = build_count_balanced_subset(ordinary, fallback)
        fallback_counts = pair_counts(fallback_balanced, "count_balance_pair_id")
        summaries.append(
            {
                "strategy": WEAKER_FALLBACK_STRATEGY,
                "selected": False,
                "weaker_fallback": True,
                **{f"{split}_pairs": fallback_counts[split] for split in SPLITS},
                **{f"{split}_count_gap_standardized": standardized_count_gap(fallback_balanced, split) for split in SPLITS},
                "support_pass": support_pass(fallback_counts),
            }
        )
        if support_pass(fallback_counts):
            selected_name = WEAKER_FALLBACK_STRATEGY
            selected_balanced = fallback_balanced
            selected_counts = fallback_counts
            used_weaker_fallback = True
            summaries[-1]["selected"] = True
            summaries[0]["selected"] = False
    return {"ordinary": ordinary, "count_balanced": selected_balanced}, summaries, selected_name, used_weaker_fallback


def run_track(track: str, frame: pd.DataFrame, models: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = split_frame(frame, "train")
    summary_rows: list[dict[str, Any]] = []
    decisions: list[pd.DataFrame] = []
    group_rows: list[pd.DataFrame] = []
    sweep_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    fitted: dict[str, Any | None] = {}
    thresholds: dict[str, float] = {}
    for model in models:
        estimator = fit_estimator(model, train)
        fitted[str(model["method"])] = estimator
        validation = split_frame(frame, "validation")
        summary, method_decisions, grouped, sweep, calibration = evaluate_estimator(
            track=track,
            model=model,
            estimator=estimator,
            eval_frame=validation,
            eval_split="validation",
            selected_threshold=None,
        )
        thresholds[str(model["method"])] = float(summary["threshold"])
        summary_rows.append(summary)
        for table, sink in [(method_decisions, decisions), (grouped, group_rows), (sweep, sweep_rows), (calibration, calibration_rows)]:
            if not table.empty:
                sink.append(table)
        holdout = split_frame(frame, "stress_holdout")
        holdout_summary, holdout_decisions, holdout_grouped, holdout_sweep, holdout_calibration = evaluate_estimator(
            track=track,
            model=model,
            estimator=estimator,
            eval_frame=holdout,
            eval_split="stress_holdout",
            selected_threshold=thresholds[str(model["method"])],
        )
        summary_rows.append(holdout_summary)
        for table, sink in [
            (holdout_decisions, decisions),
            (holdout_grouped, group_rows),
            (holdout_sweep, sweep_rows),
            (holdout_calibration, calibration_rows),
        ]:
            if not table.empty:
                sink.append(table)

    null_rows: list[dict[str, Any]] = []
    main_model = next(model for model in models if model["method"] == "ExtraTreesEventMain")
    validation = split_frame(frame, "validation")
    holdout = split_frame(frame, "stress_holdout")
    for seed in SHUFFLE_SEEDS:
        estimator = fit_estimator(main_model, train, shuffle_seed=seed)
        validation_summary, _, _, _, _ = evaluate_estimator(
            track=track,
            model={**main_model, "method": "ExtraTreesShuffledTrainLabels", "family": "null_control"},
            estimator=estimator,
            eval_frame=validation,
            eval_split="validation",
            selected_threshold=None,
        )
        holdout_summary, _, _, _, _ = evaluate_estimator(
            track=track,
            model={**main_model, "method": "ExtraTreesShuffledTrainLabels", "family": "null_control"},
            estimator=estimator,
            eval_frame=holdout,
            eval_split="stress_holdout",
            selected_threshold=float(validation_summary["threshold"]),
        )
        null_rows.append(
            {
                "track": track,
                "shuffle_seed": seed,
                "validation_hm_min_recall": float(validation_summary["hm_min_recall"]),
                "stress_holdout_hm_min_recall": float(holdout_summary["hm_min_recall"]),
                "validation_threshold": float(validation_summary["threshold"]),
            }
        )

    return (
        pd.DataFrame(summary_rows),
        pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame(),
        pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame(),
        pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame(),
        pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame(),
        pd.DataFrame(null_rows),
    )


def value_for(summary: pd.DataFrame, track: str, method: str, split: str, field: str) -> float:
    values = summary.loc[
        summary["track"].eq(track) & summary["method"].eq(method) & summary["eval_split"].eq(split),
        field,
    ]
    return float(values.iloc[0]) if not values.empty else 0.0


def selected_main_for_track(summary: pd.DataFrame, track: str) -> dict[str, Any]:
    candidates = summary[summary["track"].eq(track) & summary["method"].isin(MAIN_METHODS) & summary["eval_split"].eq("validation")]
    if candidates.empty:
        return {"method": "", "threshold": 0.5, "feature_count": 0}
    return candidates.sort_values(
        [
            "hm_min_recall",
            "worst_thickness_hm_min_recall",
            "worst_pose_hm_min_recall",
            "worst_seed_block_hm_min_recall",
            "worst_count_bin_hm_min_recall",
            "accuracy",
        ],
        ascending=False,
    ).iloc[0].to_dict()


def track_gate(track: str, summary: pd.DataFrame, nulls: pd.DataFrame) -> dict[str, Any]:
    selected = selected_main_for_track(summary, track)
    method = str(selected["method"])
    validation_main = value_for(summary, track, method, "validation", "hm_min_recall")
    stress_main = value_for(summary, track, method, "stress_holdout", "hm_min_recall")
    worst_thickness = min(
        value_for(summary, track, method, "validation", "worst_thickness_hm_min_recall"),
        value_for(summary, track, method, "stress_holdout", "worst_thickness_hm_min_recall"),
    )
    worst_pose = min(
        value_for(summary, track, method, "validation", "worst_pose_hm_min_recall"),
        value_for(summary, track, method, "stress_holdout", "worst_pose_hm_min_recall"),
    )
    worst_seed = min(
        value_for(summary, track, method, "validation", "worst_seed_block_hm_min_recall"),
        value_for(summary, track, method, "stress_holdout", "worst_seed_block_hm_min_recall"),
    )
    worst_count_bin = min(
        value_for(summary, track, method, "validation", "worst_count_bin_hm_min_recall"),
        value_for(summary, track, method, "stress_holdout", "worst_count_bin_hm_min_recall"),
    )
    validation_ece = value_for(summary, track, method, "validation", "expected_calibration_error")
    stress_ece = value_for(summary, track, method, "stress_holdout", "expected_calibration_error")
    total_count_hm = max(
        value_for(summary, track, "ExtraTreesTotalCountOnly", "validation", "hm_min_recall"),
        value_for(summary, track, "ExtraTreesTotalCountOnly", "stress_holdout", "hm_min_recall"),
    )
    lineage_hm = max(
        value_for(summary, track, "ExtraTreesLineageOnly", "validation", "hm_min_recall"),
        value_for(summary, track, "ExtraTreesLineageOnly", "stress_holdout", "hm_min_recall"),
    )
    track_null = nulls[nulls["track"].eq(track)].copy()
    null_values = pd.concat(
        [track_null["validation_hm_min_recall"], track_null["stress_holdout_hm_min_recall"]],
        ignore_index=True,
    ) if not track_null.empty else pd.Series([1.0])
    null_p95 = float(np.quantile(null_values.to_numpy(dtype=np.float64), 0.95))
    null_max = float(null_values.max())
    margin = validation_main - null_p95
    pass_items = {
        "selected_main_model_present": bool(method),
        "validation_main_hm_min_recall": validation_main >= THRESHOLDS["validation_main_hm_min_recall_min"],
        "stress_holdout_main_hm_min_recall": stress_main >= THRESHOLDS["stress_holdout_main_hm_min_recall_min"],
        "worst_thickness_hm_min_recall": worst_thickness >= THRESHOLDS["worst_thickness_hm_min_recall_min"],
        "worst_pose_hm_min_recall": worst_pose >= THRESHOLDS["worst_pose_hm_min_recall_min"],
        "worst_seed_block_hm_min_recall": worst_seed >= THRESHOLDS["worst_seed_block_hm_min_recall_min"],
        "worst_count_bin_hm_min_recall": worst_count_bin >= THRESHOLDS["worst_count_bin_hm_min_recall_min"],
        "validation_ece_below_ceiling": validation_ece <= THRESHOLDS["validation_ece_max"],
        "stress_holdout_ece_below_ceiling": stress_ece <= THRESHOLDS["stress_holdout_ece_max"],
        "total_count_only_below_ceiling": total_count_hm < THRESHOLDS["total_count_only_hm_min_recall_max"],
        "lineage_only_below_ceiling": lineage_hm < THRESHOLDS["lineage_only_hm_min_recall_max"],
        "shuffled_label_null_p95_below_ceiling": null_p95 < THRESHOLDS["shuffled_label_null_p95_max"],
        "shuffled_label_single_seed_below_ceiling": null_max < THRESHOLDS["shuffled_label_single_seed_max"],
        "real_minus_null_margin": margin >= THRESHOLDS["real_minus_null_margin_min"],
    }
    return {
        "track": track,
        "gate_passed": all(pass_items.values()),
        "selected_main_model": {"method": method, "threshold": float(selected.get("threshold", 0.5)), "feature_count": int(selected.get("feature_count", 0))},
        "validation_main_hm_min_recall": validation_main,
        "stress_holdout_main_hm_min_recall": stress_main,
        "worst_thickness_hm_min_recall": worst_thickness,
        "worst_pose_hm_min_recall": worst_pose,
        "worst_seed_block_hm_min_recall": worst_seed,
        "worst_count_bin_hm_min_recall": worst_count_bin,
        "validation_expected_calibration_error": validation_ece,
        "stress_holdout_expected_calibration_error": stress_ece,
        "total_count_only_hm_min_recall": total_count_hm,
        "lineage_only_hm_min_recall": lineage_hm,
        "shuffled_label_null_p95": null_p95,
        "shuffled_label_single_seed_max": null_max,
        "real_minus_null_margin": margin,
        "pass_items": pass_items,
        "stop_reasons": [name for name, passed in pass_items.items() if not passed],
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], selection: pd.DataFrame, count_summary: pd.DataFrame) -> None:
    lines = [
        "# v8A full-cell clean development model gate",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Ordinary gate passed: `{str(gate['ordinary_gate']['gate_passed']).lower()}`",
        f"- Count-balanced gate passed: `{str(gate['count_balanced_gate']['gate_passed']).lower()}`",
        f"- Count-balance strategy: `{gate['count_balance_strategy']}`",
        f"- Weaker fallback used: `{str(gate['weaker_count_balance_fallback_used']).lower()}`",
        "",
        "## Count-Balance Support",
        "",
        markdown_table(
            count_summary,
            [
                "strategy",
                "selected",
                "weaker_fallback",
                "train_pairs",
                "validation_pairs",
                "stress_holdout_pairs",
                "validation_count_gap_standardized",
                "stress_holdout_count_gap_standardized",
                "support_pass",
            ],
        ),
        "",
        "## Model Summary",
        "",
        markdown_table(
            selection.sort_values(["track", "eval_split", "family", "hm_min_recall"], ascending=[True, True, True, False]),
            [
                "track",
                "method",
                "eval_split",
                "family",
                "threshold",
                "hm_min_recall",
                "hematite_recall",
                "magnetite_recall",
                "worst_thickness_hm_min_recall",
                "worst_pose_hm_min_recall",
                "worst_seed_block_hm_min_recall",
                "worst_count_bin_hm_min_recall",
                "expected_calibration_error",
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
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is still development-only model diagnostics. Passing this gate would unlock stability replication, not shadow/final, not product metrics, and not a large development matrix by itself.",
            "",
        ]
    )
    (output_dir / "v8a_fullcell_clean_development_model_gate_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train development-only full-cell H/M baselines with ordinary and count-balanced gates.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--final-audit-dir", default=DEFAULT_FINAL_AUDIT_DIR)
    parser.add_argument("--count-balance-strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    final_audit_dir = as_project_path(project_root, args.final_audit_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    final_audit_gate = load_json(final_audit_dir / "v8a_fullcell_training_data_final_audit_gate.json")
    if not bool(final_audit_gate.get("training_unlocked", False)):
        raise RuntimeError(f"Final data audit did not unlock training: {final_audit_gate.get('decision')}")
    for name, payload in {"schema_gate": schema_gate, "feature_manifest": manifest, "final_audit_gate": final_audit_gate}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing full-cell training because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing full-cell training because {name} reports existing XRT cube reads.")

    raw_frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    frame, lineage_cols = add_lineage_controls(raw_frame)
    main_cols, control_cols, total_count_cols, overlap_cols, thickness_pose_cols = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features available.")
    leak_tokens = [
        "material",
        "source_id",
        "source_mode",
        "sample_id",
        "seed_block",
        "random_seed",
        "thickness_mm",
        "pose_index",
        "split",
        "origin",
        "path",
        "row_index",
        "label",
    ]
    lineage_like = [col for col in main_cols if any(token in col.lower() for token in leak_tokens)]
    if lineage_like:
        raise RuntimeError(f"Main feature columns look lineage-like: {lineage_like}")

    tracks, count_summaries, selected_strategy, fallback_used = build_training_tracks(frame, args.count_balance_strategy)
    count_summary = pd.DataFrame(count_summaries)
    selected_count_row = count_summary[count_summary["selected"].astype(bool)].iloc[0].to_dict()
    count_support_pass = bool(selected_count_row["support_pass"])
    sk = require_sklearn()
    models = build_models(sk, main_cols, total_count_cols, lineage_cols)

    selection_tables: list[pd.DataFrame] = []
    decision_tables: list[pd.DataFrame] = []
    group_tables: list[pd.DataFrame] = []
    sweep_tables: list[pd.DataFrame] = []
    calibration_tables: list[pd.DataFrame] = []
    null_tables: list[pd.DataFrame] = []
    for track, track_frame in tracks.items():
        selection, decisions, groups, sweeps, calibration, nulls = run_track(track, track_frame, models)
        selection_tables.append(selection)
        decision_tables.append(decisions)
        group_tables.append(groups)
        sweep_tables.append(sweeps)
        calibration_tables.append(calibration)
        null_tables.append(nulls)

    selection = pd.concat(selection_tables, ignore_index=True)
    decisions = pd.concat(decision_tables, ignore_index=True) if decision_tables else pd.DataFrame()
    group_recalls = pd.concat(group_tables, ignore_index=True) if group_tables else pd.DataFrame()
    threshold_sweeps = pd.concat(sweep_tables, ignore_index=True) if sweep_tables else pd.DataFrame()
    calibration_bins = pd.concat(calibration_tables, ignore_index=True) if calibration_tables else pd.DataFrame()
    null_distribution = pd.concat(null_tables, ignore_index=True) if null_tables else pd.DataFrame()

    ordinary_gate = track_gate("ordinary", selection, null_distribution)
    count_balanced_gate = track_gate("count_balanced", selection, null_distribution)
    global_pass_items = {
        "final_data_audit_training_unlocked": bool(final_audit_gate.get("training_unlocked", False)),
        "count_balanced_support_pass": count_support_pass,
        "ordinary_gate_passed": bool(ordinary_gate["gate_passed"]),
        "count_balanced_gate_passed": bool(count_balanced_gate["gate_passed"]),
        "no_lineage_like_main_features": not lineage_like,
        "development_only": True,
        "no_shadow_final": True,
    }
    stop_reasons = [name for name, passed in global_pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/train_v8a_fullcell_clean_development_model.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_fullcell_clean_development_model_gate",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "final_audit_dir": args.final_audit_dir,
        "gate_passed": gate_passed,
        "decision": "fullcell_clean_development_baseline_passed_ready_for_stability_replication"
        if gate_passed
        else "stop_or_rework_fullcell_clean_development_baseline",
        "count_balance_strategy": selected_strategy,
        "requested_count_balance_strategy": args.count_balance_strategy,
        "weaker_count_balance_fallback_used": bool(fallback_used),
        "count_balance_support": selected_count_row,
        "ordinary_gate": ordinary_gate,
        "count_balanced_gate": count_balanced_gate,
        "thresholds": THRESHOLDS,
        "pass_items": global_pass_items,
        "stop_reasons": stop_reasons,
        "feature_counts": {
            "main": int(len(main_cols)),
            "total_count": int(len(total_count_cols)),
            "lineage": int(len(lineage_cols)),
            "overlap": int(len(overlap_cols)),
            "thickness_pose": int(len(thickness_pose_cols)),
            "all_controls": int(len(control_cols)),
        },
        "input_gate_decisions": {
            "schema_gate": schema_gate.get("decision"),
            "feature_manifest_candidate_view": manifest.get("candidate_view"),
            "final_data_audit": final_audit_gate.get("decision"),
        },
        "next_allowed_stage": "stability_replication" if gate_passed else "rework_or_stop_loss_review",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest_out = {
        "generated_by": gate["generated_by"],
        "generated_at_utc": generated_at,
        "protocol_name": gate["protocol_name"],
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "gate_file": "v8a_fullcell_clean_development_model_gate.json",
        "count_balance_strategy": selected_strategy,
        "ordinary_sample_count": int(len(tracks["ordinary"])),
        "count_balanced_sample_count": int(len(tracks["count_balanced"])),
        "main_feature_columns": main_cols,
        "total_count_control_columns": total_count_cols,
        "lineage_control_columns": lineage_cols,
    }

    tracks["count_balanced"].to_csv(output_dir / "v8a_fullcell_count_balanced_training_view.csv", index=False, lineterminator="\n")
    count_summary.to_csv(output_dir / "v8a_fullcell_count_balance_support.csv", index=False, lineterminator="\n")
    selection.to_csv(output_dir / "v8a_fullcell_clean_development_model_selection.csv", index=False, lineterminator="\n")
    decisions.to_csv(output_dir / "v8a_fullcell_clean_development_model_decisions.csv", index=False, lineterminator="\n")
    group_recalls.to_csv(output_dir / "v8a_fullcell_clean_development_group_recalls.csv", index=False, lineterminator="\n")
    threshold_sweeps.to_csv(output_dir / "v8a_fullcell_clean_development_threshold_sweeps.csv", index=False, lineterminator="\n")
    calibration_bins.to_csv(output_dir / "v8a_fullcell_clean_development_calibration_bins.csv", index=False, lineterminator="\n")
    null_distribution.to_csv(output_dir / "v8a_fullcell_clean_development_shuffled_null_distribution.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_fullcell_clean_development_model_manifest.json", json_clean(manifest_out))
    write_json(output_dir / "v8a_fullcell_clean_development_model_gate.json", json_clean(gate))
    write_report(output_dir, json_clean(gate), selection, count_summary)
    print(
        "decision={decision} gate_passed={passed} ordinary={ordinary} count_balanced={balanced} strategy={strategy}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            ordinary=str(ordinary_gate["gate_passed"]).lower(),
            balanced=str(count_balanced_gate["gate_passed"]).lower(),
            strategy=selected_strategy,
        )
    )


if __name__ == "__main__":
    main()
