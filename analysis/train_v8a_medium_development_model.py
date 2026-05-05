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

from train_v8a_event_feature_smoke import (
    feature_sets,
    load_json,
    pair_recalls,
)


HM_PAIR = ("Hematite", "Magnetite")
MAIN_METHODS = {"LogisticEventMain", "ExtraTreesEventMain"}
CONTROL_METHODS = {
    "ExtraTreesTotalCountOnly",
    "ExtraTreesOverlapOnly",
    "ExtraTreesThicknessPoseOnly",
    "ExtraTreesShuffledTrainLabels",
    "ExtraTreesSourceOffLeakage",
}
THRESHOLDS = {
    "validation_main_hm_min_recall_min": 0.95,
    "stress_holdout_main_hm_min_recall_min": 0.95,
    "worst_thickness_hm_min_recall_min": 0.90,
    "worst_pose_hm_min_recall_min": 0.90,
    "worst_stress_label_hm_min_recall_min": 0.90,
    "total_count_only_hm_min_recall_max": 0.60,
    "overlap_only_hm_min_recall_max": 0.60,
    "thickness_pose_hm_min_recall_max": 0.60,
    "shuffled_label_hm_min_recall_max": 0.55,
    "source_off_hm_min_recall_max": 0.60,
    "main_minus_source_off_hm_margin_min": 0.35,
    "validation_ece_max": 0.20,
    "stress_holdout_ece_max": 0.25,
}


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Missing scikit-learn. Run with the project venv, for example "
            "`/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python analysis/train_v8a_medium_development_model.py`."
        ) from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
        "brier_score_loss": brier_score_loss,
        "log_loss": log_loss,
    }


def build_models(
    sk: dict[str, Any],
    main_cols: list[str],
    total_count_cols: list[str],
    overlap_cols: list[str],
    thickness_pose_cols: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "method": "LogisticEventMain",
            "family": "main",
            "estimator": sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=3000, class_weight="balanced", random_state=9801),
            ),
            "feature_cols": main_cols,
            "train_source_mode": "custom_diffraction_on",
            "eval_source_mode": "custom_diffraction_on",
            "shuffle_train_labels": False,
        },
        {
            "method": "ExtraTreesEventMain",
            "family": "main",
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=500,
                random_state=9802,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
            "feature_cols": main_cols,
            "train_source_mode": "custom_diffraction_on",
            "eval_source_mode": "custom_diffraction_on",
            "shuffle_train_labels": False,
        },
        {
            "method": "ExtraTreesTotalCountOnly",
            "family": "control",
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=250,
                random_state=9803,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
            "feature_cols": total_count_cols,
            "train_source_mode": "custom_diffraction_on",
            "eval_source_mode": "custom_diffraction_on",
            "shuffle_train_labels": False,
        },
        {
            "method": "ExtraTreesOverlapOnly",
            "family": "control",
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=250,
                random_state=9804,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
            "feature_cols": overlap_cols,
            "train_source_mode": "custom_diffraction_on",
            "eval_source_mode": "custom_diffraction_on",
            "shuffle_train_labels": False,
        },
        {
            "method": "ExtraTreesThicknessPoseOnly",
            "family": "control",
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=250,
                random_state=9805,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
            "feature_cols": thickness_pose_cols,
            "train_source_mode": "custom_diffraction_on",
            "eval_source_mode": "custom_diffraction_on",
            "shuffle_train_labels": False,
        },
        {
            "method": "ExtraTreesShuffledTrainLabels",
            "family": "control",
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=250,
                random_state=9806,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
            "feature_cols": main_cols,
            "train_source_mode": "custom_diffraction_on",
            "eval_source_mode": "custom_diffraction_on",
            "shuffle_train_labels": True,
        },
        {
            "method": "ExtraTreesSourceOffLeakage",
            "family": "control",
            "estimator": sk["ExtraTreesClassifier"](
                n_estimators=250,
                random_state=9807,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
            "feature_cols": main_cols,
            "train_source_mode": "custom_diffraction_off",
            "eval_source_mode": "custom_diffraction_off",
            "shuffle_train_labels": False,
        },
    ]


def selected_frame(frame: pd.DataFrame, split: str, source_mode: str) -> pd.DataFrame:
    return frame[
        frame["split"].astype(str).eq(split)
        & frame["source_mode"].astype(str).eq(source_mode)
    ].copy()


def fit_estimator(model: dict[str, Any], train: pd.DataFrame) -> Any:
    cols = model["feature_cols"]
    if not cols:
        return None
    x_train = train[cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_train = train["material"].astype(str).to_numpy()
    if len(set(y_train)) < 2:
        raise RuntimeError(f"{model['method']} needs both H/M classes in train; got {sorted(set(y_train))}")
    if model["shuffle_train_labels"]:
        y_train = np.random.default_rng(9817).permutation(y_train)
    estimator = deepcopy(model["estimator"])
    estimator.fit(x_train, y_train)
    return estimator


def magnetite_probability(estimator: Any, x: np.ndarray, classes: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(x)
    class_list = [str(item) for item in classes]
    if "Magnetite" not in class_list:
        return np.zeros(x.shape[0], dtype=np.float64)
    return probabilities[:, class_list.index("Magnetite")].astype(np.float64)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = pair_recalls(y_true, y_pred)
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    return {
        "accuracy": accuracy,
        "hematite_recall": recalls["Hematite"],
        "magnetite_recall": recalls["Magnetite"],
        "hm_min_recall": float(min(recalls.values())),
    }


def group_recall_rows(decisions: pd.DataFrame, method: str, eval_split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_defs = [
        ("thickness_mm", "thickness_mm"),
        ("pose_index", "pose_index"),
        ("stress_label", "stress_label"),
    ]
    for group_name, col in group_defs:
        if col not in decisions.columns:
            continue
        for value, group in decisions.groupby(col, sort=True):
            metrics = metrics_from_predictions(
                group["material"].astype(str).to_numpy(),
                group["prediction"].astype(str).to_numpy(),
            )
            rows.append(
                {
                    "method": method,
                    "eval_split": eval_split,
                    "group": group_name,
                    "value": value,
                    "samples": int(len(group)),
                    **metrics,
                }
            )
    return rows


def expected_calibration_error(y_true_binary: np.ndarray, prob: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    rows = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if index == bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_low": float(lo),
                    "bin_high": float(hi),
                    "samples": 0,
                    "mean_probability": 0.0,
                    "empirical_fraction": 0.0,
                    "abs_gap": 0.0,
                }
            )
            continue
        mean_prob = float(prob[mask].mean())
        empirical = float(y_true_binary[mask].mean())
        gap = abs(mean_prob - empirical)
        ece += gap * count / max(len(prob), 1)
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "samples": count,
                "mean_probability": mean_prob,
                "empirical_fraction": empirical,
                "abs_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def threshold_sweep(
    sample_frame: pd.DataFrame,
    probabilities: np.ndarray,
    method: str,
    eval_split: str,
) -> pd.DataFrame:
    y_true = sample_frame["material"].astype(str).to_numpy()
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
        pred = np.where(probabilities >= threshold, "Magnetite", "Hematite")
        metrics = metrics_from_predictions(y_true, pred)
        rows.append(
            {
                "method": method,
                "eval_split": eval_split,
                "threshold": float(threshold),
                "samples": int(len(sample_frame)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_estimator(
    model: dict[str, Any],
    estimator: Any,
    eval_frame: pd.DataFrame,
    eval_split: str,
    sk: dict[str, Any],
    selected_threshold: float | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = model["feature_cols"]
    method = str(model["method"])
    if not cols or estimator is None or eval_frame.empty:
        empty = pd.DataFrame()
        return (
            {
                "method": method,
                "eval_split": eval_split,
                "status": "not_evaluable",
                "samples": int(len(eval_frame)),
                "feature_count": int(len(cols)),
                "hm_min_recall": 0.0,
                "worst_thickness_hm_min_recall": 0.0,
                "worst_pose_hm_min_recall": 0.0,
                "worst_stress_label_hm_min_recall": 0.0,
                "threshold": float(selected_threshold) if selected_threshold is not None else 0.5,
            },
            empty,
            empty,
            empty,
            empty,
        )
    x_eval = eval_frame[cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_true = eval_frame["material"].astype(str).to_numpy()
    if hasattr(estimator, "predict_proba"):
        prob_magnetite = magnetite_probability(estimator, x_eval, estimator.classes_)
        sweep = threshold_sweep(eval_frame, prob_magnetite, method, eval_split)
        if selected_threshold is None:
            ranked_sweep = sweep.assign(threshold_distance_to_0p5=(sweep["threshold"] - 0.5).abs())
            selected = ranked_sweep.sort_values(
                ["hm_min_recall", "accuracy", "threshold_distance_to_0p5"],
                ascending=[False, False, True],
            ).iloc[0]
            threshold = float(selected["threshold"])
        else:
            threshold = float(selected_threshold)
        prediction = np.where(prob_magnetite >= threshold, "Magnetite", "Hematite")
        y_binary = (y_true == "Magnetite").astype(int)
        try:
            brier = float(sk["brier_score_loss"](y_binary, prob_magnetite))
        except ValueError:
            brier = 1.0
        try:
            logloss = float(sk["log_loss"](y_binary, prob_magnetite, labels=[0, 1]))
        except ValueError:
            logloss = float("nan")
        ece, calibration_bins = expected_calibration_error(y_binary, prob_magnetite)
        calibration_bins.insert(0, "eval_split", eval_split)
        calibration_bins.insert(0, "method", method)
    else:
        prediction = np.asarray(estimator.predict(x_eval)).astype(str)
        prob_magnetite = (prediction == "Magnetite").astype(float)
        threshold = 0.5
        sweep = threshold_sweep(eval_frame, prob_magnetite, method, eval_split)
        brier = float("nan")
        logloss = float("nan")
        ece = float("nan")
        calibration_bins = pd.DataFrame()

    decisions = eval_frame[
        [
            "sample_id",
            "split",
            "material",
            "source_mode",
            "stress_label",
            "source_id",
            "random_seed",
            "thickness_mm",
            "pose_index",
        ]
    ].copy()
    decisions["method"] = method
    decisions["threshold"] = threshold
    decisions["probability_magnetite"] = prob_magnetite
    decisions["prediction"] = prediction
    decisions["is_correct"] = decisions["material"].astype(str).to_numpy() == prediction

    grouped_rows = group_recall_rows(decisions, method, eval_split)
    grouped = pd.DataFrame(grouped_rows)
    metrics = metrics_from_predictions(y_true, prediction)

    def worst_group(group_name: str) -> float:
        values = grouped.loc[grouped["group"].eq(group_name), "hm_min_recall"] if not grouped.empty else pd.Series(dtype=float)
        return float(values.min()) if not values.empty else 0.0

    summary = {
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
        "worst_stress_label_hm_min_recall": worst_group("stress_label"),
        "brier_score": brier,
        "log_loss": logloss,
        "expected_calibration_error": ece,
    }
    return summary, decisions, grouped, sweep, calibration_bins


def load_required_gates(project_root: Path, input_dir: Path, training_gate_dir: Path, stress_gate_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    feature_manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    training_gate = load_json(training_gate_dir / "v8a_event_training_gate.json")
    stress_gate = load_json(stress_gate_dir / "v8a_event_feature_stress_gate.json")
    if not bool(schema_gate.get("gate_passed", False)):
        raise RuntimeError(f"Event schema gate did not pass: {schema_gate.get('decision')}")
    if not bool(schema_gate.get("tiny_training_gate_allowed", False)):
        raise RuntimeError(f"Input event features are not training-allowed: {schema_gate.get('stop_reasons')}")
    if not bool(training_gate.get("gate_passed", False)):
        raise RuntimeError(f"Baseline training/control gate did not pass: {training_gate.get('decision')}")
    if not bool(stress_gate.get("gate_passed", False)):
        raise RuntimeError(f"Stress gate did not pass: {stress_gate.get('decision')}")
    for name, payload in {
        "schema_gate": schema_gate,
        "feature_manifest": feature_manifest,
        "training_gate": training_gate,
        "stress_gate": stress_gate,
    }.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing Phase 4 training because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing Phase 4 training because {name} reports existing XRT cube reads.")
    return schema_gate, feature_manifest, training_gate, stress_gate


def integrity_summary(frame: pd.DataFrame, main_cols: list[str]) -> dict[str, Any]:
    leak_tokens = ["material", "source", "sample", "seed", "thickness", "pose", "split", "path", "label"]
    lineage_like = [col for col in main_cols if any(token in col.lower() for token in leak_tokens)]
    split_values = sorted(str(item).lower() for item in frame["split"].dropna().unique())
    return {
        "sample_count": int(len(frame)),
        "material_counts": {str(k): int(v) for k, v in frame["material"].value_counts(dropna=False).to_dict().items()},
        "split_counts": {str(k): int(v) for k, v in frame["split"].value_counts(dropna=False).to_dict().items()},
        "source_mode_counts": {str(k): int(v) for k, v in frame["source_mode"].value_counts(dropna=False).to_dict().items()},
        "stress_label_counts": {str(k): int(v) for k, v in frame["stress_label"].value_counts(dropna=False).to_dict().items()},
        "thickness_counts": {str(k): int(v) for k, v in frame["thickness_mm"].value_counts(dropna=False).sort_index().to_dict().items()},
        "pose_counts": {str(k): int(v) for k, v in frame["pose_index"].value_counts(dropna=False).sort_index().to_dict().items()},
        "shadow_or_final_splits": [item for item in split_values if item in {"shadow", "final"}],
        "main_feature_count": int(len(main_cols)),
        "lineage_like_main_features": lineage_like,
    }


def value_for(summary: pd.DataFrame, method: str, split: str, field: str) -> float:
    values = summary.loc[
        summary["method"].eq(method) & summary["eval_split"].eq(split),
        field,
    ]
    return float(values.iloc[0]) if not values.empty else 0.0


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], selection: pd.DataFrame, group_recalls: pd.DataFrame) -> None:
    best = gate["selected_main_model"]
    lines = [
        "# v8A medium development model gate report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only H/M medium-matrix model training evidence. This is not product accuracy, not shadow/final validation, not hardware validation, and not manuscript-grade powder XRD evidence.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Selected main model: `{best['method']}`",
        f"- Selected validation threshold: `{best['threshold']:.2f}`",
        f"- Validation H/M min recall: `{gate['validation_main_hm_min_recall']:.4f}`",
        f"- Stress-holdout H/M min recall: `{gate['stress_holdout_main_hm_min_recall']:.4f}`",
        f"- Worst by thickness H/M min recall: `{gate['worst_thickness_hm_min_recall']:.4f}`",
        f"- Worst by pose H/M min recall: `{gate['worst_pose_hm_min_recall']:.4f}`",
        f"- Worst by stress label H/M min recall: `{gate['worst_stress_label_hm_min_recall']:.4f}`",
        f"- Validation ECE: `{gate['validation_expected_calibration_error']:.4f}`",
        f"- Stress-holdout ECE: `{gate['stress_holdout_expected_calibration_error']:.4f}`",
        "",
        "## Controls",
        "",
        f"- Total-count-only H/M min recall: `{gate['total_count_only_hm_min_recall']:.4f}`",
        f"- Overlap-only H/M min recall: `{gate['overlap_only_hm_min_recall']:.4f}`",
        f"- Thickness/pose-only H/M min recall: `{gate['thickness_pose_hm_min_recall']:.4f}`",
        f"- Shuffled-label H/M min recall: `{gate['shuffled_label_hm_min_recall']:.4f}`",
        f"- Source-off H/M min recall: `{gate['source_off_hm_min_recall']:.4f}`",
        f"- Main minus source-off margin: `{gate['main_minus_source_off_hm_margin']:.4f}`",
        "",
        "## Model Summary",
        "",
        markdown_table(
            selection.sort_values(["eval_split", "family", "hm_min_recall", "accuracy"], ascending=[True, True, False, False]),
            [
                "method",
                "eval_split",
                "family",
                "threshold",
                "hm_min_recall",
                "hematite_recall",
                "magnetite_recall",
                "worst_thickness_hm_min_recall",
                "worst_pose_hm_min_recall",
                "worst_stress_label_hm_min_recall",
                "expected_calibration_error",
            ],
        ),
        "",
        "## Worst Group Recalls",
        "",
        markdown_table(
            group_recalls.sort_values("hm_min_recall").head(12),
            ["method", "eval_split", "group", "value", "samples", "hm_min_recall", "hematite_recall", "magnetite_recall"],
        ),
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        for reason in gate["stop_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_medium_development_model_gate_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run development-only model training/calibration gate for the v8A medium H/M matrix.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_medium_event_to_feature")
    parser.add_argument("--training-gate-dir", default="results/accuracy_v3/v8a_medium_event_training")
    parser.add_argument("--stress-gate-dir", default="results/accuracy_v3/v8a_medium_event_feature_stress_gate")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_medium_development_model")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    training_gate_dir = project_root / args.training_gate_dir
    stress_gate_dir = project_root / args.stress_gate_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate, feature_manifest, training_gate, stress_gate = load_required_gates(project_root, input_dir, training_gate_dir, stress_gate_dir)
    sk = require_sklearn()
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, control_cols, total_count_cols, overlap_cols, thickness_pose_cols = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features are available.")

    integrity = integrity_summary(frame, main_cols)
    if integrity["shadow_or_final_splits"]:
        raise RuntimeError(f"Refusing Phase 4 training because shadow/final splits are present: {integrity['shadow_or_final_splits']}")
    if integrity["lineage_like_main_features"]:
        raise RuntimeError(f"Refusing Phase 4 training because main features look lineage-like: {integrity['lineage_like_main_features']}")

    models = build_models(sk, main_cols, total_count_cols, overlap_cols, thickness_pose_cols)
    fitted: dict[str, Any] = {}
    selected_thresholds: dict[str, float] = {}
    summary_rows: list[dict[str, Any]] = []
    decisions: list[pd.DataFrame] = []
    group_rows: list[pd.DataFrame] = []
    sweep_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []

    for model in models:
        train = selected_frame(frame, "train", model["train_source_mode"])
        estimator = fit_estimator(model, train)
        fitted[model["method"]] = estimator
        validation = selected_frame(frame, "validation", model["eval_source_mode"])
        summary, method_decisions, grouped, sweep, calibration = evaluate_estimator(
            model,
            estimator,
            validation,
            "validation",
            sk,
            selected_threshold=None,
        )
        selected_thresholds[model["method"]] = float(summary["threshold"])
        summary_rows.append(summary)
        for table, sink in [
            (method_decisions, decisions),
            (grouped, group_rows),
            (sweep, sweep_rows),
            (calibration, calibration_rows),
        ]:
            if not table.empty:
                sink.append(table)

        stress_holdout = selected_frame(frame, "stress_holdout", model["eval_source_mode"])
        holdout_summary, holdout_decisions, holdout_grouped, holdout_sweep, holdout_calibration = evaluate_estimator(
            model,
            estimator,
            stress_holdout,
            "stress_holdout",
            sk,
            selected_threshold=selected_thresholds[model["method"]],
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

    selection = pd.DataFrame(summary_rows)
    validation_main = selection[selection["method"].isin(MAIN_METHODS) & selection["eval_split"].eq("validation")]
    selected_main = validation_main.sort_values(
        ["hm_min_recall", "worst_thickness_hm_min_recall", "worst_pose_hm_min_recall", "worst_stress_label_hm_min_recall", "accuracy"],
        ascending=False,
    ).iloc[0].to_dict()
    selected_method = str(selected_main["method"])
    validation_main_hm = value_for(selection, selected_method, "validation", "hm_min_recall")
    stress_holdout_main_hm = value_for(selection, selected_method, "stress_holdout", "hm_min_recall")
    worst_thickness = min(
        value_for(selection, selected_method, "validation", "worst_thickness_hm_min_recall"),
        value_for(selection, selected_method, "stress_holdout", "worst_thickness_hm_min_recall"),
    )
    worst_pose = min(
        value_for(selection, selected_method, "validation", "worst_pose_hm_min_recall"),
        value_for(selection, selected_method, "stress_holdout", "worst_pose_hm_min_recall"),
    )
    worst_stress_label = min(
        value_for(selection, selected_method, "validation", "worst_stress_label_hm_min_recall"),
        value_for(selection, selected_method, "stress_holdout", "worst_stress_label_hm_min_recall"),
    )
    validation_ece = value_for(selection, selected_method, "validation", "expected_calibration_error")
    stress_holdout_ece = value_for(selection, selected_method, "stress_holdout", "expected_calibration_error")
    total_count_hm = value_for(selection, "ExtraTreesTotalCountOnly", "validation", "hm_min_recall")
    overlap_hm = value_for(selection, "ExtraTreesOverlapOnly", "validation", "hm_min_recall")
    thickness_pose_hm = value_for(selection, "ExtraTreesThicknessPoseOnly", "validation", "hm_min_recall")
    shuffled_hm = value_for(selection, "ExtraTreesShuffledTrainLabels", "validation", "hm_min_recall")
    source_off_hm = value_for(selection, "ExtraTreesSourceOffLeakage", "validation", "hm_min_recall")
    margin = validation_main_hm - source_off_hm

    pass_items = {
        "schema_gate_passed": bool(schema_gate.get("gate_passed")),
        "baseline_training_gate_passed": bool(training_gate.get("gate_passed")),
        "stress_gate_passed": bool(stress_gate.get("gate_passed")),
        "development_only_no_shadow_final": not integrity["shadow_or_final_splits"],
        "no_lineage_like_main_features": not integrity["lineage_like_main_features"],
        "validation_main_hm_min_recall": validation_main_hm >= THRESHOLDS["validation_main_hm_min_recall_min"],
        "stress_holdout_main_hm_min_recall": stress_holdout_main_hm >= THRESHOLDS["stress_holdout_main_hm_min_recall_min"],
        "worst_thickness_hm_min_recall": worst_thickness >= THRESHOLDS["worst_thickness_hm_min_recall_min"],
        "worst_pose_hm_min_recall": worst_pose >= THRESHOLDS["worst_pose_hm_min_recall_min"],
        "worst_stress_label_hm_min_recall": worst_stress_label >= THRESHOLDS["worst_stress_label_hm_min_recall_min"],
        "total_count_only_below_ceiling": total_count_hm < THRESHOLDS["total_count_only_hm_min_recall_max"],
        "overlap_only_below_ceiling": overlap_hm < THRESHOLDS["overlap_only_hm_min_recall_max"],
        "thickness_pose_below_ceiling": thickness_pose_hm < THRESHOLDS["thickness_pose_hm_min_recall_max"],
        "shuffled_label_below_ceiling": shuffled_hm < THRESHOLDS["shuffled_label_hm_min_recall_max"],
        "source_off_below_ceiling": source_off_hm < THRESHOLDS["source_off_hm_min_recall_max"],
        "main_minus_source_off_margin": margin >= THRESHOLDS["main_minus_source_off_hm_margin_min"],
        "validation_ece_below_ceiling": validation_ece <= THRESHOLDS["validation_ece_max"],
        "stress_holdout_ece_below_ceiling": stress_holdout_ece <= THRESHOLDS["stress_holdout_ece_max"],
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/train_v8a_medium_development_model.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v8A_medium_development_model_training_gate",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": "development-only H/M medium-matrix model training/calibration gate; not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD",
        "input_dir": args.input_dir,
        "training_gate_dir": args.training_gate_dir,
        "stress_gate_dir": args.stress_gate_dir,
        "gate_passed": gate_passed,
        "decision": "development_model_training_passed_keep_shadow_final_sealed" if gate_passed else "stop_or_rework_medium_development_model_training",
        "selected_main_model": {
            "method": selected_method,
            "threshold": float(selected_main["threshold"]),
            "feature_count": int(selected_main["feature_count"]),
        },
        "validation_main_hm_min_recall": validation_main_hm,
        "stress_holdout_main_hm_min_recall": stress_holdout_main_hm,
        "worst_thickness_hm_min_recall": worst_thickness,
        "worst_pose_hm_min_recall": worst_pose,
        "worst_stress_label_hm_min_recall": worst_stress_label,
        "validation_expected_calibration_error": validation_ece,
        "stress_holdout_expected_calibration_error": stress_holdout_ece,
        "total_count_only_hm_min_recall": total_count_hm,
        "overlap_only_hm_min_recall": overlap_hm,
        "thickness_pose_hm_min_recall": thickness_pose_hm,
        "shuffled_label_hm_min_recall": shuffled_hm,
        "source_off_hm_min_recall": source_off_hm,
        "main_minus_source_off_hm_margin": margin,
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "integrity_summary": integrity,
        "input_gate_decisions": {
            "schema_gate": schema_gate.get("decision"),
            "baseline_training_gate": training_gate.get("decision"),
            "stress_gate": stress_gate.get("decision"),
            "feature_peak_table_id": feature_manifest.get("peak_table_id"),
            "feature_source_peak_table_ids": feature_manifest.get("source_peak_table_ids"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest = {
        "generated_by": gate["generated_by"],
        "generated_at_utc": gate["generated_at_utc"],
        "protocol_name": gate["protocol_name"],
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "sample_count": int(len(frame)),
        "main_feature_count": int(len(main_cols)),
        "control_feature_count": int(len(control_cols)),
        "gate_file": "v8a_medium_development_model_gate.json",
    }

    validation_decisions = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
    group_recalls = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    threshold_table = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    calibration_bins = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    selection.to_csv(output_dir / "v8a_medium_development_model_selection.csv", index=False, lineterminator="\n")
    validation_decisions.to_csv(output_dir / "v8a_medium_development_model_decisions.csv", index=False, lineterminator="\n")
    group_recalls.to_csv(output_dir / "v8a_medium_development_group_recalls.csv", index=False, lineterminator="\n")
    threshold_table.to_csv(output_dir / "v8a_medium_development_threshold_sweep.csv", index=False, lineterminator="\n")
    calibration_bins.to_csv(output_dir / "v8a_medium_development_calibration_bins.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_medium_development_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "v8a_medium_development_model_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(output_dir, gate, selection, group_recalls)
    print(
        "decision={decision} gate_passed={passed} selected={method} threshold={threshold:.2f} validation_hm={validation:.4f} stress_holdout_hm={holdout:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            method=selected_method,
            threshold=float(selected_main["threshold"]),
            validation=validation_main_hm,
            holdout=stress_holdout_main_hm,
        )
    )


if __name__ == "__main__":
    main()
