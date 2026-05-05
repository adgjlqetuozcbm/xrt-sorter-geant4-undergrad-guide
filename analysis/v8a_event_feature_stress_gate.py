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

from train_v8a_event_feature_smoke import (
    feature_sets,
    load_json,
    pair_recalls,
    require_sklearn,
)


MAIN_METHODS = {"LogisticEventMain", "ExtraTreesEventMain"}
CONTROL_METHODS = {
    "ExtraTreesTotalCountOnly",
    "ExtraTreesOverlapOnly",
    "ExtraTreesThicknessPoseOnly",
    "ExtraTreesShuffledTrainLabels",
    "ExtraTreesSourceOffLeakage",
}


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def rng_for(name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"v8a_stress_gate|{name}".encode("utf-8")).hexdigest()
    seed = int(digest[:12], 16) % (2**32 - 1)
    return np.random.default_rng(seed)


def main_feature_columns(frame: pd.DataFrame) -> list[str]:
    main_cols, _, _, _, _ = feature_sets(frame)
    return main_cols


def iqr(values: pd.Series) -> float:
    q75, q25 = np.percentile(values.fillna(0.0).to_numpy(dtype=np.float64), [75, 25])
    return float(max(q75 - q25, 1e-9))


def apply_stress(frame: pd.DataFrame, feature_cols: list[str], scenario: dict[str, Any]) -> pd.DataFrame:
    kind = scenario["kind"]
    if kind == "identity":
        return frame.copy()
    stressed = frame.copy()
    rng = rng_for(str(scenario["name"]))
    if kind == "multiplicative_and_additive_noise":
        mult_sigma = float(scenario.get("multiplicative_sigma", 0.0))
        add_frac = float(scenario.get("additive_fraction_of_feature_iqr", 0.0))
        for col in feature_cols:
            values = stressed[col].fillna(0.0).to_numpy(dtype=np.float64)
            multiplier = rng.lognormal(mean=0.0, sigma=mult_sigma, size=len(values))
            additive = rng.normal(0.0, add_frac * iqr(stressed[col]), size=len(values))
            stressed[col] = np.clip(values * multiplier + additive, 0.0, None)
        return stressed
    if kind == "column_scale_noise":
        sigma = float(scenario.get("column_scale_sigma", 0.0))
        for col in feature_cols:
            stressed[col] = np.clip(stressed[col].fillna(0.0).to_numpy(dtype=np.float64) * rng.lognormal(0.0, sigma), 0.0, None)
        return stressed
    if kind == "feature_smoothing":
        mix = float(scenario.get("mix_fraction", 0.0))
        row_mean = stressed[feature_cols].fillna(0.0).mean(axis=1)
        for col in feature_cols:
            stressed[col] = (1.0 - mix) * stressed[col].fillna(0.0) + mix * row_mean
        return stressed
    if kind == "background_noise":
        frac = float(scenario.get("background_fraction_of_feature_median", 0.0))
        sigma = float(scenario.get("multiplicative_sigma", 0.0))
        global_median = float(np.median(stressed[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)))
        for col in feature_cols:
            values = stressed[col].fillna(0.0).to_numpy(dtype=np.float64)
            background = abs(global_median) * frac
            stressed[col] = np.clip((values + background) * rng.lognormal(0.0, sigma, size=len(values)), 0.0, None)
        return stressed
    if kind == "suppress_overlap_features":
        factor = float(scenario.get("suppression_factor", 1.0))
        overlap_like = [col for col in feature_cols if "35p" in col or "62p" in col or "unique" in col]
        for col in overlap_like:
            stressed[col] = stressed[col].fillna(0.0) * factor
        return stressed
    raise ValueError(f"Unknown stress scenario kind: {kind}")


def build_models(sk: dict[str, Any], main_cols: list[str], total_count_cols: list[str], overlap_cols: list[str], thickness_pose_cols: list[str]) -> list[tuple[str, Any, list[str], str, str, bool]]:
    return [
        (
            "LogisticEventMain",
            sk["make_pipeline"](sk["StandardScaler"](), sk["LogisticRegression"](max_iter=2000, class_weight="balanced", random_state=9501)),
            main_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesEventMain",
            sk["ExtraTreesClassifier"](n_estimators=300, random_state=9502, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesTotalCountOnly",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9503, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            total_count_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesOverlapOnly",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9504, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            overlap_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesThicknessPoseOnly",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9505, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            thickness_pose_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            False,
        ),
        (
            "ExtraTreesShuffledTrainLabels",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9506, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            "custom_diffraction_on",
            "custom_diffraction_on",
            True,
        ),
        (
            "ExtraTreesSourceOffLeakage",
            sk["ExtraTreesClassifier"](n_estimators=200, random_state=9507, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            "custom_diffraction_off",
            "custom_diffraction_off",
            False,
        ),
    ]


def evaluate_fixed_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_cols: list[str],
    method_name: str,
    estimator: Any,
    *,
    train_source_mode: str,
    validation_source_mode: str,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 9307,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_mask = train_frame["split"].astype(str).eq("train") & train_frame["source_mode"].astype(str).eq(train_source_mode)
    validation_mask = validation_frame["split"].astype(str).eq("validation") & validation_frame["source_mode"].astype(str).eq(validation_source_mode)
    x_train = train_frame.loc[train_mask, feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    x_validation = validation_frame.loc[validation_mask, feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_train = train_frame.loc[train_mask, "material"].astype(str).to_numpy()
    y_validation = validation_frame.loc[validation_mask, "material"].astype(str).to_numpy()
    if len(set(y_train)) < 2 or len(set(y_validation)) < 2:
        raise RuntimeError(
            f"{method_name} needs both H/M classes in train and validation; train={set(y_train)} validation={set(y_validation)}"
        )
    if shuffle_train_labels:
        rng = np.random.default_rng(shuffle_seed)
        y_train = rng.permutation(y_train)
    estimator.fit(x_train, y_train)
    predictions = np.asarray(estimator.predict(x_validation)).astype(str)
    recalls = pair_recalls(y_validation, predictions)
    decisions = validation_frame.loc[
        validation_mask,
        ["sample_id", "split", "material", "source_mode", "source_id", "random_seed", "thickness_mm", "pose_index"],
    ].copy()
    decisions["method"] = method_name
    decisions["prediction"] = predictions
    decisions["is_correct"] = decisions["material"].astype(str).to_numpy() == predictions
    by_thickness = []
    for _, group in decisions.groupby("thickness_mm", sort=True):
        thickness_recalls = pair_recalls(group["material"].astype(str).to_numpy(), group["prediction"].astype(str).to_numpy())
        by_thickness.append(min(thickness_recalls.values()))
    return (
        {
            "method": method_name,
            "train_source_mode": train_source_mode,
            "validation_source_mode": validation_source_mode,
            "feature_count": int(len(feature_cols)),
            "train_samples": int(len(y_train)),
            "validation_samples": int(len(y_validation)),
            "hematite_recall": recalls["Hematite"],
            "magnetite_recall": recalls["Magnetite"],
            "hm_min_recall": float(min(recalls.values())),
            "pairwise_hm_min_recall": float(min(recalls.values())),
            "worst_thickness_hm_min_recall": float(min(by_thickness)) if by_thickness else 0.0,
        },
        decisions,
    )


def evaluate_selection(train_frame: pd.DataFrame, validation_frame: pd.DataFrame, sk: dict[str, Any], scenario_name: str) -> pd.DataFrame:
    main_cols, _, total_count_cols, overlap_cols, thickness_pose_cols = feature_sets(train_frame)
    rows = []
    for method_name, estimator, cols, train_mode, validation_mode, shuffle_labels in build_models(sk, main_cols, total_count_cols, overlap_cols, thickness_pose_cols):
        if not cols:
            rows.append({"scenario": scenario_name, "method": method_name, "hm_min_recall": 0.0, "worst_thickness_hm_min_recall": 0.0, "status": "not_evaluable_no_features"})
            continue
        row, _ = evaluate_fixed_model(
            train_frame,
            validation_frame,
            cols,
            method_name,
            estimator,
            train_source_mode=train_mode,
            validation_source_mode=validation_mode,
            shuffle_train_labels=shuffle_labels,
        )
        row["scenario"] = scenario_name
        row["status"] = "evaluated"
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_thickness(frame: pd.DataFrame, sk: dict[str, Any], main_cols: list[str]) -> pd.DataFrame:
    rows = []
    for thickness in sorted(frame["thickness_mm"].dropna().unique()):
        train_mask = (
            frame["split"].astype(str).eq("train")
            & frame["source_mode"].astype(str).eq("custom_diffraction_on")
            & ~frame["thickness_mm"].eq(thickness)
        )
        validation_mask = (
            frame["source_mode"].astype(str).eq("custom_diffraction_on")
            & frame["thickness_mm"].eq(thickness)
        )
        if train_mask.sum() < 2 or validation_mask.sum() < 2:
            continue
        estimator = sk["make_pipeline"](sk["StandardScaler"](), sk["LogisticRegression"](max_iter=2000, class_weight="balanced", random_state=9601))
        x_train = frame.loc[train_mask, main_cols].fillna(0.0).to_numpy(dtype=np.float64)
        y_train = frame.loc[train_mask, "material"].astype(str).to_numpy()
        x_val = frame.loc[validation_mask, main_cols].fillna(0.0).to_numpy(dtype=np.float64)
        y_val = frame.loc[validation_mask, "material"].astype(str).to_numpy()
        estimator.fit(x_train, y_train)
        pred = np.asarray(estimator.predict(x_val)).astype(str)
        recalls = pair_recalls(y_val, pred)
        rows.append(
            {
                "holdout": "thickness",
                "held_out_value": thickness,
                "train_samples": int(len(y_train)),
                "validation_samples": int(len(y_val)),
                "hematite_recall": recalls["Hematite"],
                "magnetite_recall": recalls["Magnetite"],
                "hm_min_recall": float(min(recalls.values())),
            }
        )
    return pd.DataFrame(rows)


def validation_seed_groups(frame: pd.DataFrame, sk: dict[str, Any], main_cols: list[str]) -> pd.DataFrame:
    rows = []
    train_mask = frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq("custom_diffraction_on")
    x_train = frame.loc[train_mask, main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_train = frame.loc[train_mask, "material"].astype(str).to_numpy()
    estimator = sk["make_pipeline"](sk["StandardScaler"](), sk["LogisticRegression"](max_iter=2000, class_weight="balanced", random_state=9602))
    estimator.fit(x_train, y_train)
    validation = frame["split"].astype(str).eq("validation") & frame["source_mode"].astype(str).eq("custom_diffraction_on")
    for seed in sorted(frame.loc[validation, "random_seed"].dropna().unique()):
        mask = validation & frame["random_seed"].eq(seed)
        x_val = frame.loc[mask, main_cols].fillna(0.0).to_numpy(dtype=np.float64)
        y_val = frame.loc[mask, "material"].astype(str).to_numpy()
        if len(set(y_val)) < 2:
            continue
        pred = np.asarray(estimator.predict(x_val)).astype(str)
        recalls = pair_recalls(y_val, pred)
        rows.append(
            {
                "holdout": "validation_seed_group",
                "held_out_value": int(seed),
                "train_samples": int(len(y_train)),
                "validation_samples": int(len(y_val)),
                "hematite_recall": recalls["Hematite"],
                "magnetite_recall": recalls["Magnetite"],
                "hm_min_recall": float(min(recalls.values())),
            }
        )
    return pd.DataFrame(rows)


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


def write_report(output_dir: Path, gate: dict[str, Any], scenario_summary: pd.DataFrame, holdout_summary: pd.DataFrame) -> None:
    lines = [
        "# v8A event-feature stress gate report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only stress gate with fixed baseline training and stressed validation features. This is not product accuracy, not shadow/final validation, not ordinary XRT H/M sorting, and not manuscript-grade powder XRD evidence.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Worst main H/M min recall: `{gate['worst_main_hm_min_recall']:.4f}`",
        f"- Worst main thickness H/M min recall: `{gate['worst_main_thickness_hm_min_recall']:.4f}`",
        f"- Worst overlap-only H/M min recall: `{gate['worst_overlap_only_hm_min_recall']:.4f}`",
        f"- Worst source-off H/M min recall: `{gate['worst_source_off_hm_min_recall']:.4f}`",
        f"- Worst leave-one-thickness H/M min recall: `{gate['worst_leave_one_thickness_hm_min_recall']:.4f}`",
        f"- Evaluation protocol: `{gate['evaluation_protocol']}`",
        "",
        "## Scenario Summary",
        "",
        markdown_table(scenario_summary, ["scenario", "best_main_hm_min_recall", "overlap_only_hm_min_recall", "source_off_hm_min_recall"]),
        "",
        "## Holdout Summary",
        "",
        markdown_table(holdout_summary, ["holdout", "held_out_value", "hm_min_recall", "hematite_recall", "magnetite_recall"]),
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_event_feature_stress_gate_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stricter development-only stress gates on v8A event-derived features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_event_feature_stress_gate_config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(project_root / args.config)
    input_dir = project_root / config["input_feature_dir"]
    output_dir = project_root / config["output_dir"]
    ensure_output_dir(output_dir, args.overwrite)
    sk = require_sklearn()
    thresholds = config["thresholds"]

    feature_manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    peak_gate_path = project_root / config["peak_provenance_audit_dir"] / "v8a_peak_provenance_gate.json"
    peak_gate = load_json(peak_gate_path)
    if not bool(schema_gate.get("gate_passed")) or not bool(schema_gate.get("tiny_training_gate_allowed")):
        raise RuntimeError("Input event-to-feature gate must pass and allow tiny training before stress gate.")
    if not bool(peak_gate.get("gate_passed")):
        raise RuntimeError(f"Peak provenance gate did not pass: {peak_gate.get('stop_reasons')}")
    if feature_manifest.get("peak_table_id") != peak_gate.get("peak_table_id"):
        raise RuntimeError(
            "Input feature manifest peak table does not match audited peak provenance gate: "
            f"{feature_manifest.get('peak_table_id')} != {peak_gate.get('peak_table_id')}"
        )
    if bool(feature_manifest.get("shadow_or_final_used")) or bool(feature_manifest.get("reads_existing_xrt_cubes")):
        raise RuntimeError("Refusing stress gate because input manifest reports shadow/final or XRT cube reads.")
    source_peak_table_ids = [str(item) for item in feature_manifest.get("source_peak_table_ids", [])]
    source_peak_table_matches_analysis = bool(feature_manifest.get("source_peak_table_matches_analysis", True))

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols = main_feature_columns(frame)
    forbidden_tokens = [str(item).lower() for item in config.get("forbidden_main_feature_name_tokens", [])]
    lineage_like = [col for col in main_cols if any(token in col.lower() for token in forbidden_tokens)]
    if lineage_like:
        raise RuntimeError(f"Main features contain lineage-like names: {lineage_like}")

    all_selection = []
    scenario_rows = []
    for scenario in config["stress_scenarios"]:
        scenario_name = str(scenario["name"])
        stressed = apply_stress(frame, main_cols, scenario)
        selection = evaluate_selection(frame, stressed, sk, scenario_name)
        all_selection.append(selection)
        best_main = selection[selection["method"].isin(MAIN_METHODS)].sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False).iloc[0]
        def hm_for(method: str) -> float:
            values = selection.loc[selection["method"].eq(method), "hm_min_recall"]
            return float(values.iloc[0]) if not values.empty else 0.0
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "best_main_method": str(best_main["method"]),
                "best_main_hm_min_recall": float(best_main["hm_min_recall"]),
                "best_main_worst_thickness_hm_min_recall": float(best_main["worst_thickness_hm_min_recall"]),
                "total_count_only_hm_min_recall": hm_for("ExtraTreesTotalCountOnly"),
                "overlap_only_hm_min_recall": hm_for("ExtraTreesOverlapOnly"),
                "thickness_pose_hm_min_recall": hm_for("ExtraTreesThicknessPoseOnly"),
                "shuffled_label_hm_min_recall": hm_for("ExtraTreesShuffledTrainLabels"),
                "source_off_hm_min_recall": hm_for("ExtraTreesSourceOffLeakage"),
            }
        )

    selection_frame = pd.concat(all_selection, ignore_index=True)
    scenario_summary = pd.DataFrame(scenario_rows)
    holdout_summary = pd.concat([leave_one_thickness(frame, sk, main_cols), validation_seed_groups(frame, sk, main_cols)], ignore_index=True)

    worst_main = float(scenario_summary["best_main_hm_min_recall"].min())
    worst_main_thickness = float(scenario_summary["best_main_worst_thickness_hm_min_recall"].min())
    worst_total_count = float(scenario_summary["total_count_only_hm_min_recall"].max())
    worst_overlap = float(scenario_summary["overlap_only_hm_min_recall"].max())
    worst_shuffled = float(scenario_summary["shuffled_label_hm_min_recall"].max())
    worst_source_off = float(scenario_summary["source_off_hm_min_recall"].max())
    worst_margin = float((scenario_summary["best_main_hm_min_recall"] - scenario_summary["source_off_hm_min_recall"]).min())
    leave_one = holdout_summary[holdout_summary["holdout"].eq("thickness")]
    seed_group = holdout_summary[holdout_summary["holdout"].eq("validation_seed_group")]
    worst_leave_one = float(leave_one["hm_min_recall"].min()) if not leave_one.empty else 0.0
    worst_seed_group = float(seed_group["hm_min_recall"].min()) if not seed_group.empty else 0.0

    stop_reasons = []
    if worst_main < float(thresholds["main_hm_min_recall_min"]):
        stop_reasons.append("Worst stressed main H/M recall is below threshold.")
    if worst_main_thickness < float(thresholds["worst_thickness_hm_min_recall_min"]):
        stop_reasons.append("Worst stressed thickness H/M recall is below threshold.")
    if worst_total_count >= float(thresholds["total_count_only_hm_min_recall_max"]):
        stop_reasons.append("Total-count-only control exceeds ceiling.")
    if worst_overlap >= float(thresholds["overlap_only_hm_min_recall_max"]):
        stop_reasons.append("Overlap-only control exceeds stricter ceiling.")
    if worst_shuffled >= float(thresholds["shuffled_label_hm_min_recall_max"]):
        stop_reasons.append("Shuffled-label control exceeds ceiling.")
    if worst_source_off >= float(thresholds["source_off_hm_min_recall_max"]):
        stop_reasons.append("Source-off leakage control exceeds ceiling.")
    if worst_margin < float(thresholds["main_minus_source_off_hm_margin_min"]):
        stop_reasons.append("Main minus source-off margin is below threshold.")
    if worst_leave_one < float(thresholds["leave_one_thickness_hm_min_recall_min"]):
        stop_reasons.append("Leave-one-thickness H/M recall is below threshold.")
    if worst_seed_group < float(thresholds["validation_seed_group_hm_min_recall_min"]):
        stop_reasons.append("Validation seed-group H/M recall is below threshold.")

    gate_passed = not stop_reasons
    if gate_passed and not source_peak_table_matches_analysis:
        decision = "proceed_to_medium_development_matrix_preregistration_requires_successor_source_regeneration"
    elif gate_passed:
        decision = "proceed_to_medium_development_matrix_preregistration"
    else:
        decision = "stop_or_rework_v8a_stress_gate"
    gate = {
        "generated_by": "analysis/v8a_event_feature_stress_gate.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v8A_event_feature_stress_gate",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": config["claim_scope"],
        "evaluation_protocol": "fixed_baseline_train_rows_unstressed_and_validation_rows_stressed",
        "gate_passed": gate_passed,
        "decision": decision,
        "peak_provenance_gate_passed": bool(peak_gate.get("gate_passed")),
        "schema_gate_passed": bool(schema_gate.get("gate_passed")),
        "input_peak_table_id": feature_manifest.get("peak_table_id"),
        "input_source_peak_table_ids": source_peak_table_ids,
        "source_peak_table_matches_analysis": source_peak_table_matches_analysis,
        "medium_matrix_requires_successor_source_regeneration": bool(gate_passed and not source_peak_table_matches_analysis),
        "lineage_like_main_features": lineage_like,
        "scenario_count": int(len(scenario_summary)),
        "main_feature_count": int(len(main_cols)),
        "worst_main_hm_min_recall": worst_main,
        "worst_main_thickness_hm_min_recall": worst_main_thickness,
        "worst_total_count_hm_min_recall": worst_total_count,
        "worst_overlap_only_hm_min_recall": worst_overlap,
        "worst_shuffled_label_hm_min_recall": worst_shuffled,
        "worst_source_off_hm_min_recall": worst_source_off,
        "worst_main_minus_source_off_margin": worst_margin,
        "worst_leave_one_thickness_hm_min_recall": worst_leave_one,
        "worst_validation_seed_group_hm_min_recall": worst_seed_group,
        "thresholds": thresholds,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }

    selection_frame.to_csv(output_dir / "v8a_event_feature_stress_model_selection.csv", index=False, lineterminator="\n")
    scenario_summary.to_csv(output_dir / "v8a_event_feature_stress_scenario_summary.csv", index=False, lineterminator="\n")
    holdout_summary.to_csv(output_dir / "v8a_event_feature_stress_holdout_summary.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_event_feature_stress_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(output_dir, gate, scenario_summary, holdout_summary)
    print(
        "decision={decision} gate_passed={passed} worst_main={main:.4f} worst_overlap={overlap:.4f} source_off={source_off:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            main=worst_main,
            overlap=worst_overlap,
            source_off=worst_source_off,
        )
    )


if __name__ == "__main__":
    main()
