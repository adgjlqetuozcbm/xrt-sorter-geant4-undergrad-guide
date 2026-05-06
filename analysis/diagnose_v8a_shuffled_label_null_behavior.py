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


CLAIM_SCOPE = (
    "development-only shuffled-label/null-behavior diagnosis for v8A H/M sidecar features; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)

THRESHOLDS = {
    "shuffle_seed_count_min": 30,
    "null_hm_min_recall_ceiling": 0.55,
    "valid_shuffle_fraction_min": 0.25,
    "selected_threshold_inflation_min": 0.10,
    "fixed_threshold_near_chance_max": 0.55,
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


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("Missing scikit-learn in the active environment.") from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = np.where(probabilities >= threshold, "Magnetite", "Hematite").astype(str)
    recalls = pair_recalls(y_true.astype(str), predictions)
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(y_true == predictions)) if len(y_true) else 0.0,
        "hematite_recall": recalls["Hematite"],
        "magnetite_recall": recalls["Magnetite"],
        "hm_min_recall": float(min(recalls.values())),
    }


def selected_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
        metric = threshold_metrics(y_true, probabilities, float(threshold))
        metric["threshold_distance_to_0p5"] = abs(float(threshold) - 0.5)
        rows.append(metric)
    selected = sorted(rows, key=lambda item: (item["hm_min_recall"], item["accuracy"], -item["threshold_distance_to_0p5"]), reverse=True)[0]
    return float(selected["threshold"]), selected


def model_specs(sk: dict[str, Any], seed: int) -> list[tuple[str, Any]]:
    return [
        (
            "Logistic",
            sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=3000, class_weight="balanced", random_state=seed),
            ),
        ),
        (
            "ExtraTrees",
            sk["ExtraTreesClassifier"](
                n_estimators=120,
                random_state=seed,
                class_weight="balanced",
                max_features="sqrt",
                n_jobs=-1,
            ),
        ),
    ]


def add_count_bin(frame: pd.DataFrame, width: float) -> pd.Series:
    return np.floor(frame["control_total_count_norm"].fillna(0.0).to_numpy(dtype=np.float64) / width).astype(int).astype(str)


def shuffled_labels(train: pd.DataFrame, seed: int, mode: str, count_bin_width: float) -> tuple[np.ndarray, float]:
    labels = train["material"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    if mode == "row_level":
        shuffled = rng.permutation(labels)
        return shuffled, float(np.mean(shuffled != labels)) if len(labels) else 0.0
    if mode == "pair_swap" and "clean_match_pair_id" in train.columns:
        result = pd.Series(labels, index=train.index, dtype=object)
        for _, group_index in train.groupby("clean_match_pair_id", sort=True).groups.items():
            group_index = list(group_index)
            current = result.loc[group_index].astype(str).to_numpy()
            if len(current) != 2 or len(set(current)) != 2:
                result.loc[group_index] = rng.permutation(current)
            elif bool(rng.integers(0, 2)):
                result.loc[group_index] = current[::-1]
        shuffled = result.loc[train.index].astype(str).to_numpy()
        return shuffled, float(np.mean(shuffled != labels)) if len(labels) else 0.0
    if mode != "within_split_thickness_pose_count_bin":
        raise ValueError(f"Unknown shuffle mode: {mode}")
    result = pd.Series(labels, index=train.index, dtype=object)
    keyed = train[["thickness_mm", "pose_index"]].copy()
    keyed["count_bin"] = add_count_bin(train, count_bin_width)
    for _, group_index in keyed.groupby(["thickness_mm", "pose_index", "count_bin"], sort=True).groups.items():
        group_index = list(group_index)
        result.loc[group_index] = rng.permutation(result.loc[group_index].astype(str).to_numpy())
    shuffled = result.loc[train.index].astype(str).to_numpy()
    return shuffled, float(np.mean(shuffled != labels)) if len(labels) else 0.0


def magnetite_probability(estimator: Any, x_values: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(x_values)
    classes = [str(item) for item in estimator.classes_]
    if "Magnetite" not in classes:
        return np.zeros(len(x_values), dtype=np.float64)
    return probabilities[:, classes.index("Magnetite")].astype(np.float64)


def evaluate_null(frame: pd.DataFrame, main_cols: list[str], seeds: list[int], count_bin_width: float, sk: dict[str, Any]) -> pd.DataFrame:
    train = frame[frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    validation = frame[frame["split"].astype(str).eq("validation") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    holdout = frame[frame["split"].astype(str).eq("stress_holdout") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    if train.empty or validation.empty or holdout.empty:
        raise RuntimeError("Train/validation/stress_holdout source-on rows are required for null diagnosis.")

    rows: list[dict[str, Any]] = []
    x_train = train[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    eval_frames = {"validation": validation, "stress_holdout": holdout}
    shuffle_modes = ["row_level", "within_split_thickness_pose_count_bin"]
    if "clean_match_pair_id" in train.columns:
        shuffle_modes.append("pair_swap")
    for shuffle_mode in shuffle_modes:
        for seed in seeds:
            y_train, effective_shuffle_fraction = shuffled_labels(train, seed, shuffle_mode, count_bin_width)
            for model_name, estimator in model_specs(sk, seed):
                fitted = deepcopy(estimator)
                fitted.fit(x_train, y_train)
                validation_threshold = 0.5
                for eval_split, eval_frame in eval_frames.items():
                    x_eval = eval_frame[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
                    y_true = eval_frame["material"].astype(str).to_numpy()
                    prob = magnetite_probability(fitted, x_eval)
                    fixed = threshold_metrics(y_true, prob, 0.5)
                    if eval_split == "validation":
                        validation_threshold, selected = selected_threshold(y_true, prob)
                    else:
                        selected = threshold_metrics(y_true, prob, validation_threshold)
                    for policy, metrics in [("fixed_0p5", fixed), ("validation_selected", selected)]:
                        rows.append(
                            {
                                "shuffle_mode": shuffle_mode,
                                "shuffle_seed": seed,
                                "model": model_name,
                                "eval_split": eval_split,
                                "threshold_policy": policy,
                                "effective_shuffle_fraction": effective_shuffle_fraction,
                                **{key: value for key, value in metrics.items() if key != "threshold_distance_to_0p5"},
                            }
                        )
    return pd.DataFrame(rows)


def summarize_null(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby(["shuffle_mode", "model", "eval_split", "threshold_policy"], sort=True)
    rows = []
    for keys, group in grouped:
        shuffle_mode, model, eval_split, threshold_policy = keys
        rows.append(
            {
                "shuffle_mode": shuffle_mode,
                "model": model,
                "eval_split": eval_split,
                "threshold_policy": threshold_policy,
                "seed_count": int(group["shuffle_seed"].nunique()),
                "hm_min_recall_mean": float(group["hm_min_recall"].mean()),
                "hm_min_recall_p95": float(group["hm_min_recall"].quantile(0.95)),
                "hm_min_recall_max": float(group["hm_min_recall"].max()),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_max": float(group["accuracy"].max()),
                "effective_shuffle_fraction_min": float(group["effective_shuffle_fraction"].min()),
                "effective_shuffle_fraction_mean": float(group["effective_shuffle_fraction"].mean()),
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


def write_report(output_dir: Path, gate: dict[str, Any], aggregate: pd.DataFrame) -> None:
    lines = [
        "# v8A shuffled-label null behavior diagnosis",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Protocol artifact suspected: `{str(gate['gate_protocol_artifact_suspected']).lower()}`",
        f"- Tree null overfit suspected: `{str(gate['tree_null_overfit_suspected']).lower()}`",
        f"- Persistent null shortcut suspected: `{str(gate['persistent_null_shortcut_suspected']).lower()}`",
        "",
        "## Null Summary",
        "",
        markdown_table(
            aggregate.sort_values("hm_min_recall_max", ascending=False),
            [
                "shuffle_mode",
                "model",
                "eval_split",
                "threshold_policy",
                "seed_count",
                "effective_shuffle_fraction_mean",
                "hm_min_recall_mean",
                "hm_min_recall_p95",
                "hm_min_recall_max",
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
    (output_dir / "v8a_shuffled_label_null_behavior_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose shuffled-label/null behavior for v8A H/M features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-seeds", default=",".join(str(seed) for seed in range(10001, 10031)))
    parser.add_argument("--count-bin-width", type=float, default=0.003)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    for name, payload in {"schema_gate": schema_gate, "manifest": manifest}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing null diagnosis because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing null diagnosis because {name} reports existing XRT cube reads.")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features available.")
    seeds = [int(item.strip()) for item in args.shuffle_seeds.split(",") if item.strip()]
    sk = require_sklearn()
    summary = evaluate_null(frame, main_cols, seeds, float(args.count_bin_width), sk)
    aggregate = summarize_null(summary)

    fixed = aggregate[aggregate["threshold_policy"].eq("fixed_0p5")]
    selected = aggregate[aggregate["threshold_policy"].eq("validation_selected")]
    valid_fixed = fixed[fixed["effective_shuffle_fraction_min"] >= THRESHOLDS["valid_shuffle_fraction_min"]]
    valid_selected = selected[selected["effective_shuffle_fraction_min"] >= THRESHOLDS["valid_shuffle_fraction_min"]]
    fixed_max = float(valid_fixed["hm_min_recall_max"].max()) if not valid_fixed.empty else 1.0
    selected_max = float(valid_selected["hm_min_recall_max"].max()) if not valid_selected.empty else 1.0
    fixed_p95 = float(valid_fixed["hm_min_recall_p95"].max()) if not valid_fixed.empty else 1.0
    selected_p95 = float(valid_selected["hm_min_recall_p95"].max()) if not valid_selected.empty else 1.0
    extra_fixed_max = float(valid_fixed[valid_fixed["model"].eq("ExtraTrees")]["hm_min_recall_max"].max()) if not valid_fixed.empty else 1.0
    logistic_fixed_max = float(valid_fixed[valid_fixed["model"].eq("Logistic")]["hm_min_recall_max"].max()) if not valid_fixed.empty else 1.0
    within_fixed_max = float(
        valid_fixed[valid_fixed["shuffle_mode"].eq("within_split_thickness_pose_count_bin")]["hm_min_recall_max"].max()
    ) if not valid_fixed.empty and not valid_fixed[valid_fixed["shuffle_mode"].eq("within_split_thickness_pose_count_bin")].empty else 0.0
    invalid_shuffle_modes = sorted(
        aggregate.loc[
            aggregate["effective_shuffle_fraction_min"] < THRESHOLDS["valid_shuffle_fraction_min"],
            "shuffle_mode",
        ].astype(str).unique().tolist()
    )
    selected_inflation = selected_max - fixed_max
    protocol_artifact = bool(
        selected_max >= THRESHOLDS["null_hm_min_recall_ceiling"]
        and fixed_max < THRESHOLDS["fixed_threshold_near_chance_max"]
        and selected_inflation >= THRESHOLDS["selected_threshold_inflation_min"]
    )
    tree_overfit = bool(
        extra_fixed_max >= THRESHOLDS["null_hm_min_recall_ceiling"]
        and logistic_fixed_max < THRESHOLDS["fixed_threshold_near_chance_max"]
    )
    persistent_shortcut = bool(within_fixed_max >= THRESHOLDS["null_hm_min_recall_ceiling"])
    pass_items = {
        "shuffle_seed_count": len(seeds) >= THRESHOLDS["shuffle_seed_count_min"],
        "valid_shuffle_mode_available": not valid_fixed.empty,
        "fixed_threshold_null_p95_below_ceiling": fixed_p95 < THRESHOLDS["null_hm_min_recall_ceiling"],
        "selected_threshold_null_p95_below_ceiling": selected_p95 < THRESHOLDS["null_hm_min_recall_ceiling"],
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/diagnose_v8a_shuffled_label_null_behavior.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_shuffled_label_null_behavior_diagnosis",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "gate_passed": gate_passed,
        "decision": "null_behavior_clean" if gate_passed else "null_behavior_root_cause_needed",
        "shuffle_seed_count": int(len(seeds)),
        "main_feature_count": int(len(main_cols)),
        "fixed_threshold_hm_min_recall_max": fixed_max,
        "selected_threshold_hm_min_recall_max": selected_max,
        "fixed_threshold_hm_min_recall_p95": fixed_p95,
        "selected_threshold_hm_min_recall_p95": selected_p95,
        "selected_minus_fixed_hm_min_recall_max": selected_inflation,
        "extratrees_fixed_threshold_hm_min_recall_max": extra_fixed_max,
        "logistic_fixed_threshold_hm_min_recall_max": logistic_fixed_max,
        "within_strata_fixed_threshold_hm_min_recall_max": within_fixed_max,
        "invalid_shuffle_modes_due_to_low_effective_shuffle": invalid_shuffle_modes,
        "valid_shuffle_fraction_min": THRESHOLDS["valid_shuffle_fraction_min"],
        "gate_protocol_artifact_suspected": protocol_artifact,
        "tree_null_overfit_suspected": tree_overfit,
        "persistent_null_shortcut_suspected": persistent_shortcut,
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    summary.to_csv(output_dir / "v8a_shuffled_label_null_behavior_rows.csv", index=False, lineterminator="\n")
    aggregate.to_csv(output_dir / "v8a_shuffled_label_null_behavior_summary.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_shuffled_label_null_behavior_gate.json", json_clean(gate))
    write_report(output_dir, gate, aggregate)
    print(
        "decision={decision} gate_passed={passed} fixed_max={fixed:.4f} selected_max={selected:.4f} tree_overfit={tree}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            fixed=fixed_max,
            selected=selected_max,
            tree=str(tree_overfit).lower(),
        )
    )


if __name__ == "__main__":
    main()
