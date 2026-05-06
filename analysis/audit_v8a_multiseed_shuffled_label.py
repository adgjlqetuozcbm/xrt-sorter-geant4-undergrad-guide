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
from train_v8a_event_feature_smoke import feature_sets, load_json, pair_recalls
from train_v8a_medium_development_model import expected_calibration_error, require_sklearn, threshold_sweep


CLAIM_SCOPE = (
    "development-only multi-seed shuffled-label sanity audit for v8A H/M sidecar features; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)

DEFAULT_SHUFFLE_SEEDS = [9817, 9827, 9837, 9847, 9857]
THRESHOLDS = {
    "shuffle_seed_count_min": 3,
    "max_validation_hm_min_recall": 0.55,
    "max_stress_holdout_hm_min_recall": 0.55,
    "train_samples_min": 100,
    "validation_samples_min": 50,
    "stress_holdout_samples_min": 50,
}


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


def build_eval_frame(frame: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    if not strategy_name:
        return frame.copy()
    balanced = build_balanced_subset(frame, strategy_by_name(strategy_name))
    source_off = frame[frame["source_mode"].astype(str).eq("custom_diffraction_off")].copy()
    source_off["count_balance_bin"] = "source_off_control"
    source_off["match_pair_id"] = "source_off_control"
    source_off["match_delta_total_count_norm"] = 0.0
    return pd.concat([balanced, source_off], ignore_index=True, sort=False)


def split_source(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame[
        frame["split"].astype(str).eq(split)
        & frame["source_mode"].astype(str).eq("custom_diffraction_on")
    ].copy()


def evaluate_shuffle_seed(frame: pd.DataFrame, main_cols: list[str], seed: int, sk: dict[str, Any]) -> list[dict[str, Any]]:
    train = split_source(frame, "train")
    validation = split_source(frame, "validation")
    holdout = split_source(frame, "stress_holdout")
    if train.empty or validation.empty or holdout.empty:
        return []
    x_train = train[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_train = train["material"].astype(str).to_numpy()
    shuffled_labels = np.random.default_rng(seed).permutation(y_train)
    estimator = sk["ExtraTreesClassifier"](
        n_estimators=350,
        random_state=seed,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
    )
    estimator.fit(x_train, shuffled_labels)

    rows: list[dict[str, Any]] = []
    selected_threshold: float | None = None
    for split_name, eval_frame in [("validation", validation), ("stress_holdout", holdout)]:
        x_eval = eval_frame[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
        probabilities = estimator.predict_proba(x_eval)
        classes = [str(item) for item in estimator.classes_]
        prob_magnetite = probabilities[:, classes.index("Magnetite")] if "Magnetite" in classes else np.zeros(len(eval_frame))
        sweep = threshold_sweep(eval_frame, prob_magnetite.astype(np.float64), "ExtraTreesMultiSeedShuffledLabels", split_name)
        if split_name == "validation":
            ranked = sweep.assign(threshold_distance_to_0p5=(sweep["threshold"] - 0.5).abs())
            selected_threshold = float(
                ranked.sort_values(["hm_min_recall", "accuracy", "threshold_distance_to_0p5"], ascending=[False, False, True]).iloc[0]["threshold"]
            )
        threshold = float(selected_threshold if selected_threshold is not None else 0.5)
        predictions = np.where(prob_magnetite >= threshold, "Magnetite", "Hematite").astype(str)
        y_true = eval_frame["material"].astype(str).to_numpy()
        recalls = pair_recalls(y_true, predictions)
        y_binary = (y_true == "Magnetite").astype(int)
        ece, _ = expected_calibration_error(y_binary, prob_magnetite.astype(np.float64))
        rows.append(
            {
                "shuffle_seed": seed,
                "eval_split": split_name,
                "threshold": threshold,
                "samples": int(len(eval_frame)),
                "train_samples": int(len(train)),
                "accuracy": float(np.mean(y_true == predictions)) if len(y_true) else 0.0,
                "hematite_recall": recalls["Hematite"],
                "magnetite_recall": recalls["Magnetite"],
                "hm_min_recall": float(min(recalls.values())),
                "expected_calibration_error": ece,
            }
        )
    return rows


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


def write_report(output_dir: Path, gate: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        "# v8A multi-seed shuffled-label audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Count-balance strategy: `{gate['count_balance_strategy'] or 'none'}`",
        f"- Shuffle seeds: `{','.join(str(item) for item in gate['shuffle_seeds'])}`",
        f"- Max validation H/M min recall: `{gate['max_validation_hm_min_recall']:.4f}`",
        f"- Max stress-holdout H/M min recall: `{gate['max_stress_holdout_hm_min_recall']:.4f}`",
        "",
        "## Seed Summary",
        "",
        markdown_table(summary.sort_values(["eval_split", "shuffle_seed"]), ["shuffle_seed", "eval_split", "threshold", "samples", "hm_min_recall", "hematite_recall", "magnetite_recall"]),
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiseed_shuffled_label_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed shuffled-label sanity audit on v8A H/M features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count-balance-strategy", default="")
    parser.add_argument("--shuffle-seeds", default=",".join(str(seed) for seed in DEFAULT_SHUFFLE_SEEDS))
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
            raise RuntimeError(f"Refusing shuffled-label audit because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing shuffled-label audit because {name} reports existing XRT cube reads.")
    if not bool(schema_gate.get("gate_passed", False)):
        raise RuntimeError(f"Input schema gate did not pass: {schema_gate.get('decision')}")

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    eval_frame = build_eval_frame(frame, args.count_balance_strategy)
    main_cols, _, _, _, _ = feature_sets(eval_frame)
    if not main_cols:
        raise RuntimeError("No diffraction_* main features are available.")
    seeds = [int(item.strip()) for item in args.shuffle_seeds.split(",") if item.strip()]
    sk = require_sklearn()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(evaluate_shuffle_seed(eval_frame, main_cols, seed, sk))
    summary = pd.DataFrame(rows)
    train = split_source(eval_frame, "train")
    validation = split_source(eval_frame, "validation")
    holdout = split_source(eval_frame, "stress_holdout")
    max_validation = float(summary.loc[summary["eval_split"].eq("validation"), "hm_min_recall"].max()) if not summary.empty else 1.0
    max_holdout = float(summary.loc[summary["eval_split"].eq("stress_holdout"), "hm_min_recall"].max()) if not summary.empty else 1.0
    pass_items = {
        "shuffle_seed_count": len(seeds) >= THRESHOLDS["shuffle_seed_count_min"],
        "train_support": len(train) >= THRESHOLDS["train_samples_min"],
        "validation_support": len(validation) >= THRESHOLDS["validation_samples_min"],
        "stress_holdout_support": len(holdout) >= THRESHOLDS["stress_holdout_samples_min"],
        "validation_shuffled_label_below_ceiling": max_validation < THRESHOLDS["max_validation_hm_min_recall"],
        "stress_holdout_shuffled_label_below_ceiling": max_holdout < THRESHOLDS["max_stress_holdout_hm_min_recall"],
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_multiseed_shuffled_label.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_multiseed_shuffled_label_sanity_gate",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "count_balance_strategy": args.count_balance_strategy,
        "gate_passed": gate_passed,
        "decision": "multiseed_shuffled_label_sanity_passed" if gate_passed else "stop_or_rework_multiseed_shuffled_label_sanity",
        "shuffle_seeds": seeds,
        "train_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "stress_holdout_samples": int(len(holdout)),
        "main_feature_count": int(len(main_cols)),
        "max_validation_hm_min_recall": max_validation,
        "max_stress_holdout_hm_min_recall": max_holdout,
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    summary.to_csv(output_dir / "v8a_multiseed_shuffled_label_summary.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_multiseed_shuffled_label_gate.json", gate)
    write_report(output_dir, gate, summary)
    print(
        "decision={decision} gate_passed={passed} strategy={strategy} max_shuffle={validation:.4f}/{holdout:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            strategy=args.count_balance_strategy or "none",
            validation=max_validation,
            holdout=max_holdout,
        )
    )


if __name__ == "__main__":
    main()
