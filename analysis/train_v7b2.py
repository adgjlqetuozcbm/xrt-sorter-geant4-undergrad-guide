from __future__ import annotations

import argparse
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd

import train_v7b as base


FULL_GATE_THRESHOLDS = {
    "macro_f1": 0.84,
    "hm_min_recall": 0.80,
    "hm_pairwise_min_recall": 0.78,
    "min_validation_support_per_class": 120,
    "runner_failures": 0,
    "runner_pending": 0,
}


def require_pilot_gate(project_root: Path, pilot_gate_path: str, allow_without_pilot: bool) -> dict:
    gate_path = project_root / pilot_gate_path
    if not gate_path.exists():
        if allow_without_pilot:
            return {"gate_passed": False, "decision": "missing_overridden_for_smoke"}
        raise FileNotFoundError(f"Missing v7B2 Pilot gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if bool(gate.get("shadow_or_final_used", False)):
        raise RuntimeError("Refusing v7B2 full training because Pilot gate reports shadow/final use.")
    if not bool(gate.get("gate_passed", False)) and not allow_without_pilot:
        raise RuntimeError(f"v7B2 Pilot did not pass; decision={gate.get('decision')}")
    return gate


def gate_report(output_dir: Path, selected: pd.Series, per_class: pd.DataFrame, pairwise: pd.DataFrame, manifest: dict, run_status: dict, pilot_gate: dict) -> dict:
    hm_row = pairwise[pairwise["pair"].eq("Hematite/Magnetite")]
    observed = {
        "method": selected["method"],
        "round_id": int(selected["round_id"]),
        "top1_accuracy": float(selected["top1_accuracy"]),
        "macro_f1": float(selected["macro_f1"]),
        "min_class_recall": float(selected["min_class_recall"]),
        "hm_min_recall": float(selected["hm_min_recall"]),
        "hm_pairwise_min_recall": float(hm_row["pair_min_recall"].iloc[0]) if not hm_row.empty else float(selected["hm_pairwise_min_recall"]),
        "min_validation_support_per_class": int(per_class["support"].min()) if not per_class.empty else 0,
        "runner_failures": int(run_status.get("failed", 0)),
        "runner_pending": run_status.get("pending"),
        "runner_completed": int(run_status.get("completed", 0)),
        "runner_expected_rows": run_status.get("expected_rows"),
        "runner_status_checked": bool(run_status.get("checked", False)),
        "pilot_gate_decision": pilot_gate.get("decision"),
        "pilot_gate_passed": bool(pilot_gate.get("gate_passed", False)),
    }
    checks = {
        "macro_f1": observed["macro_f1"] >= FULL_GATE_THRESHOLDS["macro_f1"],
        "hm_min_recall": observed["hm_min_recall"] >= FULL_GATE_THRESHOLDS["hm_min_recall"],
        "hm_pairwise_min_recall": observed["hm_pairwise_min_recall"] >= FULL_GATE_THRESHOLDS["hm_pairwise_min_recall"],
        "min_validation_support_per_class": observed["min_validation_support_per_class"] >= FULL_GATE_THRESHOLDS["min_validation_support_per_class"],
        "runner_failures": observed["runner_failures"] == FULL_GATE_THRESHOLDS["runner_failures"],
        "runner_pending": observed["runner_pending"] == FULL_GATE_THRESHOLDS["runner_pending"],
        "runner_status_available": bool(run_status.get("checked", False)),
        "shadow_or_final_not_used": not bool(manifest.get("shadow_or_final_used", False)),
        "pilot_gate_passed": bool(pilot_gate.get("gate_passed", False)),
    }
    return {
        "generated_by": "analysis/train_v7b2.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit_dir": output_dir.as_posix(),
        "thresholds": FULL_GATE_THRESHOLDS,
        "observed": observed,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "stop_rule": "Full v7B2 is allowed only after the Pilot gate passes; failed full gates must not trigger shadow/final use.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/gate full v7B2 only after the H/M physics Pilot passes.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cube-dir", default="results/accuracy_v3/v7b2_full_dev")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pilot-gate", default="results/accuracy_v3/v7b2_hm_physics_dev/v7b2_pilot_gate.json")
    parser.add_argument("--allow-without-pilot", action="store_true", help="Only for tiny smoke wiring checks; full use must not set this.")
    parser.add_argument("--methods", default="ExtraTrees,HardNegativeExtraTrees,GroupExpertExtraTrees,HMPairwiseRerankExtraTrees")
    parser.add_argument("--repeat-rounds", type=int, default=2)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--include-thickness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--status-csv", default="results/material_sorting/run_status_v7b2_full_dev.csv")
    parser.add_argument("--n-jobs", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    args = parser.parse_args()

    start_time = perf_counter()
    # Preserve extended UNC prefixes for Windows-side scientific Python.
    project_root = Path(args.project_root)
    pilot_gate = require_pilot_gate(project_root, args.pilot_gate, args.allow_without_pilot)
    cube_dir = project_root / args.cube_dir
    output_dir = project_root / (args.output_dir.strip() or args.cube_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = base.parse_str_list(args.methods)
    n_jobs = max(1, int(args.n_jobs))
    base.log_progress(f"start v7b2 cube_dir={cube_dir} methods={methods} repeat_rounds={args.repeat_rounds}", start_time=start_time)
    sk = base.require_sklearn()
    XGBClassifier = base.require_xgboost()

    cube, metadata, feature_names, cube_manifest = base.load_cube(cube_dir)
    if bool(cube_manifest.get("shadow_or_final_used", False)):
        raise RuntimeError("Shadow/final seeds are present in v7B2 full training metadata.")
    train_mask = metadata["split"].astype(str).eq("train").to_numpy()
    validation_mask = metadata["split"].astype(str).eq("validation").to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("v7B2 full training requires non-empty train and validation splits.")

    x, model_feature_names = base.build_feature_matrix(cube, metadata, feature_names, args.include_thickness)
    labels = metadata["material"].astype(str).to_numpy()
    classes = [material for material in base.TARGET_MATERIALS if material in set(labels)]
    x_train = x[train_mask]
    x_validation = x[validation_mask]
    train_meta = metadata.loc[train_mask].reset_index(drop=True)
    validation_meta = metadata.loc[validation_mask].reset_index(drop=True)
    y_train = labels[train_mask]
    y_validation = labels[validation_mask]

    rows = []
    payloads = {}
    sample_weight = None
    for round_id in range(1, max(1, int(args.repeat_rounds)) + 1):
        for method in methods:
            try:
                metrics, predictions, scores, class_array = base.evaluate_method(
                    method,
                    round_id,
                    x_train,
                    y_train,
                    train_meta,
                    x_validation,
                    y_validation,
                    sample_weight if method.startswith("HardNegative") or method == "HMPairwiseRerankExtraTrees" else None,
                    sk,
                    XGBClassifier,
                    classes,
                    n_jobs,
                )
                metrics["feature_count"] = int(x_train.shape[1])
                payloads[(method, round_id)] = (predictions, scores, class_array)
            except Exception as exc:  # noqa: BLE001
                metrics = {
                    "method": method,
                    "round_id": int(round_id),
                    "samples": int(len(y_validation)),
                    "top1_accuracy": math.nan,
                    "top3_accuracy": math.nan,
                    "macro_f1": math.nan,
                    "min_class_recall": math.nan,
                    "hm_min_recall": math.nan,
                    "hm_pairwise_min_recall": math.nan,
                    "key_hard_negative_pair_min_recall": math.nan,
                    "model_size_rank": base.MODEL_RANK.get(method, 99),
                    "feature_count": int(x_train.shape[1]),
                    "error": str(exc),
                }
            rows.append(metrics)
        selected_so_far = base.choose_model(pd.DataFrame(rows))
        key = (str(selected_so_far["method"]), int(selected_so_far["round_id"]))
        if key in payloads:
            best_predictions, best_scores, best_classes = payloads[key]
            decisions = base.decision_frame(validation_meta, best_predictions, best_scores, best_classes)
            sample_weight = base.update_weights(train_meta, validation_meta, decisions, args.hard_negative_weight)

    selection = pd.DataFrame(rows)
    selected = base.choose_model(selection)
    selected_key = (str(selected["method"]), int(selected["round_id"]))
    if selected_key not in payloads:
        raise RuntimeError(f"Selected v7B2 method has no prediction payload: {selected_key}")
    predictions, scores, class_array = payloads[selected_key]
    per_class = base.per_class_table(validation_meta, predictions, sk)
    decisions = base.decision_frame(validation_meta, predictions, scores, class_array)
    pairwise = base.pairwise_audit(validation_meta, predictions)
    view_ablation = base.evaluate_view_ablation(x_train, y_train, x_validation, y_validation, model_feature_names, sk, classes, n_jobs)
    split_audit = (
        metadata.groupby(["split", "random_seed", "material"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(["split", "random_seed", "material"])
    )
    run_status = base.runner_status(project_root, args.status_csv, cube_manifest)
    manifest = {
        "generated_by": "analysis/train_v7b2.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v7b2_full_dev_training",
        "development_only": True,
        "shadow_or_final_used": bool(cube_manifest.get("shadow_or_final_used", False)),
        "cube_dir": args.cube_dir,
        "output_dir": args.output_dir.strip() or args.cube_dir,
        "pilot_gate": pilot_gate,
        "methods": methods,
        "repeat_rounds_requested": int(args.repeat_rounds),
        "feature_count": int(x_train.shape[1]),
        "selected_method": selected["method"],
        "selected_round": int(selected["round_id"]),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "xgboost_available": XGBClassifier is not None,
        },
    }
    gate = gate_report(output_dir, selected, per_class, pairwise, manifest, run_status, pilot_gate)

    selection.to_csv(output_dir / "v7b2_model_selection.csv", index=False, lineterminator="\n")
    pd.DataFrame([selected]).to_csv(output_dir / "v7b2_development_summary.csv", index=False, lineterminator="\n")
    per_class.to_csv(output_dir / "v7b2_per_class_recall.csv", index=False, lineterminator="\n")
    decisions.to_csv(output_dir / "v7b2_validation_decisions.csv", index=False, lineterminator="\n")
    pairwise.to_csv(output_dir / "v7b2_pairwise_hard_negative_audit.csv", index=False, lineterminator="\n")
    base.failure_analysis(per_class, decisions).to_csv(output_dir / "v7b2_failure_analysis.csv", index=False, lineterminator="\n")
    view_ablation.to_csv(output_dir / "v7b2_view_ablation.csv", index=False, lineterminator="\n")
    base.confusion_matrix_table(validation_meta, predictions, sk).to_csv(output_dir / "v7b2_confusion_matrix.csv", lineterminator="\n")
    split_audit.to_csv(output_dir / "split_audit_training.csv", index=False, lineterminator="\n")
    (output_dir / "strict_generalization_manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    (output_dir / "v7b2_gate.json").write_bytes((json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    base.log_progress(f"Wrote v7B2 training audit to {output_dir}", start_time=start_time)
    base.log_progress(f"selected_method={selected['method']} gate_passed={gate['gate_passed']}", start_time=start_time)


if __name__ == "__main__":
    main()
