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
from train_v8a_multiclass_context_model import (
    HM_PAIR,
    add_lineage_controls,
    build_models,
    ensure_output_dir,
    evaluate_estimator,
    fit_estimator,
    json_clean,
    require_sklearn,
    write_json,
)


SHUFFLE_SEEDS = list(range(66001, 66011))
MAIN_METHODS = ("LogisticMulticlassMain", "ExtraTreesMulticlassMain")
THRESHOLDS = {
    "validation_worst_profile_hm_min_recall_min": 0.85,
    "stress_worst_profile_hm_min_recall_min": 0.75,
    "validation_worst_profile_macro_f1_min": 0.60,
    "stress_worst_profile_macro_f1_min": 0.55,
    "validation_overall_hm_min_recall_min": 0.90,
    "stress_overall_hm_min_recall_min": 0.85,
    "total_count_only_worst_profile_hm_max": 0.65,
    "lineage_only_worst_profile_hm_max": 0.65,
    "shuffled_label_worst_profile_hm_p95_max": 0.55,
    "real_minus_shuffled_worst_profile_hm_margin_min": 0.20,
}


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def profile_eval_frames(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    result: list[tuple[str, str, pd.DataFrame]] = []
    for split in ["validation", "stress_holdout"]:
        split_frame = frame[frame["split"].astype(str).eq(split)].copy()
        result.append((split, "__overall__", split_frame))
        for profile, group in split_frame.groupby("physical_perturbation_profile", sort=True):
            result.append((split, str(profile), group.copy()))
    return result


def summary_best(summary: pd.DataFrame, family: str = "main") -> pd.DataFrame:
    rows: list[pd.Series] = []
    subset = summary[summary["family"].astype(str).eq(family) & summary["status"].astype(str).eq("evaluated")].copy()
    for _, group in subset.groupby(["eval_split", "physical_perturbation_profile"], sort=True):
        rows.append(group.sort_values(["hm_min_recall", "macro_f1", "min_class_recall"], ascending=False).iloc[0])
    return pd.DataFrame(rows)


def fit_shuffled_estimator(model: dict[str, Any], train: pd.DataFrame, labels: list[str], shuffle_seed: int) -> Any | None:
    cols = list(model["feature_cols"])
    if not cols or train.empty:
        return None
    y_train = train["material"].astype(str).to_numpy()
    if len(set(y_train)) < 2:
        return None
    rng = np.random.default_rng(shuffle_seed)
    estimator = deepcopy(model["estimator"])
    estimator.fit(train[cols].fillna(0.0).to_numpy(dtype=np.float64), rng.permutation(y_train))
    return estimator


def shuffled_null(
    *,
    model: dict[str, Any],
    train: pd.DataFrame,
    eval_items: list[tuple[str, str, pd.DataFrame]],
    labels: list[str],
    sk: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SHUFFLE_SEEDS:
        estimator = fit_shuffled_estimator(model, train, labels, seed)
        for eval_split, profile, eval_frame in eval_items:
            if profile == "__overall__":
                continue
            summary, _, _ = evaluate_estimator(
                track="v6_shuffled_label_null",
                model=model,
                estimator=estimator,
                eval_frame=eval_frame,
                eval_split=eval_split,
                labels=labels,
                sk=sk,
            )
            summary["physical_perturbation_profile"] = profile
            summary["shuffle_seed"] = seed
            rows.append(summary)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 48) -> str:
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


def confusion_rows(decisions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decisions.empty:
        return pd.DataFrame()
    subset = decisions[
        decisions["method"].astype(str).isin(MAIN_METHODS)
        & ~decisions["physical_perturbation_profile"].astype(str).eq("__overall__")
    ].copy()
    for keys, group in subset.groupby(["method", "split", "physical_perturbation_profile"], sort=True):
        method, split, profile = keys
        for actual in labels:
            actual_group = group[group["material"].astype(str).eq(actual)]
            denom = max(len(actual_group), 1)
            for predicted in labels:
                count = int((actual_group["prediction"].astype(str) == predicted).sum())
                if count:
                    rows.append(
                        {
                            "method": method,
                            "eval_split": split,
                            "physical_perturbation_profile": profile,
                            "actual": actual,
                            "predicted": predicted,
                            "count": count,
                            "rate": float(count / denom),
                        }
                    )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, gate: dict[str, Any], best: pd.DataFrame, controls: pd.DataFrame, shuffled: pd.DataFrame) -> None:
    lines = [
        "# v8A v6 physical robustness model report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Worst validation profile H/M min recall: `{gate['validation_worst_profile_hm_min_recall']:.4f}`",
        f"- Worst stress profile H/M min recall: `{gate['stress_worst_profile_hm_min_recall']:.4f}`",
        f"- Total-count-only worst profile H/M: `{gate['total_count_only_worst_profile_hm']:.4f}`",
        f"- Lineage-only worst profile H/M: `{gate['lineage_only_worst_profile_hm']:.4f}`",
        "",
        "## Best Main Model",
        "",
        markdown_table(
            best.sort_values(["eval_split", "physical_perturbation_profile"]),
            ["eval_split", "physical_perturbation_profile", "method", "top1_accuracy", "macro_f1", "min_class_recall", "hematite_recall", "magnetite_recall", "hm_min_recall"],
            limit=48,
        ),
        "",
        "## Control Models",
        "",
        markdown_table(
            controls.sort_values(["eval_split", "physical_perturbation_profile", "method"]),
            ["eval_split", "physical_perturbation_profile", "method", "top1_accuracy", "macro_f1", "hm_min_recall"],
            limit=64,
        ),
        "",
        "## Shuffled Label Null",
        "",
        markdown_table(
            shuffled.sort_values(["eval_split", "physical_perturbation_profile", "method"]).head(24),
            ["eval_split", "physical_perturbation_profile", "method", "hm_min_recall_p95", "macro_f1_p95"],
            limit=24,
        ),
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_multiclass_context_v6_physical_robust_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def min_value(data: pd.DataFrame, column: str) -> float:
    return float(data[column].min()) if not data.empty else 0.0


def max_value(data: pd.DataFrame, column: str) -> float:
    return float(data[column].max()) if not data.empty else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate v8A v6 perturbation-augmented physical robustness models.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--final-audit-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = as_project_path(project_root, args.input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    final_gate = load_json(as_project_path(project_root, args.final_audit_gate))
    labels = sorted(frame["material"].astype(str).unique().tolist())
    sk = require_sklearn()
    main_cols, _, total_count_cols, _, _ = feature_sets(frame)
    frame, lineage_cols = add_lineage_controls(frame)
    models = build_models(sk, main_cols, total_count_cols, lineage_cols)

    train = frame[frame["split"].astype(str).eq("train")].copy()
    eval_items = profile_eval_frames(frame)
    fitted = {str(model["method"]): fit_estimator(model, train, labels) for model in models}

    summary_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    for eval_split, profile, eval_frame in eval_items:
        for model in models:
            summary, decisions, _ = evaluate_estimator(
                track="v6_physical_robust",
                model=model,
                estimator=fitted[str(model["method"])],
                eval_frame=eval_frame,
                eval_split=eval_split,
                labels=labels,
                sk=sk,
            )
            summary["physical_perturbation_profile"] = profile
            summary_rows.append(summary)
            if not decisions.empty:
                decisions["physical_perturbation_profile"] = profile
                decision_frames.append(decisions)

    summary = pd.DataFrame(summary_rows)
    best = summary_best(summary, "main")
    controls = summary[summary["family"].astype(str).eq("control")].copy()
    summary.to_csv(output_dir / "v8a_multiclass_context_v6_physical_robust_summary.csv", index=False)
    best.to_csv(output_dir / "v8a_multiclass_context_v6_physical_robust_best_main.csv", index=False)
    controls.to_csv(output_dir / "v8a_multiclass_context_v6_physical_robust_controls.csv", index=False)
    decisions_all = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    if not decisions_all.empty:
        decisions_all.to_csv(output_dir / "v8a_multiclass_context_v6_physical_robust_decisions.csv", index=False)
        confusions = confusion_rows(decisions_all, labels)
        if not confusions.empty:
            confusions.to_csv(output_dir / "v8a_multiclass_context_v6_physical_robust_confusions.csv", index=False)

    shuffled_model = [model for model in models if model["method"] == "LogisticMulticlassMain"][0]
    shuffled_raw = shuffled_null(model=shuffled_model, train=train, eval_items=eval_items, labels=labels, sk=sk)
    if not shuffled_raw.empty:
        shuffled_summary = (
            shuffled_raw.groupby(["eval_split", "physical_perturbation_profile", "method"], sort=True)
            .agg(
                hm_min_recall_p95=("hm_min_recall", lambda values: float(np.percentile(values, 95))),
                macro_f1_p95=("macro_f1", lambda values: float(np.percentile(values, 95))),
            )
            .reset_index()
        )
    else:
        shuffled_summary = pd.DataFrame(columns=["eval_split", "physical_perturbation_profile", "method", "hm_min_recall_p95", "macro_f1_p95"])
    shuffled_raw.to_csv(output_dir / "v8a_multiclass_context_v6_physical_robust_shuffled_raw.csv", index=False)
    shuffled_summary.to_csv(output_dir / "v8a_multiclass_context_v6_physical_robust_shuffled_summary.csv", index=False)

    profile_best = best[~best["physical_perturbation_profile"].astype(str).eq("__overall__")]
    validation_profiles = profile_best[profile_best["eval_split"].astype(str).eq("validation")]
    stress_profiles = profile_best[profile_best["eval_split"].astype(str).eq("stress_holdout")]
    validation_overall = best[
        best["eval_split"].astype(str).eq("validation")
        & best["physical_perturbation_profile"].astype(str).eq("__overall__")
    ]
    stress_overall = best[
        best["eval_split"].astype(str).eq("stress_holdout")
        & best["physical_perturbation_profile"].astype(str).eq("__overall__")
    ]

    total_count_controls = controls[
        controls["method"].astype(str).eq("ExtraTreesTotalCountOnly")
        & ~controls["physical_perturbation_profile"].astype(str).eq("__overall__")
    ]
    lineage_controls = controls[
        controls["method"].astype(str).eq("ExtraTreesLineageOnly")
        & ~controls["physical_perturbation_profile"].astype(str).eq("__overall__")
    ]
    shuffled_worst_p95 = max_value(shuffled_summary, "hm_min_recall_p95")
    real_worst_hm = min(min_value(validation_profiles, "hm_min_recall"), min_value(stress_profiles, "hm_min_recall"))

    stop_reasons: list[str] = []
    if not bool(final_gate.get("training_unlocked", False)):
        stop_reasons.append("final_audit_did_not_unlock_training")
    validation_worst_hm = min_value(validation_profiles, "hm_min_recall")
    stress_worst_hm = min_value(stress_profiles, "hm_min_recall")
    validation_worst_macro = min_value(validation_profiles, "macro_f1")
    stress_worst_macro = min_value(stress_profiles, "macro_f1")
    validation_overall_hm = min_value(validation_overall, "hm_min_recall")
    stress_overall_hm = min_value(stress_overall, "hm_min_recall")
    total_count_worst = max_value(total_count_controls, "hm_min_recall")
    lineage_worst = max_value(lineage_controls, "hm_min_recall")
    if validation_worst_hm < THRESHOLDS["validation_worst_profile_hm_min_recall_min"]:
        stop_reasons.append("validation_worst_profile_hm_min_recall_below_threshold")
    if stress_worst_hm < THRESHOLDS["stress_worst_profile_hm_min_recall_min"]:
        stop_reasons.append("stress_worst_profile_hm_min_recall_below_threshold")
    if validation_worst_macro < THRESHOLDS["validation_worst_profile_macro_f1_min"]:
        stop_reasons.append("validation_worst_profile_macro_f1_below_threshold")
    if stress_worst_macro < THRESHOLDS["stress_worst_profile_macro_f1_min"]:
        stop_reasons.append("stress_worst_profile_macro_f1_below_threshold")
    if validation_overall_hm < THRESHOLDS["validation_overall_hm_min_recall_min"]:
        stop_reasons.append("validation_overall_hm_min_recall_below_threshold")
    if stress_overall_hm < THRESHOLDS["stress_overall_hm_min_recall_min"]:
        stop_reasons.append("stress_overall_hm_min_recall_below_threshold")
    if total_count_worst > THRESHOLDS["total_count_only_worst_profile_hm_max"]:
        stop_reasons.append("total_count_only_worst_profile_hm_above_shortcut_threshold")
    if lineage_worst > THRESHOLDS["lineage_only_worst_profile_hm_max"]:
        stop_reasons.append("lineage_only_worst_profile_hm_above_shortcut_threshold")
    if shuffled_worst_p95 > THRESHOLDS["shuffled_label_worst_profile_hm_p95_max"]:
        stop_reasons.append("shuffled_label_worst_profile_hm_p95_above_threshold")
    if real_worst_hm - shuffled_worst_p95 < THRESHOLDS["real_minus_shuffled_worst_profile_hm_margin_min"]:
        stop_reasons.append("real_minus_shuffled_worst_profile_hm_margin_too_small")

    gate_passed = not stop_reasons
    gate = {
        "generated_by": "analysis/train_v8a_multiclass_context_v6_physical_robust.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "claim_scope": "development-only perturbation-augmented physical robustness diagnostic; not hardware validation or product accuracy",
        "gate_passed": gate_passed,
        "decision": "v6_physical_robust_gate_passed_ready_for_scalability_scout" if gate_passed else "stop_v6_physical_robust_gate",
        "input_dir": args.input_dir,
        "sample_count": int(len(frame)),
        "train_samples": int(len(train)),
        "train_profiles": sorted(train["physical_perturbation_profile"].astype(str).unique().tolist()),
        "materials": labels,
        "hm_pair": list(HM_PAIR),
        "main_feature_count": int(len(main_cols)),
        "validation_worst_profile_hm_min_recall": validation_worst_hm,
        "stress_worst_profile_hm_min_recall": stress_worst_hm,
        "validation_worst_profile_macro_f1": validation_worst_macro,
        "stress_worst_profile_macro_f1": stress_worst_macro,
        "validation_overall_hm_min_recall": validation_overall_hm,
        "stress_overall_hm_min_recall": stress_overall_hm,
        "total_count_only_worst_profile_hm": total_count_worst,
        "lineage_only_worst_profile_hm": lineage_worst,
        "shuffled_label_worst_profile_hm_p95": shuffled_worst_p95,
        "real_minus_shuffled_worst_profile_hm_margin": real_worst_hm - shuffled_worst_p95,
        "thresholds": THRESHOLDS,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(output_dir / "v8a_multiclass_context_v6_physical_robust_gate.json", json_clean(gate))
    write_report(output_dir, gate, best, controls, shuffled_summary)
    print(
        "decision={decision} gate_passed={passed} val_worst_hm={val:.4f} stress_worst_hm={stress:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            val=validation_worst_hm,
            stress=stress_worst_hm,
        )
    )


if __name__ == "__main__":
    main()
