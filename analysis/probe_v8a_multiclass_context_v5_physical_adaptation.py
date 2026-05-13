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
    add_lineage_controls,
    build_models,
    ensure_output_dir,
    evaluate_estimator,
    fit_estimator,
    json_clean,
    require_sklearn,
    write_json,
)


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def adaptation_train_frame(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    nominal_train = frame[
        frame["split"].astype(str).eq("train")
        & frame["physical_perturbation_profile"].astype(str).eq("nominal")
    ].copy()
    if protocol == "nominal_train_only":
        return nominal_train
    validation = frame[frame["split"].astype(str).eq("validation")].copy()
    if protocol == "nominal_train_plus_validation_all_profiles":
        return pd.concat([nominal_train, validation], ignore_index=True)
    if protocol == "nominal_train_plus_validation_resolution_and_combined":
        selected = validation[
            validation["physical_perturbation_profile"].astype(str).isin(["resolution_blur_moderate", "combined_moderate"])
        ].copy()
        return pd.concat([nominal_train, selected], ignore_index=True)
    raise ValueError(f"Unknown adaptation protocol: {protocol}")


def best_main_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    subset = summary[summary["family"].astype(str).eq("main") & summary["status"].astype(str).eq("evaluated")].copy()
    for _, group in subset.groupby(["protocol", "physical_perturbation_profile"], sort=True):
        rows.append(group.sort_values(["hm_min_recall", "macro_f1", "min_class_recall"], ascending=False).iloc[0])
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


def write_report(output_dir: Path, gate: dict[str, Any], best: pd.DataFrame) -> None:
    lines = [
        "# v8A v5 physical perturbation adaptation probe",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "This is a development-only probe: validation perturbation rows are intentionally used for adaptation training, and only stress_holdout is evaluated.",
        "",
        "## Best Main Model on Stress Holdout",
        "",
        markdown_table(
            best.sort_values(["protocol", "physical_perturbation_profile"]),
            ["protocol", "physical_perturbation_profile", "method", "top1_accuracy", "macro_f1", "min_class_recall", "hematite_recall", "magnetite_recall", "hm_min_recall"],
            limit=48,
        ),
        "",
    ]
    (output_dir / "v8a_multiclass_context_v5_physical_adaptation_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether v5 physical perturbation failures can be repaired with perturbation-augmented training.")
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
    models = [model for model in build_models(sk, main_cols, total_count_cols, lineage_cols) if model["family"] == "main"]

    protocols = [
        "nominal_train_only",
        "nominal_train_plus_validation_resolution_and_combined",
        "nominal_train_plus_validation_all_profiles",
    ]
    stress = frame[frame["split"].astype(str).eq("stress_holdout")].copy()
    summary_rows: list[dict[str, Any]] = []
    for protocol in protocols:
        train = adaptation_train_frame(frame, protocol)
        for model in models:
            estimator = fit_estimator(model, train, labels)
            for profile, eval_frame in stress.groupby("physical_perturbation_profile", sort=True):
                summary, _, _ = evaluate_estimator(
                    track="v5_physical_adaptation_probe",
                    model=model,
                    estimator=estimator,
                    eval_frame=eval_frame.copy(),
                    eval_split="stress_holdout",
                    labels=labels,
                    sk=sk,
                )
                summary["protocol"] = protocol
                summary["physical_perturbation_profile"] = str(profile)
                summary["train_samples"] = int(len(train))
                summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    best = best_main_rows(summary)
    summary.to_csv(output_dir / "v8a_multiclass_context_v5_physical_adaptation_summary.csv", index=False)
    best.to_csv(output_dir / "v8a_multiclass_context_v5_physical_adaptation_best_main.csv", index=False)

    aggregate = (
        best[~best["physical_perturbation_profile"].astype(str).eq("nominal")]
        .groupby("protocol")
        .agg(
            worst_perturbed_hm_min_recall=("hm_min_recall", "min"),
            worst_perturbed_macro_f1=("macro_f1", "min"),
            mean_perturbed_hm_min_recall=("hm_min_recall", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "v8a_multiclass_context_v5_physical_adaptation_aggregate.csv", index=False)

    gate = {
        "generated_by": "analysis/probe_v8a_multiclass_context_v5_physical_adaptation.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "uses_validation_for_adaptation_training": True,
        "claim_scope": "development-only adaptation probe; not final validation or product accuracy",
        "source_final_audit_training_unlocked": bool(final_gate.get("training_unlocked", False)),
        "protocols": protocols,
        "sample_count": int(len(frame)),
        "materials": labels,
        "main_feature_count": int(len(main_cols)),
        "aggregate": aggregate.to_dict(orient="records"),
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(output_dir / "v8a_multiclass_context_v5_physical_adaptation_gate.json", json_clean(gate))
    write_report(output_dir, gate, best)
    print("decision=v5_physical_adaptation_probe_completed protocols=" + str(len(protocols)))


if __name__ == "__main__":
    main()
