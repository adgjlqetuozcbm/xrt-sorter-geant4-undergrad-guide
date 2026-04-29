from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import material_sorting_selected_rebuild as selected
import material_sorting_v2 as v2
from strict_generalization_audit import parse_int_list, parse_raw_dirs, write_csv


HM_PAIR = ["Hematite", "Magnetite"]
DEFAULT_METHODS = ["LogisticRegression", "SVM_Linear", "ExtraTrees", "HistGradientBoosting"]
MODEL_SIZE_RANK = {
    "LogisticRegression": 0,
    "SVM_Linear": 1,
    "HistGradientBoosting": 2,
    "ExtraTrees": 3,
}


def build_frame(project_root: Path, raw_dirs: list[Path], photon_budget: int) -> tuple[pd.DataFrame, dict]:
    old_budget = v2.PHOTONS_PER_SAMPLE
    v2.PHOTONS_PER_SAMPLE = photon_budget
    try:
        material_records: list[v2.RunRecord] = []
        calibration_records: list[v2.RunRecord] = []
        for raw_dir in raw_dirs:
            material_part, calibration_part = v2.discover_records(project_root, raw_dir)
            material_records.extend(record for record in material_part if record.material in HM_PAIR)
            calibration_records.extend(calibration_part)
        calibration_sources = {record.source_id for record in material_records}
        calibration_records = [
            record for record in calibration_records if record.source_id in calibration_sources
        ]
        calibration = v2.calibration_table(calibration_records)
        samples = pd.concat([v2.aggregate_run(record) for record in material_records], ignore_index=True)
        calibrated = v2.apply_calibration(samples, calibration)
        fused, table_mode = v2.fuse_sources(calibrated)
        status = {
            "material_metadata_found": len(material_records),
            "calibration_metadata_found": len(calibration_records),
            "materials_found": sorted({record.material for record in material_records}),
            "sources_found": sorted({record.source_id for record in material_records}),
            "thicknesses_found": sorted({float(record.thickness_mm) for record in material_records}),
            "seeds_found": sorted({int(record.random_seed) for record in material_records}),
            "table_mode": table_mode,
            "rows": int(len(fused)),
        }
        return fused, status
    finally:
        v2.PHOTONS_PER_SAMPLE = old_budget


def append_dictionary(train: pd.DataFrame, validation: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_aug, validation_aug, feature_cols, _ = selected.append_dictionary(train, validation, base_cols)
    return train_aug, validation_aug, feature_cols


def mode_feature_columns(feature_cols: list[str], feature_mode: str) -> list[str]:
    if feature_mode == "thickness_blind":
        return [col for col in feature_cols if col != "thickness_mm"]
    return feature_cols


def evaluate_predictions(method: str, frame: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray, classes: np.ndarray, sk) -> dict:
    y_true = frame["material"].astype(str).to_numpy()
    recalls = sk["recall_score"](y_true, predictions, labels=np.array(HM_PAIR), average=None, zero_division=0)
    return {
        "method": method,
        "samples": int(len(frame)),
        "top1_accuracy": float(np.mean(y_true == predictions)),
        "top3_accuracy": v2.topk_accuracy(y_true, scores, classes, min(2, len(classes))),
        "macro_f1": float(sk["f1_score"](y_true, predictions, labels=np.array(HM_PAIR), average="macro", zero_division=0)),
        "hematite_recall": float(recalls[0]),
        "magnetite_recall": float(recalls[1]),
        "hm_min_recall": float(np.min(recalls)),
        "min_class_recall": float(np.min(recalls)),
    }


def score_method(method: str, train: pd.DataFrame, validation: pd.DataFrame, feature_cols: list[str], sk) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    _, predictions, scores, classes = v2.train_and_score(method, train, validation, feature_cols, sk)
    metrics = evaluate_predictions(method, validation, predictions, scores, classes, sk)
    metrics["feature_count"] = int(len(feature_cols))
    return metrics, predictions, scores, classes


def evaluate_methods(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    methods: list[str],
    feature_mode: str,
    sk,
) -> pd.DataFrame:
    cols = mode_feature_columns(feature_cols, feature_mode)
    rows = []
    for method in methods:
        try:
            metrics, _, _, _ = score_method(method, train, validation, cols, sk)
        except Exception as exc:  # noqa: BLE001
            metrics = {
                "method": method,
                "samples": int(len(validation)),
                "top1_accuracy": math.nan,
                "top3_accuracy": math.nan,
                "macro_f1": math.nan,
                "hematite_recall": math.nan,
                "magnetite_recall": math.nan,
                "hm_min_recall": math.nan,
                "min_class_recall": math.nan,
                "feature_count": int(len(cols)),
                "error": str(exc),
            }
        metrics["feature_mode"] = feature_mode
        metrics["model_size_rank"] = MODEL_SIZE_RANK.get(method, 99)
        rows.append(metrics)
    return pd.DataFrame(rows)


def choose_model(table: pd.DataFrame) -> dict:
    ranked = table.dropna(subset=["hm_min_recall", "macro_f1", "top1_accuracy"]).sort_values(
        ["hm_min_recall", "macro_f1", "top1_accuracy", "model_size_rank"],
        ascending=[False, False, False, True],
    )
    if ranked.empty:
        raise RuntimeError("No v6c candidate produced finite metrics.")
    return ranked.iloc[0].to_dict()


def per_class_table(frame: pd.DataFrame, predictions: np.ndarray, split: str, sk) -> pd.DataFrame:
    y_true = frame["material"].astype(str).to_numpy()
    recalls = sk["recall_score"](y_true, predictions, labels=np.array(HM_PAIR), average=None, zero_division=0)
    support = frame["material"].value_counts().to_dict()
    return pd.DataFrame(
        [
            {"split": split, "material": material, "support": int(support.get(material, 0)), "recall": float(recall)}
            for material, recall in zip(HM_PAIR, recalls)
        ]
    )


def feature_family_ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    feature_mode: str,
    sk,
) -> pd.DataFrame:
    rows = []
    cols = mode_feature_columns(feature_cols, feature_mode)
    for family in v2.FEATURE_FAMILY_ORDER:
        family_cols = [col for col in cols if v2.feature_family(col) == family]
        if not family_cols:
            continue
        metrics, _, _, _ = score_method("ExtraTrees", train, validation, family_cols, sk)
        metrics["feature_mode"] = feature_mode
        metrics["ablation_type"] = "feature_family_only"
        metrics["ablation_value"] = family
        rows.append(metrics)
    return pd.DataFrame(rows)


def source_ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    feature_mode: str,
    sk,
) -> pd.DataFrame:
    cols = mode_feature_columns(feature_cols, feature_mode)
    sources = sorted({col.split("__", 1)[0] for col in cols if "__" in col and col.startswith("mono_")})
    rows = []
    global_cols = [col for col in cols if "__" not in col or col.startswith("dict_")]
    for source in sources:
        source_cols = [col for col in cols if col.startswith(f"{source}__")]
        dual_cols = [col for col in cols if source in col and col.startswith("dual_energy__")]
        selected_cols = list(dict.fromkeys([*source_cols, *dual_cols, *global_cols]))
        if not selected_cols:
            continue
        metrics, _, _, _ = score_method("ExtraTrees", train, validation, selected_cols, sk)
        metrics["feature_mode"] = feature_mode
        metrics["ablation_type"] = "source_plus_global"
        metrics["ablation_value"] = source
        rows.append(metrics)
    return pd.DataFrame(rows)


def failure_analysis(per_class: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in per_class.itertuples(index=False):
        part = decisions[decisions["material"].astype(str).eq(str(row.material))]
        confusions = part.loc[~part["is_correct"], "predicted_material"].value_counts().to_dict()
        rows.append(
            {
                "material": row.material,
                "support": int(row.support),
                "recall": float(row.recall),
                "miss_count": int(round(int(row.support) * (1.0 - float(row.recall)))),
                "common_confusions": ";".join(f"{name}:{int(count)}" for name, count in confusions.items()),
                "failure_status": "pass" if float(row.recall) >= 0.75 else "fail",
                "next_action": "inspect_source_geometry_and_scatter_features" if float(row.recall) < 0.75 else "monitor",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="H/M-first v6c development audit with thickness-aware/blind summaries.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/v6c_hm_source_design")
    parser.add_argument("--raw-dirs", default="")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v6c_hm_source_design")
    parser.add_argument("--photon-budget", type=int, default=5000)
    parser.add_argument("--train-seeds", default=",".join(str(seed) for seed in range(2101, 2113)))
    parser.add_argument("--validation-seeds", default=",".join(str(seed) for seed in range(2201, 2207)))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--protocol-name", default="v6c_hm_source_design_development")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dirs = parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs)
    train_seeds = parse_int_list(args.train_seeds)
    validation_seeds = parse_int_list(args.validation_seeds)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    sk = v2.require_sklearn()

    frame, status = build_frame(project_root, raw_dirs, args.photon_budget)
    seed_series = frame["random_seed"].astype(int)
    train = frame[seed_series.isin(train_seeds)].copy()
    validation = frame[seed_series.isin(validation_seeds)].copy()
    if train.empty or validation.empty:
        raise ValueError("Train and validation splits must both be non-empty.")

    base_cols = v2.numeric_feature_columns(frame)
    train_aug, validation_aug, feature_cols = append_dictionary(train, validation, base_cols)
    selection = pd.concat(
        [
            evaluate_methods(train_aug, validation_aug, feature_cols, methods, "thickness_aware", sk),
            evaluate_methods(train_aug, validation_aug, feature_cols, methods, "thickness_blind", sk),
        ],
        ignore_index=True,
    )
    selected = choose_model(selection[selection["feature_mode"].eq("thickness_aware")])
    selected_method = str(selected["method"])
    aware_cols = mode_feature_columns(feature_cols, "thickness_aware")
    blind_cols = mode_feature_columns(feature_cols, "thickness_blind")
    aware_metrics, aware_predictions, aware_scores, aware_classes = score_method(selected_method, train_aug, validation_aug, aware_cols, sk)
    blind_metrics, blind_predictions, blind_scores, blind_classes = score_method(selected_method, train_aug, validation_aug, blind_cols, sk)
    aware_metrics["feature_mode"] = "thickness_aware"
    blind_metrics["feature_mode"] = "thickness_blind"
    aware_metrics["base_feature_count"] = int(len(base_cols))
    blind_metrics["base_feature_count"] = int(len(base_cols))

    per_class = per_class_table(validation_aug, aware_predictions, "validation", sk)
    decisions = v2.decision_frame(validation_aug, aware_predictions, aware_scores, aware_classes, probability_threshold=0.0, margin_threshold=0.0)
    pairwise = pd.DataFrame(
        [
            {
                "split": "validation",
                "method": selected_method,
                "feature_mode": "thickness_aware",
                "samples": int(len(validation_aug)),
                "hm_min_recall": aware_metrics["hm_min_recall"],
                "hematite_recall": aware_metrics["hematite_recall"],
                "magnetite_recall": aware_metrics["magnetite_recall"],
                "macro_f1": aware_metrics["macro_f1"],
                "top1_accuracy": aware_metrics["top1_accuracy"],
            },
            {
                "split": "validation",
                "method": selected_method,
                "feature_mode": "thickness_blind",
                "samples": int(len(validation_aug)),
                "hm_min_recall": blind_metrics["hm_min_recall"],
                "hematite_recall": blind_metrics["hematite_recall"],
                "magnetite_recall": blind_metrics["magnetite_recall"],
                "macro_f1": blind_metrics["macro_f1"],
                "top1_accuracy": blind_metrics["top1_accuracy"],
            },
        ]
    )
    split_audit = (
        frame.assign(split_role=frame["random_seed"].astype(int).map({**{seed: "train" for seed in train_seeds}, **{seed: "validation" for seed in validation_seeds}}).fillna("unused"))
        .groupby(["split_role", "random_seed", "material"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
    )
    registry = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocol_name": args.protocol_name,
                "evaluation_stage": "development_validation_only",
                "selected_method": selected_method,
                "thickness_aware_hm_min_recall": aware_metrics["hm_min_recall"],
                "thickness_blind_hm_min_recall": blind_metrics["hm_min_recall"],
                "pairwise_hm_min_recall": float(pairwise["hm_min_recall"].min()),
                "min_class_support_observed": int(per_class["support"].min()),
                "claim_safe": False,
                "next_action": "shadow_only_if_v6c_gate_passes",
            }
        ]
    )

    write_csv(selection, output_dir / "validation_model_selection.csv")
    write_csv(pd.DataFrame([aware_metrics]), output_dir / "development_validation_summary.csv")
    write_csv(pd.DataFrame([blind_metrics]), output_dir / "development_validation_summary_thickness_blind.csv")
    write_csv(per_class, output_dir / "per_class_recall_validation.csv")
    write_csv(decisions, output_dir / "validation_decisions.csv")
    write_csv(pairwise, output_dir / "hm_pairwise_audit.csv")
    write_csv(feature_family_ablation(train_aug, validation_aug, feature_cols, "thickness_aware", sk), output_dir / "feature_family_ablation.csv")
    write_csv(source_ablation(train_aug, validation_aug, feature_cols, "thickness_aware", sk), output_dir / "source_ablation.csv")
    write_csv(split_audit, output_dir / "split_audit.csv")
    write_csv(registry, output_dir / "experiment_registry.csv")
    write_csv(failure_analysis(per_class, decisions), output_dir / "failure_analysis.csv")

    manifest = {
        "generated_by": "analysis/hm_v6c_development_audit.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": args.protocol_name,
        "development_only": True,
        "shadow_or_final_used": False,
        "raw_dirs": [path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else path.as_posix() for path in raw_dirs],
        "output_dir": args.output_dir,
        "photon_budget": args.photon_budget,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "methods": methods,
        "selected_method": selected_method,
        "status": status,
        "feature_counts": {
            "base": int(len(base_cols)),
            "model": int(len(feature_cols)),
            "thickness_blind": int(len(blind_cols)),
        },
        "development_validation_metrics": {
            "thickness_aware": aware_metrics,
            "thickness_blind": blind_metrics,
        },
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    (output_dir / "strict_generalization_manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    )
    print(f"Wrote v6c H/M development audit to {output_dir}")
    print(f"selected_method={selected_method} claim_safe=False")


if __name__ == "__main__":
    main()
