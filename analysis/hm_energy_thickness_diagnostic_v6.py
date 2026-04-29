from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import material_sorting_v2 as v2


HM_PAIR = ["Hematite", "Magnetite"]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_raw_dirs(project_root: Path, raw_dir: str, raw_dirs: str) -> list[Path]:
    values = [item.strip() for item in raw_dirs.split(",") if item.strip()] if raw_dirs.strip() else [raw_dir]
    return [project_root / value for value in values]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(data: dict, path: Path) -> None:
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def source_energy_kev(source_id: str) -> float:
    if source_id.startswith("mono_") and source_id.endswith("kev"):
        return float(source_id.removeprefix("mono_").removesuffix("kev"))
    return math.nan


def build_calibrated_hm_frame(
    project_root: Path,
    raw_dirs: list[Path],
    seeds: list[int],
    photons_per_sample: int,
) -> tuple[pd.DataFrame, dict]:
    old_budget = v2.PHOTONS_PER_SAMPLE
    v2.PHOTONS_PER_SAMPLE = photons_per_sample
    try:
        material_records: list[v2.RunRecord] = []
        calibration_records: list[v2.RunRecord] = []
        for raw_dir in raw_dirs:
            material_part, calibration_part = v2.discover_records(project_root, raw_dir)
            material_records.extend(
                record
                for record in material_part
                if record.material in HM_PAIR and int(record.random_seed) in set(seeds)
            )
            calibration_records.extend(
                record for record in calibration_part if int(record.random_seed) in set(seeds)
            )
        if not material_records:
            raise ValueError("No Hematite/Magnetite material records found for the requested seeds.")
        needed_sources = {record.source_id for record in material_records}
        calibration_records = [
            record for record in calibration_records if record.source_id in needed_sources
        ]
        calibration = v2.calibration_table(calibration_records)
        samples = pd.concat([v2.aggregate_run(record) for record in material_records], ignore_index=True)
        calibrated = v2.apply_calibration(samples, calibration)
        inventory = {
            "material_records": len(material_records),
            "calibration_records": len(calibration_records),
            "rows_after_sampling": int(len(calibrated)),
            "materials": sorted(calibrated["material"].unique().tolist()),
            "sources": sorted(calibrated["source_id"].unique().tolist()),
            "thicknesses_mm": sorted(float(value) for value in calibrated["thickness_mm"].unique()),
            "seeds": sorted(int(value) for value in calibrated["random_seed"].unique()),
        }
        return calibrated.replace([np.inf, -np.inf], np.nan).fillna(0.0), inventory
    finally:
        v2.PHOTONS_PER_SAMPLE = old_budget


def label_vector(frame: pd.DataFrame) -> np.ndarray:
    return (frame["material"].astype(str).to_numpy() == "Magnetite").astype(int)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = v2.numeric_feature_columns(frame)
    return v2.physics_feature_columns(cols)


def predict_proba_for_magnetite(model, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[cols])
    for idx, label in enumerate(model.classes_):
        if str(label) == "Magnetite":
            return raw[:, idx]
    return np.full(len(frame), 0.5)


def score_pairwise_group(
    group_type: str,
    source_id: str,
    thickness_mm: float | str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    random_state: int,
) -> list[dict]:
    cols = feature_columns(pd.concat([train, validation], ignore_index=True))
    if not cols or train["material"].nunique() < 2 or validation["material"].nunique() < 2:
        return [
            {
                "group_type": group_type,
                "source_id": source_id,
                "energy_keV": source_energy_kev(source_id),
                "thickness_mm": thickness_mm,
                "model": "not_scored",
                "feature_count": len(cols),
                "train_samples": int(len(train)),
                "validation_samples": int(len(validation)),
                "top1_accuracy": math.nan,
                "macro_f1": math.nan,
                "hematite_recall": math.nan,
                "magnetite_recall": math.nan,
                "hm_min_recall": math.nan,
                "roc_auc_magnetite": math.nan,
            }
        ]

    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state),
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=800,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",
        ),
    }

    rows = []
    y_true = validation["material"].astype(str).to_numpy()
    y_auc = label_vector(validation)
    for model_name, model in models.items():
        model.fit(train[cols], train["material"].astype(str))
        pred = model.predict(validation[cols])
        recalls = recall_score(y_true, pred, labels=HM_PAIR, average=None, zero_division=0)
        try:
            auc = float(roc_auc_score(y_auc, predict_proba_for_magnetite(model, validation, cols)))
        except ValueError:
            auc = math.nan
        rows.append(
            {
                "group_type": group_type,
                "source_id": source_id,
                "energy_keV": source_energy_kev(source_id),
                "thickness_mm": thickness_mm,
                "model": model_name,
                "feature_count": len(cols),
                "train_samples": int(len(train)),
                "validation_samples": int(len(validation)),
                "top1_accuracy": float(np.mean(y_true == pred)),
                "macro_f1": float(f1_score(y_true, pred, labels=HM_PAIR, average="macro", zero_division=0)),
                "hematite_recall": float(recalls[0]),
                "magnetite_recall": float(recalls[1]),
                "hm_min_recall": float(np.min(recalls)),
                "roc_auc_magnetite": auc,
            }
        )
    return rows


def signed_auc(values: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    if len(np.unique(y)) < 2 or float(np.nanstd(values)) < 1e-12:
        return math.nan, 1
    auc = float(roc_auc_score(y, values))
    if auc < 0.5:
        return 1.0 - auc, -1
    return auc, 1


def validation_auc_with_orientation(values: np.ndarray, y: np.ndarray, sign: int) -> float:
    if len(np.unique(y)) < 2 or float(np.nanstd(values)) < 1e-12:
        return math.nan
    return float(roc_auc_score(y, values * sign))


def top_feature_separability(
    group_type: str,
    source_id: str,
    thickness_mm: float | str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    top_n: int,
) -> list[dict]:
    cols = feature_columns(pd.concat([train, validation], ignore_index=True))
    if not cols:
        return []
    y_train = label_vector(train)
    y_validation = label_vector(validation)
    rows = []
    for col in cols:
        train_values = train[col].to_numpy(dtype=float)
        validation_values = validation[col].to_numpy(dtype=float)
        train_auc, sign = signed_auc(train_values, y_train)
        if math.isnan(train_auc):
            continue
        h = train.loc[train["material"] == "Hematite", col].to_numpy(dtype=float)
        m = train.loc[train["material"] == "Magnetite", col].to_numpy(dtype=float)
        pooled = math.sqrt((float(np.var(h)) + float(np.var(m))) / 2.0)
        cohen_d = (float(np.mean(m)) - float(np.mean(h))) / pooled if pooled > 1e-12 else math.nan
        rows.append(
            {
                "group_type": group_type,
                "source_id": source_id,
                "energy_keV": source_energy_kev(source_id),
                "thickness_mm": thickness_mm,
                "feature": col,
                "feature_family": v2.feature_family(col),
                "train_oriented_auc": train_auc,
                "validation_auc_same_orientation": validation_auc_with_orientation(
                    validation_values,
                    y_validation,
                    sign,
                ),
                "orientation_sign_for_magnetite": sign,
                "train_cohen_d_magnetite_minus_hematite": cohen_d,
            }
        )
    rows.sort(key=lambda row: row["train_oriented_auc"], reverse=True)
    return rows[:top_n]


def split_frame(frame: pd.DataFrame, train_seeds: list[int], validation_seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_series = frame["random_seed"].astype(int)
    train = frame[seed_series.isin(train_seeds)].copy()
    validation = frame[seed_series.isin(validation_seeds)].copy()
    return train, validation


def source_rows(frame: pd.DataFrame, train_seeds: list[int], validation_seeds: list[int], random_state: int) -> pd.DataFrame:
    rows = []
    for source_id, part in frame.groupby("source_id"):
        train, validation = split_frame(part, train_seeds, validation_seeds)
        rows.extend(score_pairwise_group("single_source_all_thickness", str(source_id), "all", train, validation, random_state))
    return pd.DataFrame(rows).sort_values(["hm_min_recall", "roc_auc_magnetite"], ascending=[False, False])


def source_thickness_rows(frame: pd.DataFrame, train_seeds: list[int], validation_seeds: list[int], random_state: int) -> pd.DataFrame:
    rows = []
    for (source_id, thickness), part in frame.groupby(["source_id", "thickness_mm"]):
        train, validation = split_frame(part, train_seeds, validation_seeds)
        rows.extend(score_pairwise_group("single_source_fixed_thickness", str(source_id), float(thickness), train, validation, random_state))
    return pd.DataFrame(rows).sort_values(["hm_min_recall", "roc_auc_magnetite"], ascending=[False, False])


def separability_rows(frame: pd.DataFrame, train_seeds: list[int], validation_seeds: list[int], top_n: int) -> pd.DataFrame:
    rows = []
    for source_id, part in frame.groupby("source_id"):
        train, validation = split_frame(part, train_seeds, validation_seeds)
        rows.extend(top_feature_separability("single_source_all_thickness", str(source_id), "all", train, validation, top_n))
    for (source_id, thickness), part in frame.groupby(["source_id", "thickness_mm"]):
        train, validation = split_frame(part, train_seeds, validation_seeds)
        rows.extend(top_feature_separability("single_source_fixed_thickness", str(source_id), float(thickness), train, validation, top_n))
    return pd.DataFrame(rows).sort_values(["validation_auc_same_orientation", "train_oriented_auc"], ascending=[False, False])


def fused_rows(frame: pd.DataFrame, train_seeds: list[int], validation_seeds: list[int], random_state: int) -> pd.DataFrame:
    fused, table_mode = v2.fuse_sources(frame)
    rows = []
    train, validation = split_frame(fused, train_seeds, validation_seeds)
    for row in score_pairwise_group("all_sources_fused", table_mode, "all", train, validation, random_state):
        row["table_mode"] = table_mode
        rows.append(row)
    for thickness, part in fused.groupby("thickness_mm"):
        train_part, validation_part = split_frame(part, train_seeds, validation_seeds)
        for row in score_pairwise_group("all_sources_fused_fixed_thickness", table_mode, float(thickness), train_part, validation_part, random_state):
            row["table_mode"] = table_mode
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["hm_min_recall", "roc_auc_magnetite"], ascending=[False, False])


def validation_error_by_thickness(decisions_path: Path) -> pd.DataFrame:
    if not decisions_path.exists():
        return pd.DataFrame()
    decisions = pd.read_csv(decisions_path)
    hm = decisions[decisions["material"].isin(HM_PAIR)].copy()
    if hm.empty:
        return pd.DataFrame()
    grouped = (
        hm.groupby(["material", "thickness_mm", "predicted_material"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["material", "thickness_mm", "predicted_material"])
    )
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="Development-only H/M energy-thickness diagnostic for accuracy sprint v6.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/v5_hm_lowwide")
    parser.add_argument("--raw-dirs", default="")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v5_hm_lowwide/hm_diagnostic")
    parser.add_argument("--train-seeds", default="1501,1502,1503,1504,1505,1506,1507,1508,1509,1510,1511,1512,1513,1514,1515,1516,1517,1518,1519,1520")
    parser.add_argument("--validation-seeds", default="1601,1602,1603,1604,1605,1606,1607,1608,1609,1610")
    parser.add_argument("--photon-budget", type=int, default=5000)
    parser.add_argument("--top-features", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validation-decisions", default="results/accuracy_v3/v5_hm_lowwide/validation_decisions.csv")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = parse_int_list(args.train_seeds)
    validation_seeds = parse_int_list(args.validation_seeds)
    raw_dirs = parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs)

    frame, inventory = build_calibrated_hm_frame(project_root, raw_dirs, train_seeds + validation_seeds, args.photon_budget)

    source_table = source_rows(frame, train_seeds, validation_seeds, args.random_state)
    source_thickness_table = source_thickness_rows(frame, train_seeds, validation_seeds, args.random_state)
    separability_table = separability_rows(frame, train_seeds, validation_seeds, args.top_features)
    fused_table = fused_rows(frame, train_seeds, validation_seeds, args.random_state)
    error_table = validation_error_by_thickness(project_root / args.validation_decisions)

    write_csv(source_table, output_dir / "hm_single_source_pairwise_models.csv")
    write_csv(source_thickness_table, output_dir / "hm_single_source_thickness_pairwise_models.csv")
    write_csv(separability_table, output_dir / "hm_feature_separability.csv")
    write_csv(fused_table, output_dir / "hm_fused_pairwise_by_thickness.csv")
    if not error_table.empty:
        write_csv(error_table, output_dir / "hm_validation_error_by_thickness.csv")

    manifest = {
        "generated_by": "analysis/hm_energy_thickness_diagnostic_v6.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "shadow_or_final_used": False,
        "raw_dirs": [path.as_posix() for path in raw_dirs],
        "output_dir": args.output_dir,
        "photon_budget": args.photon_budget,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "inventory": inventory,
        "outputs": {
            "single_source": "hm_single_source_pairwise_models.csv",
            "single_source_thickness": "hm_single_source_thickness_pairwise_models.csv",
            "feature_separability": "hm_feature_separability.csv",
            "fused_pairwise": "hm_fused_pairwise_by_thickness.csv",
            "validation_error_by_thickness": "hm_validation_error_by_thickness.csv" if not error_table.empty else "",
        },
    }
    write_json(manifest, output_dir / "hm_diagnostic_manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
