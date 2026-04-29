from __future__ import annotations

import argparse
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import material_sorting_selected_rebuild as selected
import material_sorting_v2 as v2


DEFAULT_METHODS = [
    "ExtraTrees",
    "HistGradientBoosting",
    "HierarchicalExtraTrees",
    "HMExpertHierarchicalExtraTrees",
    "HematiteMagnetiteRecallExtraTrees",
    "HematitePriorityExtraTrees",
    "HMStrongRecallExtraTrees",
    "HighGroupRecallExtraTrees",
    "HMAllEnergyExpertExtraTrees",
    "HMLowEnergyExpertExtraTrees",
    "HMThicknessGroupedExpertExtraTrees",
    "HMPairwiseVotingExpertExtraTrees",
    "XGBoostGPU",
]
HM_PAIR = ["Hematite", "Magnetite"]
ACCEPTANCE_TARGETS = {
    "top1_accuracy": 0.88,
    "macro_f1": 0.84,
    "min_class_recall": 0.75,
}
HM_DIFFERENTIAL_SOURCES = ["mono_120kev", "mono_150kev", "mono_200kev"]
HM_DIFFERENTIAL_SOURCE_PAIRS = [
    ("mono_200kev", "mono_150kev"),
    ("mono_200kev", "mono_120kev"),
    ("mono_150kev", "mono_120kev"),
]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def parse_raw_dirs(project_root: Path, raw_dir: str, raw_dirs: str) -> list[Path]:
    values = parse_str_list(raw_dirs) if raw_dirs.strip() else [raw_dir]
    return [project_root / value for value in values]


def discover_status(material_records: list[v2.RunRecord], calibration_records: list[v2.RunRecord], rows: int, table_mode: str) -> dict:
    material_keys = {
        (record.material, round(record.thickness_mm, 3), record.source_id, record.random_seed)
        for record in material_records
    }
    calibration_keys = {(record.source_id, record.random_seed) for record in calibration_records}
    duplicate_material_keys = len(material_keys) != len(material_records)
    duplicate_calibration_keys = len(calibration_keys) != len(calibration_records)
    return {
        "material_metadata_found": len(material_records),
        "calibration_metadata_found": len(calibration_records),
        "materials_found": sorted({record.material for record in material_records}),
        "sources_found": sorted({record.source_id for record in material_records}),
        "thicknesses_found": sorted({record.thickness_mm for record in material_records}),
        "seeds_found": sorted({record.random_seed for record in material_records}),
        "duplicate_material_keys": duplicate_material_keys,
        "duplicate_calibration_keys": duplicate_calibration_keys,
        "table_mode": table_mode,
        "rows": int(rows),
    }


def build_frame_from_raw_dirs(project_root: Path, raw_dirs: list[Path], photons_per_sample: int) -> tuple[pd.DataFrame, dict]:
    old_budget = v2.PHOTONS_PER_SAMPLE
    v2.PHOTONS_PER_SAMPLE = photons_per_sample
    try:
        material_records: list[v2.RunRecord] = []
        calibration_records: list[v2.RunRecord] = []
        for raw_dir in raw_dirs:
            material_part, calibration_part = v2.discover_records(project_root, raw_dir)
            material_records.extend(material_part)
            calibration_records.extend(calibration_part)
        if not material_records:
            raise ValueError(f"No material records found in raw dirs: {[path.as_posix() for path in raw_dirs]}")
        calibration = v2.calibration_table(calibration_records)
        samples = pd.concat([v2.aggregate_run(record) for record in material_records], ignore_index=True)
        calibrated = v2.apply_calibration(samples, calibration)
        fused, table_mode = v2.fuse_sources(calibrated)
        return fused, discover_status(material_records, calibration_records, len(fused), table_mode)
    finally:
        v2.PHOTONS_PER_SAMPLE = old_budget


def split_audit_frame(frame: pd.DataFrame, train_seeds: list[int], validation_seeds: list[int], test_seeds: list[int]) -> pd.DataFrame:
    roles = {}
    for seed in train_seeds:
        roles[int(seed)] = "train"
    for seed in validation_seeds:
        roles[int(seed)] = "validation"
    for seed in test_seeds:
        roles[int(seed)] = "test"
    audit = frame.copy()
    audit["random_seed"] = audit["random_seed"].astype(int)
    audit["split_role"] = audit["random_seed"].map(roles).fillna("unused")
    return (
        audit.groupby(["split_role", "random_seed", "material"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(["split_role", "random_seed", "material"])
    )


def check_split_integrity(train_seeds: list[int], validation_seeds: list[int], test_seeds: list[int]) -> dict:
    train_set = set(train_seeds)
    validation_set = set(validation_seeds)
    test_set = set(test_seeds)
    return {
        "train_validation_overlap": sorted(train_set & validation_set),
        "train_test_overlap": sorted(train_set & test_set),
        "validation_test_overlap": sorted(validation_set & test_set),
        "split_is_disjoint": not (train_set & validation_set or train_set & test_set or validation_set & test_set),
    }


def evaluation_labels(frame: pd.DataFrame) -> np.ndarray:
    observed = set(frame["material"].astype(str))
    if observed == set(v2.TARGET_MATERIALS):
        return np.array(v2.TARGET_MATERIALS)
    return np.array([material for material in v2.TARGET_MATERIALS if material in observed])


def evaluate_context_scores(
    method: str,
    frame: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sk,
) -> dict:
    labels = evaluation_labels(frame)
    y_true = frame["material"].astype(str).to_numpy()
    recalls = sk["recall_score"](y_true, predictions, labels=labels, average=None, zero_division=0)
    return {
        "method": method,
        "samples": int(len(frame)),
        "evaluated_materials": ";".join(labels),
        "top1_accuracy": float(np.mean(y_true == predictions)),
        "top3_accuracy": v2.topk_accuracy(y_true, scores, classes, min(3, len(classes))),
        "macro_f1": float(sk["f1_score"](y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "min_class_recall": float(np.min(recalls)) if len(recalls) else math.nan,
    }


def per_class_recall_context(frame: pd.DataFrame, predictions: np.ndarray, split: str, sk) -> pd.DataFrame:
    labels = evaluation_labels(frame)
    y_true = frame["material"].astype(str).to_numpy()
    recalls = sk["recall_score"](y_true, predictions, labels=labels, average=None, zero_division=0)
    support = frame["material"].value_counts().to_dict()
    return pd.DataFrame(
        [
            {
                "split": split,
                "material": material,
                "support": int(support.get(material, 0)),
                "recall": float(recall),
            }
            for material, recall in zip(labels, recalls)
        ]
    )


def hm_expert_feature_columns(feature_cols: list[str]) -> list[str]:
    preferred_families = {
        "attenuation",
        "thickness_normalized_attenuation",
        "spectral_shape",
        "scatter_direct",
        "detector_response",
        "source_fusion",
        "dictionary_distance",
    }
    cols = [
        col
        for col in feature_cols
        if v2.feature_family(col) in preferred_families or col.startswith("hmphys__")
    ]
    return cols or feature_cols


def log_positive(value: pd.Series) -> pd.Series:
    return np.log(np.clip(value.astype(float), 1e-6, 1e6))


def log1p_positive(value: pd.Series) -> pd.Series:
    return np.log1p(value.astype(float).clip(lower=0.0))


def add_hm_differential_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add a small pre-registered set of H/M high-energy tail contrast features."""
    result = frame.copy()
    created: list[str] = []

    def add_column(name: str, values: pd.Series) -> None:
        result[name] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        created.append(name)

    def has_col(source: str, feature: str) -> bool:
        return f"{source}__{feature}" in result.columns

    def col(source: str, feature: str) -> str:
        return f"{source}__{feature}"

    for high_source, low_source in HM_DIFFERENTIAL_SOURCE_PAIRS:
        for feature in ["T_e_120_inf", "primary_transmission_rate", "detector_gamma_rate"]:
            if has_col(high_source, feature) and has_col(low_source, feature):
                add_column(
                    f"hmphys__log_ratio__{high_source}__over__{low_source}__{feature}",
                    log_positive(result[col(high_source, feature)]) - log_positive(result[col(low_source, feature)]),
                )
        for feature in ["A_e_120_inf", "A_per_mm_e_120_inf"]:
            if has_col(high_source, feature) and has_col(low_source, feature):
                add_column(
                    f"hmphys__delta__{high_source}__minus__{low_source}__{feature}",
                    result[col(high_source, feature)].astype(float) - result[col(low_source, feature)].astype(float),
                )
        for feature in ["Rsig020_e_120_inf", "Rsig050_e_120_inf", "direct_primary_count"]:
            if has_col(high_source, feature) and has_col(low_source, feature):
                add_column(
                    f"hmphys__log1p_ratio__{high_source}__over__{low_source}__{feature}",
                    log1p_positive(result[col(high_source, feature)]) - log1p_positive(result[col(low_source, feature)]),
                )

    for source in HM_DIFFERENTIAL_SOURCES:
        if has_col(source, "T_e_120_inf") and has_col(source, "T_low_0_80"):
            add_column(
                f"hmphys__tail_over_low__{source}__T_e_120_inf",
                log_positive(result[col(source, "T_e_120_inf")]) - log_positive(result[col(source, "T_low_0_80")]),
            )
        for sigma in ["Rsig020", "Rsig050"]:
            tail = f"{sigma}_e_120_inf"
            shoulder = f"{sigma}_e_080_090"
            if has_col(source, tail) and has_col(source, shoulder):
                add_column(
                    f"hmphys__tail_over_shoulder__{source}__{sigma}",
                    log1p_positive(result[col(source, tail)]) - log1p_positive(result[col(source, shoulder)]),
                )
        if has_col(source, "direct_primary_count") and has_col(source, "scattered_primary_count"):
            add_column(
                f"hmphys__direct_minus_scatter__{source}",
                log1p_positive(result[col(source, "direct_primary_count")])
                - log1p_positive(result[col(source, "scattered_primary_count")]),
            )

    return result, created


def add_hm_metrics(metrics: dict, eval_frame: pd.DataFrame, predictions: np.ndarray, sk) -> dict:
    per_class = v2.per_class_recall_table(eval_frame, predictions, "eval", sk)
    recall_by_material = {
        str(row.material): float(row.recall)
        for row in per_class.itertuples(index=False)
    }
    metrics = dict(metrics)
    metrics["hematite_recall"] = recall_by_material.get("Hematite", math.nan)
    metrics["magnetite_recall"] = recall_by_material.get("Magnetite", math.nan)
    hm_values = [metrics["hematite_recall"], metrics["magnetite_recall"]]
    metrics["hm_min_recall"] = float(np.nanmin(hm_values)) if not all(math.isnan(value) for value in hm_values) else math.nan
    return metrics


def source_energy_from_feature(col: str) -> float | None:
    if "__" not in col:
        return None
    source_id = col.split("__", 1)[0]
    if not (source_id.startswith("mono_") and source_id.endswith("kev")):
        return None
    try:
        return float(source_id.removeprefix("mono_").removesuffix("kev"))
    except ValueError:
        return None


def low_energy_hm_feature_columns(feature_cols: list[str], max_energy_keV: float = 50.0) -> list[str]:
    hm_cols = hm_expert_feature_columns(feature_cols)
    cols = []
    for col in hm_cols:
        energy = source_energy_from_feature(col)
        if energy is not None and energy <= max_energy_keV:
            cols.append(col)
        elif energy is None and col.startswith("dict_"):
            cols.append(col)
    return cols or hm_cols


def fit_hm_pairwise_extra_trees(
    train: pd.DataFrame,
    feature_cols: list[str],
    sk,
    random_state: int,
):
    train_hm = train[train["material"].astype(str).isin(HM_PAIR)].copy()
    if train_hm["material"].nunique() < 2:
        return None
    model = sk["ExtraTreesClassifier"](
        n_estimators=1800,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )
    model.fit(train_hm[feature_cols], train_hm["material"])
    return model


def hm_probability_matrix(model, eval_frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    probs = np.full((len(eval_frame), len(HM_PAIR)), 0.5, dtype=float)
    if model is None or eval_frame.empty:
        return probs
    raw = model.predict_proba(eval_frame[feature_cols])
    probs = np.zeros((len(eval_frame), len(HM_PAIR)), dtype=float)
    for local_index, material in enumerate(model.classes_):
        if str(material) in HM_PAIR:
            probs[:, HM_PAIR.index(str(material))] = raw[:, local_index]
    sums = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, sums, out=np.full_like(probs, 0.5), where=sums > 0)


def hm_pairwise_probabilities(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    sk,
    random_state: int,
) -> np.ndarray:
    model = fit_hm_pairwise_extra_trees(train, feature_cols, sk, random_state)
    return hm_probability_matrix(model, eval_frame, feature_cols)


def thickness_group_hm_probabilities(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    sk,
    random_state: int,
) -> np.ndarray:
    probs = hm_pairwise_probabilities(train, eval_frame, feature_cols, sk, random_state)
    train_hm = train[train["material"].astype(str).isin(HM_PAIR)].copy()
    if train_hm.empty or "thickness_mm" not in train_hm.columns:
        return probs
    for offset, thickness in enumerate(sorted(train_hm["thickness_mm"].astype(float).unique())):
        part = train_hm[np.isclose(train_hm["thickness_mm"].astype(float), thickness)].copy()
        if part["material"].nunique() < 2:
            continue
        mask = np.isclose(eval_frame["thickness_mm"].astype(float), thickness)
        if not mask.any():
            continue
        model = fit_hm_pairwise_extra_trees(part, feature_cols, sk, random_state + offset + 1)
        probs[mask, :] = hm_probability_matrix(model, eval_frame.loc[mask], feature_cols)
    return probs


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    sums = scores.sum(axis=1, keepdims=True)
    return np.divide(scores, sums, out=np.zeros_like(scores), where=sums > 0)


def redistribute_hm_mass(base_scores: np.ndarray, base_classes: np.ndarray, hm_probs: np.ndarray) -> np.ndarray:
    if "Hematite" not in base_classes or "Magnetite" not in base_classes:
        return normalize_scores(np.array(base_scores, copy=True))
    scores = np.array(base_scores, copy=True)
    h_index = int(np.where(base_classes == "Hematite")[0][0])
    m_index = int(np.where(base_classes == "Magnetite")[0][0])
    hm_mass = scores[:, h_index] + scores[:, m_index]
    scores[:, h_index] = hm_mass * hm_probs[:, 0]
    scores[:, m_index] = hm_mass * hm_probs[:, 1]
    return normalize_scores(scores)


def score_hm_pairwise_ensemble_extra_trees(
    method: str,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    sk,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    all_hm_cols = hm_expert_feature_columns(feature_cols)
    low_hm_cols = low_energy_hm_feature_columns(feature_cols)
    base_cols = low_hm_cols if method == "HMLowEnergyExpertExtraTrees" else feature_cols
    random_state = {
        "HMAllEnergyExpertExtraTrees": 353,
        "HMLowEnergyExpertExtraTrees": 359,
        "HMThicknessGroupedExpertExtraTrees": 367,
        "HMPairwiseVotingExpertExtraTrees": 373,
    }[method]
    base_model = sk["ExtraTreesClassifier"](
        n_estimators=2000,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )
    base_model.fit(train[base_cols], train["material"])
    base_scores = base_model.predict_proba(eval_frame[base_cols])
    classes = np.array(base_model.classes_)

    if method == "HMAllEnergyExpertExtraTrees":
        hm_probs = hm_pairwise_probabilities(train, eval_frame, all_hm_cols, sk, 383)
    elif method == "HMLowEnergyExpertExtraTrees":
        hm_probs = hm_pairwise_probabilities(train, eval_frame, low_hm_cols, sk, 389)
    elif method == "HMThicknessGroupedExpertExtraTrees":
        hm_probs = thickness_group_hm_probabilities(train, eval_frame, all_hm_cols, sk, 397)
    else:
        hm_votes = [
            hm_pairwise_probabilities(train, eval_frame, all_hm_cols, sk, 401),
            hm_pairwise_probabilities(train, eval_frame, low_hm_cols, sk, 409),
            thickness_group_hm_probabilities(train, eval_frame, all_hm_cols, sk, 419),
        ]
        hm_probs = normalize_scores(np.mean(hm_votes, axis=0))

    scores = redistribute_hm_mass(base_scores, classes, hm_probs)
    predictions = classes[np.argmax(scores, axis=1)]
    metrics = add_hm_metrics(evaluate_context_scores(method, eval_frame, predictions, scores, classes, sk), eval_frame, predictions, sk)
    metrics["hm_expert_feature_count"] = int(len(all_hm_cols))
    metrics["hm_low_energy_feature_count"] = int(len(low_hm_cols))
    metrics["base_model_feature_count"] = int(len(base_cols))
    return metrics, predictions, scores, classes


def score_weighted_extra_trees(
    method: str,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    sk,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    class_weight = {
        "Quartz": 1.0,
        "Calcite": 1.0,
        "Orthoclase": 1.0,
        "Albite": 1.0,
        "Dolomite": 1.0,
        "Pyrite": 1.5,
        "Hematite": 4.0,
        "Magnetite": 4.0,
        "Chalcopyrite": 1.5,
        "Galena": 1.0,
    }
    random_state = 137
    if method == "HematitePriorityExtraTrees":
        class_weight.update({"Hematite": 8.0, "Magnetite": 3.0, "Pyrite": 1.2, "Chalcopyrite": 1.2})
        random_state = 271
    elif method == "HMStrongRecallExtraTrees":
        class_weight.update({"Hematite": 8.0, "Magnetite": 8.0, "Pyrite": 1.0, "Chalcopyrite": 1.0})
        random_state = 277
    model = sk["ExtraTreesClassifier"](
        n_estimators=1600,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
        min_samples_leaf=1,
    )
    model.fit(train[feature_cols], train["material"])
    predictions = model.predict(eval_frame[feature_cols])
    scores = model.predict_proba(eval_frame[feature_cols])
    classes = np.array(model.classes_)
    metrics = add_hm_metrics(evaluate_context_scores(method, eval_frame, predictions, scores, classes, sk), eval_frame, predictions, sk)
    return metrics, predictions, scores, classes


def score_high_group_recall_extra_trees(
    method: str,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    group_map: dict[str, str],
    sk,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    classes = np.array(v2.TARGET_MATERIALS)
    train_groups = train["material"].astype(str).map(group_map)
    group_model = sk["ExtraTreesClassifier"](n_estimators=1200, random_state=151, n_jobs=-1, class_weight="balanced")
    group_model.fit(train[feature_cols], train_groups)
    group_scores = group_model.predict_proba(eval_frame[feature_cols])
    group_classes = np.array(group_model.classes_)
    scores = np.zeros((len(eval_frame), len(classes)), dtype=float)
    for group in group_classes:
        materials = [material for material in classes if group_map.get(material) == group]
        part = train[train["material"].isin(materials)].copy()
        if group == "high_absorption":
            class_weight = {
                "Pyrite": 1.5,
                "Hematite": 4.0,
                "Magnetite": 4.0,
                "Chalcopyrite": 1.5,
                "Galena": 1.0,
            }
            random_state = 157
        else:
            class_weight = "balanced"
            random_state = 163
        material_model = sk["ExtraTreesClassifier"](
            n_estimators=1800,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight,
            min_samples_leaf=1,
        )
        material_model.fit(part[feature_cols], part["material"])
        material_scores = material_model.predict_proba(eval_frame[feature_cols])
        material_classes = np.array(material_model.classes_)
        group_index = np.where(group_classes == group)[0][0]
        for local_index, material in enumerate(material_classes):
            global_index = np.where(classes == material)[0][0]
            scores[:, global_index] = group_scores[:, group_index] * material_scores[:, local_index]
    scores_sum = scores.sum(axis=1, keepdims=True)
    scores = np.divide(scores, scores_sum, out=np.zeros_like(scores), where=scores_sum > 0)
    predictions = classes[np.argmax(scores, axis=1)]
    metrics = add_hm_metrics(evaluate_context_scores(method, eval_frame, predictions, scores, classes, sk), eval_frame, predictions, sk)
    return metrics, predictions, scores, classes


def score_hm_expert_hierarchical_extra_trees(
    method: str,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    group_map: dict[str, str],
    sk,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    classes = np.array(v2.TARGET_MATERIALS)
    train_groups = train["material"].astype(str).map(group_map)
    group_model = sk["ExtraTreesClassifier"](n_estimators=1400, random_state=211, n_jobs=-1, class_weight="balanced")
    group_model.fit(train[feature_cols], train_groups)
    group_scores = group_model.predict_proba(eval_frame[feature_cols])
    group_classes = np.array(group_model.classes_)
    scores = np.zeros((len(eval_frame), len(classes)), dtype=float)

    for group in group_classes:
        materials = [material for material in classes if group_map.get(material) == group]
        part = train[train["material"].isin(materials)].copy()
        if group == "high_absorption":
            class_weight = {
                "Pyrite": 1.5,
                "Hematite": 3.0,
                "Magnetite": 3.0,
                "Chalcopyrite": 1.5,
                "Galena": 1.0,
            }
            random_state = 223
        else:
            class_weight = "balanced"
            random_state = 227
        material_model = sk["ExtraTreesClassifier"](
            n_estimators=1800,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight,
            min_samples_leaf=1,
            max_features="sqrt",
        )
        material_model.fit(part[feature_cols], part["material"])
        material_scores = material_model.predict_proba(eval_frame[feature_cols])
        material_classes = np.array(material_model.classes_)
        group_index = np.where(group_classes == group)[0][0]
        for local_index, material in enumerate(material_classes):
            global_index = np.where(classes == material)[0][0]
            scores[:, global_index] = group_scores[:, group_index] * material_scores[:, local_index]

    hm_train = train[train["material"].isin(HM_PAIR)].copy()
    if hm_train["material"].nunique() == 2:
        hm_cols = hm_expert_feature_columns(feature_cols)
        hm_model = sk["ExtraTreesClassifier"](
            n_estimators=2400,
            random_state=239,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=1,
            max_features="sqrt",
        )
        hm_model.fit(hm_train[hm_cols], hm_train["material"])
        hm_scores = hm_model.predict_proba(eval_frame[hm_cols])
        hm_classes = np.array(hm_model.classes_)
        h_index = np.where(classes == "Hematite")[0][0]
        m_index = np.where(classes == "Magnetite")[0][0]
        hm_mass = scores[:, h_index] + scores[:, m_index]
        scores[:, h_index] = 0.0
        scores[:, m_index] = 0.0
        for local_index, material in enumerate(hm_classes):
            global_index = np.where(classes == material)[0][0]
            scores[:, global_index] = hm_mass * hm_scores[:, local_index]

    scores_sum = scores.sum(axis=1, keepdims=True)
    scores = np.divide(scores, scores_sum, out=np.zeros_like(scores), where=scores_sum > 0)
    predictions = classes[np.argmax(scores, axis=1)]
    metrics = add_hm_metrics(evaluate_context_scores(method, eval_frame, predictions, scores, classes, sk), eval_frame, predictions, sk)
    metrics["hm_expert_feature_count"] = len(hm_expert_feature_columns(feature_cols))
    return metrics, predictions, scores, classes


def score_method(
    method: str,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    group_map: dict[str, str],
    sk,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    if method in {"HematiteMagnetiteRecallExtraTrees", "HematitePriorityExtraTrees", "HMStrongRecallExtraTrees"}:
        return score_weighted_extra_trees(method, train, eval_frame, feature_cols, sk)
    if method == "HighGroupRecallExtraTrees":
        return score_high_group_recall_extra_trees(method, train, eval_frame, feature_cols, group_map, sk)
    if method == "HMExpertHierarchicalExtraTrees":
        return score_hm_expert_hierarchical_extra_trees(method, train, eval_frame, feature_cols, group_map, sk)
    if method in {
        "HMAllEnergyExpertExtraTrees",
        "HMLowEnergyExpertExtraTrees",
        "HMThicknessGroupedExpertExtraTrees",
        "HMPairwiseVotingExpertExtraTrees",
    }:
        return score_hm_pairwise_ensemble_extra_trees(method, train, eval_frame, feature_cols, sk)
    _, predictions, scores, classes, _ = selected.score_method(method, train, eval_frame, feature_cols, group_map, sk)
    metrics = evaluate_context_scores(method, eval_frame, predictions, scores, classes, sk)
    return add_hm_metrics(metrics, eval_frame, predictions, sk), predictions, scores, classes


def score_methods(
    methods: list[str],
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    group_map: dict[str, str],
    sk,
) -> pd.DataFrame:
    rows = []
    for method in methods:
        try:
            metrics, _, _, _ = score_method(method, train, eval_frame, feature_cols, group_map, sk)
        except Exception as exc:  # noqa: BLE001
            metrics = {
                "method": method,
                "samples": int(len(eval_frame)),
                "top1_accuracy": math.nan,
                "top3_accuracy": math.nan,
                "macro_f1": math.nan,
                "min_class_recall": math.nan,
                "hematite_recall": math.nan,
                "magnetite_recall": math.nan,
                "hm_min_recall": math.nan,
                "error": str(exc),
            }
        rows.append(metrics)
    return pd.DataFrame(rows)


def choose_validation_method(validation_table: pd.DataFrame) -> dict:
    ranked = validation_table.dropna(subset=["top1_accuracy", "macro_f1", "min_class_recall"]).sort_values(
        ["hm_min_recall", "min_class_recall", "macro_f1", "top1_accuracy"],
        ascending=[False, False, False, False],
    )
    if ranked.empty:
        raise RuntimeError("No validation method produced finite metrics.")
    return ranked.iloc[0].to_dict()


def evaluate_locked_split(
    frame: pd.DataFrame,
    train_seeds: list[int],
    validation_seeds: list[int],
    test_seeds: list[int],
    methods: list[str],
    group_map: dict[str, str],
    sk,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    seed_series = frame["random_seed"].astype(int)
    train = frame[seed_series.isin(train_seeds)].copy()
    validation = frame[seed_series.isin(validation_seeds)].copy()
    test = frame[seed_series.isin(test_seeds)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty.")

    base_cols = v2.numeric_feature_columns(frame)
    train_aug, validation_aug, feature_cols, _ = selected.append_dictionary(train, validation, base_cols)
    validation_table = score_methods(methods, train_aug, validation_aug, feature_cols, group_map, sk)
    selected_method = str(choose_validation_method(validation_table)["method"])

    final_train = pd.concat([train, validation], ignore_index=True)
    final_train_aug, test_aug, final_feature_cols, _ = selected.append_dictionary(final_train, test, base_cols)
    final_metrics, predictions, scores, classes = score_method(
        selected_method,
        final_train_aug,
        test_aug,
        final_feature_cols,
        group_map,
        sk,
    )
    final_metrics["base_feature_count"] = int(len(base_cols))
    final_metrics["model_feature_count"] = int(len(final_feature_cols))
    per_class = per_class_recall_context(test_aug, predictions, "test", sk)
    decisions = v2.decision_frame(test_aug, predictions, scores, classes, probability_threshold=0.0, margin_threshold=0.0)
    return validation_table, final_metrics, per_class, decisions


def pairwise_top_features(model, feature_cols: list[str], limit: int = 15) -> tuple[str, str, str]:
    if not hasattr(model, "feature_importances_"):
        return "", "", ""
    importances = np.asarray(model.feature_importances_, dtype=float)
    if importances.size != len(feature_cols):
        return "", "", ""
    order = np.argsort(importances)[::-1][:limit]
    top = [(feature_cols[index], float(importances[index])) for index in order if importances[index] > 0]
    top_features = ";".join(f"{name}:{value:.4f}" for name, value in top)
    source_scores: dict[str, float] = {}
    family_scores: dict[str, float] = {}
    for name, value in top:
        source = name.split("__", 1)[0] if "__" in name else "global"
        source_scores[source] = source_scores.get(source, 0.0) + value
        family = v2.feature_family(name)
        family_scores[family] = family_scores.get(family, 0.0) + value
    top_sources = ";".join(f"{name}:{value:.4f}" for name, value in sorted(source_scores.items(), key=lambda item: item[1], reverse=True))
    top_families = ";".join(f"{name}:{value:.4f}" for name, value in sorted(family_scores.items(), key=lambda item: item[1], reverse=True))
    return top_features, top_sources, top_families


def hm_pairwise_audit_frame(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    sk,
    split: str,
) -> pd.DataFrame:
    train_hm = train[train["material"].astype(str).isin(HM_PAIR)].copy()
    eval_hm = eval_frame[eval_frame["material"].astype(str).isin(HM_PAIR)].copy()
    if train_hm["material"].nunique() < 2 or eval_hm.empty:
        return pd.DataFrame(
            [
                {
                    "split": split,
                    "method": "HMPairwiseExtraTrees",
                    "samples": int(len(eval_hm)),
                    "status": "insufficient_hm_classes",
                }
            ]
        )
    hm_cols = hm_expert_feature_columns(feature_cols)
    model = sk["ExtraTreesClassifier"](
        n_estimators=1600,
        random_state=331,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )
    model.fit(train_hm[hm_cols], train_hm["material"])
    predictions = model.predict(eval_hm[hm_cols])
    scores = model.predict_proba(eval_hm[hm_cols])
    classes = np.array(model.classes_)
    y_true = eval_hm["material"].astype(str).to_numpy()
    recalls = sk["recall_score"](y_true, predictions, labels=np.array(HM_PAIR), average=None, zero_division=0)
    roc_auc = math.nan
    try:
        from sklearn.metrics import roc_auc_score

        magnetite_index = int(np.where(classes == "Magnetite")[0][0])
        roc_auc = float(roc_auc_score((y_true == "Magnetite").astype(int), scores[:, magnetite_index]))
    except Exception:
        roc_auc = math.nan
    decisions = v2.decision_frame(eval_hm, predictions, scores, classes, probability_threshold=0.0, margin_threshold=0.0)
    confusion = decisions.loc[~decisions["is_correct"], "predicted_material"].value_counts().to_dict()
    top_features, top_sources, top_families = pairwise_top_features(model, hm_cols)
    return pd.DataFrame(
        [
            {
                "split": split,
                "method": "HMPairwiseExtraTrees",
                "samples": int(len(eval_hm)),
                "status": "ok",
                "top1_accuracy": float(np.mean(y_true == predictions)),
                "macro_f1": float(sk["f1_score"](y_true, predictions, labels=np.array(HM_PAIR), average="macro", zero_division=0)),
                "hematite_recall": float(recalls[0]),
                "magnetite_recall": float(recalls[1]),
                "hm_min_recall": float(np.min(recalls)),
                "roc_auc_magnetite": roc_auc,
                "feature_count": int(len(hm_cols)),
                "confusions": ";".join(f"{name}:{int(count)}" for name, count in confusion.items()),
                "top_features": top_features,
                "top_sources": top_sources,
                "top_feature_families": top_families,
            }
        ]
    )


def build_augmented_splits(
    frame: pd.DataFrame,
    train_seeds: list[int],
    validation_seeds: list[int],
    test_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    seed_series = frame["random_seed"].astype(int)
    train = frame[seed_series.isin(train_seeds)].copy()
    validation = frame[seed_series.isin(validation_seeds)].copy()
    test = frame[seed_series.isin(test_seeds)].copy() if test_seeds else pd.DataFrame()
    base_cols = v2.numeric_feature_columns(frame)
    train_aug, validation_aug, feature_cols, _ = selected.append_dictionary(train, validation, base_cols)
    if not test.empty:
        final_train = pd.concat([train, validation], ignore_index=True)
        final_train_aug, test_aug, final_feature_cols, _ = selected.append_dictionary(final_train, test, base_cols)
    else:
        final_train_aug = pd.DataFrame()
        test_aug = pd.DataFrame()
        final_feature_cols = feature_cols
    return train_aug, validation_aug, test_aug, feature_cols, final_feature_cols


def evaluate_development_split(
    frame: pd.DataFrame,
    train_seeds: list[int],
    validation_seeds: list[int],
    methods: list[str],
    group_map: dict[str, str],
    sk,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_series = frame["random_seed"].astype(int)
    train = frame[seed_series.isin(train_seeds)].copy()
    validation = frame[seed_series.isin(validation_seeds)].copy()
    if train.empty or validation.empty:
        raise ValueError("Train and validation splits must be non-empty for development-only audit.")
    base_cols = v2.numeric_feature_columns(frame)
    train_aug, validation_aug, feature_cols, _ = selected.append_dictionary(train, validation, base_cols)
    validation_table = score_methods(methods, train_aug, validation_aug, feature_cols, group_map, sk)
    selected_method = str(choose_validation_method(validation_table)["method"])
    validation_metrics, predictions, scores, classes = score_method(
        selected_method,
        train_aug,
        validation_aug,
        feature_cols,
        group_map,
        sk,
    )
    validation_metrics["base_feature_count"] = int(len(base_cols))
    validation_metrics["model_feature_count"] = int(len(feature_cols))
    per_class = per_class_recall_context(validation_aug, predictions, "validation", sk)
    decisions = v2.decision_frame(validation_aug, predictions, scores, classes, probability_threshold=0.0, margin_threshold=0.0)
    pairwise = hm_pairwise_audit_frame(train_aug, validation_aug, feature_cols, sk, "validation")
    return validation_table, validation_metrics, per_class, decisions, pairwise


def evaluate_rotating_splits(
    frame: pd.DataFrame,
    methods: list[str],
    group_map: dict[str, str],
    sk,
) -> pd.DataFrame:
    seeds = sorted(frame["random_seed"].astype(int).unique())
    rows = []
    for index, test_seed in enumerate(seeds):
        validation_seed = seeds[(index + 1) % len(seeds)]
        train_seeds = [seed for seed in seeds if seed not in {test_seed, validation_seed}]
        validation_table, final_metrics, per_class, _ = evaluate_locked_split(
            frame,
            train_seeds,
            [validation_seed],
            [test_seed],
            methods,
            group_map,
            sk,
        )
        selected_method = str(choose_validation_method(validation_table)["method"])
        worst = per_class.sort_values(["recall", "material"]).head(3)
        rows.append(
            {
                "test_seed": int(test_seed),
                "validation_seed": int(validation_seed),
                "train_seeds": ";".join(str(seed) for seed in train_seeds),
                "selected_method": selected_method,
                "top1_accuracy": final_metrics["top1_accuracy"],
                "top3_accuracy": final_metrics["top3_accuracy"],
                "macro_f1": final_metrics["macro_f1"],
                "min_class_recall": final_metrics["min_class_recall"],
                "worst_classes": ";".join(
                    f"{row.material}:{float(row.recall):.3f}" for row in worst.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows)


def common_confusions(decisions: pd.DataFrame, material: str) -> dict[str, int]:
    part = decisions[decisions["material"].astype(str).eq(material)]
    if part.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in part.loc[~part["is_correct"], "predicted_material"].value_counts().head(5).to_dict().items()
    }


def physics_hypothesis(material: str, confusions: dict[str, int], group_map: dict[str, str]) -> str:
    confused_materials = set(confusions)
    if material in HM_PAIR or confused_materials.intersection(HM_PAIR):
        return "iron_oxide_pair_similarity_current_spectral_features_insufficient"
    if group_map.get(material) == "low_absorption":
        return "low_absorption_silicate_carbonate_overlap"
    if group_map.get(material) == "high_absorption":
        return "high_absorption_sulfide_oxide_overlap"
    return "class_specific_overlap_requires_feature_ablation"


def next_action_for_material(material: str, recall: float, confusions: dict[str, int]) -> str:
    if recall >= ACCEPTANCE_TARGETS["min_class_recall"]:
        return "monitor_no_immediate_change"
    if material in HM_PAIR or set(confusions).intersection(HM_PAIR):
        return "prioritize_hm_expert_features_energy_response_and_pairwise_validation"
    if confusions:
        return "inspect_confusion_pair_feature_ablation_and_group_expert"
    return "increase_support_or_check_empty_prediction_path"


def failure_analysis_frame(per_class: pd.DataFrame, decisions: pd.DataFrame, group_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for row in per_class.itertuples(index=False):
        material = str(row.material)
        confusions = common_confusions(decisions, material)
        recall = float(row.recall)
        rows.append(
            {
                "material": material,
                "group_label": group_map.get(material, ""),
                "support": int(row.support),
                "recall": recall,
                "miss_count": int(round(int(row.support) * (1.0 - recall))),
                "common_confusions": ";".join(f"{name}:{count}" for name, count in confusions.items()),
                "failure_status": "pass" if recall >= ACCEPTANCE_TARGETS["min_class_recall"] else "fail",
                "physics_hypothesis": physics_hypothesis(material, confusions, group_map),
                "next_action": next_action_for_material(material, recall, confusions),
            }
        )
    return pd.DataFrame(rows).sort_values(["failure_status", "recall", "material"])


def registry_failure_reason(
    final_metrics: dict,
    min_support: int,
    min_support_required: int,
    reused_test: list[int],
    integrity: dict,
    development_only: bool,
) -> str:
    reasons = []
    if reused_test:
        reasons.append(f"reused_or_burned_test_seeds={';'.join(str(seed) for seed in reused_test)}")
    if not integrity["split_is_disjoint"]:
        reasons.append("split_overlap_detected")
    if min_support < min_support_required:
        reasons.append(f"per_class_support_below_{min_support_required}={min_support}")
    if development_only:
        for metric in ["hematite_recall", "magnetite_recall", "hm_min_recall"]:
            value = float(final_metrics.get(metric, math.nan))
            if math.isnan(value) or value < ACCEPTANCE_TARGETS["min_class_recall"]:
                reasons.append(f"development_{metric}_below_{ACCEPTANCE_TARGETS['min_class_recall']:g}")
    else:
        for metric, threshold in ACCEPTANCE_TARGETS.items():
            value = float(final_metrics.get(metric, math.nan))
            if math.isnan(value) or value < threshold:
                reasons.append(f"{metric}_below_{threshold:g}")
        if float(final_metrics.get("hm_min_recall", math.nan)) < ACCEPTANCE_TARGETS["min_class_recall"]:
            reasons.append("hm_min_recall_below_target")
    return ";".join(reasons) if reasons else "none"


def experiment_registry_frame(
    args: argparse.Namespace,
    status: dict,
    selected_method: str,
    validation_table: pd.DataFrame,
    final_metrics: dict,
    min_support: int,
    reused_test: list[int],
    integrity: dict,
    claim_safe: bool,
) -> pd.DataFrame:
    selected_validation = choose_validation_method(validation_table)
    development_only = bool(getattr(args, "development_only", False))
    return pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocol_name": args.protocol_name,
                "evaluation_stage": "development_validation_only" if development_only else "locked_final_test",
                "hypothesis": "layered_absorption_group_with_hm_expert_and_enriched_spectral_features",
                "change_summary": "strict audit with validation-only model selection and paper ledger outputs",
                "feature_engineering": "hm_differential" if getattr(args, "enable_hm_differential_features", False) else "baseline",
                "hm_differential_feature_count": status.get("hm_differential_feature_count", 0),
                "raw_dirs": args.raw_dirs or args.raw_dir,
                "photon_budget": int(args.photon_budget),
                "train_seeds": args.train_seeds,
                "validation_seeds": args.validation_seeds,
                "test_seeds": args.test_seeds,
                "burned_test_seeds": args.burned_test_seeds,
                "contains_burned_test_seed": bool(reused_test),
                "table_mode": status.get("table_mode", ""),
                "selected_method": selected_method,
                "validation_top1_accuracy": selected_validation.get("top1_accuracy", math.nan),
                "validation_macro_f1": selected_validation.get("macro_f1", math.nan),
                "validation_min_class_recall": selected_validation.get("min_class_recall", math.nan),
                "validation_hematite_recall": selected_validation.get("hematite_recall", math.nan),
                "validation_magnetite_recall": selected_validation.get("magnetite_recall", math.nan),
                "development_top1_accuracy": final_metrics.get("top1_accuracy", math.nan) if development_only else math.nan,
                "development_macro_f1": final_metrics.get("macro_f1", math.nan) if development_only else math.nan,
                "development_min_class_recall": final_metrics.get("min_class_recall", math.nan) if development_only else math.nan,
                "development_hematite_recall": final_metrics.get("hematite_recall", math.nan) if development_only else math.nan,
                "development_magnetite_recall": final_metrics.get("magnetite_recall", math.nan) if development_only else math.nan,
                "final_top1_accuracy": math.nan if development_only else final_metrics.get("top1_accuracy", math.nan),
                "final_macro_f1": math.nan if development_only else final_metrics.get("macro_f1", math.nan),
                "final_min_class_recall": math.nan if development_only else final_metrics.get("min_class_recall", math.nan),
                "final_hematite_recall": math.nan if development_only else final_metrics.get("hematite_recall", math.nan),
                "final_magnetite_recall": math.nan if development_only else final_metrics.get("magnetite_recall", math.nan),
                "model_feature_count": final_metrics.get("model_feature_count", math.nan),
                "min_class_support_observed": min_support,
                "claim_safe": claim_safe,
                "failure_reason": registry_failure_reason(
                    final_metrics,
                    min_support,
                    int(args.min_class_support),
                    reused_test,
                    integrity,
                    development_only,
                ),
                "next_action": (
                    "freeze_and_report"
                    if claim_safe
                    else (
                        "continue_hm_train_validation_iteration_without_opening_final_test"
                        if development_only
                        else "continue_train_validation_iteration_without_reusing_final_test"
                    )
                ),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict seed-holdout generalization audit for material sorting.")
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/selected_rebuild")
    parser.add_argument(
        "--raw-dirs",
        default="",
        help="Comma-separated raw dirs to merge. Use this to train on burned development seeds and test on a new profile.",
    )
    parser.add_argument("--output-dir", default="results/material_sorting_strict_generalization")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--photon-budget", type=int, default=5000)
    parser.add_argument("--train-seeds", default="101,202,303")
    parser.add_argument("--validation-seeds", default="404")
    parser.add_argument("--test-seeds", default="505")
    parser.add_argument("--burned-test-seeds", default="303,505")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--min-class-support", type=int, default=40)
    parser.add_argument("--rotate-existing-seeds", action="store_true")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Evaluate train/validation only and write a development ledger without opening or claiming a final test.",
    )
    parser.add_argument(
        "--enable-hm-differential-features",
        action="store_true",
        help="Add pre-registered high-energy H/M differential features before model selection.",
    )
    parser.add_argument("--protocol-name", default="strict_generalization_seed_holdout")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = parse_int_list(args.train_seeds)
    validation_seeds = parse_int_list(args.validation_seeds)
    test_seeds = parse_int_list(args.test_seeds)
    burned_test_seeds = set(parse_int_list(args.burned_test_seeds))
    methods = parse_str_list(args.methods)

    sk = v2.require_sklearn()
    group_map = selected.material_group_map(project_root)
    raw_dirs = parse_raw_dirs(project_root, args.raw_dir, args.raw_dirs)
    frame, status = build_frame_from_raw_dirs(project_root, raw_dirs, args.photon_budget)
    hm_differential_features: list[str] = []
    if args.enable_hm_differential_features:
        frame, hm_differential_features = add_hm_differential_features(frame)
    status["hm_differential_features_enabled"] = bool(args.enable_hm_differential_features)
    status["hm_differential_feature_count"] = int(len(hm_differential_features))
    integrity = check_split_integrity(train_seeds, validation_seeds, test_seeds)

    if args.development_only:
        validation_table, validation_metrics, per_class, decisions, pairwise = evaluate_development_split(
            frame,
            train_seeds,
            validation_seeds,
            methods,
            group_map,
            sk,
        )
        selected_method = str(choose_validation_method(validation_table)["method"])
        split_audit = split_audit_frame(frame, train_seeds, validation_seeds, [])
        min_support = int(per_class["support"].min()) if not per_class.empty else 0
        passes_metrics = bool(
            validation_metrics["top1_accuracy"] >= ACCEPTANCE_TARGETS["top1_accuracy"]
            and validation_metrics["macro_f1"] >= ACCEPTANCE_TARGETS["macro_f1"]
            and validation_metrics["min_class_recall"] >= ACCEPTANCE_TARGETS["min_class_recall"]
        )
        passes_support = bool(min_support >= args.min_class_support)
        write_csv(validation_table, output_dir / "validation_model_selection.csv")
        write_csv(pd.DataFrame([validation_metrics]), output_dir / "development_validation_summary.csv")
        write_csv(per_class, output_dir / "per_class_recall_validation.csv")
        write_csv(decisions, output_dir / "validation_decisions.csv")
        write_csv(pairwise, output_dir / "hm_pairwise_audit.csv")
        write_csv(split_audit, output_dir / "split_audit.csv")
        write_csv(
            experiment_registry_frame(
                args,
                status,
                selected_method,
                validation_table,
                validation_metrics,
                min_support,
                [],
                integrity,
                False,
            ),
            output_dir / "experiment_registry.csv",
        )
        failure_analysis = failure_analysis_frame(per_class, decisions, group_map)
        write_csv(failure_analysis, output_dir / "failure_analysis.csv")
        manifest = {
            "package": "xrt-sorter-geant4-undergrad-guide",
            "generated_by": "analysis/strict_generalization_audit.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol_name": args.protocol_name,
            "development_only": True,
            "raw_dir": args.raw_dir,
            "raw_dirs": [path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else path.as_posix() for path in raw_dirs],
            "output_dir": args.output_dir,
            "photon_budget": args.photon_budget,
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "test_seeds": [],
            "burned_test_seeds": sorted(burned_test_seeds),
            "split_integrity": integrity,
            "methods": methods,
            "feature_engineering": {
                "hm_differential_features_enabled": bool(args.enable_hm_differential_features),
                "hm_differential_feature_count": int(len(hm_differential_features)),
                "hm_differential_features": hm_differential_features,
            },
            "model_selection_policy": "development-only validation: rank by hm_min_recall, then min_class_recall, then macro_f1, then top1_accuracy",
            "selected_method": selected_method,
            "selected_validation_metrics": choose_validation_method(validation_table),
            "status": status,
            "acceptance_targets": ACCEPTANCE_TARGETS,
            "min_class_support_required": args.min_class_support,
            "min_class_support_observed": min_support,
            "development_validation_metrics": validation_metrics,
            "passes_metric_targets": passes_metrics,
            "passes_support_target": passes_support,
            "claim_safe_automatic_ten_material_sorting": False,
            "stage_conclusion": "development_validation_passed_not_final_claim" if passes_metrics and passes_support else "development_validation_failed_or_incomplete",
            "paper_ledger_outputs": {
                "experiment_registry": (output_dir / "experiment_registry.csv").relative_to(project_root).as_posix()
                if (output_dir / "experiment_registry.csv").is_relative_to(project_root)
                else (output_dir / "experiment_registry.csv").as_posix(),
                "failure_analysis": (output_dir / "failure_analysis.csv").relative_to(project_root).as_posix()
                if (output_dir / "failure_analysis.csv").is_relative_to(project_root)
                else (output_dir / "failure_analysis.csv").as_posix(),
                "hm_pairwise_audit": (output_dir / "hm_pairwise_audit.csv").relative_to(project_root).as_posix()
                if (output_dir / "hm_pairwise_audit.csv").is_relative_to(project_root)
                else (output_dir / "hm_pairwise_audit.csv").as_posix(),
            },
            "software": {"python": platform.python_version(), "pandas": pd.__version__},
        }
        v2.write_manifest(output_dir / "strict_generalization_manifest.json", manifest)
        print(f"Wrote development-only strict audit to {output_dir}")
        print(f"selected_method={selected_method} claim_safe=False")
        return

    validation_table, final_metrics, per_class, decisions = evaluate_locked_split(
        frame,
        train_seeds,
        validation_seeds,
        test_seeds,
        methods,
        group_map,
        sk,
    )
    selected_method = str(choose_validation_method(validation_table)["method"])
    split_audit = split_audit_frame(frame, train_seeds, validation_seeds, test_seeds)
    min_support = int(per_class["support"].min()) if not per_class.empty else 0
    unseen_test = sorted(set(test_seeds) - burned_test_seeds)
    reused_test = sorted(set(test_seeds) & burned_test_seeds)
    passes_metrics = bool(
        final_metrics["top1_accuracy"] >= ACCEPTANCE_TARGETS["top1_accuracy"]
        and final_metrics["macro_f1"] >= ACCEPTANCE_TARGETS["macro_f1"]
        and final_metrics["min_class_recall"] >= ACCEPTANCE_TARGETS["min_class_recall"]
    )
    passes_support = bool(min_support >= args.min_class_support)
    claim_safe = bool(passes_metrics and passes_support and integrity["split_is_disjoint"] and not reused_test)

    write_csv(validation_table, output_dir / "validation_model_selection.csv")
    write_csv(pd.DataFrame([final_metrics]), output_dir / "final_test_summary.csv")
    write_csv(per_class, output_dir / "per_class_recall_final_test.csv")
    write_csv(decisions, output_dir / "final_test_decisions.csv")
    write_csv(split_audit, output_dir / "split_audit.csv")
    train_aug, validation_aug, test_aug, feature_cols, final_feature_cols = build_augmented_splits(
        frame,
        train_seeds,
        validation_seeds,
        test_seeds,
    )
    pairwise_frames = [
        hm_pairwise_audit_frame(train_aug, validation_aug, feature_cols, sk, "validation"),
        hm_pairwise_audit_frame(
            pd.concat([train_aug, validation_aug], ignore_index=True),
            test_aug,
            final_feature_cols,
            sk,
            "test",
        ),
    ]
    write_csv(pd.concat(pairwise_frames, ignore_index=True), output_dir / "hm_pairwise_audit.csv")
    write_csv(
        experiment_registry_frame(
            args,
            status,
            selected_method,
            validation_table,
            final_metrics,
            min_support,
            reused_test,
            integrity,
            claim_safe,
        ),
        output_dir / "experiment_registry.csv",
    )
    failure_analysis = failure_analysis_frame(per_class, decisions, group_map)
    write_csv(failure_analysis, output_dir / "failure_analysis.csv")

    rotation_summary = pd.DataFrame()
    if args.rotate_existing_seeds:
        rotation_summary = evaluate_rotating_splits(frame, methods, group_map, sk)
        write_csv(rotation_summary, output_dir / "rotating_seed_audit.csv")

    manifest = {
        "package": "xrt-sorter-geant4-undergrad-guide",
        "generated_by": "analysis/strict_generalization_audit.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": args.protocol_name,
        "raw_dir": args.raw_dir,
        "raw_dirs": [path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else path.as_posix() for path in raw_dirs],
        "output_dir": args.output_dir,
        "photon_budget": args.photon_budget,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
        "burned_test_seeds": sorted(burned_test_seeds),
        "unseen_test_seeds": unseen_test,
        "reused_test_seeds": reused_test,
        "split_integrity": integrity,
        "methods": methods,
        "feature_engineering": {
            "hm_differential_features_enabled": bool(args.enable_hm_differential_features),
            "hm_differential_feature_count": int(len(hm_differential_features)),
            "hm_differential_features": hm_differential_features,
        },
        "model_selection_policy": "validation-only: rank by hm_min_recall, then min_class_recall, then macro_f1, then top1_accuracy",
        "selected_method": selected_method,
        "selected_validation_metrics": choose_validation_method(validation_table),
        "status": status,
        "acceptance_targets": ACCEPTANCE_TARGETS,
        "min_class_support_required": args.min_class_support,
        "min_class_support_observed": min_support,
        "final_test_metrics": final_metrics,
        "passes_metric_targets": passes_metrics,
        "passes_support_target": passes_support,
        "claim_safe_automatic_ten_material_sorting": claim_safe,
        "paper_ledger_outputs": {
            "experiment_registry": (output_dir / "experiment_registry.csv").relative_to(project_root).as_posix()
            if (output_dir / "experiment_registry.csv").is_relative_to(project_root)
            else (output_dir / "experiment_registry.csv").as_posix(),
            "failure_analysis": (output_dir / "failure_analysis.csv").relative_to(project_root).as_posix()
            if (output_dir / "failure_analysis.csv").is_relative_to(project_root)
            else (output_dir / "failure_analysis.csv").as_posix(),
            "hm_pairwise_audit": (output_dir / "hm_pairwise_audit.csv").relative_to(project_root).as_posix()
            if (output_dir / "hm_pairwise_audit.csv").is_relative_to(project_root)
            else (output_dir / "hm_pairwise_audit.csv").as_posix(),
        },
        "stage_conclusion": "accepted_claim_safe" if claim_safe else "diagnostic_or_failed_not_claim_safe",
        "rotation_summary_rows": int(len(rotation_summary)),
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    v2.write_manifest(output_dir / "strict_generalization_manifest.json", manifest)
    print(f"Wrote strict generalization audit to {output_dir}")
    print(f"selected_method={selected_method} claim_safe={claim_safe}")


if __name__ == "__main__":
    main()
