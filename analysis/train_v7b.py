from __future__ import annotations

import argparse
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import material_sorting_v2 as v2


TARGET_MATERIALS = list(v2.TARGET_MATERIALS)
HM_PAIR = ("Hematite", "Magnetite")
KEY_HARD_PAIRS = [
    ("Hematite", "Magnetite"),
    ("Pyrite", "Chalcopyrite"),
    ("Calcite", "Dolomite"),
    ("Orthoclase", "Albite"),
    ("Pyrite", "Galena"),
    ("Chalcopyrite", "Galena"),
]
DEFAULT_METHODS = [
    "ExtraTrees",
    "HistGradientBoosting",
    "XGBoost",
    "HardNegativeExtraTrees",
    "HardNegativeXGBoost",
    "GroupExpertExtraTrees",
    "HMPairwiseRerankExtraTrees",
]
MODEL_RANK = {
    "ExtraTrees": 0,
    "HistGradientBoosting": 1,
    "XGBoost": 2,
    "GroupExpertExtraTrees": 3,
    "HardNegativeExtraTrees": 4,
    "HardNegativeXGBoost": 5,
    "HMPairwiseRerankExtraTrees": 6,
}
DEFAULT_SHADOW_SEEDS = set(range(4301, 4307))


def log_progress(message: str, *, start_time: float | None = None) -> None:
    elapsed = ""
    if start_time is not None:
        elapsed = f" elapsed={perf_counter() - start_time:.1f}s"
    print(f"[v7b-train] {datetime.now().isoformat(timespec='seconds')} {message}{elapsed}", flush=True)


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def require_sklearn():
    sk = v2.require_sklearn()
    from sklearn.metrics import precision_recall_fscore_support

    sk["precision_recall_fscore_support"] = precision_recall_fscore_support
    return sk


def require_xgboost():
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError:
        return None
    return XGBClassifier


def load_cube(cube_dir: Path) -> tuple[np.ndarray, pd.DataFrame, list[str], dict]:
    data = np.load(cube_dir / "measurement_cube.npz", allow_pickle=True)
    cube = data["X"].astype(np.float32)
    metadata = pd.read_csv(cube_dir / "sample_metadata.csv")
    feature_names = [str(item) for item in data["feature_names"].tolist()]
    manifest_path = cube_dir / "measurement_cube_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return cube, metadata, feature_names, manifest


def build_feature_matrix(
    cube: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: list[str],
    include_thickness: bool,
) -> tuple[np.ndarray, list[str]]:
    x = cube.reshape((cube.shape[0], -1)).astype(np.float32)
    names = list(feature_names)
    if include_thickness:
        x = np.column_stack([x, metadata["thickness_mm"].astype(float).to_numpy(dtype=np.float32)])
        names.append("metadata__thickness_mm")
    return x, names


def labels_to_int(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(classes)}
    return np.array([mapping[str(label)] for label in labels], dtype=int)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores[scores < 0.0] = 0.0
    sums = scores.sum(axis=1, keepdims=True)
    if scores.shape[1] == 0:
        return scores
    return np.divide(scores, sums, out=np.full_like(scores, 1.0 / scores.shape[1]), where=sums > 0)


def align_scores(raw_scores: np.ndarray, raw_classes: np.ndarray, target_classes: list[str]) -> np.ndarray:
    aligned = np.zeros((raw_scores.shape[0], len(target_classes)), dtype=float)
    target_index = {label: index for index, label in enumerate(target_classes)}
    for raw_index, raw_label in enumerate(raw_classes.astype(str)):
        if raw_label in target_index:
            aligned[:, target_index[raw_label]] = raw_scores[:, raw_index]
    return normalize_scores(aligned)


def topk_accuracy(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray, k: int) -> float:
    if scores.size == 0 or len(y_true) == 0:
        return math.nan
    k = max(1, min(k, scores.shape[1]))
    order = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    return float(np.mean([truth in classes[row] for truth, row in zip(y_true, order)]))


def pair_recalls(y_true: np.ndarray, predictions: np.ndarray, pair: tuple[str, str]) -> dict:
    recalls = {}
    for material in pair:
        mask = y_true == material
        recalls[material] = float(np.mean(predictions[mask] == material)) if mask.any() else 0.0
    return recalls


def pair_min_recall(y_true: np.ndarray, predictions: np.ndarray, pair: tuple[str, str]) -> float:
    recalls = pair_recalls(y_true, predictions, pair)
    return float(min(recalls.values())) if recalls else 0.0


def evaluate_predictions(
    method: str,
    round_id: int,
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sk,
) -> dict:
    recalls = sk["recall_score"](y_true, predictions, labels=np.array(TARGET_MATERIALS), average=None, zero_division=0)
    recall_map = {material: float(recall) for material, recall in zip(TARGET_MATERIALS, recalls)}
    hard_pair_values = [pair_min_recall(y_true, predictions, pair) for pair in KEY_HARD_PAIRS]
    return {
        "method": method,
        "round_id": int(round_id),
        "samples": int(len(y_true)),
        "top1_accuracy": float(np.mean(y_true == predictions)) if len(y_true) else math.nan,
        "top3_accuracy": topk_accuracy(y_true, scores, classes, min(3, len(classes))),
        "macro_f1": float(sk["f1_score"](y_true, predictions, labels=np.array(TARGET_MATERIALS), average="macro", zero_division=0)),
        "min_class_recall": float(np.min(recalls)) if len(recalls) else 0.0,
        "hm_min_recall": float(min(recall_map.get(HM_PAIR[0], 0.0), recall_map.get(HM_PAIR[1], 0.0))),
        "hm_pairwise_min_recall": pair_min_recall(y_true, predictions, HM_PAIR),
        "key_hard_negative_pair_min_recall": float(min(hard_pair_values)) if hard_pair_values else 0.0,
        "model_size_rank": MODEL_RANK.get(method, 99),
    }


def make_extra_trees(sk, random_state: int, weighted: bool = False, n_jobs: int = -1):
    return sk["ExtraTreesClassifier"](
        n_estimators=900 if weighted else 700,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )


def make_hist_gradient_boosting(sk, random_state: int):
    return sk["HistGradientBoostingClassifier"](
        max_iter=220,
        learning_rate=0.035,
        l2_regularization=0.08,
        max_leaf_nodes=31,
        random_state=random_state,
    )


def fit_predict_sklearn(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    classes: list[str],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_weight is None:
        model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    predictions = np.asarray(model.predict(x_eval)).astype(str)
    if hasattr(model, "predict_proba"):
        scores = align_scores(np.asarray(model.predict_proba(x_eval), dtype=float), np.asarray(model.classes_), classes)
    else:
        scores = np.zeros((len(predictions), len(classes)), dtype=float)
        index = {label: i for i, label in enumerate(classes)}
        for row, prediction in enumerate(predictions):
            if prediction in index:
                scores[row, index[prediction]] = 1.0
        scores = normalize_scores(scores)
    return predictions, scores, np.array(classes)


def fit_predict_xgboost(
    XGBClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    classes: list[str],
    sample_weight: np.ndarray | None,
    random_state: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_train_int = labels_to_int(y_train, classes)
    model = XGBClassifier(
        n_estimators=420,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train_int, sample_weight=sample_weight)
    scores = normalize_scores(np.asarray(model.predict_proba(x_eval), dtype=float))
    class_array = np.array(classes)
    predictions = class_array[np.argmax(scores, axis=1)]
    return predictions, scores, class_array


def fit_predict_group_expert(
    sk,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_meta: pd.DataFrame,
    x_eval: np.ndarray,
    classes: list[str],
    random_state: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_train = train_meta["group_label"].fillna(train_meta["category"]).astype(str).replace("", "unknown").to_numpy()
    group_model = make_extra_trees(sk, random_state, n_jobs=n_jobs)
    group_model.fit(x_train, group_train)
    group_predictions = group_model.predict(x_eval).astype(str)
    experts = {}
    constants = {}
    for group in sorted(set(group_train)):
        mask = group_train == group
        labels = sorted(set(y_train[mask].astype(str)))
        if len(labels) == 1:
            constants[group] = labels[0]
        else:
            model = make_extra_trees(sk, random_state + len(experts) + 1, n_jobs=n_jobs)
            model.fit(x_train[mask], y_train[mask])
            experts[group] = model

    scores = np.zeros((x_eval.shape[0], len(classes)), dtype=float)
    class_index = {label: index for index, label in enumerate(classes)}
    for row_index, group in enumerate(group_predictions):
        if group in constants:
            scores[row_index, class_index[constants[group]]] = 1.0
            continue
        model = experts.get(group)
        if model is None:
            continue
        raw = np.asarray(model.predict_proba(x_eval[row_index : row_index + 1]), dtype=float)
        aligned = align_scores(raw, np.asarray(model.classes_), classes)
        scores[row_index] = aligned[0]
    scores = normalize_scores(scores)
    class_array = np.array(classes)
    predictions = class_array[np.argmax(scores, axis=1)]
    return predictions, scores, class_array


def fit_predict_hm_rerank(
    sk,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    classes: list[str],
    sample_weight: np.ndarray | None,
    random_state: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = make_extra_trees(sk, random_state, n_jobs=n_jobs)
    base.fit(x_train, y_train, sample_weight=sample_weight)
    base_scores = align_scores(np.asarray(base.predict_proba(x_eval), dtype=float), np.asarray(base.classes_), classes)
    hm_mask = np.isin(y_train.astype(str), np.array(HM_PAIR))
    if hm_mask.sum() < 4:
        class_array = np.array(classes)
        return class_array[np.argmax(base_scores, axis=1)], base_scores, class_array

    pair_model = make_extra_trees(sk, random_state + 91, weighted=True, n_jobs=n_jobs)
    pair_weight = sample_weight[hm_mask] if sample_weight is not None else None
    pair_model.fit(x_train[hm_mask], y_train[hm_mask], sample_weight=pair_weight)
    pair_scores = align_scores(np.asarray(pair_model.predict_proba(x_eval), dtype=float), np.asarray(pair_model.classes_), classes)

    class_array = np.array(classes)
    hm_indices = [classes.index(HM_PAIR[0]), classes.index(HM_PAIR[1])]
    reranked = base_scores.copy()
    order = np.argsort(base_scores, axis=1)[:, ::-1]
    for row in range(base_scores.shape[0]):
        top3 = set(class_array[order[row, : min(3, len(classes))]])
        if HM_PAIR[0] not in top3 or HM_PAIR[1] not in top3:
            continue
        hm_mass = max(float(base_scores[row, hm_indices].sum()), 0.05)
        reranked[row, hm_indices[0]] = pair_scores[row, hm_indices[0]] * hm_mass
        reranked[row, hm_indices[1]] = pair_scores[row, hm_indices[1]] * hm_mass
    reranked = normalize_scores(reranked)
    return class_array[np.argmax(reranked, axis=1)], reranked, class_array


def evaluate_method(
    method: str,
    round_id: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_meta: pd.DataFrame,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    sample_weight: np.ndarray | None,
    sk,
    XGBClassifier,
    classes: list[str],
    n_jobs: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    if method in {"XGBoost", "HardNegativeXGBoost"} and XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")
    if method == "ExtraTrees":
        predictions, scores, class_array = fit_predict_sklearn(make_extra_trees(sk, 700 + round_id, n_jobs=n_jobs), x_train, y_train, x_eval, classes)
    elif method == "HardNegativeExtraTrees":
        predictions, scores, class_array = fit_predict_sklearn(
            make_extra_trees(sk, 1700 + round_id, weighted=True, n_jobs=n_jobs),
            x_train,
            y_train,
            x_eval,
            classes,
            sample_weight,
        )
    elif method == "HistGradientBoosting":
        predictions, scores, class_array = fit_predict_sklearn(
            make_hist_gradient_boosting(sk, 900 + round_id),
            x_train,
            y_train,
            x_eval,
            classes,
        )
    elif method == "XGBoost":
        predictions, scores, class_array = fit_predict_xgboost(XGBClassifier, x_train, y_train, x_eval, classes, None, 1100 + round_id, n_jobs)
    elif method == "HardNegativeXGBoost":
        predictions, scores, class_array = fit_predict_xgboost(
            XGBClassifier,
            x_train,
            y_train,
            x_eval,
            classes,
            sample_weight,
            2100 + round_id,
            n_jobs,
        )
    elif method == "GroupExpertExtraTrees":
        predictions, scores, class_array = fit_predict_group_expert(sk, x_train, y_train, train_meta, x_eval, classes, 2600 + round_id, n_jobs)
    elif method == "HMPairwiseRerankExtraTrees":
        predictions, scores, class_array = fit_predict_hm_rerank(sk, x_train, y_train, x_eval, classes, sample_weight, 3000 + round_id, n_jobs)
    else:
        raise ValueError(f"Unknown v7B method: {method}")
    metrics = evaluate_predictions(method, round_id, y_eval, predictions, scores, class_array, sk)
    return metrics, predictions, scores, class_array


def choose_model(table: pd.DataFrame) -> dict:
    ranked = table.dropna(
        subset=[
            "hm_min_recall",
            "hm_pairwise_min_recall",
            "key_hard_negative_pair_min_recall",
            "macro_f1",
            "top1_accuracy",
        ]
    ).sort_values(
        [
            "hm_min_recall",
            "hm_pairwise_min_recall",
            "key_hard_negative_pair_min_recall",
            "macro_f1",
            "top1_accuracy",
            "min_class_recall",
            "model_size_rank",
        ],
        ascending=[False, False, False, False, False, False, True],
    )
    if ranked.empty:
        raise RuntimeError("No v7B method produced finite metrics.")
    return ranked.iloc[0].to_dict()


def decision_frame(metadata: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    order = np.argsort(scores, axis=1)[:, ::-1]
    top1 = scores[np.arange(len(scores)), order[:, 0]]
    top2 = scores[np.arange(len(scores)), order[:, 1]] if scores.shape[1] > 1 else top1
    rows = []
    for idx, row in enumerate(metadata.reset_index(drop=True).itertuples(index=False)):
        rows.append(
            {
                "material": row.material,
                "predicted_material": str(predictions[idx]),
                "is_correct": bool(str(row.material) == str(predictions[idx])),
                "top1_score": float(top1[idx]),
                "top2_score": float(top2[idx]),
                "score_margin": float(top1[idx] - top2[idx]),
                "top3_candidates": ";".join(classes[order[idx, : min(3, scores.shape[1])]].astype(str)),
                "group_label": getattr(row, "group_label", ""),
                "category": getattr(row, "category", ""),
                "thickness_mm": float(row.thickness_mm),
                "random_seed": int(row.random_seed),
                "sample_id": int(row.sample_id),
                "split": row.split,
            }
        )
    return pd.DataFrame(rows)


def per_class_table(metadata: pd.DataFrame, predictions: np.ndarray, sk) -> pd.DataFrame:
    y_true = metadata["material"].astype(str).to_numpy()
    precision, recall, f1, support = sk["precision_recall_fscore_support"](
        y_true,
        predictions,
        labels=np.array(TARGET_MATERIALS),
        zero_division=0,
    )
    return pd.DataFrame(
        [
            {
                "split": "validation",
                "material": material,
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, material in enumerate(TARGET_MATERIALS)
        ]
    )


def pairwise_audit(metadata: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    y_true = metadata["material"].astype(str).to_numpy()
    rows = []
    for pair in KEY_HARD_PAIRS:
        recalls = pair_recalls(y_true, predictions, pair)
        pair_mask = np.isin(y_true, np.array(pair))
        confusions = {}
        if pair_mask.any():
            misses = pd.DataFrame({"true": y_true[pair_mask], "pred": predictions[pair_mask]})
            confusions = misses.loc[misses["true"] != misses["pred"], "pred"].value_counts().to_dict()
        rows.append(
            {
                "split": "validation",
                "pair": f"{pair[0]}/{pair[1]}",
                "material_a": pair[0],
                "material_b": pair[1],
                "recall_a": float(recalls.get(pair[0], 0.0)),
                "recall_b": float(recalls.get(pair[1], 0.0)),
                "pair_min_recall": float(min(recalls.values())) if recalls else 0.0,
                "support": int(pair_mask.sum()),
                "common_wrong_predictions": ";".join(f"{name}:{int(count)}" for name, count in confusions.items()),
            }
        )
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
                "miss_count": int((~part["is_correct"]).sum()),
                "common_confusions": ";".join(f"{name}:{int(count)}" for name, count in confusions.items()),
                "failure_status": "pass" if float(row.recall) >= 0.70 else "fail",
                "next_action": "inspect_hard_pair_or_group_expert" if float(row.recall) < 0.70 else "monitor",
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_table(metadata: pd.DataFrame, predictions: np.ndarray, sk) -> pd.DataFrame:
    cm = sk["confusion_matrix"](metadata["material"].astype(str).to_numpy(), predictions, labels=np.array(TARGET_MATERIALS))
    return pd.DataFrame(cm, index=TARGET_MATERIALS, columns=TARGET_MATERIALS)


def update_weights(train_meta: pd.DataFrame, validation_meta: pd.DataFrame, decisions: pd.DataFrame, base_weight: float) -> np.ndarray:
    weights = np.array(
        train_meta.get("sample_weight_base", pd.Series(np.ones(len(train_meta)))).astype(float).to_numpy(),
        dtype=float,
        copy=True,
    )
    merged = validation_meta.reset_index(drop=True).copy()
    merged["is_correct"] = decisions["is_correct"].to_numpy(dtype=bool)
    material_recall = merged.groupby("material")["is_correct"].mean().to_dict()
    pair_penalty_materials = set()
    for pair in KEY_HARD_PAIRS:
        recalls = {material: float(material_recall.get(material, 1.0)) for material in pair}
        if min(recalls.values()) < 0.75:
            pair_penalty_materials.update(pair)
    for index, row in enumerate(train_meta.itertuples(index=False)):
        recall = float(material_recall.get(row.material, 1.0))
        if recall < 0.80:
            weights[index] += base_weight * (0.80 - recall) / 0.80
        if row.material in pair_penalty_materials:
            weights[index] += 0.5 * base_weight
        if row.material in HM_PAIR:
            weights[index] += 0.5 * base_weight
    return weights


def view_feature_indices(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {
        "all": list(range(len(feature_names))),
        "transmission_only": [i for i, name in enumerate(feature_names) if "__transmission__" in name or name == "metadata__thickness_mm"],
        "side_scatter_only": [i for i, name in enumerate(feature_names) if "__side_scatter__" in name or name == "metadata__thickness_mm"],
        "high_energy_only": [
            i
            for i, name in enumerate(feature_names)
            if any(tag in name for tag in ["mono_120kev", "mono_150kev", "mono_200kev"]) or name == "metadata__thickness_mm"
        ],
        "oblique_only": [i for i, name in enumerate(feature_names) if "oblique_" in name or name == "metadata__thickness_mm"],
        "oblique_20deg_only": [i for i, name in enumerate(feature_names) if "oblique_20deg" in name or name == "metadata__thickness_mm"],
        "normal_wide_only": [i for i, name in enumerate(feature_names) if "normal_wide" in name or name == "metadata__thickness_mm"],
    }
    return {name: indices for name, indices in groups.items() if indices}


def evaluate_view_ablation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: list[str],
    sk,
    classes: list[str],
    n_jobs: int,
) -> pd.DataFrame:
    rows = []
    for view_name, indices in view_feature_indices(feature_names).items():
        model = make_extra_trees(sk, 4100 + len(rows), n_jobs=n_jobs)
        predictions, scores, class_array = fit_predict_sklearn(model, x_train[:, indices], y_train, x_eval[:, indices], classes)
        metrics = evaluate_predictions("ExtraTrees", 0, y_eval, predictions, scores, class_array, sk)
        metrics["view_name"] = view_name
        metrics["feature_count"] = int(len(indices))
        rows.append(metrics)
    return pd.DataFrame(rows)


def runner_status(project_root: Path, status_csv: str, cube_manifest: dict) -> dict:
    status_path = project_root / status_csv
    if not status_path.exists():
        return {"checked": False, "rows": 0, "completed": 0, "failed": 0, "pending": None, "expected_rows": None}
    table = pd.read_csv(status_path)
    if "returncode" not in table.columns:
        return {"checked": False, "rows": int(len(table)), "completed": 0, "failed": 0, "pending": None, "expected_rows": None}
    rc = table["returncode"].astype(str)
    completed = int(rc.eq("0").sum())
    failed = int((~rc.isin(["", "0", "nan", "None"])).sum())
    expected = cube_manifest.get("records_used")
    if expected is not None:
        expected = int(expected) + int(cube_manifest.get("calibration_records_used", 0))
    pending = None if expected is None else max(int(expected) - completed - failed, 0)
    return {
        "checked": True,
        "rows": int(len(table)),
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "expected_rows": expected,
    }


def gate_report(
    output_dir: Path,
    selected: dict,
    per_class: pd.DataFrame,
    pairwise: pd.DataFrame,
    manifest: dict,
    run_status: dict,
) -> dict:
    thresholds = {
        "top1_accuracy": 0.85,
        "macro_f1": 0.82,
        "min_class_recall": 0.70,
        "hm_min_recall": 0.80,
        "hm_pairwise_min_recall": 0.78,
        "key_hard_negative_pair_min_recall": 0.75,
        "min_validation_support_per_class": 100,
        "runner_failures": 0,
        "runner_pending": 0,
    }
    hm_row = pairwise[pairwise["pair"].eq("Hematite/Magnetite")]
    observed = {
        "method": selected["method"],
        "round_id": int(selected["round_id"]),
        "top1_accuracy": float(selected["top1_accuracy"]),
        "macro_f1": float(selected["macro_f1"]),
        "min_class_recall": float(selected["min_class_recall"]),
        "hm_min_recall": float(selected["hm_min_recall"]),
        "hm_pairwise_min_recall": float(hm_row["pair_min_recall"].iloc[0]) if not hm_row.empty else float(selected["hm_pairwise_min_recall"]),
        "key_hard_negative_pair_min_recall": float(pairwise["pair_min_recall"].min()) if not pairwise.empty else 0.0,
        "min_validation_support_per_class": int(per_class["support"].min()) if not per_class.empty else 0,
        "runner_failures": int(run_status.get("failed", 0)),
        "runner_pending": run_status.get("pending"),
        "runner_completed": int(run_status.get("completed", 0)),
        "runner_expected_rows": run_status.get("expected_rows"),
        "runner_status_checked": bool(run_status.get("checked", False)),
    }
    checks = {
        "top1_accuracy": observed["top1_accuracy"] >= thresholds["top1_accuracy"],
        "macro_f1": observed["macro_f1"] >= thresholds["macro_f1"],
        "min_class_recall": observed["min_class_recall"] >= thresholds["min_class_recall"],
        "hm_min_recall": observed["hm_min_recall"] >= thresholds["hm_min_recall"],
        "hm_pairwise_min_recall": observed["hm_pairwise_min_recall"] >= thresholds["hm_pairwise_min_recall"],
        "key_hard_negative_pair_min_recall": observed["key_hard_negative_pair_min_recall"] >= thresholds["key_hard_negative_pair_min_recall"],
        "min_validation_support_per_class": observed["min_validation_support_per_class"] >= thresholds["min_validation_support_per_class"],
        "runner_failures": observed["runner_failures"] == thresholds["runner_failures"],
        "runner_pending": observed["runner_pending"] == thresholds["runner_pending"],
        "runner_status_available": bool(run_status.get("checked", False)),
        "shadow_or_final_not_used": not bool(manifest.get("shadow_or_final_used", False)),
    }
    return {
        "generated_by": "analysis/train_v7b.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit_dir": output_dir.as_posix(),
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "stop_rule": "If three v7B development rounds fail, stop tuning this validation set and design v7B2 physical matrix.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v7B ten-material hard-negative models on measurement cubes.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cube-dir", default="results/accuracy_v3/v7b_hard_negative_dev")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--repeat-rounds", type=int, default=3)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--include-thickness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--status-csv", default="results/material_sorting/run_status_v7b_hard_negative_dev.csv")
    parser.add_argument("--n-jobs", type=int, default=min(4, max(1, os.cpu_count() or 1)), help="Worker threads for tree/XGBoost models. Use a small value to avoid saturating the workstation.")
    args = parser.parse_args()

    start_time = perf_counter()
    project_root = Path(args.project_root).resolve()
    cube_dir = project_root / args.cube_dir
    output_dir = project_root / (args.output_dir.strip() or args.cube_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_str_list(args.methods)
    n_jobs = max(1, int(args.n_jobs))
    log_progress(f"start cube_dir={cube_dir} output_dir={output_dir} methods={methods} repeat_rounds={args.repeat_rounds} n_jobs={n_jobs}", start_time=start_time)
    sk = require_sklearn()
    XGBClassifier = require_xgboost()

    cube, metadata, feature_names, cube_manifest = load_cube(cube_dir)
    log_progress(f"loaded cube_shape={cube.shape} metadata_rows={len(metadata)} feature_names={len(feature_names)}", start_time=start_time)
    if metadata["random_seed"].isin(DEFAULT_SHADOW_SEEDS).any() or bool(cube_manifest.get("shadow_or_final_used", False)):
        raise RuntimeError("Shadow/final seeds are present in v7B training metadata.")
    train_mask = metadata["split"].astype(str).eq("train").to_numpy()
    validation_mask = metadata["split"].astype(str).eq("validation").to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("v7B training requires non-empty train and validation splits.")

    x, model_feature_names = build_feature_matrix(cube, metadata, feature_names, args.include_thickness)
    labels = metadata["material"].astype(str).to_numpy()
    classes = [material for material in TARGET_MATERIALS if material in set(labels)]
    missing = sorted(set(TARGET_MATERIALS) - set(classes))
    if missing:
        print(f"Warning: target materials missing from cube metadata: {missing}")

    x_train = x[train_mask]
    x_validation = x[validation_mask]
    train_meta = metadata.loc[train_mask].reset_index(drop=True)
    validation_meta = metadata.loc[validation_mask].reset_index(drop=True)
    y_train = labels[train_mask]
    y_validation = labels[validation_mask]
    log_progress(
        f"built features train={x_train.shape} validation={x_validation.shape} classes={len(classes)} include_thickness={args.include_thickness}",
        start_time=start_time,
    )

    rows = []
    payloads = {}
    sample_weight: np.ndarray | None = None
    rounds = max(1, int(args.repeat_rounds))
    for round_id in range(1, rounds + 1):
        log_progress(f"round={round_id}/{rounds} start", start_time=start_time)
        for method in methods:
            log_progress(f"round={round_id} method={method} start", start_time=start_time)
            try:
                metrics, predictions, scores, class_array = evaluate_method(
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
                log_progress(
                    f"round={round_id} method={method} done top1={metrics['top1_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} hm_min={metrics['hm_min_recall']:.4f}",
                    start_time=start_time,
                )
            except Exception as exc:  # noqa: BLE001
                log_progress(f"round={round_id} method={method} error={exc}", start_time=start_time)
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
                    "model_size_rank": MODEL_RANK.get(method, 99),
                    "feature_count": int(x_train.shape[1]),
                    "error": str(exc),
                }
            rows.append(metrics)
        selected_so_far = choose_model(pd.DataFrame(rows))
        log_progress(
            f"round={round_id} selected_so_far={selected_so_far['method']} hm_min={float(selected_so_far['hm_min_recall']):.4f} macro_f1={float(selected_so_far['macro_f1']):.4f} top1={float(selected_so_far['top1_accuracy']):.4f}",
            start_time=start_time,
        )
        if (
            float(selected_so_far["hm_min_recall"]) >= 0.80
            and float(selected_so_far["hm_pairwise_min_recall"]) >= 0.78
            and float(selected_so_far["macro_f1"]) >= 0.82
            and float(selected_so_far["top1_accuracy"]) >= 0.85
        ):
            break
        key = (str(selected_so_far["method"]), int(selected_so_far["round_id"]))
        if key in payloads:
            best_predictions, best_scores, best_classes = payloads[key]
            decisions = decision_frame(validation_meta, best_predictions, best_scores, best_classes)
            sample_weight = update_weights(train_meta, validation_meta, decisions, args.hard_negative_weight)

    selection = pd.DataFrame(rows)
    selected = choose_model(selection)
    selected_key = (str(selected["method"]), int(selected["round_id"]))
    if selected_key not in payloads:
        raise RuntimeError(f"Selected v7B method has no prediction payload: {selected_key}")
    predictions, scores, class_array = payloads[selected_key]

    log_progress(f"selected method={selected['method']} round={int(selected['round_id'])}; writing audit tables", start_time=start_time)
    per_class = per_class_table(validation_meta, predictions, sk)
    decisions = decision_frame(validation_meta, predictions, scores, class_array)
    pairwise = pairwise_audit(validation_meta, predictions)
    view_ablation = evaluate_view_ablation(x_train, y_train, x_validation, y_validation, model_feature_names, sk, classes, n_jobs)
    split_audit = (
        metadata.groupby(["split", "random_seed", "material"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(["split", "random_seed", "material"])
    )
    run_status = runner_status(project_root, args.status_csv, cube_manifest)
    manifest = {
        "generated_by": "analysis/train_v7b.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v7b_hard_negative_dev_training",
        "development_only": True,
        "shadow_or_final_used": bool(cube_manifest.get("shadow_or_final_used", False)),
        "cube_dir": args.cube_dir,
        "output_dir": args.output_dir.strip() or args.cube_dir,
        "methods": methods,
        "repeat_rounds_requested": int(args.repeat_rounds),
        "repeat_rounds_observed": int(selection["round_id"].max()),
        "hard_negative_weight": float(args.hard_negative_weight),
        "tensor_shape": cube_manifest.get("tensor_shape", list(cube.shape)),
        "records_used": cube_manifest.get("records_used"),
        "calibration_records_used": cube_manifest.get("calibration_records_used"),
        "feature_count": int(x_train.shape[1]),
        "selected_method": selected["method"],
        "selected_round": int(selected["round_id"]),
        "key_hard_pairs": [list(pair) for pair in KEY_HARD_PAIRS],
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "xgboost_available": XGBClassifier is not None,
        },
    }
    gate = gate_report(output_dir, selected, per_class, pairwise, manifest, run_status)

    selection.to_csv(output_dir / "v7b_model_selection.csv", index=False, lineterminator="\n")
    pd.DataFrame([selected]).to_csv(output_dir / "v7b_development_summary.csv", index=False, lineterminator="\n")
    per_class.to_csv(output_dir / "v7b_per_class_recall.csv", index=False, lineterminator="\n")
    decisions.to_csv(output_dir / "validation_decisions.csv", index=False, lineterminator="\n")
    pairwise.to_csv(output_dir / "v7b_pairwise_hard_negative_audit.csv", index=False, lineterminator="\n")
    failure_analysis(per_class, decisions).to_csv(output_dir / "v7b_failure_analysis.csv", index=False, lineterminator="\n")
    view_ablation.to_csv(output_dir / "v7b_view_ablation.csv", index=False, lineterminator="\n")
    confusion_matrix_table(validation_meta, predictions, sk).to_csv(output_dir / "v7b_confusion_matrix.csv", lineterminator="\n")
    split_audit.to_csv(output_dir / "split_audit_training.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [
            {
                "timestamp_utc": manifest["generated_at_utc"],
                "protocol_name": manifest["protocol_name"],
                "evaluation_stage": "development_validation_only",
                "selected_method": selected["method"],
                "selected_round": int(selected["round_id"]),
                "top1_accuracy": float(selected["top1_accuracy"]),
                "macro_f1": float(selected["macro_f1"]),
                "hm_min_recall": float(selected["hm_min_recall"]),
                "hm_pairwise_min_recall": float(selected["hm_pairwise_min_recall"]),
                "gate_passed": bool(gate["gate_passed"]),
                "claim_safe": False,
                "next_action": "v7C_or_shadow_prep_if_pass" if gate["gate_passed"] else "v7B_round_or_v7B2_physics_matrix",
            }
        ]
    ).to_csv(output_dir / "experiment_registry.csv", index=False, lineterminator="\n")
    (output_dir / "strict_generalization_manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    (output_dir / "v7b_gate.json").write_bytes((json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    log_progress(f"Wrote v7B training audit to {output_dir}", start_time=start_time)
    log_progress(f"selected_method={selected['method']} round={int(selected['round_id'])} gate_passed={gate['gate_passed']}", start_time=start_time)


if __name__ == "__main__":
    main()
