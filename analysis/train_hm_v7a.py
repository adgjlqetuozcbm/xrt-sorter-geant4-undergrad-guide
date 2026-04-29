from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import material_sorting_v2 as v2


HM_PAIR = ["Hematite", "Magnetite"]
DEFAULT_METHODS = ["ExtraTrees", "XGBoost", "PCAExtraTrees", "HardNegativeExtraTrees", "HardNegativeXGBoost"]
MODEL_RANK = {
    "ExtraTrees": 0,
    "PCAExtraTrees": 1,
    "XGBoost": 2,
    "HardNegativeExtraTrees": 3,
    "HardNegativeXGBoost": 4,
}


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def require_sklearn():
    sk = v2.require_sklearn()
    from sklearn.decomposition import PCA

    from sklearn.metrics import f1_score, recall_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sk["PCA"] = PCA
    sk["f1_score"] = f1_score
    sk["recall_score"] = recall_score
    sk["make_pipeline"] = make_pipeline
    sk["StandardScaler"] = StandardScaler
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


def build_feature_matrix(cube: np.ndarray, metadata: pd.DataFrame, feature_names: list[str], include_thickness: bool) -> tuple[np.ndarray, list[str]]:
    x = cube.reshape((cube.shape[0], -1)).astype(np.float32)
    names = list(feature_names)
    if include_thickness:
        x = np.column_stack([x, metadata["thickness_mm"].astype(float).to_numpy(dtype=np.float32)])
        names.append("metadata__thickness_mm")
    return x, names


def label_arrays(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels = metadata["material"].astype(str).to_numpy()
    mapping = {material: index for index, material in enumerate(HM_PAIR)}
    y = np.array([mapping[label] for label in labels], dtype=int)
    return labels, y, mapping


def topk_accuracy(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray, k: int) -> float:
    if scores.size == 0:
        return math.nan
    order = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    return float(np.mean([truth in classes[row] for truth, row in zip(y_true, order)]))


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    sums = scores.sum(axis=1, keepdims=True)
    return np.divide(scores, sums, out=np.full_like(scores, 1.0 / scores.shape[1]), where=sums > 0)


def evaluate_predictions(method: str, round_id: int, y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray, classes: np.ndarray, sk) -> dict:
    recalls = sk["recall_score"](y_true, predictions, labels=np.array(HM_PAIR), average=None, zero_division=0)
    return {
        "method": method,
        "round_id": int(round_id),
        "samples": int(len(y_true)),
        "top1_accuracy": float(np.mean(y_true == predictions)),
        "top2_accuracy": topk_accuracy(y_true, scores, classes, min(2, len(classes))),
        "macro_f1": float(sk["f1_score"](y_true, predictions, labels=np.array(HM_PAIR), average="macro", zero_division=0)),
        "hematite_recall": float(recalls[0]),
        "magnetite_recall": float(recalls[1]),
        "hm_min_recall": float(np.min(recalls)),
        "pairwise_hm_min_recall": float(np.min(recalls)),
        "model_size_rank": MODEL_RANK.get(method, 99),
    }


def make_extra_trees(sk, random_state: int, weighted: bool = False):
    return sk["ExtraTreesClassifier"](
        n_estimators=900 if weighted else 700,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )


def fit_predict_sklearn(model, x_train: np.ndarray, labels_train: np.ndarray, x_eval: np.ndarray, sample_weight: np.ndarray | None = None):
    if sample_weight is None:
        model.fit(x_train, labels_train)
    else:
        model.fit(x_train, labels_train, sample_weight=sample_weight)
    predictions = np.asarray(model.predict(x_eval)).astype(str)
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(x_eval), dtype=float)
        classes = np.asarray(model.classes_).astype(str)
    else:
        classes = np.array(HM_PAIR)
        scores = np.zeros((len(predictions), len(classes)), dtype=float)
        for index, prediction in enumerate(predictions):
            scores[index, int(np.where(classes == prediction)[0][0])] = 1.0
    return predictions, normalize_scores(scores), classes


def fit_predict_xgboost(XGBClassifier, x_train: np.ndarray, y_train_int: np.ndarray, x_eval: np.ndarray, sample_weight: np.ndarray | None, random_state: int):
    model = XGBClassifier(
        n_estimators=450,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train_int, sample_weight=sample_weight)
    magnetite_probability = np.asarray(model.predict_proba(x_eval), dtype=float)[:, 1]
    scores = np.column_stack([1.0 - magnetite_probability, magnetite_probability])
    classes = np.array(HM_PAIR)
    predictions = classes[np.argmax(scores, axis=1)]
    return predictions, normalize_scores(scores), classes


def evaluate_method(
    method: str,
    round_id: int,
    x_train: np.ndarray,
    y_train_labels: np.ndarray,
    y_train_int: np.ndarray,
    x_eval: np.ndarray,
    y_eval_labels: np.ndarray,
    sample_weight: np.ndarray | None,
    sk,
    XGBClassifier,
    pca_components: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    if method == "XGBoost" and XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")
    if method == "HardNegativeXGBoost" and XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")

    if method == "ExtraTrees":
        predictions, scores, classes = fit_predict_sklearn(make_extra_trees(sk, 700 + round_id), x_train, y_train_labels, x_eval)
    elif method == "HardNegativeExtraTrees":
        predictions, scores, classes = fit_predict_sklearn(make_extra_trees(sk, 1700 + round_id, weighted=True), x_train, y_train_labels, x_eval, sample_weight)
    elif method == "PCAExtraTrees":
        components = max(2, min(pca_components, x_train.shape[0] - 1, x_train.shape[1]))
        model = sk["make_pipeline"](
            sk["StandardScaler"](),
            sk["PCA"](n_components=components, random_state=900 + round_id),
            make_extra_trees(sk, 1900 + round_id),
        )
        predictions, scores, classes = fit_predict_sklearn(model, x_train, y_train_labels, x_eval)
    elif method == "XGBoost":
        predictions, scores, classes = fit_predict_xgboost(XGBClassifier, x_train, y_train_int, x_eval, None, 1100 + round_id)
    elif method == "HardNegativeXGBoost":
        predictions, scores, classes = fit_predict_xgboost(XGBClassifier, x_train, y_train_int, x_eval, sample_weight, 2100 + round_id)
    else:
        raise ValueError(f"Unknown v7A method: {method}")

    metrics = evaluate_predictions(method, round_id, y_eval_labels, predictions, scores, classes, sk)
    return metrics, predictions, scores, classes


def choose_model(table: pd.DataFrame) -> dict:
    ranked = table.dropna(subset=["hm_min_recall", "pairwise_hm_min_recall", "macro_f1", "top1_accuracy"]).sort_values(
        ["hm_min_recall", "pairwise_hm_min_recall", "macro_f1", "top1_accuracy", "model_size_rank"],
        ascending=[False, False, False, False, True],
    )
    if ranked.empty:
        raise RuntimeError("No v7A method produced finite metrics.")
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
                "top2_candidates": ";".join(classes[order[idx, :2]].astype(str)),
                "thickness_mm": float(row.thickness_mm),
                "random_seed": int(row.random_seed),
                "sample_id": int(row.sample_id),
                "split": row.split,
            }
        )
    return pd.DataFrame(rows)


def per_class_table(metadata: pd.DataFrame, predictions: np.ndarray, sk) -> pd.DataFrame:
    y_true = metadata["material"].astype(str).to_numpy()
    recalls = sk["recall_score"](y_true, predictions, labels=np.array(HM_PAIR), average=None, zero_division=0)
    support = metadata["material"].value_counts().to_dict()
    return pd.DataFrame(
        [
            {"split": "validation", "material": material, "support": int(support.get(material, 0)), "recall": float(recall)}
            for material, recall in zip(HM_PAIR, recalls)
        ]
    )


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
                "failure_status": "pass" if float(row.recall) >= 0.72 else "fail",
                "next_action": "repeat_hard_negative_training_or_v7b" if float(row.recall) < 0.72 else "monitor",
            }
        )
    return pd.DataFrame(rows)


def update_weights(
    train_meta: pd.DataFrame,
    validation_meta: pd.DataFrame,
    decisions: pd.DataFrame,
    base_weight: float,
) -> np.ndarray:
    weights = np.ones(len(train_meta), dtype=float)
    merged = validation_meta.reset_index(drop=True).copy()
    merged["is_correct"] = decisions["is_correct"].to_numpy(dtype=bool)
    material_recall = merged.groupby("material")["is_correct"].mean().to_dict()
    thickness_recall = merged.groupby(["material", "thickness_mm"])["is_correct"].mean().to_dict()
    for index, row in enumerate(train_meta.itertuples(index=False)):
        recall = float(material_recall.get(row.material, 1.0))
        group_recall = float(thickness_recall.get((row.material, float(row.thickness_mm)), recall))
        if recall < 0.75:
            weights[index] += base_weight * (0.75 - recall) / 0.75
        if group_recall < 0.72:
            weights[index] += 0.5 * base_weight * (0.72 - group_recall) / 0.72
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
        "normal_wide_only": [i for i, name in enumerate(feature_names) if "normal_wide" in name or name == "metadata__thickness_mm"],
    }
    return {name: indices for name, indices in groups.items() if indices}


def evaluate_view_ablation(
    x_train: np.ndarray,
    y_train_labels: np.ndarray,
    x_eval: np.ndarray,
    y_eval_labels: np.ndarray,
    feature_names: list[str],
    sk,
) -> pd.DataFrame:
    rows = []
    for view_name, indices in view_feature_indices(feature_names).items():
        model = make_extra_trees(sk, 3100 + len(rows))
        predictions, scores, classes = fit_predict_sklearn(model, x_train[:, indices], y_train_labels, x_eval[:, indices])
        metrics = evaluate_predictions("ExtraTrees", 0, y_eval_labels, predictions, scores, classes, sk)
        metrics["view_name"] = view_name
        metrics["feature_count"] = int(len(indices))
        rows.append(metrics)
    return pd.DataFrame(rows)


def gate_report(output_dir: Path, selected: dict, per_class: pd.DataFrame, manifest: dict) -> dict:
    thresholds = {
        "hm_min_recall": 0.75,
        "pairwise_hm_min_recall": 0.72,
        "hematite_recall": 0.72,
        "magnetite_recall": 0.72,
        "min_class_support": 60,
    }
    observed = {
        "method": selected["method"],
        "round_id": int(selected["round_id"]),
        "hm_min_recall": float(selected["hm_min_recall"]),
        "pairwise_hm_min_recall": float(selected["pairwise_hm_min_recall"]),
        "hematite_recall": float(selected["hematite_recall"]),
        "magnetite_recall": float(selected["magnetite_recall"]),
        "min_class_support": int(per_class["support"].min()) if not per_class.empty else 0,
    }
    checks = {
        "hm_min_recall": observed["hm_min_recall"] >= thresholds["hm_min_recall"],
        "pairwise_hm_min_recall": observed["pairwise_hm_min_recall"] >= thresholds["pairwise_hm_min_recall"],
        "hematite_recall": observed["hematite_recall"] >= thresholds["hematite_recall"],
        "magnetite_recall": observed["magnetite_recall"] >= thresholds["magnetite_recall"],
        "min_class_support": observed["min_class_support"] >= thresholds["min_class_support"],
        "shadow_or_final_not_used": not bool(manifest.get("shadow_or_final_used", False)),
    }
    report = {
        "generated_by": "analysis/train_hm_v7a.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit_dir": output_dir.as_posix(),
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "repeat_rule": "If gate fails, repeat hard-negative training up to 3 rounds; after 3 failed rounds move to v7B.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v7A H/M models on exported measurement cubes.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cube-dir", default="results/accuracy_v3/v7a_hm_measurement_cube")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--repeat-rounds", type=int, default=3)
    parser.add_argument("--hard-negative-weight", type=float, default=4.0)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--include-thickness", action="store_true", default=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    cube_dir = project_root / args.cube_dir
    output_dir = project_root / (args.output_dir.strip() or args.cube_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_str_list(args.methods)
    sk = require_sklearn()
    XGBClassifier = require_xgboost()

    cube, metadata, feature_names, cube_manifest = load_cube(cube_dir)
    x, model_feature_names = build_feature_matrix(cube, metadata, feature_names, args.include_thickness)
    labels, y_int, _ = label_arrays(metadata)
    train_mask = metadata["split"].astype(str).eq("train").to_numpy()
    validation_mask = metadata["split"].astype(str).eq("validation").to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("v7A training requires non-empty train and validation splits.")
    if metadata.loc[~(train_mask | validation_mask), "random_seed"].isin(range(2301, 2307)).any():
        raise RuntimeError("Shadow seeds are present in v7A training metadata.")

    x_train = x[train_mask]
    x_validation = x[validation_mask]
    train_meta = metadata.loc[train_mask].reset_index(drop=True)
    validation_meta = metadata.loc[validation_mask].reset_index(drop=True)
    y_train_labels = labels[train_mask]
    y_validation_labels = labels[validation_mask]
    y_train_int = y_int[train_mask]

    rows = []
    best_payload = None
    sample_weight: np.ndarray | None = None
    rounds = max(1, int(args.repeat_rounds))
    for round_id in range(1, rounds + 1):
        for method in methods:
            try:
                metrics, predictions, scores, classes = evaluate_method(
                    method,
                    round_id,
                    x_train,
                    y_train_labels,
                    y_train_int,
                    x_validation,
                    y_validation_labels,
                    sample_weight if method.startswith("HardNegative") else None,
                    sk,
                    XGBClassifier,
                    args.pca_components,
                )
                metrics["feature_count"] = int(x_train.shape[1])
            except Exception as exc:  # noqa: BLE001
                metrics = {
                    "method": method,
                    "round_id": int(round_id),
                    "samples": int(len(y_validation_labels)),
                    "top1_accuracy": math.nan,
                    "top2_accuracy": math.nan,
                    "macro_f1": math.nan,
                    "hematite_recall": math.nan,
                    "magnetite_recall": math.nan,
                    "hm_min_recall": math.nan,
                    "pairwise_hm_min_recall": math.nan,
                    "model_size_rank": MODEL_RANK.get(method, 99),
                    "feature_count": int(x_train.shape[1]),
                    "error": str(exc),
                }
                predictions = np.array([], dtype=str)
                scores = np.zeros((0, 2), dtype=float)
                classes = np.array(HM_PAIR)
            rows.append(metrics)
            current_table = pd.DataFrame(rows)
            current_selected = choose_model(current_table)
            if best_payload is None or (
                str(metrics["method"]) == str(current_selected["method"]) and int(metrics["round_id"]) == int(current_selected["round_id"])
            ):
                if len(predictions):
                    best_payload = (current_selected, predictions, scores, classes)
        selected_so_far = choose_model(pd.DataFrame(rows))
        if float(selected_so_far["hm_min_recall"]) >= 0.75 and float(selected_so_far["pairwise_hm_min_recall"]) >= 0.72:
            break
        if best_payload is not None:
            _, best_predictions, best_scores, best_classes = best_payload
            decisions = decision_frame(validation_meta, best_predictions, best_scores, best_classes)
            sample_weight = update_weights(train_meta, validation_meta, decisions, args.hard_negative_weight)

    selection = pd.DataFrame(rows)
    selected = choose_model(selection)
    if best_payload is None or str(best_payload[0]["method"]) != str(selected["method"]) or int(best_payload[0]["round_id"]) != int(selected["round_id"]):
        metrics, predictions, scores, classes = evaluate_method(
            str(selected["method"]),
            int(selected["round_id"]),
            x_train,
            y_train_labels,
            y_train_int,
            x_validation,
            y_validation_labels,
            sample_weight if str(selected["method"]).startswith("HardNegative") else None,
            sk,
            XGBClassifier,
            args.pca_components,
        )
    else:
        _, predictions, scores, classes = best_payload

    per_class = per_class_table(validation_meta, predictions, sk)
    decisions = decision_frame(validation_meta, predictions, scores, classes)
    pairwise = pd.DataFrame(
        [
            {
                "split": "validation",
                "method": selected["method"],
                "round_id": int(selected["round_id"]),
                "samples": int(len(validation_meta)),
                "hm_min_recall": float(selected["hm_min_recall"]),
                "pairwise_hm_min_recall": float(selected["pairwise_hm_min_recall"]),
                "hematite_recall": float(selected["hematite_recall"]),
                "magnetite_recall": float(selected["magnetite_recall"]),
                "macro_f1": float(selected["macro_f1"]),
                "top1_accuracy": float(selected["top1_accuracy"]),
            }
        ]
    )
    view_ablation = evaluate_view_ablation(x_train, y_train_labels, x_validation, y_validation_labels, model_feature_names, sk)
    split_audit = (
        metadata.groupby(["split", "random_seed", "material"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
        .sort_values(["split", "random_seed", "material"])
    )
    manifest = {
        "generated_by": "analysis/train_hm_v7a.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v7a_hm_measurement_cube_training",
        "development_only": True,
        "shadow_or_final_used": bool(cube_manifest.get("shadow_or_final_used", False)),
        "cube_dir": args.cube_dir,
        "output_dir": args.output_dir.strip() or args.cube_dir,
        "methods": methods,
        "repeat_rounds_requested": int(args.repeat_rounds),
        "repeat_rounds_observed": int(selection["round_id"].max()),
        "hard_negative_weight": float(args.hard_negative_weight),
        "tensor_shape": cube_manifest.get("tensor_shape", list(cube.shape)),
        "feature_count": int(x_train.shape[1]),
        "selected_method": selected["method"],
        "selected_round": int(selected["round_id"]),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "xgboost_available": XGBClassifier is not None,
        },
    }
    gate = gate_report(output_dir, selected, per_class, manifest)

    selection.to_csv(output_dir / "v7a_model_selection.csv", index=False, lineterminator="\n")
    pd.DataFrame([selected]).to_csv(output_dir / "v7a_development_summary.csv", index=False, lineterminator="\n")
    per_class.to_csv(output_dir / "per_class_recall_validation.csv", index=False, lineterminator="\n")
    decisions.to_csv(output_dir / "validation_decisions.csv", index=False, lineterminator="\n")
    pairwise.to_csv(output_dir / "v7a_hm_pairwise_audit.csv", index=False, lineterminator="\n")
    failure_analysis(per_class, decisions).to_csv(output_dir / "v7a_failure_analysis.csv", index=False, lineterminator="\n")
    view_ablation.to_csv(output_dir / "v7a_view_ablation.csv", index=False, lineterminator="\n")
    split_audit.to_csv(output_dir / "split_audit_training.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [
            {
                "timestamp_utc": manifest["generated_at_utc"],
                "protocol_name": manifest["protocol_name"],
                "evaluation_stage": "development_validation_only",
                "selected_method": selected["method"],
                "selected_round": int(selected["round_id"]),
                "hm_min_recall": float(selected["hm_min_recall"]),
                "pairwise_hm_min_recall": float(selected["pairwise_hm_min_recall"]),
                "gate_passed": bool(gate["gate_passed"]),
                "claim_safe": False,
                "next_action": "v7B_if_v7A_repeat_fails" if not gate["gate_passed"] else "prepare_v7B_ten_material_development",
            }
        ]
    ).to_csv(output_dir / "experiment_registry.csv", index=False, lineterminator="\n")
    (output_dir / "strict_generalization_manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    (output_dir / "v7a_gate.json").write_bytes(
        (json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    print(f"Wrote v7A training audit to {output_dir}")
    print(f"selected_method={selected['method']} round={int(selected['round_id'])} gate_passed={gate['gate_passed']}")


if __name__ == "__main__":
    main()
