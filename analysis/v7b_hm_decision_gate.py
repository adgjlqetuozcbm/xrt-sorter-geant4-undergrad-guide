from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import train_v7b as v7b


HM_PAIR = v7b.HM_PAIR
TARGET_MATERIALS = v7b.TARGET_MATERIALS
KEY_HARD_PAIRS = v7b.KEY_HARD_PAIRS
BASELINE_HM_MIN_RECALL = 0.6083333333333333
EPS = 1e-9
VARIANT_RANK = {"normal_narrow": 0, "normal_wide": 1, "oblique_10deg": 2, "oblique_20deg": 3}


def log_progress(message: str, *, start_time: float | None = None) -> None:
    elapsed = ""
    if start_time is not None:
        elapsed = f" elapsed={perf_counter() - start_time:.1f}s"
    print(f"[v7b-hm-decision] {datetime.now().isoformat(timespec='seconds')} {message}{elapsed}", flush=True)


def parse_feature_name(name: str) -> dict[str, str]:
    if name == "metadata__thickness_mm":
        return {"kind": "metadata", "source_id": "", "detector": "", "energy": "", "variant": ""}
    parts = str(name).split("__", 2)
    if len(parts) < 2:
        return {"kind": "unknown", "source_id": "", "detector": "", "energy": "", "variant": ""}
    source_id, detector = parts[0], parts[1]
    match = re.match(r"mono_([0-9p.]+)kev_(.+)", source_id)
    energy = ""
    variant = ""
    if match:
        energy = match.group(1).replace("p", ".")
        variant = match.group(2)
    return {"kind": "cube", "source_id": source_id, "detector": detector, "energy": energy, "variant": variant}


def source_sort_key(source_id: str) -> tuple[float, int, str]:
    parsed = parse_feature_name(f"{source_id}__transmission__dummy")
    try:
        energy = float(parsed["energy"]) if parsed["energy"] else math.inf
    except ValueError:
        energy = math.inf
    return energy, VARIANT_RANK.get(parsed["variant"], 99), source_id


def feature_index_table(feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(feature_names):
        parsed = parse_feature_name(name)
        parsed["index"] = int(index)
        parsed["feature_name"] = name
        rows.append(parsed)
    return pd.DataFrame(rows)


def indices_from_mask(table: pd.DataFrame, mask: pd.Series) -> list[int]:
    metadata_mask = table["kind"].eq("metadata")
    return table.loc[mask | metadata_mask, "index"].astype(int).tolist()


def focused_feature_views(feature_names: list[str]) -> dict[str, list[int]]:
    table = feature_index_table(feature_names)
    cube = table["kind"].eq("cube")
    is_trans = table["detector"].eq("transmission")
    is_side = table["detector"].eq("side_scatter")
    is_high = table["energy"].isin(["120", "150", "200", "120.0", "150.0", "200.0"])
    is_oblique = table["variant"].str.startswith("oblique_", na=False)
    is_oblique20 = table["variant"].eq("oblique_20deg")
    is_normal_wide = table["variant"].eq("normal_wide")
    is_normal_narrow = table["variant"].eq("normal_narrow")
    views = {
        "all": table["index"].astype(int).tolist(),
        "transmission_only": indices_from_mask(table, cube & is_trans),
        "side_scatter_only": indices_from_mask(table, cube & is_side),
        "transmission_high_energy": indices_from_mask(table, cube & is_trans & is_high),
        "transmission_oblique": indices_from_mask(table, cube & is_trans & is_oblique),
        "transmission_oblique_20deg": indices_from_mask(table, cube & is_trans & is_oblique20),
        "transmission_normal_wide": indices_from_mask(table, cube & is_trans & is_normal_wide),
        "transmission_normal_narrow": indices_from_mask(table, cube & is_trans & is_normal_narrow),
        "oblique_only": indices_from_mask(table, cube & is_oblique),
        "high_energy_only": indices_from_mask(table, cube & is_high),
    }
    return {name: sorted(set(indices)) for name, indices in views.items() if indices}


def source_audit_views(feature_names: list[str]) -> dict[str, tuple[str, list[int]]]:
    table = feature_index_table(feature_names)
    cube = table["kind"].eq("cube")
    views: dict[str, tuple[str, list[int]]] = {}
    for detector in sorted(item for item in table.loc[cube, "detector"].dropna().unique() if item):
        mask = cube & table["detector"].eq(detector)
        views[f"detector::{detector}"] = ("detector", indices_from_mask(table, mask))
    for energy in sorted((item for item in table.loc[cube, "energy"].dropna().unique() if item), key=lambda x: float(x)):
        mask = cube & table["energy"].eq(energy)
        views[f"energy::{energy}kev"] = ("energy", indices_from_mask(table, mask))
    for variant in sorted(item for item in table.loc[cube, "variant"].dropna().unique() if item):
        mask = cube & table["variant"].eq(variant)
        views[f"variant::{variant}"] = ("variant", indices_from_mask(table, mask))
    for source_id in sorted((item for item in table.loc[cube, "source_id"].dropna().unique() if item), key=source_sort_key):
        mask = cube & table["source_id"].eq(source_id)
        views[f"source::{source_id}"] = ("source", indices_from_mask(table, mask))
    return {name: (kind, sorted(set(indices))) for name, (kind, indices) in views.items() if indices}


def make_extra_trees(sk, random_state: int, n_jobs: int, n_estimators: int = 520):
    return sk["ExtraTreesClassifier"](
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )


def evaluate_full_model(
    method: str,
    round_id: int,
    view_name: str,
    indices: list[int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    sk,
    classes: list[str],
    n_jobs: int,
    random_state: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model = make_extra_trees(sk, random_state=random_state, n_jobs=n_jobs)
    predictions, scores, class_array = v7b.fit_predict_sklearn(
        model,
        x_train[:, indices],
        y_train,
        x_validation[:, indices],
        classes,
    )
    metrics = v7b.evaluate_predictions(method, round_id, y_validation, predictions, scores, class_array, sk)
    metrics["view_name"] = view_name
    metrics["feature_count"] = int(len(indices))
    metrics["candidate_policy"] = ""
    return metrics, predictions, scores, class_array


def pairwise_rerank_scores(
    base_scores: np.ndarray,
    base_predictions: np.ndarray,
    pair_scores: np.ndarray,
    classes: list[str],
    policy: str,
) -> np.ndarray:
    class_array = np.array(classes)
    hm_indices = [classes.index(HM_PAIR[0]), classes.index(HM_PAIR[1])]
    order = np.argsort(base_scores, axis=1)[:, ::-1]
    reranked = base_scores.copy()
    for row in range(base_scores.shape[0]):
        top3 = set(class_array[order[row, : min(3, len(classes))]])
        apply_pair = False
        if policy == "base_hm_only":
            apply_pair = str(base_predictions[row]) in HM_PAIR
        elif policy == "both_hm_top3":
            apply_pair = HM_PAIR[0] in top3 and HM_PAIR[1] in top3
        elif policy == "either_hm_top3":
            apply_pair = HM_PAIR[0] in top3 or HM_PAIR[1] in top3
        else:
            raise ValueError(f"Unknown rerank policy: {policy}")
        if not apply_pair:
            continue
        hm_mass = max(float(base_scores[row, hm_indices].sum()), 0.15)
        reranked[row, hm_indices[0]] = pair_scores[row, hm_indices[0]] * hm_mass
        reranked[row, hm_indices[1]] = pair_scores[row, hm_indices[1]] * hm_mass
    return v7b.normalize_scores(reranked)


def evaluate_pairwise_rerank(
    method: str,
    round_id: int,
    view_name: str,
    indices: list[int],
    policy: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    sk,
    classes: list[str],
    n_jobs: int,
    random_state: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    base_model = make_extra_trees(sk, random_state=random_state, n_jobs=n_jobs)
    base_predictions, base_scores, class_array = v7b.fit_predict_sklearn(
        base_model,
        x_train[:, indices],
        y_train,
        x_validation[:, indices],
        classes,
    )
    hm_train_mask = np.isin(y_train.astype(str), np.array(HM_PAIR))
    pair_model = make_extra_trees(sk, random_state=random_state + 97, n_jobs=n_jobs, n_estimators=420)
    pair_model.fit(x_train[hm_train_mask][:, indices], y_train[hm_train_mask])
    pair_raw = np.asarray(pair_model.predict_proba(x_validation[:, indices]), dtype=float)
    pair_scores = v7b.align_scores(pair_raw, np.asarray(pair_model.classes_), classes)
    reranked = pairwise_rerank_scores(base_scores, base_predictions, pair_scores, classes, policy)
    predictions = np.array(classes)[np.argmax(reranked, axis=1)]
    metrics = v7b.evaluate_predictions(method, round_id, y_validation, predictions, reranked, class_array, sk)
    metrics["view_name"] = view_name
    metrics["feature_count"] = int(len(indices))
    metrics["candidate_policy"] = policy
    return metrics, predictions, reranked, class_array


def hm_pair_only_diagnostics(
    view_indices: dict[str, list[int]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    sk,
    n_jobs: int,
) -> pd.DataFrame:
    rows = []
    hm_train_mask = np.isin(y_train.astype(str), np.array(HM_PAIR))
    hm_validation_mask = np.isin(y_validation.astype(str), np.array(HM_PAIR))
    pair_classes = list(HM_PAIR)
    for offset, (view_name, indices) in enumerate(view_indices.items()):
        model = make_extra_trees(sk, random_state=6200 + offset, n_jobs=n_jobs, n_estimators=520)
        model.fit(x_train[hm_train_mask][:, indices], y_train[hm_train_mask])
        predictions, scores, class_array = v7b.fit_predict_sklearn(
            model,
            x_train[hm_train_mask][:, indices],
            y_train[hm_train_mask],
            x_validation[hm_validation_mask][:, indices],
            pair_classes,
        )
        y_pair = y_validation[hm_validation_mask]
        recalls = v7b.pair_recalls(y_pair, predictions, HM_PAIR)
        rows.append(
            {
                "view_name": view_name,
                "feature_count": int(len(indices)),
                "support": int(hm_validation_mask.sum()),
                "pair_accuracy": float(np.mean(y_pair == predictions)),
                "hematite_recall": float(recalls.get(HM_PAIR[0], 0.0)),
                "magnetite_recall": float(recalls.get(HM_PAIR[1], 0.0)),
                "hm_pair_only_min_recall": float(min(recalls.values())) if recalls else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["hm_pair_only_min_recall", "pair_accuracy"], ascending=[False, False])


def hm_by_thickness(metadata: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    rows = []
    y_true = metadata["material"].astype(str).to_numpy()
    for material in HM_PAIR:
        for thickness in sorted(metadata.loc[metadata["material"].astype(str).eq(material), "thickness_mm"].astype(float).unique()):
            mask = (y_true == material) & np.isclose(metadata["thickness_mm"].astype(float).to_numpy(), float(thickness))
            support = int(mask.sum())
            correct = int(np.sum(predictions[mask] == material)) if support else 0
            rows.append(
                {
                    "material": material,
                    "thickness_mm": float(thickness),
                    "support": support,
                    "recall": float(correct / support) if support else 0.0,
                    "miss_count": int(support - correct),
                }
            )
    return pd.DataFrame(rows)


def hm_topk_audit(metadata: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    y_true = metadata["material"].astype(str).to_numpy()
    order = np.argsort(scores, axis=1)[:, ::-1]
    rows = []
    for material in [HM_PAIR[0], HM_PAIR[1], "H/M pooled"]:
        if material == "H/M pooled":
            mask = np.isin(y_true, np.array(HM_PAIR))
        else:
            mask = y_true == material
        support = int(mask.sum())
        if support == 0:
            continue
        top1 = classes[order[mask, :1]]
        top2 = classes[order[mask, : min(2, len(classes))]]
        top3 = classes[order[mask, : min(3, len(classes))]]
        truths = y_true[mask]
        correct_top1 = np.array([truth in row for truth, row in zip(truths, top1)])
        correct_top2 = np.array([truth in row for truth, row in zip(truths, top2)])
        correct_top3 = np.array([truth in row for truth, row in zip(truths, top3)])
        both_hm_top3 = np.array([HM_PAIR[0] in row and HM_PAIR[1] in row for row in top3])
        reciprocal = int(np.sum(((truths == HM_PAIR[0]) & (predictions[mask] == HM_PAIR[1])) | ((truths == HM_PAIR[1]) & (predictions[mask] == HM_PAIR[0]))))
        rows.append(
            {
                "material": material,
                "support": support,
                "top1_contains_truth": float(correct_top1.mean()),
                "top2_contains_truth": float(correct_top2.mean()),
                "top3_contains_truth": float(correct_top3.mean()),
                "both_hm_in_top3": float(both_hm_top3.mean()),
                "reciprocal_hm_misses": reciprocal,
            }
        )
    return pd.DataFrame(rows)


def evaluate_source_views(
    audit_views: dict[str, tuple[str, list[int]]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    sk,
    classes: list[str],
    n_jobs: int,
    start_time: float,
) -> pd.DataFrame:
    rows = []
    for offset, (view_name, (view_type, indices)) in enumerate(audit_views.items()):
        log_progress(f"source/view audit {offset + 1}/{len(audit_views)} view={view_name}", start_time=start_time)
        metrics, _, _, _ = evaluate_full_model(
            method="SourceViewExtraTrees",
            round_id=0,
            view_name=view_name,
            indices=indices,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            sk=sk,
            classes=classes,
            n_jobs=n_jobs,
            random_state=7100 + offset,
        )
        metrics["view_type"] = view_type
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["hm_min_recall", "macro_f1", "top1_accuracy"], ascending=[False, False, False])


def rank_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values(
        ["hm_min_recall", "hm_pairwise_min_recall", "key_hard_negative_pair_min_recall", "macro_f1", "top1_accuracy"],
        ascending=[False, False, False, False, False],
    )


def gate_from_selected(
    selected: dict,
    topk: pd.DataFrame,
    by_thickness: pd.DataFrame,
    source_audit: pd.DataFrame,
    pair_only: pd.DataFrame,
    baseline_gate: dict,
    manifest: dict,
) -> dict:
    hm_thickness_min = float(by_thickness["recall"].min()) if not by_thickness.empty else 0.0
    hm_top3 = float(topk.loc[topk["material"].eq("H/M pooled"), "top3_contains_truth"].iloc[0]) if not topk.empty else 0.0
    hm_top2 = float(topk.loc[topk["material"].eq("H/M pooled"), "top2_contains_truth"].iloc[0]) if not topk.empty else 0.0
    max_view_hm = float(source_audit["hm_min_recall"].max()) if not source_audit.empty else 0.0
    max_pair_only = float(pair_only["hm_pair_only_min_recall"].max()) if not pair_only.empty else 0.0
    hm_min = float(selected["hm_min_recall"])
    improvement = hm_min - BASELINE_HM_MIN_RECALL
    structural_signal = bool(max_view_hm >= 0.70 or max_pair_only >= 0.72 or hm_top3 >= 0.85)

    if (
        float(selected["top1_accuracy"]) >= 0.85
        and float(selected["macro_f1"]) >= 0.82
        and float(selected["min_class_recall"]) >= 0.70
        and hm_min >= 0.80
        and float(selected["hm_pairwise_min_recall"]) >= 0.78
        and float(selected["key_hard_negative_pair_min_recall"]) >= 0.75
        and hm_thickness_min >= 0.70
    ):
        next_decision = "no_advanced_model_needed"
    elif 0.70 <= hm_min < 0.80 and structural_signal:
        next_decision = "start_v7c_advanced_model"
    elif hm_min <= 0.65 and max_view_hm <= 0.68 and max_pair_only <= 0.68:
        next_decision = "prioritize_v7b2_physics_matrix"
    elif 0.65 < hm_min < 0.70:
        next_decision = "allow_one_small_v7c_feasibility"
    else:
        next_decision = "diagnose_before_large_advanced_model"

    observed = {
        "method": selected["method"],
        "round_id": int(selected["round_id"]),
        "view_name": selected.get("view_name", ""),
        "candidate_policy": selected.get("candidate_policy", ""),
        "top1_accuracy": float(selected["top1_accuracy"]),
        "macro_f1": float(selected["macro_f1"]),
        "min_class_recall": float(selected["min_class_recall"]),
        "hm_min_recall": hm_min,
        "hm_pairwise_min_recall": float(selected["hm_pairwise_min_recall"]),
        "key_hard_negative_pair_min_recall": float(selected["key_hard_negative_pair_min_recall"]),
        "hm_improvement_over_r1": float(improvement),
        "hm_by_thickness_min_recall": hm_thickness_min,
        "hm_top2_contains_truth": hm_top2,
        "hm_top3_contains_truth": hm_top3,
        "max_source_view_hm_min_recall": max_view_hm,
        "max_pair_only_hm_min_recall": max_pair_only,
    }
    checks = {
        "r2_success_hm_gain": bool(improvement >= 0.08 and hm_min >= 0.69),
        "r2_success_top1": bool(float(selected["top1_accuracy"]) >= 0.85),
        "r2_success_macro_f1": bool(float(selected["macro_f1"]) >= 0.84),
        "no_advanced_model_top1": bool(float(selected["top1_accuracy"]) >= 0.85),
        "no_advanced_model_macro_f1": bool(float(selected["macro_f1"]) >= 0.82),
        "no_advanced_model_min_class_recall": bool(float(selected["min_class_recall"]) >= 0.70),
        "no_advanced_model_hm_min_recall": bool(hm_min >= 0.80),
        "no_advanced_model_hm_pairwise": bool(float(selected["hm_pairwise_min_recall"]) >= 0.78),
        "no_advanced_model_key_pair": bool(float(selected["key_hard_negative_pair_min_recall"]) >= 0.75),
        "no_advanced_model_hm_thickness": bool(hm_thickness_min >= 0.70),
        "shadow_or_final_not_used": not bool(manifest.get("shadow_or_final_used", False)),
    }
    return {
        "generated_by": "analysis/v7b_hm_decision_gate.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "v7b_r2_r3_hm_decision_gate",
        "baseline_gate": baseline_gate.get("observed", {}),
        "observed": observed,
        "checks": checks,
        "gate_passed": all(
            [
                checks["no_advanced_model_top1"],
                checks["no_advanced_model_macro_f1"],
                checks["no_advanced_model_min_class_recall"],
                checks["no_advanced_model_hm_min_recall"],
                checks["no_advanced_model_hm_pairwise"],
                checks["no_advanced_model_key_pair"],
                checks["no_advanced_model_hm_thickness"],
                checks["shadow_or_final_not_used"],
            ]
        ),
        "advanced_model_decision": next_decision,
        "structural_signal_for_v7c": structural_signal,
        "stop_rule": "Do not run shadow/final. If R2/R3 cannot move H/M beyond the decision thresholds, redesign v7B2 physics before large advanced models.",
    }


def write_report(output_dir: Path, gate: dict, selected: dict) -> None:
    observed = gate["observed"]
    text = f"""# v7B H/M Decision Gate

Generated: {gate['generated_at_utc']}

Selected method: {selected['method']} round {int(selected['round_id'])}

- top1_accuracy: {observed['top1_accuracy']:.4f}
- macro_f1: {observed['macro_f1']:.4f}
- H/M min recall: {observed['hm_min_recall']:.4f}
- H/M improvement over R1: {observed['hm_improvement_over_r1']:.4f}
- H/M top-3 contains truth: {observed['hm_top3_contains_truth']:.4f}
- max source/view H/M min recall: {observed['max_source_view_hm_min_recall']:.4f}
- max H/M-only pair min recall: {observed['max_pair_only_hm_min_recall']:.4f}

Decision: `{gate['advanced_model_decision']}`

Gate passed: `{gate['gate_passed']}`

Shadow/final used: `false`
"""
    (output_dir / "v7b_hm_decision_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v7B R2/R3 H/M decision gate and advanced-model trigger audit.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cube-dir", default="results/accuracy_v3/v7b_hard_negative_dev")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--n-jobs", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--skip-source-audit", action="store_true")
    args = parser.parse_args()

    start_time = perf_counter()
    project_root = Path(args.project_root).resolve()
    cube_dir = project_root / args.cube_dir
    output_dir = project_root / (args.output_dir.strip() or args.cube_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = max(1, int(args.n_jobs))
    log_progress(f"start cube_dir={cube_dir} output_dir={output_dir} n_jobs={n_jobs}", start_time=start_time)

    sk = v7b.require_sklearn()
    cube, metadata, raw_feature_names, cube_manifest = v7b.load_cube(cube_dir)
    if metadata["random_seed"].isin(v7b.DEFAULT_SHADOW_SEEDS).any() or bool(cube_manifest.get("shadow_or_final_used", False)):
        raise RuntimeError("Shadow/final seeds are present in v7B decision gate inputs.")

    x, feature_names = v7b.build_feature_matrix(cube, metadata, raw_feature_names, include_thickness=True)
    labels = metadata["material"].astype(str).to_numpy()
    classes = [material for material in TARGET_MATERIALS if material in set(labels)]
    train_mask = metadata["split"].astype(str).eq("train").to_numpy()
    validation_mask = metadata["split"].astype(str).eq("validation").to_numpy()
    x_train = x[train_mask]
    x_validation = x[validation_mask]
    y_train = labels[train_mask]
    y_validation = labels[validation_mask]
    validation_meta = metadata.loc[validation_mask].reset_index(drop=True)
    log_progress(f"loaded train={x_train.shape} validation={x_validation.shape} classes={len(classes)}", start_time=start_time)

    baseline_gate_path = output_dir / "v7b_gate.json"
    baseline_gate = json.loads(baseline_gate_path.read_text(encoding="utf-8")) if baseline_gate_path.exists() else {}
    if baseline_gate_path.exists():
        (output_dir / "v7b_r1_gate_baseline.json").write_text(json.dumps(baseline_gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    views = focused_feature_views(feature_names)
    full_model_specs: list[tuple[str, str, str, list[int]]] = [
        ("R2ExtraTreesAll", "all", "", views["all"]),
        ("R2ExtraTreesTransmission", "transmission_only", "", views["transmission_only"]),
        ("R2ExtraTreesTransmissionHighEnergy", "transmission_high_energy", "", views["transmission_high_energy"]),
        ("R2ExtraTreesTransmissionOblique", "transmission_oblique", "", views["transmission_oblique"]),
        ("R2ExtraTreesTransmissionOblique20", "transmission_oblique_20deg", "", views["transmission_oblique_20deg"]),
        ("R2ExtraTreesTransmissionWide", "transmission_normal_wide", "", views["transmission_normal_wide"]),
        ("R2ExtraTreesObliqueOnly", "oblique_only", "", views["oblique_only"]),
    ]
    pairwise_specs: list[tuple[str, str, str, list[int]]] = [
        ("R2HMPairwiseTransmissionBaseHM", "transmission_only", "base_hm_only", views["transmission_only"]),
        ("R2HMPairwiseTransmissionBothTop3", "transmission_only", "both_hm_top3", views["transmission_only"]),
        ("R2HMPairwiseAllBaseHM", "all", "base_hm_only", views["all"]),
    ]

    rows = []
    payloads: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for round_id in [2]:
        for offset, (method, view_name, _, indices) in enumerate(full_model_specs):
            log_progress(f"round={round_id} method={method} view={view_name} start", start_time=start_time)
            metrics, predictions, scores, class_array = evaluate_full_model(
                method,
                round_id,
                view_name,
                indices,
                x_train,
                y_train,
                x_validation,
                y_validation,
                sk,
                classes,
                n_jobs,
                8200 + offset,
            )
            rows.append(metrics)
            payloads[(method, round_id)] = (predictions, scores, class_array)
            log_progress(f"round={round_id} method={method} done hm_min={metrics['hm_min_recall']:.4f} top1={metrics['top1_accuracy']:.4f}", start_time=start_time)
        for offset, (method, view_name, policy, indices) in enumerate(pairwise_specs):
            log_progress(f"round={round_id} method={method} policy={policy} start", start_time=start_time)
            metrics, predictions, scores, class_array = evaluate_pairwise_rerank(
                method,
                round_id,
                view_name,
                indices,
                policy,
                x_train,
                y_train,
                x_validation,
                y_validation,
                sk,
                classes,
                n_jobs,
                9100 + offset,
            )
            rows.append(metrics)
            payloads[(method, round_id)] = (predictions, scores, class_array)
            log_progress(f"round={round_id} method={method} done hm_min={metrics['hm_min_recall']:.4f} top1={metrics['top1_accuracy']:.4f}", start_time=start_time)

    selection = rank_table(pd.DataFrame(rows))
    best_r2 = selection.iloc[0].to_dict()
    run_r3 = bool(float(best_r2["hm_min_recall"]) >= 0.72 and float(best_r2["hm_min_recall"]) - BASELINE_HM_MIN_RECALL >= 0.08 and int(args.max_rounds) >= 3)
    if run_r3:
        top_methods = selection.head(3)
        for row_offset, row in enumerate(top_methods.itertuples(index=False)):
            method = str(row.method).replace("R2", "R3", 1)
            view_name = str(row.view_name)
            policy = str(row.candidate_policy)
            indices = views[view_name]
            log_progress(f"round=3 method={method} view={view_name} policy={policy} start", start_time=start_time)
            if policy:
                metrics, predictions, scores, class_array = evaluate_pairwise_rerank(
                    method,
                    3,
                    view_name,
                    indices,
                    policy,
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    sk,
                    classes,
                    n_jobs,
                    10100 + row_offset,
                )
            else:
                metrics, predictions, scores, class_array = evaluate_full_model(
                    method,
                    3,
                    view_name,
                    indices,
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    sk,
                    classes,
                    n_jobs,
                    10100 + row_offset,
                )
            rows.append(metrics)
            payloads[(method, 3)] = (predictions, scores, class_array)
            log_progress(f"round=3 method={method} done hm_min={metrics['hm_min_recall']:.4f} top1={metrics['top1_accuracy']:.4f}", start_time=start_time)
        selection = rank_table(pd.DataFrame(rows))

    selected = selection.iloc[0].to_dict()
    selected_key = (str(selected["method"]), int(selected["round_id"]))
    predictions, scores, class_array = payloads[selected_key]
    topk = hm_topk_audit(validation_meta, predictions, scores, class_array)
    by_thickness = hm_by_thickness(validation_meta, predictions)
    pair_only = hm_pair_only_diagnostics(views, x_train, y_train, x_validation, y_validation, sk, n_jobs)
    if args.skip_source_audit:
        source_audit = pd.DataFrame()
    else:
        source_audit = evaluate_source_views(source_audit_views(feature_names), x_train, y_train, x_validation, y_validation, sk, classes, n_jobs, start_time)

    gate = gate_from_selected(selected, topk, by_thickness, source_audit, pair_only, baseline_gate, cube_manifest)

    selection.to_csv(output_dir / "v7b_r2_r3_model_selection.csv", index=False, lineterminator="\n")
    selection.loc[selection["view_name"].astype(str).ne("")].to_csv(
        output_dir / "v7b_hm_view_ablation_summary.csv",
        index=False,
        lineterminator="\n",
    )
    by_thickness.to_csv(output_dir / "v7b_hm_by_thickness_recall.csv", index=False, lineterminator="\n")
    source_audit.to_csv(output_dir / "v7b_hm_by_source_energy_angle_view.csv", index=False, lineterminator="\n")
    pair_only.to_csv(output_dir / "v7b_hm_pair_only_expert_diagnostic.csv", index=False, lineterminator="\n")
    topk.to_csv(output_dir / "v7b_hm_topk_containment_audit.csv", index=False, lineterminator="\n")
    v7b.pairwise_audit(validation_meta, predictions).to_csv(output_dir / "v7b_hm_reciprocal_confusion_audit.csv", index=False, lineterminator="\n")
    (output_dir / "v7b_r2_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "v7b_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(output_dir, gate, selected)
    log_progress(f"selected={selected['method']} round={int(selected['round_id'])} hm_min={float(selected['hm_min_recall']):.4f} decision={gate['advanced_model_decision']}", start_time=start_time)


if __name__ == "__main__":
    main()
