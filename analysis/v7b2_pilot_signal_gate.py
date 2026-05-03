from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HM_PAIR = ("Hematite", "Magnetite")
V7B_BASELINE_HM_MIN_RECALL = 0.6083333333333333
PASS_HM_MIN_RECALL = 0.70
PASS_IMPROVEMENT = 0.08
STOP_HM_MIN_RECALL = 0.68
FULL_SOURCE_LIMIT = 18


def require_sklearn():
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import recall_score
    except ModuleNotFoundError as exc:
        raise SystemExit("scikit-learn is required for v7B2 pilot signal gate.") from exc
    return {"ExtraTreesClassifier": ExtraTreesClassifier, "recall_score": recall_score}


def load_features(feature_dir: Path) -> pd.DataFrame:
    path = feature_dir / "v7b2_physical_view_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing physical feature table: {path}")
    frame = pd.read_csv(path)
    if set(frame["material"].astype(str)) != set(HM_PAIR):
        raise RuntimeError("v7B2 Pilot gate expects H/M-only physical features.")
    return frame


def pivot_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    feature_metrics = [
        "hit_rate_sum",
        "calibrated_hit_ratio_mean",
        "attenuation_mean",
        "path_norm_attenuation",
        "energy_slope_path_norm_attenuation",
        "angle_sensitivity_path_norm_attenuation",
        "scatter_to_direct",
        "tail120_rate_sum",
        "detector_reliable",
    ]
    id_cols = ["sample_index", "material", "split", "random_seed", "sample_id", "thickness_mm"]
    available_metrics = [metric for metric in feature_metrics if metric in frame.columns]
    if not available_metrics:
        raise RuntimeError("No preregistered v7B2 physical metrics are present.")
    long = frame[id_cols + ["source_id", "detector_id", *available_metrics]].copy()
    melted = long.melt(
        id_vars=id_cols + ["source_id", "detector_id"],
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    )
    melted["feature"] = melted["source_id"].astype(str) + "__" + melted["detector_id"].astype(str) + "__" + melted["metric"].astype(str)
    table = (
        melted.pivot_table(index=id_cols, columns="feature", values="value", aggfunc="mean", fill_value=0.0)
        .reset_index()
        .sort_values(["split", "material", "thickness_mm", "random_seed", "sample_id"])
    )
    names = [str(col) for col in table.columns if col not in id_cols]
    x = table[names].to_numpy(dtype=np.float32)
    thickness = table["thickness_mm"].to_numpy(dtype=np.float32).reshape(-1, 1)
    x = np.hstack([x, thickness])
    names.append("metadata__thickness_mm")
    y = table["material"].astype(str).to_numpy()
    split = table["split"].astype(str).to_numpy()
    return x, y, split, table[id_cols], names


def pair_recalls(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    recalls = {}
    for material in HM_PAIR:
        mask = y_true == material
        recalls[material] = float(np.mean(predictions[mask] == material)) if mask.any() else 0.0
    return recalls


def evaluate_model(x: np.ndarray, y: np.ndarray, split: np.ndarray, sk) -> tuple[dict, pd.DataFrame]:
    train_mask = split == "train"
    validation_mask = split == "validation"
    if not train_mask.any() or not validation_mask.any():
        raise RuntimeError("v7B2 Pilot gate requires both train and validation splits.")
    model = sk["ExtraTreesClassifier"](
        n_estimators=900,
        random_state=7202,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )
    model.fit(x[train_mask], y[train_mask])
    predictions = np.asarray(model.predict(x[validation_mask])).astype(str)
    y_validation = y[validation_mask]
    recalls = pair_recalls(y_validation, predictions)
    hm_min = float(min(recalls.values()))
    improvement = hm_min - V7B_BASELINE_HM_MIN_RECALL
    decision = "gray_zone_v7c_sidecar_only"
    if hm_min >= PASS_HM_MIN_RECALL and improvement >= PASS_IMPROVEMENT:
        decision = "proceed_to_full_v7b2"
    elif hm_min < STOP_HM_MIN_RECALL:
        decision = "stop_physics_expansion_write_limitation"
    decisions = pd.DataFrame(
        {
            "material": y_validation,
            "prediction": predictions,
            "is_correct": y_validation == predictions,
        }
    )
    return (
        {
            "method": "V7B2PhysicalExtraTrees",
            "validation_samples": int(len(y_validation)),
            "hematite_recall": recalls["Hematite"],
            "magnetite_recall": recalls["Magnetite"],
            "hm_min_recall": hm_min,
            "v7b_baseline_hm_min_recall": V7B_BASELINE_HM_MIN_RECALL,
            "hm_min_recall_improvement": improvement,
            "pass_hm_min_recall_threshold": PASS_HM_MIN_RECALL,
            "pass_improvement_threshold": PASS_IMPROVEMENT,
            "stop_hm_min_recall_threshold": STOP_HM_MIN_RECALL,
            "decision": decision,
            "gate_passed": decision == "proceed_to_full_v7b2",
        },
        decisions,
    )


def score_views(contrast: pd.DataFrame) -> pd.DataFrame:
    if contrast.empty:
        return contrast
    score_cols = [col for col in contrast.columns if col.endswith("_effect_size")]
    frame = contrast.copy()
    frame["preregistered_source_score"] = frame[score_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).max(axis=1)
    source_scores = (
        frame.groupby(["source_id", "energy_keV", "source_variant", "incidence_angle_deg"], as_index=False)["preregistered_source_score"]
        .max()
        .sort_values(["preregistered_source_score", "energy_keV", "incidence_angle_deg"], ascending=[False, True, True])
    )
    source_scores["selected_for_full_v7b2"] = source_scores.groupby(lambda _: True).cumcount() < FULL_SOURCE_LIMIT
    return source_scores


def write_limitation(feature_dir: Path, gate: dict) -> None:
    text = f"""# v7B2 Pilot limitation note

The v7B2 H/M physical pilot did not reach the preregistered signal threshold.

- H/M min recall: `{gate['hm_min_recall']:.4f}`
- Improvement over v7B baseline: `{gate['hm_min_recall_improvement']:.4f}`
- Stop threshold: `<{STOP_HM_MIN_RECALL:.2f}`

This is development-only evidence. It does not show that XRT cannot separate Hematite/Magnetite in general; it only shows that the current simulated v7B2 Pilot observations did not create enough H/M contrast to justify expanding to full v7B2.
"""
    (feature_dir / "v7b2_pilot_limitation_note.md").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered v7B2 H/M Pilot signal gate.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--feature-dir", default="results/accuracy_v3/v7b2_hm_physics_dev")
    args = parser.parse_args()

    sk = require_sklearn()
    # Preserve extended UNC prefixes for Windows-side scientific Python.
    project_root = Path(args.project_root)
    feature_dir = project_root / args.feature_dir
    frame = load_features(feature_dir)
    x, y, split, sample_meta, feature_names = pivot_features(frame)
    gate, decisions = evaluate_model(x, y, split, sk)
    decisions = pd.concat([sample_meta[sample_meta["split"].eq("validation")].reset_index(drop=True), decisions], axis=1)
    decisions.to_csv(feature_dir / "v7b2_pilot_validation_decisions.csv", index=False, lineterminator="\n")

    contrast_path = feature_dir / "v7b2_hm_physical_contrast_by_view.csv"
    source_scores = score_views(pd.read_csv(contrast_path)) if contrast_path.exists() else pd.DataFrame()
    source_scores.to_csv(feature_dir / "v7b2_pilot_source_ranking.csv", index=False, lineterminator="\n")
    selected_sources = source_scores[source_scores.get("selected_for_full_v7b2", pd.Series(dtype=bool)).eq(True)]["source_id"].astype(str).tolist()
    gate.update(
        {
            "generated_by": "analysis/v7b2_pilot_signal_gate.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol_name": "v7b2_hm_physics_dev",
            "shadow_or_final_used": False,
            "feature_count": int(x.shape[1]),
            "selected_full_v7b2_source_ids": selected_sources[:FULL_SOURCE_LIMIT],
            "selected_full_v7b2_source_count": int(min(len(selected_sources), FULL_SOURCE_LIMIT)),
            "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        }
    )
    if gate["decision"] == "stop_physics_expansion_write_limitation":
        write_limitation(feature_dir, gate)
    (feature_dir / "v7b2_pilot_gate.json").write_bytes((json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(f"decision={gate['decision']} hm_min_recall={gate['hm_min_recall']:.4f} improvement={gate['hm_min_recall_improvement']:.4f}")


if __name__ == "__main__":
    main()
