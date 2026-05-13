from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import platform

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from train_v8a_event_feature_smoke import feature_sets

HM_PAIR = ("Hematite", "Magnetite")
PROFILE_COL = "physical_perturbation_profile"


def recall_by_class(y_true, y_pred, labels):
    out = {}
    for label in labels:
        mask = y_true == label
        out[label] = float(np.mean(y_pred[mask] == label)) if mask.any() else 0.0
    return out


def eval_model(name, estimator, train, eval_items, labels, feature_cols):
    X_train = train[feature_cols].fillna(0.0).to_numpy(float)
    y_train = train["material"].astype(str).to_numpy()
    estimator.fit(X_train, y_train)
    rows = []
    for split, profile, frame in eval_items:
        X = frame[feature_cols].fillna(0.0).to_numpy(float)
        y = frame["material"].astype(str).to_numpy()
        pred = estimator.predict(X)
        recalls = recall_by_class(y, pred, labels)
        rows.append({
            "method": name,
            "eval_split": split,
            "physical_perturbation_profile": profile,
            "samples": len(y),
            "top1_accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "min_class_recall": float(min(recalls.values())),
            "hematite_recall": recalls.get("Hematite", 0.0),
            "magnetite_recall": recalls.get("Magnetite", 0.0),
            "hm_min_recall": float(min(recalls.get(x, 0.0) for x in HM_PAIR)),
        })
    return pd.DataFrame(rows)


def main():
    root = Path.cwd()
    input_dir = root / "results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_event_to_feature"
    out = root / "results/accuracy_v3/v8a_multiclass_context_v8_model_sweep_reporting_probe"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    labels = sorted(frame["material"].astype(str).unique())
    main_cols, *_ = feature_sets(frame)
    train = frame[frame["split"].astype(str).eq("train")].copy()
    eval_items = []
    for split in ["validation", "stress_holdout"]:
        sf = frame[frame["split"].astype(str).eq(split)].copy()
        eval_items.append((split, "__overall__", sf))
        for profile, group in sf.groupby(PROFILE_COL, sort=True):
            eval_items.append((split, str(profile), group.copy()))
    models = [
        ("Logistic_C1_balanced", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=9101))),
        ("Logistic_C0p3_balanced", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3, random_state=9102))),
        ("LinearSVC_C0p3_balanced", make_pipeline(StandardScaler(), LinearSVC(C=0.3, class_weight="balanced", random_state=9103, max_iter=10000))),
        ("ExtraTrees_sqrt_1200", ExtraTreesClassifier(n_estimators=1200, random_state=9104, class_weight="balanced", max_features="sqrt", min_samples_leaf=1, n_jobs=-1)),
        ("ExtraTrees_log2_leaf2", ExtraTreesClassifier(n_estimators=1200, random_state=9105, class_weight="balanced", max_features="log2", min_samples_leaf=2, n_jobs=-1)),
        ("RandomForest_sqrt_800", RandomForestClassifier(n_estimators=800, random_state=9106, class_weight="balanced", max_features="sqrt", min_samples_leaf=1, n_jobs=-1)),
        ("HistGB_l2", HistGradientBoostingClassifier(random_state=9107, l2_regularization=0.2, max_iter=400, learning_rate=0.05)),
    ]
    rows = []
    for name, estimator in models:
        print("running", name, flush=True)
        rows.append(eval_model(name, estimator, train, eval_items, labels, main_cols))
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(out / "v8_model_sweep_summary.csv", index=False)
    stress = result[(result.eval_split == "stress_holdout") & (result.physical_perturbation_profile != "__overall__")]
    overall = result[(result.eval_split == "stress_holdout") & (result.physical_perturbation_profile == "__overall__")]
    ranking = []
    for method, group in stress.groupby("method"):
        overall_row = overall[overall.method == method].iloc[0].to_dict()
        ranking.append({
            "method": method,
            "stress_worst_profile_hm": float(group.hm_min_recall.min()),
            "stress_worst_profile_macro_f1": float(group.macro_f1.min()),
            "stress_overall_hm": float(overall_row["hm_min_recall"]),
            "stress_overall_macro_f1": float(overall_row["macro_f1"]),
            "stress_overall_top1": float(overall_row["top1_accuracy"]),
        })
    ranking = pd.DataFrame(ranking).sort_values(["stress_worst_profile_hm", "stress_overall_hm", "stress_overall_macro_f1"], ascending=False)
    ranking.to_csv(out / "v8_model_sweep_ranking.csv", index=False)
    gate = {
        "generated_by": "inline_v8_model_sweep_reporting_probe",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only": True,
        "best_method": ranking.iloc[0].to_dict(),
        "top_methods": ranking.head(10).to_dict(orient="records"),
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out / "v8_model_sweep_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ranking.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
