from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import platform

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_v8a_event_feature_smoke import feature_sets

HM_PAIR = ("Hematite", "Magnetite")


def recall_by_class(y_true, y_pred, labels):
    out = {}
    for label in labels:
        mask = y_true == label
        out[label] = float(np.mean(y_pred[mask] == label)) if mask.any() else 0.0
    return out


def metrics(y, pred, labels):
    recalls = recall_by_class(y, pred, labels)
    return {
        "top1_accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "min_class_recall": float(min(recalls.values())),
        "hematite_recall": recalls.get("Hematite", 0.0),
        "magnetite_recall": recalls.get("Magnetite", 0.0),
        "hm_min_recall": float(min(recalls.get(x, 0.0) for x in HM_PAIR)),
    }


def eval_estimator(name, estimator, train, eval_items, labels, cols):
    estimator.fit(train[cols].fillna(0.0).to_numpy(float), train["material"].astype(str).to_numpy())
    rows=[]
    for split, profile, frame in eval_items:
        y=frame["material"].astype(str).to_numpy()
        pred=estimator.predict(frame[cols].fillna(0.0).to_numpy(float))
        row={"method":name,"eval_split":split,"physical_perturbation_profile":profile,"samples":len(y)}
        row.update(metrics(y,pred,labels))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    root=Path.cwd()
    frame=pd.read_csv(root/"results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_event_to_feature/v8a_event_sidecar_features.csv")
    labels=sorted(frame["material"].astype(str).unique())
    cols,*_=feature_sets(frame)
    train=frame[frame["split"].astype(str).eq("train")].copy()
    eval_items=[]
    for split in ["validation","stress_holdout"]:
        sf=frame[frame["split"].astype(str).eq(split)].copy()
        eval_items.append((split,"__overall__",sf))
        eval_items.extend((str(p), str(p), g.copy()) for p,g in [])
        for profile, group in sf.groupby("physical_perturbation_profile", sort=True):
            eval_items.append((split,str(profile),group.copy()))
    out=root/"results/accuracy_v3/v8a_multiclass_context_v8_ensemble_relaxed_probe"
    out.mkdir(parents=True, exist_ok=True)

    def logi(C=1.0):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=C, random_state=9201))
    def hist():
        return HistGradientBoostingClassifier(random_state=9202, l2_regularization=0.2, max_iter=400, learning_rate=0.05)
    def et():
        return ExtraTreesClassifier(n_estimators=1000, random_state=9203, class_weight="balanced", max_features="sqrt", n_jobs=-1)

    models=[]
    for lw, hw in [(1,1),(2,1),(3,1),(1,2),(1,3),(4,1),(1,4)]:
        models.append((f"Vote_Logistic{lw}_HistGB{hw}", VotingClassifier([("log", logi()), ("hist", hist())], voting="soft", weights=[lw,hw], n_jobs=None)))
    for lw, hw, ew in [(1,1,1),(2,1,1),(1,2,1),(1,1,2),(3,1,1),(1,3,1)]:
        models.append((f"Vote_Log{lw}_Hist{hw}_ET{ew}", VotingClassifier([("log", logi()), ("hist", hist()), ("et", et())], voting="soft", weights=[lw,hw,ew], n_jobs=None)))

    all_rows=[]
    ranking=[]
    for name, est in models:
        print("running", name, flush=True)
        summary=eval_estimator(name, est, train, eval_items, labels, cols)
        all_rows.append(summary)
        stress=summary[(summary.eval_split=="stress_holdout") & (summary.physical_perturbation_profile!="__overall__")]
        val_over=summary[(summary.eval_split=="validation") & (summary.physical_perturbation_profile=="__overall__")].iloc[0]
        stress_over=summary[(summary.eval_split=="stress_holdout") & (summary.physical_perturbation_profile=="__overall__")].iloc[0]
        ranking.append({
            "method":name,
            "validation_overall_hm":float(val_over.hm_min_recall),
            "validation_overall_macro_f1":float(val_over.macro_f1),
            "stress_overall_hm":float(stress_over.hm_min_recall),
            "stress_overall_macro_f1":float(stress_over.macro_f1),
            "stress_overall_top1":float(stress_over.top1_accuracy),
            "stress_worst_profile_hm":float(stress.hm_min_recall.min()),
            "stress_worst_profile_macro_f1":float(stress.macro_f1.min()),
        })
    all_summary=pd.concat(all_rows, ignore_index=True)
    ranking=pd.DataFrame(ranking).sort_values(["stress_worst_profile_hm","validation_overall_hm","stress_overall_hm","stress_overall_macro_f1"], ascending=False)
    all_summary.to_csv(out/"v8_ensemble_relaxed_summary.csv", index=False)
    ranking.to_csv(out/"v8_ensemble_relaxed_ranking.csv", index=False)
    gate={
        "generated_by":"inline_v8_ensemble_relaxed_probe",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only":True,
        "top_methods":ranking.head(12).to_dict(orient="records"),
        "software":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__},
    }
    (out/"v8_ensemble_relaxed_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(ranking.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
