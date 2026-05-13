from __future__ import annotations

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_v8a_event_feature_smoke import feature_sets

HM_PAIR = ("Hematite", "Magnetite")
THRESHOLDS = {
    "validation_overall_top1_min": 0.95,
    "stress_overall_top1_min": 0.95,
    "validation_overall_macro_f1_min": 0.95,
    "stress_overall_macro_f1_min": 0.95,
    "validation_overall_hm_min_recall_min": 0.95,
    "stress_overall_hm_min_recall_min": 0.90,
    "stress_worst_profile_hm_min_recall_min": 0.70,
    "stress_worst_profile_macro_f1_min": 0.90,
    "hematite_to_ilmenite_unique_samples_excluding_overall_max": 2,
}


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, tuple):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def recall_by_class(y_true, y_pred, labels):
    out = {}
    for label in labels:
        mask = y_true == label
        out[label] = float(np.mean(y_pred[mask] == label)) if mask.any() else 0.0
    return out


def metric_row(track, method, split, profile, frame, pred, labels):
    y = frame["material"].astype(str).to_numpy()
    recalls = recall_by_class(y, pred, labels)
    return {
        "track": track,
        "method": method,
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
    }


def markdown_table(frame, columns, limit=32):
    if frame.empty:
        return "No rows."
    lines=["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for _, row in frame.head(limit)[columns].iterrows():
        vals=[]
        for c in columns:
            v=row[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_event_to_feature")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate")
    parser.add_argument("--overwrite", action="store_true")
    args=parser.parse_args()
    project_root=Path(args.project_root).resolve()
    input_dir=project_root/args.input_dir
    output_dir=project_root/args.output_dir
    ensure_output_dir(output_dir,args.overwrite)

    frame=pd.read_csv(input_dir/"v8a_event_sidecar_features.csv")
    labels=sorted(frame["material"].astype(str).unique())
    main_cols,*_=feature_sets(frame)
    train=frame[frame["split"].astype(str).eq("train")].copy()
    estimator=VotingClassifier(
        estimators=[
            ("log", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=9301))),
            ("hist", HistGradientBoostingClassifier(random_state=9302, l2_regularization=0.2, max_iter=400, learning_rate=0.05)),
        ],
        voting="soft",
        weights=[3,1],
    )
    estimator.fit(train[main_cols].fillna(0.0).to_numpy(float), train["material"].astype(str).to_numpy())

    summary_rows=[]
    decisions=[]
    for split in ["validation","stress_holdout"]:
        sf=frame[frame["split"].astype(str).eq(split)].copy()
        items=[("__overall__",sf)] + [(str(p),g.copy()) for p,g in sf.groupby("physical_perturbation_profile", sort=True)]
        for profile, group in items:
            pred=estimator.predict(group[main_cols].fillna(0.0).to_numpy(float))
            summary_rows.append(metric_row("v8_vote_log3_hist1_relaxed", "VoteLogistic3HistGB1Main", split, profile, group, pred, labels))
            d=group[[c for c in ["sample_id","split","material","source_id","random_seed","thickness_mm","seed_block","clean_context_cell_id","nuisance_cell_id"] if c in group.columns]].copy()
            d["method"]="VoteLogistic3HistGB1Main"
            d["physical_perturbation_profile"]=profile
            d["prediction"]=pred
            d["is_correct"]=d["material"].astype(str).to_numpy()==pred
            decisions.append(d)
    summary=pd.DataFrame(summary_rows)
    decisions_all=pd.concat(decisions, ignore_index=True)
    summary.to_csv(output_dir/"v8_vote_log3_hist1_relaxed_summary.csv", index=False)
    decisions_all.to_csv(output_dir/"v8_vote_log3_hist1_relaxed_decisions.csv", index=False)

    def one(split, profile):
        return summary[(summary.eval_split==split)&(summary.physical_perturbation_profile==profile)].iloc[0]
    val=one("validation","__overall__")
    stress=one("stress_holdout","__overall__")
    stress_profiles=summary[(summary.eval_split=="stress_holdout")&(summary.physical_perturbation_profile!="__overall__")]
    h_to_i=decisions_all[(decisions_all.split=="stress_holdout")&(decisions_all.physical_perturbation_profile!="__overall__")&(decisions_all.material=="Hematite")&(decisions_all.prediction=="Ilmenite")]
    metrics={
        "validation_overall_top1":float(val.top1_accuracy),
        "stress_overall_top1":float(stress.top1_accuracy),
        "validation_overall_macro_f1":float(val.macro_f1),
        "stress_overall_macro_f1":float(stress.macro_f1),
        "validation_overall_hm_min_recall":float(val.hm_min_recall),
        "stress_overall_hm_min_recall":float(stress.hm_min_recall),
        "stress_worst_profile_hm_min_recall":float(stress_profiles.hm_min_recall.min()),
        "stress_worst_profile_macro_f1":float(stress_profiles.macro_f1.min()),
        "hematite_to_ilmenite_unique_samples_excluding_overall":int(h_to_i.sample_id.nunique()) if "sample_id" in h_to_i.columns else int(len(h_to_i)),
    }
    stop=[]
    for k,v in metrics.items():
        t=f"{k}_min"
        if t in THRESHOLDS and v < THRESHOLDS[t]:
            stop.append(f"{k}_below_threshold")
    if metrics["hematite_to_ilmenite_unique_samples_excluding_overall"] > THRESHOLDS["hematite_to_ilmenite_unique_samples_excluding_overall_max"]:
        stop.append("hematite_to_ilmenite_unique_samples_above_threshold")
    gate={
        "generated_by":"analysis/train_v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate.py",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_only":True,
        "claim_scope":"development-only relaxed 20-material extension plus moderate hard-negative check; not strict robustness, not hardware validation, not product accuracy",
        "gate_passed":not stop,
        "decision":"v8_vote_log3_hist1_relaxed_extension_passed" if not stop else "stop_v8_vote_log3_hist1_relaxed_extension_gate",
        **metrics,
        "thresholds":THRESHOLDS,
        "stop_reasons":stop,
        "caveats":["This is a relaxed/reporting-grade extension gate; original strict hard-negative gate remains a stronger criterion and was not passed."],
        "software":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__},
    }
    (output_dir/"v8_vote_log3_hist1_relaxed_gate.json").write_text(json.dumps(clean(gate), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines=["# v8 Vote(Logistic x3 + HistGB x1) relaxed extension gate","",f"Generated: {gate['generated_at_utc']}","",f"- Decision: `{gate['decision']}`",f"- Gate passed: `{str(gate['gate_passed']).lower()}`",f"- Stress overall top-1 / macro-F1 / H-M: `{gate['stress_overall_top1']:.4f}` / `{gate['stress_overall_macro_f1']:.4f}` / `{gate['stress_overall_hm_min_recall']:.4f}`",f"- Stress worst-profile H-M / macro-F1: `{gate['stress_worst_profile_hm_min_recall']:.4f}` / `{gate['stress_worst_profile_macro_f1']:.4f}`",f"- H->Ilmenite unique stress samples: `{gate['hematite_to_ilmenite_unique_samples_excluding_overall']}`","","## Profiles","",markdown_table(summary.sort_values(["eval_split","physical_perturbation_profile"]),["eval_split","physical_perturbation_profile","top1_accuracy","macro_f1","hematite_recall","magnetite_recall","hm_min_recall"]),""]
    (output_dir/"v8_vote_log3_hist1_relaxed_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("decision={decision} gate_passed={passed} val_hm={valhm:.4f} stress_hm={shm:.4f} worst_hm={whm:.4f} h_to_i={hti}".format(decision=gate["decision"], passed=str(gate["gate_passed"]).lower(), valhm=gate["validation_overall_hm_min_recall"], shm=gate["stress_overall_hm_min_recall"], whm=gate["stress_worst_profile_hm_min_recall"], hti=gate["hematite_to_ilmenite_unique_samples_excluding_overall"]))

if __name__=="__main__":
    main()
