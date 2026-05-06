from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnose_v8a_shuffled_label_null_behavior import selected_threshold, threshold_metrics
from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only feature-sufficiency probe for v8A H/M sidecar features; "
    "not a promotion gate, product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)

THRESHOLDS = {
    "real_label_hm_min_recall_min": 0.90,
    "real_minus_null_margin_min": 0.35,
    "null_hm_min_recall_max": 0.55,
}


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("Missing scikit-learn in the active environment.") from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "MLPClassifier": MLPClassifier,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def labels_to_binary(labels: pd.Series) -> np.ndarray:
    return (labels.astype(str).to_numpy() == "Magnetite").astype(int)


def evaluate_estimator(name: str, estimator: Any, train: pd.DataFrame, validation: pd.DataFrame, cols: list[str], *, shuffle_seed: int | None = None) -> dict[str, Any]:
    x_train = train[cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_train = train["material"].astype(str).to_numpy()
    if shuffle_seed is not None:
        y_train = np.random.default_rng(shuffle_seed).permutation(y_train)
    x_validation = validation[cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_validation = validation["material"].astype(str).to_numpy()
    fitted = deepcopy(estimator)
    fitted.fit(x_train, y_train)
    prob = fitted.predict_proba(x_validation)
    classes = [str(item) for item in fitted.classes_]
    prob_magnetite = prob[:, classes.index("Magnetite")] if "Magnetite" in classes else np.zeros(len(validation))
    threshold, selected = selected_threshold(y_validation, prob_magnetite.astype(np.float64))
    fixed = threshold_metrics(y_validation, prob_magnetite.astype(np.float64), 0.5)
    selected["threshold"] = threshold
    return {
        "model": name,
        "label_mode": "shuffled" if shuffle_seed is not None else "real",
        "shuffle_seed": shuffle_seed,
        "fixed_hm_min_recall": float(fixed["hm_min_recall"]),
        "selected_hm_min_recall": float(selected["hm_min_recall"]),
        "selected_accuracy": float(selected["accuracy"]),
        "selected_threshold": float(threshold),
    }


def model_specs(sk: dict[str, Any], seed: int) -> list[tuple[str, Any]]:
    return [
        (
            "Logistic",
            sk["make_pipeline"](sk["StandardScaler"](), sk["LogisticRegression"](max_iter=3000, class_weight="balanced", random_state=seed)),
        ),
        (
            "ExtraTrees",
            sk["ExtraTreesClassifier"](n_estimators=220, random_state=seed, class_weight="balanced", max_features="sqrt", n_jobs=-1),
        ),
        (
            "TabularMLP",
            sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["MLPClassifier"](hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3, max_iter=600, random_state=seed),
            ),
        ),
    ]


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        "# v8A feature sufficiency model probe",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Probe blocked: `{str(gate['probe_blocked_by_root_cause_gates']).lower()}`",
        f"- Clean signal confirmed: `{str(gate['clean_signal_confirmed']).lower()}`",
        "",
        "## Model Summary",
        "",
        markdown_table(summary, ["model", "real_selected_hm_min_recall", "null_selected_hm_min_recall_max", "real_minus_null_margin"]),
        "",
        "## Notes",
        "",
        "This probe is diagnostic only. Advanced models do not unlock promotion unless null controls are already clean.",
        "",
    ]
    (output_dir / "v8a_feature_sufficiency_probe_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or block v8A feature sufficiency probes under root-cause guardrails.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--null-gate", required=True)
    parser.add_argument("--shortcut-gate", required=True)
    parser.add_argument("--stress-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force-diagnostic-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    gates = {
        "null": load_json(project_root / args.null_gate),
        "shortcut": load_json(project_root / args.shortcut_gate),
        "stress": load_json(project_root / args.stress_gate),
    }
    root_clean = all(bool(gate.get("gate_passed", False)) for gate in gates.values())
    probe_blocked = not root_clean and not args.force_diagnostic_run
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = pd.DataFrame()
    if not probe_blocked:
        schema_gate = load_json(project_root / args.input_dir / "v8a_event_schema_gate.json")
        manifest = load_json(project_root / args.input_dir / "v8a_event_feature_manifest.json")
        for name, payload in {"schema_gate": schema_gate, "manifest": manifest}.items():
            if bool(payload.get("shadow_or_final_used", False)) or bool(payload.get("reads_existing_xrt_cubes", False)):
                raise RuntimeError(f"Refusing sufficiency probe because {name} reports forbidden inputs.")
        frame = pd.read_csv(project_root / args.input_dir / "v8a_event_sidecar_features.csv")
        train = frame[frame["split"].astype(str).eq("train") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
        validation = frame[frame["split"].astype(str).eq("validation") & frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
        main_cols, _, _, _, _ = feature_sets(frame)
        sk = require_sklearn()
        rows = []
        for name, estimator in model_specs(sk, 13001):
            rows.append(evaluate_estimator(name, estimator, train, validation, main_cols))
            for seed in [13011, 13021, 13031]:
                rows.append(evaluate_estimator(name, estimator, train, validation, main_cols, shuffle_seed=seed))
        raw = pd.DataFrame(rows)
        raw.to_csv(output_dir / "v8a_feature_sufficiency_probe_rows.csv", index=False, lineterminator="\n")
        summary_rows = []
        for model, group in raw.groupby("model", sort=True):
            real = group[group["label_mode"].eq("real")]["selected_hm_min_recall"].max()
            null = group[group["label_mode"].eq("shuffled")]["selected_hm_min_recall"].max()
            summary_rows.append(
                {
                    "model": model,
                    "real_selected_hm_min_recall": float(real),
                    "null_selected_hm_min_recall_max": float(null),
                    "real_minus_null_margin": float(real - null),
                }
            )
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(output_dir / "v8a_feature_sufficiency_probe_summary.csv", index=False, lineterminator="\n")
    clean_signal = bool(
        not summary.empty
        and (summary["real_selected_hm_min_recall"] >= THRESHOLDS["real_label_hm_min_recall_min"]).any()
        and (summary["null_selected_hm_min_recall_max"] < THRESHOLDS["null_hm_min_recall_max"]).all()
        and (summary["real_minus_null_margin"] >= THRESHOLDS["real_minus_null_margin_min"]).any()
    )
    gate = {
        "generated_by": "analysis/probe_v8a_feature_sufficiency_models.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_feature_sufficiency_model_probe",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "probe_blocked_by_root_cause_gates": probe_blocked,
        "force_diagnostic_run": bool(args.force_diagnostic_run),
        "torch_available": importlib.util.find_spec("torch") is not None,
        "implemented_models": ["Logistic", "ExtraTrees", "TabularMLP"],
        "deferred_models": ["1D CNN", "2D CNN", "small Transformer"],
        "gate_passed": clean_signal and root_clean,
        "clean_signal_confirmed": clean_signal,
        "decision": "feature_sufficiency_probe_blocked_until_root_cause_clean"
        if probe_blocked
        else ("clean_signal_confirmed_ready_for_v3_prereg" if clean_signal and root_clean else "feature_sufficiency_probe_diagnostic_only_not_promotion"),
        "root_cause_gate_inputs": {name: gate.get("decision") for name, gate in gates.items()},
        "thresholds": THRESHOLDS,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(output_dir / "v8a_feature_sufficiency_probe_gate.json", gate)
    write_report(output_dir, gate, summary)
    print(
        "decision={decision} blocked={blocked} clean_signal={clean}".format(
            decision=gate["decision"],
            blocked=str(probe_blocked).lower(),
            clean=str(clean_signal).lower(),
        )
    )


if __name__ == "__main__":
    main()
