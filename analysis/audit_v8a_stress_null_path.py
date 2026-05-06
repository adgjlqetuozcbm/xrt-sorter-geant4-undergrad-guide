from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnose_v8a_shuffled_label_null_behavior import (
    magnetite_probability,
    model_specs,
    selected_threshold,
    shuffled_labels,
    threshold_metrics,
)
from train_v8a_event_feature_smoke import feature_sets, load_json
from v8a_event_feature_stress_gate import apply_stress


CLAIM_SCOPE = (
    "development-only stress-path null audit for v8A H/M sidecar features; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)

THRESHOLDS = {
    "stress_null_increase_max": 0.10,
    "stress_null_hm_min_recall_ceiling": 0.55,
    "material_delta_gap_max": 1.0,
    "residual_delta_gap_max": 1.0,
}


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("Missing scikit-learn in the active environment.") from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def standardized_gap(left: pd.Series, right: pd.Series) -> float:
    left_values = left.fillna(0.0).to_numpy(dtype=np.float64)
    right_values = right.fillna(0.0).to_numpy(dtype=np.float64)
    pooled = np.sqrt(0.5 * (np.var(left_values) + np.var(right_values)) + 1e-12)
    return float(abs(float(np.mean(right_values) - np.mean(left_values))) / pooled)


def evaluate_scenario_null(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    main_cols: list[str],
    seeds: list[int],
    sk: dict[str, Any],
) -> pd.DataFrame:
    train = train_frame[train_frame["split"].astype(str).eq("train") & train_frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()
    validation = validation_frame[
        validation_frame["split"].astype(str).eq("validation")
        & validation_frame["source_mode"].astype(str).eq("custom_diffraction_on")
    ].copy()
    if train.empty or validation.empty:
        raise RuntimeError("Train and validation source-on rows are required for stress null audit.")
    x_train = train[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    x_validation = validation[main_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_validation = validation["material"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        y_train = shuffled_labels(train, seed, "row_level", 0.003)
        for model_name, estimator in model_specs(sk, seed):
            fitted = deepcopy(estimator)
            fitted.fit(x_train, y_train)
            prob = magnetite_probability(fitted, x_validation)
            fixed = threshold_metrics(y_validation, prob, 0.5)
            selected_t, selected = selected_threshold(y_validation, prob)
            for policy, metrics in [("fixed_0p5", fixed), ("validation_selected", selected)]:
                rows.append(
                    {
                        "shuffle_seed": seed,
                        "model": model_name,
                        "threshold_policy": policy,
                        **{key: value for key, value in metrics.items() if key != "threshold_distance_to_0p5"},
                    }
                )
    return pd.DataFrame(rows)


def stress_delta_gaps(original: pd.DataFrame, stressed: pd.DataFrame, main_cols: list[str]) -> pd.DataFrame:
    rows = []
    eval_mask = original["split"].astype(str).eq("validation") & original["source_mode"].astype(str).eq("custom_diffraction_on")
    base = original.loc[eval_mask].copy()
    changed = stressed.loc[eval_mask].copy()
    for col in main_cols:
        delta = changed[col].fillna(0.0).astype(float) - base[col].fillna(0.0).astype(float)
        h = delta[base["material"].astype(str).eq("Hematite")]
        m = delta[base["material"].astype(str).eq("Magnetite")]
        rows.append(
            {
                "feature": col,
                "is_residual_feature": "residual" in col,
                "delta_abs_mean": float(delta.abs().mean()),
                "material_delta_gap_abs": standardized_gap(h, m) if len(h) and len(m) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("material_delta_gap_abs", ascending=False)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 18) -> str:
    if frame.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.head(limit)[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], scenario_summary: pd.DataFrame, delta_summary: pd.DataFrame) -> None:
    lines = [
        "# v8A stress-path null audit",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Stress generator artifact suspected: `{str(gate['stress_generator_artifact_suspected']).lower()}`",
        f"- Residual stress artifact suspected: `{str(gate['residual_stress_artifact_suspected']).lower()}`",
        "",
        "## Scenario Null Summary",
        "",
        markdown_table(
            scenario_summary.sort_values("selected_hm_min_recall_max", ascending=False),
            ["scenario", "fixed_hm_min_recall_max", "selected_hm_min_recall_max", "selected_minus_baseline"],
        ),
        "",
        "## Largest Stress Delta Gaps",
        "",
        markdown_table(delta_summary.sort_values("material_delta_gap_abs", ascending=False), ["scenario", "feature", "is_residual_feature", "material_delta_gap_abs"], limit=12),
        "",
        "## Stop Reasons",
        "",
    ]
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.append("")
    (output_dir / "v8a_stress_null_path_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether stress scenarios create null-path artifacts.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--stress-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shuffle-seeds", default=",".join(str(seed) for seed in range(12001, 12031)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    for name, payload in {"schema_gate": schema_gate, "manifest": manifest}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing stress null audit because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing stress null audit because {name} reports existing XRT cube reads.")
    config = load_json(project_root / args.stress_config)
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    main_cols, _, _, _, _ = feature_sets(frame)
    seeds = [int(item.strip()) for item in args.shuffle_seeds.split(",") if item.strip()]
    sk = require_sklearn()
    scenario_rows = []
    null_tables = []
    delta_tables = []
    baseline_selected = 0.0
    for scenario in config["stress_scenarios"]:
        scenario_name = str(scenario["name"])
        stressed = apply_stress(frame, main_cols, scenario)
        null_rows = evaluate_scenario_null(frame, stressed, main_cols, seeds, sk)
        null_rows.insert(0, "scenario", scenario_name)
        null_tables.append(null_rows)
        fixed_max = float(null_rows[null_rows["threshold_policy"].eq("fixed_0p5")]["hm_min_recall"].max())
        selected_max = float(null_rows[null_rows["threshold_policy"].eq("validation_selected")]["hm_min_recall"].max())
        if scenario_name == "baseline_identity":
            baseline_selected = selected_max
        deltas = stress_delta_gaps(frame, stressed, main_cols)
        deltas.insert(0, "scenario", scenario_name)
        delta_tables.append(deltas)
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "fixed_hm_min_recall_max": fixed_max,
                "selected_hm_min_recall_max": selected_max,
                "selected_minus_baseline": 0.0,
                "max_material_delta_gap_abs": float(deltas["material_delta_gap_abs"].max()) if not deltas.empty else 0.0,
                "max_residual_delta_gap_abs": float(deltas[deltas["is_residual_feature"]]["material_delta_gap_abs"].max())
                if bool(deltas["is_residual_feature"].any())
                else 0.0,
            }
        )
    scenario_summary = pd.DataFrame(scenario_rows)
    scenario_summary["selected_minus_baseline"] = scenario_summary["selected_hm_min_recall_max"] - baseline_selected
    null_table = pd.concat(null_tables, ignore_index=True)
    delta_summary = pd.concat(delta_tables, ignore_index=True)
    stress_artifact = bool(
        (scenario_summary["selected_hm_min_recall_max"] >= THRESHOLDS["stress_null_hm_min_recall_ceiling"]).any()
        and (scenario_summary["selected_minus_baseline"] > THRESHOLDS["stress_null_increase_max"]).any()
    )
    material_delta_artifact = bool(scenario_summary["max_material_delta_gap_abs"].max() > THRESHOLDS["material_delta_gap_max"])
    residual_artifact = bool(scenario_summary["max_residual_delta_gap_abs"].max() > THRESHOLDS["residual_delta_gap_max"])
    pass_items = {
        "stress_null_not_increased": not stress_artifact,
        "stress_material_delta_gap_below_ceiling": not material_delta_artifact,
        "residual_delta_gap_below_ceiling": not residual_artifact,
    }
    stop_reasons = [name for name, passed in pass_items.items() if not passed]
    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_stress_null_path.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_stress_null_path_audit",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "stress_config": args.stress_config,
        "gate_passed": gate_passed,
        "decision": "stress_null_path_clean" if gate_passed else "stress_null_path_artifact_found",
        "shuffle_seed_count": int(len(seeds)),
        "scenario_count": int(len(scenario_summary)),
        "baseline_selected_null_hm_min_recall_max": float(baseline_selected),
        "max_selected_null_hm_min_recall": float(scenario_summary["selected_hm_min_recall_max"].max()),
        "max_selected_minus_baseline": float(scenario_summary["selected_minus_baseline"].max()),
        "max_material_delta_gap_abs": float(scenario_summary["max_material_delta_gap_abs"].max()),
        "max_residual_delta_gap_abs": float(scenario_summary["max_residual_delta_gap_abs"].max()),
        "stress_generator_artifact_suspected": stress_artifact or material_delta_artifact,
        "residual_stress_artifact_suspected": residual_artifact,
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    null_table.to_csv(output_dir / "v8a_stress_null_path_rows.csv", index=False, lineterminator="\n")
    scenario_summary.to_csv(output_dir / "v8a_stress_null_path_scenario_summary.csv", index=False, lineterminator="\n")
    delta_summary.to_csv(output_dir / "v8a_stress_null_path_delta_gaps.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_stress_null_path_gate.json", json_clean(gate))
    write_report(output_dir, gate, scenario_summary, delta_summary)
    print(
        "decision={decision} gate_passed={passed} max_null={null:.4f} max_delta={delta:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            null=gate["max_selected_null_hm_min_recall"],
            delta=gate["max_material_delta_gap_abs"],
        )
    )


if __name__ == "__main__":
    main()
