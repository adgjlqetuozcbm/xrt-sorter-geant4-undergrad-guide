from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v8a_custom_diffraction_integration_smoke import feature_sets
from v8a_transport_sidecar_smoke import (
    MAX_CONTROL_HM_MIN_RECALL,
    MAX_OVERLAP_ONLY_HM_MIN_RECALL,
    MAX_SHUFFLED_LABEL_HM_MIN_RECALL,
    MAX_TOTAL_COUNT_HM_MIN_RECALL,
    PASS_FEATURE_AUC,
    PASS_FEATURE_D_PRIME,
    PASS_HM_MIN_RECALL,
    PASS_WORST_THICKNESS_HM_MIN_RECALL,
    ensure_output_dir,
    evaluate_model,
    markdown_table,
    observability_metrics,
    require_sklearn,
)


MAX_LEAKAGE_OFF_HM_MIN_RECALL = 0.75
MIN_MAIN_MINUS_LEAKAGE_MARGIN = 0.20


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_input_gate(input_dir: Path, leakage_dir: Path) -> tuple[dict, dict, dict, dict]:
    gate = load_json(input_dir / "v8a_integration_gate.json")
    manifest = load_json(input_dir / "v8a_integration_manifest.json")
    leakage_gate = load_json(leakage_dir / "v8a_integration_gate.json")
    leakage_manifest = load_json(leakage_dir / "v8a_integration_manifest.json")
    if not bool(gate.get("gate_passed", False)):
        raise RuntimeError(f"Integration smoke did not pass: {gate.get('decision')}")
    if not bool(gate.get("diffraction_source_enabled", False)):
        raise RuntimeError("Training input must have diffraction_source_enabled=true.")
    if bool(gate.get("shadow_or_final_used", False)) or bool(manifest.get("shadow_or_final_used", False)):
        raise RuntimeError("Refusing v8A training because input reports shadow/final use.")
    if bool(manifest.get("reads_existing_xrt_cubes", False)):
        raise RuntimeError("Refusing v8A training because input reports existing XRT cube reads.")
    if bool(leakage_manifest.get("diffraction_source_enabled", True)):
        raise RuntimeError("Leakage input must have diffraction_source_enabled=false.")
    if not bool(leakage_gate.get("leakage_control_passed", False)):
        raise RuntimeError(f"Leakage-off control did not pass: {leakage_gate.get('decision')}")
    return gate, manifest, leakage_gate, leakage_manifest


def evaluate_training(frame: pd.DataFrame, sk: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    main_cols, control_cols, total_count_cols, overlap_cols = feature_sets(frame)
    models = [
        (
            "ExtraTreesSidecarMain",
            sk["ExtraTreesClassifier"](n_estimators=800, random_state=9102, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            False,
        ),
        (
            "LogisticSidecarMain",
            sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=2000, class_weight="balanced", random_state=9103),
            ),
            main_cols,
            False,
        ),
        (
            "ExtraTreesControlOnly",
            sk["ExtraTreesClassifier"](n_estimators=500, random_state=9104, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            control_cols,
            False,
        ),
        (
            "ExtraTreesTotalCountOnly",
            sk["ExtraTreesClassifier"](n_estimators=500, random_state=9105, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            total_count_cols,
            False,
        ),
        (
            "ExtraTreesShuffledTrainLabels",
            sk["ExtraTreesClassifier"](n_estimators=500, random_state=9106, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            main_cols,
            True,
        ),
        (
            "ExtraTreesOverlapOnly",
            sk["ExtraTreesClassifier"](n_estimators=500, random_state=9107, class_weight="balanced", max_features="sqrt", n_jobs=-1),
            overlap_cols,
            False,
        ),
    ]
    rows = []
    decisions = []
    for method_name, estimator, cols, shuffle_labels in models:
        if not cols:
            raise SystemExit(f"No features available for {method_name}.")
        row, method_decisions = evaluate_model(frame, cols, method_name, estimator, shuffle_train_labels=shuffle_labels)
        rows.append(row)
        decisions.append(method_decisions)
    selection = pd.DataFrame(rows)
    validation_decisions = pd.concat(decisions, ignore_index=True)
    metrics = observability_metrics(frame, main_cols, sk["roc_auc_score"])
    main_models = selection[selection["method"].isin(["ExtraTreesSidecarMain", "LogisticSidecarMain"])]
    best_main = main_models.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False).iloc[0].to_dict()
    return selection, validation_decisions, metrics, best_main


def best_leakage_hm(leakage_frame: pd.DataFrame, sk: dict) -> float:
    main_cols = feature_sets(leakage_frame)[0]
    rows = []
    for method_name, estimator in [
        ("ExtraTreesLeakageMain", sk["ExtraTreesClassifier"](n_estimators=500, random_state=9111, class_weight="balanced", max_features="sqrt", n_jobs=-1)),
        (
            "LogisticLeakageMain",
            sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=2000, class_weight="balanced", random_state=9112),
            ),
        ),
    ]:
        row, _ = evaluate_model(leakage_frame, main_cols, method_name, estimator)
        rows.append(row)
    return float(pd.DataFrame(rows)["hm_min_recall"].max())


def write_report(output_dir: Path, gate: dict, selection: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# v8A sidecar small training gate report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        "Scope: development-only small H/M sidecar training after custom diffraction integration smoke. This is not shadow/final evidence and not a full material matrix.",
        "",
        "## Gate",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Best main method: `{gate['best_main_method']}`",
        f"- Main H/M min recall: `{gate['best_main_hm_min_recall']:.4f}`",
        f"- Worst-thickness H/M min recall: `{gate['best_main_worst_thickness_hm_min_recall']:.4f}`",
        f"- Control-only H/M min recall: `{gate['control_hm_min_recall']:.4f}`",
        f"- Total-count-only H/M min recall: `{gate['total_count_hm_min_recall']:.4f}`",
        f"- Shuffled-label H/M min recall: `{gate['shuffled_label_hm_min_recall']:.4f}`",
        f"- Overlap-only H/M min recall: `{gate['overlap_only_hm_min_recall']:.4f}`",
        f"- Leakage-off best H/M min recall: `{gate['leakage_off_best_hm_min_recall']:.4f}`",
        "",
        "## Model Selection",
        "",
        markdown_table(
            selection.sort_values(["hm_min_recall", "worst_thickness_hm_min_recall"], ascending=False),
            ["method", "hm_min_recall", "hematite_recall", "magnetite_recall", "worst_thickness_hm_min_recall"],
        ),
        "",
        "## Top Features",
        "",
        markdown_table(metrics.head(8), ["feature", "oriented_auc", "d_prime_abs"]),
        "",
    ]
    (output_dir / "v8a_sidecar_training_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the small v8A development-only sidecar gate after integration smoke.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_custom_diffraction_integration_smoke")
    parser.add_argument("--leakage-dir", default="results/accuracy_v3/v8a_custom_diffraction_integration_smoke_leakage_off")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_sidecar_training_smoke")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    input_dir = project_root / args.input_dir
    leakage_dir = project_root / args.leakage_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)
    input_gate, input_manifest, leakage_gate, leakage_manifest = require_input_gate(input_dir, leakage_dir)
    sk = require_sklearn()
    frame = pd.read_csv(input_dir / "v8a_integration_sidecar_features.csv")
    leakage_frame = pd.read_csv(leakage_dir / "v8a_integration_sidecar_features.csv")
    selection, validation_decisions, metrics, best_main = evaluate_training(frame, sk)
    leakage_hm = best_leakage_hm(leakage_frame, sk)

    control_hm = float(selection.loc[selection["method"].eq("ExtraTreesControlOnly"), "hm_min_recall"].iloc[0])
    total_count_hm = float(selection.loc[selection["method"].eq("ExtraTreesTotalCountOnly"), "hm_min_recall"].iloc[0])
    shuffled_hm = float(selection.loc[selection["method"].eq("ExtraTreesShuffledTrainLabels"), "hm_min_recall"].iloc[0])
    overlap_hm = float(selection.loc[selection["method"].eq("ExtraTreesOverlapOnly"), "hm_min_recall"].iloc[0])
    best_feature = metrics.iloc[0].to_dict()
    physical_pass = bool(best_feature["oriented_auc"] >= PASS_FEATURE_AUC or best_feature["d_prime_abs"] >= PASS_FEATURE_D_PRIME)
    ml_pass = bool(
        best_main["hm_min_recall"] >= PASS_HM_MIN_RECALL
        and best_main["pairwise_hm_min_recall"] >= PASS_HM_MIN_RECALL
        and best_main["worst_thickness_hm_min_recall"] >= PASS_WORST_THICKNESS_HM_MIN_RECALL
    )
    guard_pass = bool(
        control_hm < MAX_CONTROL_HM_MIN_RECALL
        and total_count_hm < MAX_TOTAL_COUNT_HM_MIN_RECALL
        and shuffled_hm < MAX_SHUFFLED_LABEL_HM_MIN_RECALL
        and overlap_hm < MAX_OVERLAP_ONLY_HM_MIN_RECALL
    )
    leakage_pass = bool(
        leakage_hm < MAX_LEAKAGE_OFF_HM_MIN_RECALL
        and float(best_main["hm_min_recall"]) - leakage_hm >= MIN_MAIN_MINUS_LEAKAGE_MARGIN
    )
    manifest_pass = bool(
        input_manifest.get("development_only")
        and leakage_manifest.get("development_only")
        and not input_manifest.get("shadow_or_final_used")
        and not leakage_manifest.get("shadow_or_final_used")
        and not input_manifest.get("reads_existing_xrt_cubes")
        and not leakage_manifest.get("reads_existing_xrt_cubes")
        and input_manifest.get("bin_axis") == "q_a_inv"
    )
    gate_passed = bool(physical_pass and ml_pass and guard_pass and leakage_pass and manifest_pass)
    gate = {
        "generated_by": "analysis/train_v8a_sidecar_smoke.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v8A_sidecar_small_training_smoke",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "claim_scope": "small_development_sidecar_training_only_not_full_matrix_not_shadow_final",
        "input_integration_gate": input_gate,
        "input_leakage_gate": leakage_gate,
        "gate_passed": gate_passed,
        "decision": "proceed_to_minimal_geant4_custom_diffraction_design_review" if gate_passed else "stop_or_refine_v8a_sidecar_training",
        "physical_observability_pass": physical_pass,
        "ml_pass": ml_pass,
        "guard_pass": guard_pass,
        "leakage_pass": leakage_pass,
        "manifest_pass": manifest_pass,
        "best_feature": str(best_feature["feature"]),
        "best_feature_oriented_auc": float(best_feature["oriented_auc"]),
        "best_feature_d_prime": float(best_feature["d_prime_abs"]),
        "best_main_method": str(best_main["method"]),
        "best_main_hm_min_recall": float(best_main["hm_min_recall"]),
        "best_main_pairwise_hm_min_recall": float(best_main["pairwise_hm_min_recall"]),
        "best_main_worst_thickness_hm_min_recall": float(best_main["worst_thickness_hm_min_recall"]),
        "best_main_hematite_recall": float(best_main["hematite_recall"]),
        "best_main_magnetite_recall": float(best_main["magnetite_recall"]),
        "control_hm_min_recall": control_hm,
        "total_count_hm_min_recall": total_count_hm,
        "shuffled_label_hm_min_recall": shuffled_hm,
        "overlap_only_hm_min_recall": overlap_hm,
        "leakage_off_best_hm_min_recall": leakage_hm,
        "main_minus_leakage_hm_margin": float(best_main["hm_min_recall"]) - leakage_hm,
        "thresholds": {
            "pass_feature_auc": PASS_FEATURE_AUC,
            "pass_feature_d_prime": PASS_FEATURE_D_PRIME,
            "pass_hm_min_recall": PASS_HM_MIN_RECALL,
            "pass_worst_thickness_hm_min_recall": PASS_WORST_THICKNESS_HM_MIN_RECALL,
            "max_control_hm_min_recall": MAX_CONTROL_HM_MIN_RECALL,
            "max_total_count_hm_min_recall": MAX_TOTAL_COUNT_HM_MIN_RECALL,
            "max_shuffled_label_hm_min_recall": MAX_SHUFFLED_LABEL_HM_MIN_RECALL,
            "max_overlap_only_hm_min_recall": MAX_OVERLAP_ONLY_HM_MIN_RECALL,
            "max_leakage_off_hm_min_recall": MAX_LEAKAGE_OFF_HM_MIN_RECALL,
            "min_main_minus_leakage_margin": MIN_MAIN_MINUS_LEAKAGE_MARGIN,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest = {
        "generated_by": "analysis/train_v8a_sidecar_smoke.py",
        "generated_at_utc": gate["generated_at_utc"],
        "protocol_name": "v8A_sidecar_small_training_smoke",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "input_dir": args.input_dir,
        "leakage_dir": args.leakage_dir,
        "output_dir": args.output_dir,
        "feature_count": int(len(frame.columns)),
        "sample_count": int(len(frame)),
        "validation_sample_count": int(frame["split"].astype(str).eq("validation").sum()),
        "gate_file": "v8a_sidecar_training_gate.json",
    }

    selection.to_csv(output_dir / "v8a_sidecar_training_model_selection.csv", index=False, lineterminator="\n")
    validation_decisions.to_csv(output_dir / "v8a_sidecar_training_validation_decisions.csv", index=False, lineterminator="\n")
    metrics.to_csv(output_dir / "v8a_sidecar_training_observability_metrics.csv", index=False, lineterminator="\n")
    (output_dir / "v8a_sidecar_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "v8a_sidecar_training_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(output_dir, gate, selection, metrics)
    print(
        "decision={decision} gate_passed={passed} main_hm={hm:.4f} leakage_hm={leakage:.4f}".format(
            decision=gate["decision"],
            passed=str(gate_passed).lower(),
            hm=gate["best_main_hm_min_recall"],
            leakage=leakage_hm,
        )
    )


if __name__ == "__main__":
    main()
