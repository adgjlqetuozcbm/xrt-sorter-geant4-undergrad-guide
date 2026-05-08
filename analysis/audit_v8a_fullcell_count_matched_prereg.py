from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_fullcell_clean_development_model import (
    COUNT_BALANCE_STRATEGIES,
    SPLITS,
    build_count_balanced_subset,
    pair_counts,
    standardized_count_gap,
    support_pass,
)
from train_v8a_event_feature_smoke import load_json


CLAIM_SCOPE = (
    "development-only full-cell H/M count-matched preregistration audit; "
    "does not run Geant4, does not unlock training, does not use shadow/final, "
    "and does not support product accuracy, hardware validation, full ten-material, "
    "or manuscript-grade powder XRD claims"
)

WATCH_STRATEGIES = {
    "fixed_bin_width_0p015",
    "fixed_bin_width_0p020",
    "fixed_bin_width_0p040",
    "fixed_bin_width_0p050",
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
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def summarize_strategy(frame: pd.DataFrame, strategy: dict[str, Any]) -> dict[str, Any]:
    balanced = build_count_balanced_subset(frame, strategy)
    counts = pair_counts(balanced, "count_balance_pair_id")
    return {
        "strategy": strategy["strategy"],
        "kind": strategy["kind"],
        "bin_width": strategy.get("bin_width"),
        **{f"{split}_pairs": int(counts[split]) for split in SPLITS},
        **{f"{split}_count_gap_standardized": float(standardized_count_gap(balanced, split)) for split in SPLITS},
        "support_pass": bool(support_pass(counts)),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v8A full-cell count-matched training preregistration before any rerun.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="analysis/configs/v8a_fullcell_count_matched_training_prereg_config.json")
    parser.add_argument("--source-matrix-config", default="analysis/configs/v8a_clean_hm_fullcell_countmatched_matrix_config.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_fullcell_count_matched_training_prereg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = load_json(project_root / args.config)
    matrix_config = load_json(project_root / args.source_matrix_config)
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    input_dir = project_root / config["canonical_input_view"]
    final_audit_gate = load_json(project_root / config["final_data_audit"] / "v8a_fullcell_training_data_final_audit_gate.json")
    baseline_gate = load_json(project_root / config["previous_baseline_gate"])
    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    source_on = frame[frame["source_mode"].astype(str).eq("custom_diffraction_on")].copy()

    strategy_rows = [
        summarize_strategy(source_on, strategy)
        for strategy in COUNT_BALANCE_STRATEGIES
        if strategy["strategy"] in WATCH_STRATEGIES
    ]
    strategy_frame = pd.DataFrame(strategy_rows).sort_values("bin_width")
    primary = strategy_frame[strategy_frame["strategy"].eq(config["primary_count_balance_strategy"]["strategy"])].iloc[0].to_dict()
    strict = strategy_frame[strategy_frame["strategy"].eq(config["strict_sensitivity_strategy"]["strategy"])].iloc[0].to_dict()
    wide_support = strategy_frame[strategy_frame["support_pass"].astype(bool)].copy()

    minimum = config["minimum_support"]
    source_matrix_targets = matrix_config["count_matched_training_targets"]
    stop_reasons: list[str] = []
    prereg_reasons: list[str] = []
    if not bool(schema_gate.get("gate_passed", False)):
        stop_reasons.append(f"Input schema gate did not pass: {schema_gate.get('decision')}.")
    if not bool(final_audit_gate.get("training_unlocked", False)):
        stop_reasons.append(f"Final data audit did not unlock diagnostics: {final_audit_gate.get('decision')}.")
    if bool(schema_gate.get("shadow_or_final_used", False)) or bool(final_audit_gate.get("shadow_or_final_used", False)):
        stop_reasons.append("An input gate reports shadow/final use.")
    if bool(schema_gate.get("reads_existing_xrt_cubes", False)):
        stop_reasons.append("Input schema reports existing XRT cube reads.")
    if bool(primary["support_pass"]):
        prereg_reasons.append("Primary 0.020 support unexpectedly passes on the current view; rerun baseline before source-side matrix.")
    else:
        prereg_reasons.append("Primary 0.020 support is below 300/200/200 on the current view.")
    if bool(strict["support_pass"]):
        prereg_reasons.append("Strict 0.015 support unexpectedly passes on the current view; report this before any source-side matrix.")
    else:
        prereg_reasons.append("Strict 0.015 support is below 300/200/200 on the current view.")
    if bool(baseline_gate.get("gate_passed", True)):
        stop_reasons.append("Previous full-cell baseline gate unexpectedly passed; this prereg stop should be revisited.")
    if str(baseline_gate.get("decision")) != config["previous_baseline_decision"]:
        stop_reasons.append(
            f"Previous baseline decision mismatch: {baseline_gate.get('decision')} != {config['previous_baseline_decision']}."
        )

    current_view_training_submittable = bool(primary["support_pass"]) and bool(strict["support_pass"]) and bool(baseline_gate.get("gate_passed", False))
    source_matrix_shape_ok = (
        int(matrix_config["pair_replicates_per_cell"]) == int(config["source_side_matrix_preregistration"]["pair_replicates_per_full_cell"])
        and int(matrix_config["expected_rows"]["total"]) == int(config["source_side_matrix_preregistration"]["expected_rows"]["total"])
        and float(source_matrix_targets["primary_window"]) == float(config["primary_count_balance_strategy"]["bin_width"])
        and float(source_matrix_targets["strict_sensitivity_window"]) == float(config["strict_sensitivity_strategy"]["bin_width"])
    )
    if not source_matrix_shape_ok:
        stop_reasons.append("Source-side count-matched matrix config does not match preregistered support target shape.")

    gate_passed = not stop_reasons
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    decision = (
        "current_view_not_training_submittable_source_matrix_preregistered"
        if gate_passed and not current_view_training_submittable
        else "stop_fullcell_count_matched_prereg"
    )
    gate = {
        "generated_by": "analysis/audit_v8a_fullcell_count_matched_prereg.py",
        "generated_at_utc": generated_at,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "training_unlocked": False,
        "claim_scope": CLAIM_SCOPE,
        "gate_passed": gate_passed,
        "decision": decision,
        "current_view_training_submittable": current_view_training_submittable,
        "canonical_input_view": config["canonical_input_view"],
        "minimum_support": minimum,
        "strategy_support": json_clean(strategy_frame.to_dict(orient="records")),
        "primary_strategy": json_clean(primary),
        "strict_sensitivity_strategy": json_clean(strict),
        "wide_support_strategies": json_clean(wide_support.to_dict(orient="records")),
        "previous_baseline_decision": baseline_gate.get("decision"),
        "previous_ordinary_gate_passed": bool(baseline_gate.get("ordinary_gate", {}).get("gate_passed", False)),
        "previous_count_balanced_gate_passed": bool(baseline_gate.get("count_balanced_gate", {}).get("gate_passed", False)),
        "previous_ordinary_total_count_only_hm_min_recall": baseline_gate.get("ordinary_gate", {}).get("total_count_only_hm_min_recall"),
        "previous_count_balanced_total_count_only_hm_min_recall": baseline_gate.get("count_balanced_gate", {}).get("total_count_only_hm_min_recall"),
        "previous_ordinary_shuffled_label_null_p95": baseline_gate.get("ordinary_gate", {}).get("shuffled_label_null_p95"),
        "previous_count_balanced_shuffled_label_null_p95": baseline_gate.get("count_balanced_gate", {}).get("shuffled_label_null_p95"),
        "source_side_matrix_config": args.source_matrix_config,
        "source_side_matrix_profile": matrix_config["profile"],
        "source_side_pair_replicates_per_cell": int(matrix_config["pair_replicates_per_cell"]),
        "source_side_expected_rows": matrix_config["expected_rows"],
        "source_side_expected_strict_pairs": matrix_config["expected_strict_pairs"],
        "source_side_matrix_shape_ok": source_matrix_shape_ok,
        "next_allowed_stage": "generate_source_side_countmatched_matrix_preflight_only" if gate_passed else "fix_prereg_before_any_run",
        "stop_reasons": stop_reasons,
        "prereg_reasons": prereg_reasons,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(output_dir / "v8a_fullcell_count_matched_training_prereg_gate.json", json_clean(gate))
    strategy_frame.to_csv(output_dir / "v8a_fullcell_count_matched_support_sweep.csv", index=False, lineterminator="\n")

    lines = [
        "# v8A full-cell count-matched training preregistration audit",
        "",
        f"Generated: {generated_at}",
        "",
        "Scope: development-only preregistration audit. This does not run Geant4, does not unlock training, and does not touch shadow/final.",
        "",
        f"- Decision: `{decision}`",
        f"- Gate passed: `{str(gate_passed).lower()}`",
        f"- Current view training-submittable: `{str(current_view_training_submittable).lower()}`",
        f"- Next allowed stage: `{gate['next_allowed_stage']}`",
        "",
        "## Current View Support",
        "",
        markdown_table(
            strategy_frame,
            [
                "strategy",
                "train_pairs",
                "validation_pairs",
                "stress_holdout_pairs",
                "train_count_gap_standardized",
                "validation_count_gap_standardized",
                "stress_holdout_count_gap_standardized",
                "support_pass",
            ],
        ),
        "",
        "## Baseline Controls",
        "",
        f"- Previous baseline decision: `{baseline_gate.get('decision')}`",
        f"- Ordinary total-count-only H/M min recall: `{gate['previous_ordinary_total_count_only_hm_min_recall']}`",
        f"- Ordinary shuffled-label null p95: `{gate['previous_ordinary_shuffled_label_null_p95']}`",
        f"- Count-balanced total-count-only H/M min recall: `{gate['previous_count_balanced_total_count_only_hm_min_recall']}`",
        f"- Count-balanced shuffled-label null p95: `{gate['previous_count_balanced_shuffled_label_null_p95']}`",
        "",
        "## Source-Side Prereg",
        "",
        f"- Config: `{args.source_matrix_config}`",
        f"- Profile: `{matrix_config['profile']}`",
        f"- Pair replicates per full nuisance cell: `{matrix_config['pair_replicates_per_cell']}`",
        f"- Expected rows: `{matrix_config['expected_rows']}`",
        f"- Expected strict pairs: `{matrix_config['expected_strict_pairs']}`",
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in stop_reasons) if stop_reasons else lines.append("- None.")
    lines.extend(["", "## Prereg Reasons", ""])
    lines.extend(f"- {reason}" for reason in prereg_reasons)
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- The current admitted full-cell view remains a diagnostic input, not training evidence.",
            "- The new source-side matrix config may be generated and preflighted only after this gate passes.",
            "- Training remains locked until new Geant4 outputs, feature extraction, final audit, and ordinary/count-balanced gates pass.",
        ]
    )
    (output_dir / "v8a_fullcell_count_matched_training_prereg_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "decision={decision} gate_passed={passed} current_view_training_submittable={submittable} next={next_stage}".format(
            decision=decision,
            passed=str(gate_passed).lower(),
            submittable=str(current_view_training_submittable).lower(),
            next_stage=gate["next_allowed_stage"],
        )
    )


if __name__ == "__main__":
    main()
