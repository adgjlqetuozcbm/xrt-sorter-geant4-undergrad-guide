from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from v8a_event_to_feature_pipeline import build_control_audit, relpath


CLAIM_BOUNDARY = (
    "development-only medium-plus-count-overlap feature combination; not product accuracy, "
    "not shadow/final validation, not hardware validation, and not manuscript-grade powder XRD"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite to replace development artifacts.")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def as_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def require_no_shadow_or_xrt(name: str, payload: dict[str, Any]) -> None:
    if bool(payload.get("shadow_or_final_used", False)):
        raise RuntimeError(f"Refusing combined features because {name} reports shadow/final use.")
    if bool(payload.get("reads_existing_xrt_cubes", False)):
        raise RuntimeError(f"Refusing combined features because {name} reports existing XRT cube reads.")


def read_feature_dir(path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    features = pd.read_csv(path / "v8a_event_sidecar_features.csv")
    long_table = pd.read_csv(path / "v8a_event_sidecar_long.csv")
    manifest = load_json(path / "v8a_event_feature_manifest.json")
    gate = load_json(path / "v8a_event_schema_gate.json")
    features.insert(0, "combined_feature_origin", label)
    long_table.insert(0, "combined_feature_origin", label)
    return features, long_table, manifest, gate


def ordered_union(left: list[str], right: list[str]) -> list[str]:
    values: list[str] = []
    for item in left + right:
        if item not in values:
            values.append(item)
    return values


def write_control_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    pd.DataFrame(rows, columns=fields).to_csv(path, index=False, lineterminator="\n")


def boolean_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[column].astype(str).str.lower().isin({"true", "1", "yes", "y"})


def write_report(output_dir: Path, gate: dict[str, Any], manifest: dict[str, Any], control_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v8A medium-plus-count-overlap feature combination report",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_BOUNDARY}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Training gate allowed: `{str(gate['tiny_training_gate_allowed']).lower()}`",
        f"- Medium rows: `{gate['medium_sample_count']}`",
        f"- Extension rows: `{gate['extension_sample_count']}`",
        f"- Combined rows: `{manifest['sample_count']}`",
        f"- Source-on rows: `{gate['control_summary']['source_on_rows']}`",
        f"- Source-off rows: `{gate['control_summary']['source_off_rows']}`",
        f"- Shadow/final used: `{str(gate['shadow_or_final_used']).lower()}`",
        "",
        "## Control Audit",
        "",
        "| control_group | status | metric | value | details |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in control_rows:
        lines.append(
            "| {control_group} | {status} | {metric} | {value} | {details} |".format(
                control_group=row.get("control_group", ""),
                status=row.get("status", ""),
                metric=row.get("metric", ""),
                value=row.get("value", ""),
                details=str(row.get("details", "")).replace("|", "/"),
            )
        )
    lines.extend(["", "## Stop Reasons", ""])
    if gate["stop_reasons"]:
        lines.extend(f"- {reason}" for reason in gate["stop_reasons"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This combined feature directory is a development-only input for renewed stress/count gates. It does not unlock shadow/final and must not be reported as product accuracy, hardware validation, or publishable powder-XRD simulation.",
            "",
        ]
    )
    (output_dir / "v8a_medium_plus_count_overlap_schema_gate_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine v8A medium features with count-overlap extension source-on features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--medium-input-dir", default="results/accuracy_v3/v8a_medium_event_to_feature")
    parser.add_argument("--extension-input-dir", default="results/accuracy_v3/v8a_count_overlap_extension_event_to_feature")
    parser.add_argument("--extension-config", default="analysis/configs/v8a_count_overlap_extension_config.json")
    parser.add_argument("--extension-prereg-dir", default="results/accuracy_v3/v8a_count_overlap_extension_prereg")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_event_to_feature")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    medium_dir = as_project_path(project_root, args.medium_input_dir)
    extension_dir = as_project_path(project_root, args.extension_input_dir)
    output_dir = as_project_path(project_root, args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    config = load_json(as_project_path(project_root, args.extension_config))
    prereg_gate = load_json(as_project_path(project_root, args.extension_prereg_dir) / "v8a_count_overlap_extension_prereg_gate.json")
    medium_features, medium_long, medium_manifest, medium_gate = read_feature_dir(medium_dir, "medium")
    extension_features, extension_long, extension_manifest, extension_gate = read_feature_dir(extension_dir, "count_overlap_extension")

    require_no_shadow_or_xrt("medium manifest", medium_manifest)
    require_no_shadow_or_xrt("medium gate", medium_gate)
    require_no_shadow_or_xrt("extension manifest", extension_manifest)
    require_no_shadow_or_xrt("extension gate", extension_gate)
    require_no_shadow_or_xrt("extension prereg gate", prereg_gate)
    if not bool(medium_gate.get("gate_passed", False)) or not bool(medium_gate.get("tiny_training_gate_allowed", False)):
        raise RuntimeError(f"Medium schema/training gate is not usable: {medium_gate.get('decision')}")
    if not bool(prereg_gate.get("gate_passed", False)) or bool(prereg_gate.get("training_unlocked", True)):
        raise RuntimeError(f"Extension prereg gate is not a passed, training-locked prereg artifact: {prereg_gate.get('decision')}")
    if str(extension_manifest.get("profile")) != str(config["profile"]):
        raise RuntimeError(f"Extension feature profile mismatch: {extension_manifest.get('profile')} != {config['profile']}")
    if bool(extension_gate.get("gate_passed", False)):
        raise RuntimeError("Extension-only schema gate unexpectedly passed; this combiner expects source-on-only extension rows.")

    medium_peak_id = str(medium_manifest.get("peak_table_id"))
    extension_peak_id = str(extension_manifest.get("peak_table_id"))
    required_peak_id = str(config["required_peak_table_id"])
    if medium_peak_id != required_peak_id or extension_peak_id != required_peak_id:
        raise RuntimeError(f"Peak table mismatch: medium={medium_peak_id} extension={extension_peak_id} required={required_peak_id}")
    if not bool(medium_manifest.get("source_peak_table_matches_analysis", False)):
        raise RuntimeError("Medium source peak table does not match analysis peak table.")
    if not bool(extension_manifest.get("source_peak_table_matches_analysis", False)):
        raise RuntimeError("Extension source peak table does not match analysis peak table.")
    if set(extension_features["source_mode"].astype(str)) != {"custom_diffraction_on"}:
        raise RuntimeError("Extension features must be source-on only.")
    if "custom_diffraction_off" not in set(medium_features["source_mode"].astype(str)):
        raise RuntimeError("Medium features must provide the source-off control rows for the combined gate.")

    for field in ["bin_axis", "q_bin_width_a_inv", "d_bin_width_a", "peak_window_a_inv"]:
        if medium_manifest.get(field) != extension_manifest.get(field):
            raise RuntimeError(f"Feature-grid mismatch for {field}: {medium_manifest.get(field)} != {extension_manifest.get(field)}")

    combined_features = pd.concat([medium_features, extension_features], ignore_index=True, sort=False)
    combined_long = pd.concat([medium_long, extension_long], ignore_index=True, sort=False)
    if combined_features["sample_id"].duplicated().any():
        duplicates = sorted(combined_features.loc[combined_features["sample_id"].duplicated(), "sample_id"].astype(str).unique())
        raise RuntimeError(f"Duplicate sample_id values after combination: {duplicates[:5]}")

    development_only = bool(boolean_series(combined_features, "development_only", True).all())
    shadow_or_final_used = bool(boolean_series(combined_features, "shadow_or_final_used", False).any())
    peak_id_matches = set(combined_features["peak_table_id"].astype(str)) == {required_peak_id}
    source_peak_table_ids = sorted(set(combined_features["source_peak_table_id"].astype(str)))
    source_peak_table_matches_analysis = source_peak_table_ids == [required_peak_id]
    control_rows, control_summary = build_control_audit(combined_features.to_dict(orient="records"))
    source_lineage_ok = control_summary["source_on_rows"] > 0 and control_summary["source_off_rows"] > 0
    schema_gate_passed = bool(
        development_only
        and not shadow_or_final_used
        and peak_id_matches
        and source_peak_table_matches_analysis
        and source_lineage_ok
        and control_summary["source_on_signal_gt_source_off"]
        and control_summary["source_off_low"]
    )
    tiny_training_gate_allowed = bool(schema_gate_passed and control_summary["balanced_training_support"])

    stop_reasons: list[str] = []
    if not development_only:
        stop_reasons.append("At least one combined row is not development_only.")
    if shadow_or_final_used:
        stop_reasons.append("At least one combined row reports shadow/final usage.")
    if not peak_id_matches:
        stop_reasons.append("Combined rows do not all use the required successor peak table id.")
    if not source_peak_table_matches_analysis:
        stop_reasons.append("Combined source peak table ids do not match the analysis peak table.")
    if not source_lineage_ok:
        stop_reasons.append("Combined rows do not contain both source-on and source-off controls.")
    if not control_summary["source_on_signal_gt_source_off"]:
        stop_reasons.append("Source-on peak-window signal is not above source-off.")
    if not control_summary["source_off_low"]:
        stop_reasons.append("Source-off peak-window signal is not low enough.")
    if not control_summary["balanced_training_support"]:
        stop_reasons.append("Combined rows lack balanced H/M source-on/source-off support across train/validation splits.")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ordered_columns = ordered_union(list(medium_features.columns), list(extension_features.columns))
    combined_features = combined_features.reindex(columns=ordered_columns)
    combined_long = combined_long.reindex(columns=ordered_union(list(medium_long.columns), list(extension_long.columns)))
    main_feature_columns = [col for col in combined_features.columns if col.startswith("diffraction_")]
    control_feature_columns = [col for col in combined_features.columns if col.startswith("control_")]

    manifest = {
        "generated_by": "analysis/combine_v8a_medium_count_overlap_features.py",
        "generated_at_utc": generated_at,
        "profile": "v8a_medium_plus_count_overlap_extension",
        "input_profiles": [medium_manifest.get("profile"), extension_manifest.get("profile")],
        "development_only": development_only,
        "shadow_or_final_used": shadow_or_final_used,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "bin_axis": medium_manifest.get("bin_axis"),
        "q_bin_width_a_inv": medium_manifest.get("q_bin_width_a_inv"),
        "d_bin_width_a": medium_manifest.get("d_bin_width_a"),
        "peak_window_a_inv": medium_manifest.get("peak_window_a_inv"),
        "peak_table_id": required_peak_id,
        "source_peak_table_ids": source_peak_table_ids,
        "source_peak_table_matches_analysis": source_peak_table_matches_analysis,
        "peak_manifest_path": medium_manifest.get("peak_manifest_path"),
        "schema_contract_path": medium_manifest.get("schema_contract_path"),
        "sample_count": int(len(combined_features)),
        "medium_sample_count": int(len(medium_features)),
        "extension_sample_count": int(len(extension_features)),
        "long_row_count": int(len(combined_long)),
        "feature_column_count": int(len(ordered_columns)),
        "main_feature_columns": main_feature_columns,
        "control_feature_columns": control_feature_columns,
        "lineage_columns_excluded_from_main_features": medium_manifest.get("lineage_columns_excluded_from_main_features", []),
        "input_dirs": {
            "medium": relpath(medium_dir, project_root),
            "extension": relpath(extension_dir, project_root),
        },
        "claim_scope": CLAIM_BOUNDARY,
    }
    gate = {
        "generated_by": "analysis/combine_v8a_medium_count_overlap_features.py",
        "generated_at_utc": generated_at,
        "gate_passed": schema_gate_passed,
        "tiny_training_gate_allowed": tiny_training_gate_allowed,
        "decision": (
            "combined_schema_control_gate_passed_ready_for_phase4_rework"
            if tiny_training_gate_allowed
            else "stop_or_rework_medium_plus_count_overlap_schema_control_gate"
        ),
        "development_only": development_only,
        "shadow_or_final_used": shadow_or_final_used,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "bin_axis": medium_manifest.get("bin_axis"),
        "peak_table_id": required_peak_id,
        "source_peak_table_ids": source_peak_table_ids,
        "source_peak_table_matches_analysis": source_peak_table_matches_analysis,
        "schema_contract_ok": True,
        "long_schema_ok": True,
        "peak_table_id_matches": peak_id_matches,
        "source_lineage_ok": source_lineage_ok,
        "medium_sample_count": int(len(medium_features)),
        "extension_sample_count": int(len(extension_features)),
        "control_summary": control_summary,
        "stop_reasons": stop_reasons,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    combined_features.to_csv(output_dir / "v8a_event_sidecar_features.csv", index=False, lineterminator="\n")
    combined_long.to_csv(output_dir / "v8a_event_sidecar_long.csv", index=False, lineterminator="\n")
    write_control_audit(output_dir / "v8a_event_control_audit.csv", control_rows)
    write_json(output_dir / "v8a_event_feature_manifest.json", manifest)
    write_json(output_dir / "v8a_event_schema_gate.json", gate)
    write_report(output_dir, gate, manifest, control_rows)
    print(
        "decision={decision} gate_passed={passed} combined_samples={samples} extension_samples={extension}".format(
            decision=gate["decision"],
            passed=str(schema_gate_passed).lower(),
            samples=manifest["sample_count"],
            extension=manifest["extension_sample_count"],
        )
    )


if __name__ == "__main__":
    main()
