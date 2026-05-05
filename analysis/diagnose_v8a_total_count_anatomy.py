from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_v8a_event_feature_smoke import feature_sets, load_json


CLAIM_SCOPE = (
    "development-only total-count anatomy diagnostic over v8A H/M event features; "
    "not product accuracy, hardware validation, shadow/final validation, or manuscript-grade powder XRD"
)


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


def standardized_gap(hematite: pd.Series, magnetite: pd.Series) -> float:
    h = hematite.dropna().to_numpy(dtype=np.float64)
    m = magnetite.dropna().to_numpy(dtype=np.float64)
    if len(h) == 0 or len(m) == 0:
        return 0.0
    pooled = np.sqrt(0.5 * (np.var(h) + np.var(m)) + 1e-12)
    return float(abs(float(np.mean(m) - np.mean(h))) / pooled)


def distribution_rows(frame: pd.DataFrame, total_col: str) -> pd.DataFrame:
    rows = []
    group_cols = ["split", "material"]
    for keys, group in frame.groupby(group_cols, sort=True, dropna=False):
        values = group[total_col].dropna().to_numpy(dtype=np.float64)
        split, material = keys
        rows.append(
            {
                "split": split,
                "material": material,
                "samples": int(len(group)),
                "mean": float(np.mean(values)) if len(values) else 0.0,
                "std": float(np.std(values)) if len(values) else 0.0,
                "median": float(np.median(values)) if len(values) else 0.0,
                "p05": float(np.quantile(values, 0.05)) if len(values) else 0.0,
                "p95": float(np.quantile(values, 0.95)) if len(values) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def gap_rows(frame: pd.DataFrame, total_col: str) -> pd.DataFrame:
    rows = []
    strata = [
        ("overall", []),
        ("split", ["split"]),
        ("split_thickness", ["split", "thickness_mm"]),
        ("split_pose", ["split", "pose_index"]),
        ("split_source_mode", ["split", "source_mode"]),
        ("split_stress_label", ["split", "stress_label"]),
        ("split_thickness_pose", ["split", "thickness_mm", "pose_index"]),
    ]
    for stratum_name, group_cols in strata:
        grouped = [((), frame)] if not group_cols else frame.groupby(group_cols, sort=True, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            h = group[group["material"].astype(str).eq("Hematite")][total_col]
            m = group[group["material"].astype(str).eq("Magnetite")][total_col]
            if len(h) == 0 or len(m) == 0:
                continue
            row = {
                "stratum": stratum_name,
                "group_key": "|".join(str(item) for item in keys) if keys else "all",
                "hematite_samples": int(len(h)),
                "magnetite_samples": int(len(m)),
                "hematite_mean": float(h.mean()),
                "magnetite_mean": float(m.mean()),
                "mean_gap_magnetite_minus_hematite": float(m.mean() - h.mean()),
                "standardized_gap_abs": standardized_gap(h, m),
            }
            for col, value in zip(group_cols, keys):
                row[str(col)] = value
            rows.append(row)
    return pd.DataFrame(rows)


def feature_correlations(frame: pd.DataFrame, main_cols: list[str], total_col: str) -> pd.DataFrame:
    rows = []
    scopes = [("all_source_on", frame["source_mode"].astype(str).eq("custom_diffraction_on"))]
    for split in ["train", "validation", "stress_holdout"]:
        scopes.append((f"{split}_source_on", frame["split"].astype(str).eq(split) & frame["source_mode"].astype(str).eq("custom_diffraction_on")))
    for scope, mask in scopes:
        subset = frame.loc[mask].copy()
        total = subset[total_col].astype(float)
        for col in main_cols:
            values = subset[col].astype(float)
            pearson = float(values.corr(total, method="pearson")) if len(subset) > 2 else 0.0
            spearman = float(values.corr(total, method="spearman")) if len(subset) > 2 else 0.0
            rows.append(
                {
                    "scope": scope,
                    "feature": col,
                    "samples": int(len(subset)),
                    "pearson": pearson if np.isfinite(pearson) else 0.0,
                    "spearman": spearman if np.isfinite(spearman) else 0.0,
                    "abs_pearson": abs(pearson) if np.isfinite(pearson) else 0.0,
                    "abs_spearman": abs(spearman) if np.isfinite(spearman) else 0.0,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    max_corr = result.groupby("feature", as_index=False).agg(
        max_abs_pearson=("abs_pearson", "max"),
        max_abs_spearman=("abs_spearman", "max"),
    )
    result = result.merge(max_corr, on="feature", how="left")
    return result.sort_values(["max_abs_pearson", "max_abs_spearman", "feature"], ascending=[False, False, True])


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 12) -> str:
    if frame.empty:
        return ""
    view = frame.head(limit)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view[columns].iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            rendered.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, gate: dict[str, Any], gaps: pd.DataFrame, correlations: pd.DataFrame) -> None:
    lines = [
        "# v8A total-count anatomy diagnostic",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Diagnostic passed: `{str(gate['gate_passed']).lower()}`",
        f"- Existing Phase 4 total-count-only H/M min recall: `{gate['existing_phase4_total_count_only_hm_min_recall']:.4f}`",
        f"- Max standardized total-count gap: `{gate['max_standardized_total_count_gap']:.4f}`",
        f"- Max main-feature abs Pearson correlation to total count: `{gate['max_main_feature_abs_pearson_to_total_count']:.4f}`",
        "",
        "## Largest Total-Count Gaps",
        "",
        markdown_table(
            gaps.sort_values("standardized_gap_abs", ascending=False),
            ["stratum", "group_key", "hematite_samples", "magnetite_samples", "standardized_gap_abs", "mean_gap_magnetite_minus_hematite"],
        ),
        "",
        "## Most Count-Like Main Features",
        "",
        markdown_table(
            correlations.drop_duplicates("feature").sort_values(["max_abs_pearson", "max_abs_spearman"], ascending=False),
            ["feature", "max_abs_pearson", "max_abs_spearman"],
        ),
        "",
        "## Claim Boundary",
        "",
        "This diagnostic explains confounding risk only. It does not promote the model, does not unlock shadow/final, and does not support product or hardware claims.",
        "",
    ]
    (output_dir / "v8a_total_count_anatomy_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose total-count shortcut anatomy for v8A H/M development features.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_event_to_feature")
    parser.add_argument("--phase4-gate", default="results/accuracy_v3/v8a_medium_plus_count_overlap_development_model/v8a_medium_development_model_gate.json")
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_medium_plus_count_overlap_total_count_anatomy")
    parser.add_argument("--total-count-column", default="control_total_count_norm")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    schema_gate = load_json(input_dir / "v8a_event_schema_gate.json")
    manifest = load_json(input_dir / "v8a_event_feature_manifest.json")
    if bool(schema_gate.get("shadow_or_final_used")) or bool(manifest.get("shadow_or_final_used")):
        raise RuntimeError("Refusing total-count anatomy because input reports shadow/final use.")
    if bool(schema_gate.get("reads_existing_xrt_cubes")) or bool(manifest.get("reads_existing_xrt_cubes")):
        raise RuntimeError("Refusing total-count anatomy because input reports existing XRT cube reads.")

    frame = pd.read_csv(input_dir / "v8a_event_sidecar_features.csv")
    if args.total_count_column not in frame.columns:
        raise RuntimeError(f"Missing total-count column: {args.total_count_column}")
    main_cols, _, _, _, _ = feature_sets(frame)
    distributions = distribution_rows(frame, args.total_count_column)
    gaps = gap_rows(frame, args.total_count_column)
    correlations = feature_correlations(frame, main_cols, args.total_count_column)

    phase4_gate_path = project_root / args.phase4_gate
    phase4_gate = load_json(phase4_gate_path) if phase4_gate_path.exists() else {}
    existing_total_count_hm = float(phase4_gate.get("total_count_only_hm_min_recall", 0.0))
    max_gap = float(gaps["standardized_gap_abs"].max()) if not gaps.empty else 0.0
    max_corr = float(correlations["max_abs_pearson"].max()) if not correlations.empty else 0.0
    high_corr_fraction = (
        float((correlations.drop_duplicates("feature")["max_abs_pearson"] >= 0.90).mean())
        if not correlations.empty
        else 0.0
    )
    if not main_cols or (max_corr >= 0.98 and high_corr_fraction >= 0.75):
        decision = "stop_sidecar_claim_too_count_coupled"
    elif existing_total_count_hm >= 0.60 or max_corr >= 0.75 or max_gap >= 0.50:
        decision = "feature_rework_needed"
    else:
        decision = "feature_rework_not_needed"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/diagnose_v8a_total_count_anatomy.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_total_count_anatomy_diagnostic",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "input_dir": args.input_dir,
        "phase4_gate": args.phase4_gate,
        "gate_passed": True,
        "decision": decision,
        "total_count_column": args.total_count_column,
        "sample_count": int(len(frame)),
        "main_feature_count": int(len(main_cols)),
        "existing_phase4_total_count_only_hm_min_recall": existing_total_count_hm,
        "max_standardized_total_count_gap": max_gap,
        "max_main_feature_abs_pearson_to_total_count": max_corr,
        "fraction_main_features_abs_pearson_ge_0p90": high_corr_fraction,
        "training_unlocked": False,
        "stop_reasons": [] if decision != "stop_sidecar_claim_too_count_coupled" else ["main_features_too_count_coupled"],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    distributions.to_csv(output_dir / "v8a_total_count_distributions.csv", index=False, lineterminator="\n")
    gaps.to_csv(output_dir / "v8a_total_count_gap_by_stratum.csv", index=False, lineterminator="\n")
    correlations.to_csv(output_dir / "v8a_total_count_feature_correlations.csv", index=False, lineterminator="\n")
    write_json(output_dir / "v8a_total_count_anatomy_gate.json", json_clean(gate))
    write_report(output_dir, gate, gaps, correlations)
    print(
        "decision={decision} phase4_total_count={total:.4f} max_gap={gap:.4f} max_corr={corr:.4f}".format(
            decision=decision,
            total=existing_total_count_hm,
            gap=max_gap,
            corr=max_corr,
        )
    )


if __name__ == "__main__":
    main()
