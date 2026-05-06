from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLAIM_SCOPE = (
    "development-only paired-clean null-tail anatomy for v8A H/M sidecar features; "
    "not training evidence, product accuracy, hardware validation, shadow/final validation, "
    "or manuscript-grade powder XRD"
)

THRESHOLDS = {
    "tail_threshold": 0.55,
    "tail_row_count_stop_min": 1,
    "tail_seed_concentration_share_max": 0.35,
    "tail_mode_concentration_share_max": 0.80,
    "tail_split_concentration_share_max": 0.80,
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def value_counts(rows: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    counts = Counter(rows[column].astype(str).fillna("missing")) if column in rows else Counter()
    total = max(sum(counts.values()), 1)
    return [
        {"value": key, "count": int(count), "share": float(count / total)}
        for key, count in counts.most_common()
    ]


def grouped_tail_summary(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=columns + ["tail_count", "tail_share", "hm_min_recall_mean", "hm_min_recall_max"])
    grouped = (
        rows.groupby(columns, dropna=False, sort=True)
        .agg(
            tail_count=("hm_min_recall", "size"),
            hm_min_recall_mean=("hm_min_recall", "mean"),
            hm_min_recall_max=("hm_min_recall", "max"),
        )
        .reset_index()
    )
    grouped["tail_share"] = grouped["tail_count"] / max(len(rows), 1)
    return grouped.sort_values(["tail_count", "hm_min_recall_max"], ascending=[False, False])


def write_report(output_dir: Path, gate: dict[str, Any], top_rows: pd.DataFrame) -> None:
    if top_rows.empty:
        top_tail_text = "No rows above threshold."
    else:
        show_cols = [
            col
            for col in [
                "shuffle_seed",
                "shuffle_mode",
                "model",
                "eval_split",
                "threshold_policy",
                "hm_min_recall",
                "hematite_recall",
                "magnetite_recall",
            ]
            if col in top_rows.columns
        ]
        top_tail_text = top_rows.head(20)[show_cols].to_csv(index=False, lineterminator="\n")
    lines = [
        "# v8A paired-clean null-tail anatomy",
        "",
        f"Generated: {gate['generated_at_utc']}",
        "",
        f"Scope: {CLAIM_SCOPE}.",
        "",
        "## Decision",
        "",
        f"- Decision: `{gate['decision']}`",
        f"- Gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- Tail threshold: `{gate['tail_threshold']}`",
        f"- Tail rows: `{gate['tail_row_count']}`",
        f"- Max tail hm min recall: `{gate['max_tail_hm_min_recall']:.4f}`",
        f"- Top seed share: `{gate['top_tail_seed_share']:.4f}`",
        f"- Top mode share: `{gate['top_tail_mode_share']:.4f}`",
        f"- Top split share: `{gate['top_tail_split_share']:.4f}`",
        "",
        "## Stop Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate["stop_reasons"]) if gate["stop_reasons"] else lines.append("- None.")
    lines.extend(
        [
        "",
        "## Top Tail Rows",
        "",
        "```csv",
        top_tail_text.rstrip(),
        "```",
        "",
        "## Claim Boundary",
            "",
            "This anatomy explains why the null gate is still blocked. It does not unlock training or any product/hardware/shadow/final claim.",
        ]
    )
    (output_dir / "v8a_paired_null_tail_anatomy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit where paired-clean null p95/tail failures concentrate.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--null-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tail-threshold", type=float, default=THRESHOLDS["tail_threshold"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    null_dir = project_root / args.null_dir
    feature_dir = project_root / args.feature_dir
    output_dir = project_root / args.output_dir
    ensure_output_dir(output_dir, args.overwrite)

    null_gate = load_json(null_dir / "v8a_paired_clean_null_behavior_gate.json")
    schema_gate = load_json(feature_dir / "v8a_event_schema_gate.json")
    manifest = load_json(feature_dir / "v8a_event_feature_manifest.json")
    for name, payload in {"null_gate": null_gate, "schema_gate": schema_gate, "manifest": manifest}.items():
        if bool(payload.get("shadow_or_final_used", False)):
            raise RuntimeError(f"Refusing tail anatomy because {name} reports shadow/final use.")
        if bool(payload.get("reads_existing_xrt_cubes", False)):
            raise RuntimeError(f"Refusing tail anatomy because {name} reports existing XRT cube reads.")

    rows = pd.read_csv(null_dir / "v8a_paired_clean_null_behavior_rows.csv")
    summary = pd.read_csv(null_dir / "v8a_paired_clean_null_behavior_summary.csv")
    tail_threshold = float(args.tail_threshold)
    tail = rows[rows["hm_min_recall"].fillna(0.0).astype(float) > tail_threshold].copy()
    top_rows = tail.sort_values("hm_min_recall", ascending=False).head(200)

    tail_count = int(len(tail))
    max_tail = float(tail["hm_min_recall"].max()) if tail_count else 0.0
    seed_counts = value_counts(tail, "shuffle_seed")
    mode_counts = value_counts(tail, "shuffle_mode")
    split_counts = value_counts(tail, "eval_split")
    model_counts = value_counts(tail, "model")
    policy_counts = value_counts(tail, "threshold_policy")
    top_seed_share = float(seed_counts[0]["share"]) if seed_counts else 0.0
    top_mode_share = float(mode_counts[0]["share"]) if mode_counts else 0.0
    top_split_share = float(split_counts[0]["share"]) if split_counts else 0.0

    grouped_tail_summary(tail, ["shuffle_mode", "model", "eval_split", "threshold_policy"]).to_csv(
        output_dir / "v8a_paired_null_tail_by_mode_model_split.csv", index=False, lineterminator="\n"
    )
    grouped_tail_summary(tail, ["shuffle_seed", "shuffle_mode", "model", "eval_split"]).to_csv(
        output_dir / "v8a_paired_null_tail_by_seed.csv", index=False, lineterminator="\n"
    )
    top_rows.to_csv(output_dir / "v8a_paired_null_tail_top_rows.csv", index=False, lineterminator="\n")
    summary.to_csv(output_dir / "v8a_paired_null_tail_source_summary.csv", index=False, lineterminator="\n")

    pass_items = {
        "no_tail_rows_above_threshold": tail_count < THRESHOLDS["tail_row_count_stop_min"],
        "tail_not_seed_concentrated": top_seed_share <= THRESHOLDS["tail_seed_concentration_share_max"],
        "tail_not_mode_concentrated": top_mode_share <= THRESHOLDS["tail_mode_concentration_share_max"],
        "tail_not_split_concentrated": top_split_share <= THRESHOLDS["tail_split_concentration_share_max"],
    }
    failure_labels = {
        "no_tail_rows_above_threshold": "null_tail_rows_above_threshold",
        "tail_not_seed_concentrated": "null_tail_seed_concentration_detected",
        "tail_not_mode_concentrated": "null_tail_mode_concentration_detected",
        "tail_not_split_concentrated": "null_tail_split_concentration_detected",
    }
    stop_reasons = [failure_labels[name] for name, passed in pass_items.items() if not passed]
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = {
        "generated_by": "analysis/audit_v8a_paired_null_tail_anatomy.py",
        "generated_at_utc": generated_at,
        "protocol_name": "v8A_paired_clean_null_tail_anatomy",
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "runs_geant4": False,
        "claim_scope": CLAIM_SCOPE,
        "null_dir": args.null_dir,
        "feature_dir": args.feature_dir,
        "tail_threshold": tail_threshold,
        "tail_row_count": tail_count,
        "total_null_rows": int(len(rows)),
        "max_tail_hm_min_recall": max_tail,
        "top_tail_seed_share": top_seed_share,
        "top_tail_mode_share": top_mode_share,
        "top_tail_split_share": top_split_share,
        "tail_seed_counts": seed_counts[:20],
        "tail_mode_counts": mode_counts,
        "tail_split_counts": split_counts,
        "tail_model_counts": model_counts,
        "tail_threshold_policy_counts": policy_counts,
        "null_gate_decision": null_gate.get("decision"),
        "null_gate_passed": bool(null_gate.get("gate_passed", False)),
        "source_null_summary": json_clean(summary.to_dict(orient="records")),
        "thresholds": THRESHOLDS,
        "pass_items": pass_items,
        "stop_reasons": stop_reasons,
        "gate_passed": not stop_reasons,
        "decision": "null_tail_clean" if not stop_reasons else "null_tail_anatomy_blocks_training",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(output_dir / "v8a_paired_null_tail_anatomy_gate.json", json_clean(gate))
    write_report(output_dir, gate, top_rows)
    print(
        "decision={decision} tail_rows={tail_rows} max_tail={max_tail:.4f} top_mode_share={mode_share:.4f}".format(
            decision=gate["decision"],
            tail_rows=tail_count,
            max_tail=max_tail,
            mode_share=top_mode_share,
        )
    )


if __name__ == "__main__":
    main()
