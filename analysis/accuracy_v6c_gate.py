from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def read_single_row(path: Path) -> dict:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty metrics file: {path}")
    return frame.iloc[0].to_dict()


def min_support(path: Path) -> int:
    frame = pd.read_csv(path)
    if frame.empty or "support" not in frame:
        return 0
    return int(frame["support"].min())


def status_counts(path: Path) -> dict:
    if not path.exists():
        return {"status_file_exists": False}
    frame = pd.read_csv(path)
    completed = int((frame["returncode"].astype(str) == "0").sum())
    failed = int((~frame["returncode"].astype(str).isin(["", "0"])).sum())
    return {
        "status_file_exists": True,
        "rows_in_status_file": int(len(frame)),
        "completed": completed,
        "failed": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v6c H/M source-design development gate.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--status-profile", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    audit_dir = (project_root / args.audit_dir).resolve()
    aware = read_single_row(audit_dir / "development_validation_summary.csv")
    blind = read_single_row(audit_dir / "development_validation_summary_thickness_blind.csv")
    pairwise = pd.read_csv(audit_dir / "hm_pairwise_audit.csv")
    observed_pairwise = float(pairwise["hm_min_recall"].min()) if not pairwise.empty else float("nan")
    observed_support = min_support(audit_dir / "per_class_recall_validation.csv")
    thresholds = {
        "thickness_aware_hm_min_recall": 0.80,
        "thickness_blind_hm_min_recall": 0.75,
        "pairwise_hm_min_recall": 0.75,
        "min_class_support": 120,
    }
    checks = {
        "thickness_aware_hm_min_recall": float(aware.get("hm_min_recall", float("nan"))) >= thresholds["thickness_aware_hm_min_recall"],
        "thickness_blind_hm_min_recall": float(blind.get("hm_min_recall", float("nan"))) >= thresholds["thickness_blind_hm_min_recall"],
        "pairwise_hm_min_recall": observed_pairwise >= thresholds["pairwise_hm_min_recall"],
        "min_class_support": observed_support >= thresholds["min_class_support"],
    }
    runner_status = {}
    if args.status_profile.strip():
        runner_status = status_counts(project_root / "results" / "material_sorting" / f"run_status_{args.status_profile.strip()}.csv")
        if runner_status.get("status_file_exists"):
            checks["runner_failed_zero"] = runner_status.get("failed", 1) == 0
    report = {
        "generated_by": "analysis/accuracy_v6c_gate.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit_dir": audit_dir.relative_to(project_root).as_posix() if audit_dir.is_relative_to(project_root) else audit_dir.as_posix(),
        "thresholds": thresholds,
        "observed": {
            "method": aware.get("method", ""),
            "thickness_aware_hm_min_recall": float(aware.get("hm_min_recall", float("nan"))),
            "thickness_blind_hm_min_recall": float(blind.get("hm_min_recall", float("nan"))),
            "pairwise_hm_min_recall": observed_pairwise,
            "min_class_support": observed_support,
        },
        "runner_status": runner_status,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output_json:
        output = (project_root / args.output_json).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((text + "\n").encode("utf-8"))
    print(text)
    if not report["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
