from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


GATES = {
    "hm_development": {
        "summary_file": "development_validation_summary.csv",
        "summary_split": "development",
        "top1_accuracy": 0.88,
        "macro_f1": 0.84,
        "min_class_recall": 0.75,
        "hm_min_recall": 0.80,
        "pairwise_hm_min_recall": 0.75,
        "min_class_support": 40,
    },
    "hm_shadow": {
        "summary_file": "development_validation_summary.csv",
        "summary_split": "development",
        "top1_accuracy": 0.88,
        "macro_f1": 0.84,
        "min_class_recall": 0.75,
        "hm_min_recall": 0.80,
        "pairwise_hm_min_recall": 0.75,
        "min_class_support": 40,
    },
    "full_validation": {
        "summary_file": "development_validation_summary.csv",
        "summary_split": "development",
        "top1_accuracy": 0.88,
        "macro_f1": 0.84,
        "min_class_recall": 0.75,
        "hm_min_recall": 0.75,
        "pairwise_hm_min_recall": 0.75,
        "min_class_support": 40,
    },
    "final_locked": {
        "summary_file": "final_test_summary.csv",
        "summary_split": "final",
        "top1_accuracy": 0.88,
        "macro_f1": 0.84,
        "min_class_recall": 0.75,
        "hm_min_recall": 0.75,
        "pairwise_hm_min_recall": 0.75,
        "min_class_support": 40,
    },
}


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


def pairwise_min(path: Path, split: str | None = None) -> float:
    frame = pd.read_csv(path)
    if split and "split" in frame:
        frame = frame[frame["split"].astype(str).eq(split)]
    if frame.empty or "hm_min_recall" not in frame:
        return float("nan")
    return float(frame["hm_min_recall"].min())


def status_counts(status_path: Path) -> dict:
    if not status_path.exists():
        return {"status_file_exists": False}
    frame = pd.read_csv(status_path)
    completed = int((frame["returncode"].astype(str) == "0").sum())
    failed = int((~frame["returncode"].astype(str).isin(["", "0"])).sum())
    return {
        "status_file_exists": True,
        "rows_in_status_file": int(len(frame)),
        "completed": completed,
        "failed": failed,
    }


def gate_report(project_root: Path, audit_dir: Path, stage: str, status_profile: str | None) -> dict:
    gate = GATES[stage]
    summary = read_single_row(audit_dir / gate["summary_file"])
    per_class_file = "per_class_recall_final_test.csv" if stage == "final_locked" else "per_class_recall_validation.csv"
    observed_support = min_support(audit_dir / per_class_file)
    observed_pairwise = pairwise_min(audit_dir / "hm_pairwise_audit.csv", "test" if stage == "final_locked" else "validation")
    checks = {
        "top1_accuracy": float(summary.get("top1_accuracy", float("nan"))) >= gate["top1_accuracy"],
        "macro_f1": float(summary.get("macro_f1", float("nan"))) >= gate["macro_f1"],
        "min_class_recall": float(summary.get("min_class_recall", float("nan"))) >= gate["min_class_recall"],
        "hm_min_recall": float(summary.get("hm_min_recall", float("nan"))) >= gate["hm_min_recall"],
        "pairwise_hm_min_recall": observed_pairwise >= gate["pairwise_hm_min_recall"],
        "min_class_support": observed_support >= gate["min_class_support"],
    }
    status = {}
    if status_profile:
        status = status_counts(project_root / "results" / "material_sorting" / f"run_status_{status_profile}.csv")
        if status.get("status_file_exists"):
            checks["runner_failed_zero"] = status.get("failed", 1) == 0
    report = {
        "generated_by": "analysis/accuracy_v6_gate.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "audit_dir": audit_dir.relative_to(project_root).as_posix() if audit_dir.is_relative_to(project_root) else audit_dir.as_posix(),
        "thresholds": gate,
        "observed": {
            "method": summary.get("method", ""),
            "top1_accuracy": float(summary.get("top1_accuracy", float("nan"))),
            "macro_f1": float(summary.get("macro_f1", float("nan"))),
            "min_class_recall": float(summary.get("min_class_recall", float("nan"))),
            "hm_min_recall": float(summary.get("hm_min_recall", float("nan"))),
            "pairwise_hm_min_recall": observed_pairwise,
            "min_class_support": observed_support,
        },
        "runner_status": status,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate G4 Accuracy Sprint v6 gate outputs.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--stage", choices=sorted(GATES), required=True)
    parser.add_argument("--status-profile", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    audit_dir = (project_root / args.audit_dir).resolve()
    report = gate_report(project_root, audit_dir, args.stage, args.status_profile.strip() or None)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        output = (project_root / args.output_json).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((text + "\n").encode("utf-8"))
    print(text)
    if not report["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
