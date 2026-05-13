from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROFILE = "v8a_multiclass_context_v1_cif_literature"
PEAK_MANIFEST = "source_models/config/diffraction_peak_tables/ten_material_powder_peaks_cif_or_literature_v8a_manifest.json"
EVENT_TO_FEATURE_DIR = "results/accuracy_v3/v8a_multiclass_context_v1_event_to_feature"
SHORTCUT_DIR = "results/accuracy_v3/v8a_multiclass_context_v1_feature_shortcut_structure"
FINAL_AUDIT_DIR = "results/accuracy_v3/v8a_multiclass_context_v1_training_data_final_audit"
MODEL_DIR = "results/accuracy_v3/v8a_multiclass_context_v1_development_model"
EXPECTED_ROWS = 5040


STATUS_RE = re.compile(r"selected_rows=(?P<selected>\d+)\s+completed=(?P<completed>\d+)\s+failed=(?P<failed>\d+)\s+pending=(?P<pending>\d+)")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(cmd: list[str], project_root: Path, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"[{timestamp()}] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )


def status(project_root: Path, python_exe: str) -> dict[str, int]:
    proc = run(
        [
            python_exe,
            "analysis/run_material_sorting_matrix.py",
            "--profile",
            PROFILE,
            "--status-only",
        ],
        project_root,
        capture=True,
    )
    output = proc.stdout or ""
    print(output.strip(), flush=True)
    match = STATUS_RE.search(output)
    if match is None:
        raise RuntimeError(f"Could not parse run status output: {output}")
    return {key: int(value) for key, value in match.groupdict().items()}


def runner_active() -> bool:
    proc = subprocess.run(
        ["pgrep", "-f", f"run_material_sorting_matrix.py --profile {PROFILE}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return bool(proc.stdout.strip())


def wait_for_geant4(project_root: Path, python_exe: str, poll_seconds: int) -> None:
    while True:
        current = status(project_root, python_exe)
        if current["failed"] > 0:
            raise RuntimeError(f"Geant4 status reports failed rows: {current}")
        if current["completed"] >= EXPECTED_ROWS and current["pending"] == 0:
            print(f"[{timestamp()}] Geant4 complete: {current}", flush=True)
            return
        if not runner_active():
            print(f"[{timestamp()}] Geant4 runner is not active; resuming profile {PROFILE}.", flush=True)
            resume = subprocess.Popen(
                [python_exe, "analysis/run_material_sorting_matrix.py", "--profile", PROFILE],
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            print(f"[{timestamp()}] resumed_pid={resume.pid}", flush=True)
        print(f"[{timestamp()}] waiting {poll_seconds}s before next status check", flush=True)
        time.sleep(poll_seconds)


def postprocess(project_root: Path, python_exe: str, overwrite: bool) -> None:
    overwrite_flag = ["--overwrite"] if overwrite else []
    commands = [
        [
            python_exe,
            "analysis/v8a_event_to_feature_pipeline.py",
            "--profile",
            PROFILE,
            "--peak-manifest",
            PEAK_MANIFEST,
            "--output-dir",
            EVENT_TO_FEATURE_DIR,
            *overwrite_flag,
        ],
        [
            python_exe,
            "analysis/audit_v8a_feature_shortcut_structure.py",
            "--input-dir",
            EVENT_TO_FEATURE_DIR,
            "--output-dir",
            SHORTCUT_DIR,
            *overwrite_flag,
        ],
        [
            python_exe,
            "analysis/audit_v8a_multiclass_context_training_data_final.py",
            "--input-dir",
            EVENT_TO_FEATURE_DIR,
            "--output-dir",
            FINAL_AUDIT_DIR,
            *overwrite_flag,
        ],
        [
            python_exe,
            "analysis/train_v8a_multiclass_context_model.py",
            "--input-dir",
            EVENT_TO_FEATURE_DIR,
            "--final-audit-gate",
            f"{FINAL_AUDIT_DIR}/v8a_multiclass_context_training_data_final_gate.json",
            "--output-dir",
            MODEL_DIR,
            *overwrite_flag,
        ],
    ]
    for cmd in commands:
        proc = run(cmd, project_root)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with returncode={proc.returncode}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for v8A multiclass context Geant4, then run the downstream development-only pipeline.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--overwrite-stage-outputs", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if args.log_file:
        log_path = project_root / args.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log_handle
        sys.stderr = log_handle
    python_exe = args.python_exe
    wait_for_geant4(project_root, python_exe, max(int(args.poll_seconds), 30))
    postprocess(project_root, python_exe, args.overwrite_stage_outputs)
    print(f"[{timestamp()}] v8A multiclass context pipeline complete", flush=True)


if __name__ == "__main__":
    main()
