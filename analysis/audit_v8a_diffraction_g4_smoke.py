from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROFILE = "v8a_custom_diffraction_g4_smoke"
HIGH_ANGLE_DEG = 2.0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_key_value(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip().lower()] = value.strip()
    return values


def safe_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def safe_int(value: str, fallback: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def summarize_hits(hits_path: Path, metadata_path: Path) -> dict[str, float | int | str]:
    hits = read_csv(hits_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_events = int(metadata.get("n_events", 0))
    high_angle_primary = 0
    side_scatter_hits = 0
    transmission_hits = 0
    direct_primary = 0
    scattered_primary = 0
    theta_values: list[float] = []

    for hit in hits:
        detector_id = hit.get("detector_id", "")
        is_primary = safe_int(hit.get("is_primary", "0")) == 1
        theta = safe_float(hit.get("theta_deg", "-1"), -1.0)
        if detector_id == "side_scatter":
            side_scatter_hits += 1
        if detector_id == "transmission":
            transmission_hits += 1
        if safe_int(hit.get("is_direct_primary", "0")) == 1:
            direct_primary += 1
        if safe_int(hit.get("is_scattered_primary", "0")) == 1:
            scattered_primary += 1
        if is_primary and theta >= 0.0:
            theta_values.append(theta)
            if theta >= HIGH_ANGLE_DEG:
                high_angle_primary += 1

    denominator = max(n_events, 1)
    return {
        "n_events": n_events,
        "hit_rows": len(hits),
        "transmission_hits": transmission_hits,
        "side_scatter_hits": side_scatter_hits,
        "direct_primary_hits": direct_primary,
        "scattered_primary_hits": scattered_primary,
        "high_angle_primary_hits": high_angle_primary,
        "side_scatter_rate": side_scatter_hits / denominator,
        "high_angle_primary_rate": high_angle_primary / denominator,
        "direct_primary_rate": direct_primary / denominator,
        "median_primary_theta_deg": median(theta_values),
        "max_primary_theta_deg": max(theta_values) if theta_values else -1.0,
    }


def output_paths(project_root: Path, config_path: Path) -> tuple[Path, Path]:
    config = read_key_value(config_path)
    output_dir = Path(config.get("output_dir", "."))
    if not output_dir.is_absolute():
        output_dir = project_root / "build" / output_dir
    output_prefix = config["output_prefix"]
    return output_dir / f"{output_prefix}_hits.csv", output_dir / f"{output_prefix}_metadata.json"


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the minimal v8A Geant4 custom-diffraction smoke boundary.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--output-dir", default="results/accuracy_v3/v8a_diffraction_g4_smoke")
    parser.add_argument("--min-completed-rows", type=int, default=6)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    matrix_path = project_root / "source_models" / "config" / "material_sorting_matrix" / args.profile / "material_sorting_matrix.csv"
    status_path = project_root / "results" / "material_sorting" / f"run_status_{args.profile}.csv"
    output_dir = project_root / args.output_dir
    matrix_rows = read_csv(matrix_path)
    status_rows = read_csv(status_path)
    completed_keys = {
        (
            row.get("run_role", "material"),
            row.get("material", ""),
            row.get("source_id", ""),
            row.get("thickness_mm", ""),
            row.get("random_seed", ""),
        )
        for row in status_rows
        if row.get("returncode", "") == "0"
    }

    summaries: list[dict[str, object]] = []
    for row in matrix_rows:
        key = (
            row.get("run_role", "material"),
            row.get("material", ""),
            row.get("source_id", ""),
            row.get("thickness_mm", ""),
            row.get("random_seed", ""),
        )
        if key not in completed_keys:
            continue
        config_path = project_root / row["config_path"]
        hits_path, metadata_path = output_paths(project_root, config_path)
        if not hits_path.exists() or not metadata_path.exists():
            continue
        summary = {
            "profile": args.profile,
            "split": row.get("split", ""),
            "material": row.get("material", ""),
            "source_id": row.get("source_id", ""),
            "source_mode": row.get("source_mode", ""),
            "stress_label": row.get("stress_label", ""),
            "source_energy_kev": row.get("source_energy_kev", ""),
            "thickness_mm": row.get("thickness_mm", ""),
            "random_seed": row.get("random_seed", ""),
            "hits_path": str(hits_path.relative_to(project_root)),
            "metadata_path": str(metadata_path.relative_to(project_root)),
        }
        summary.update(summarize_hits(hits_path, metadata_path))
        summaries.append(summary)

    on_rates = [
        float(row["high_angle_primary_rate"])
        for row in summaries
        if row.get("source_mode") == "on"
    ]
    off_rates = [
        float(row["high_angle_primary_rate"])
        for row in summaries
        if row.get("source_mode") == "off"
    ]
    source_on_median = median(on_rates)
    leakage_off_median = median(off_rates)
    source_on_min = min(on_rates) if on_rates else 0.0
    leakage_off_max = max(off_rates) if off_rates else 0.0
    gate_passed = (
        len(summaries) >= args.min_completed_rows
        and bool(on_rates)
        and bool(off_rates)
        and source_on_median >= 0.05
        and source_on_min >= 0.02
        and leakage_off_max < 0.02
        and source_on_median - leakage_off_median >= 0.05
    )
    decision = (
        "proceed_to_geant4_diffraction_output_schema_review"
        if gate_passed
        else "stop_or_repeat_minimal_g4_boundary_smoke"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "v8a_g4_smoke_run_summary.csv", summaries)
    manifest = {
        "generated_by": "analysis/audit_v8a_diffraction_g4_smoke.py",
        "profile": args.profile,
        "development_only": True,
        "shadow_or_final_used": False,
        "reads_existing_xrt_cubes": False,
        "bin_axis": "q_a_inv",
        "completed_rows_audited": len(summaries),
        "source_on_rows": len(on_rates),
        "source_off_rows": len(off_rates),
        "high_angle_deg": HIGH_ANGLE_DEG,
    }
    gate = {
        **manifest,
        "gate_passed": gate_passed,
        "decision": decision,
        "source_on_high_angle_primary_rate_median": source_on_median,
        "source_on_high_angle_primary_rate_min": source_on_min,
        "leakage_off_high_angle_primary_rate_median": leakage_off_median,
        "leakage_off_high_angle_primary_rate_max": leakage_off_max,
        "source_on_minus_leakage_off_median_margin": source_on_median - leakage_off_median,
        "claim_scope": "minimal Geant4 source/geometry/detector boundary smoke only; not H/M accuracy evidence",
    }
    (output_dir / "v8a_g4_smoke_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "v8a_g4_smoke_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report = [
        "# v8A Geant4 custom diffraction boundary smoke",
        "",
        f"- decision: `{decision}`",
        f"- gate_passed: `{str(gate_passed).lower()}`",
        f"- completed rows audited: `{len(summaries)}`",
        f"- source-on high-angle primary median: `{source_on_median:.6f}`",
        f"- leakage-off high-angle primary max: `{leakage_off_max:.6f}`",
        "",
        "This is a minimal source/geometry/detector boundary check. It does not use shadow/final data, does not read existing XRT cubes, and does not support a final H/M accuracy claim.",
    ]
    (output_dir / "v8a_g4_smoke_gate_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
