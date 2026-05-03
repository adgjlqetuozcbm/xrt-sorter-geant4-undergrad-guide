from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HM_PAIR = ("Hematite", "Magnetite")
EPS = 1e-6
ANGLE_BASELINE_VARIANT = "normal_narrow"


def parse_source_id(source_id: str) -> dict:
    raw = source_id.removeprefix("mono_")
    energy_text, _, variant = raw.partition("kev")
    variant = variant.removeprefix("_") or "normal_narrow"
    try:
        energy = float(energy_text.replace("p", "."))
    except ValueError:
        energy = math.nan
    angle = 0.0
    if variant.startswith("oblique_") and variant.endswith("deg"):
        try:
            angle = float(variant.removeprefix("oblique_").removesuffix("deg"))
        except ValueError:
            angle = math.nan
    return {"source_id": source_id, "energy_keV": energy, "source_variant": variant, "incidence_angle_deg": angle}


def load_cube(cube_dir: Path) -> tuple[np.ndarray, pd.DataFrame, list[str], list[str], list[str], dict]:
    data = np.load(cube_dir / "measurement_cube.npz", allow_pickle=True)
    cube = data["X"].astype(np.float32)
    source_ids = [str(item) for item in data["source_ids"].tolist()]
    detector_ids = [str(item) for item in data["detector_ids"].tolist()]
    channels = [str(item) for item in data["channels"].tolist()]
    metadata = pd.read_csv(cube_dir / "sample_metadata.csv")
    manifest_path = cube_dir / "measurement_cube_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return cube, metadata, source_ids, detector_ids, channels, manifest


def channel_index(channels: list[str], name: str) -> int:
    if name not in channels:
        raise ValueError(f"Missing required cube channel: {name}")
    return channels.index(name)


def summarize_view(cube: np.ndarray, source_ids: list[str], detector_ids: list[str], channels: list[str], metadata: pd.DataFrame) -> pd.DataFrame:
    attenuation_i = channel_index(channels, "attenuation")
    hit_rate_i = channel_index(channels, "hit_rate")
    ratio_i = channel_index(channels, "calibrated_hit_ratio")
    energy_i = channel_index(channels, "energy_mean_keV")
    direct_i = channel_index(channels, "direct_primary_rate")
    scatter_i = channel_index(channels, "scattered_primary_rate")
    tail_i = channel_index(channels, "tail120_rate")
    rows = []
    source_meta = {item["source_id"]: item for item in (parse_source_id(source_id) for source_id in source_ids)}
    for sample_index in range(cube.shape[0]):
        sample = metadata.iloc[sample_index]
        thickness = max(float(sample["thickness_mm"]), EPS)
        for source_index, source_id in enumerate(source_ids):
            source = source_meta[source_id]
            angle_rad = math.radians(float(source["incidence_angle_deg"]) if not math.isnan(source["incidence_angle_deg"]) else 0.0)
            path_length = thickness / max(math.cos(angle_rad), 0.15)
            for detector_index, detector_id in enumerate(detector_ids):
                view = cube[sample_index, source_index, detector_index, :, :, :]
                hit_rate = float(view[:, :, hit_rate_i].sum())
                attenuation = float(view[:, :, attenuation_i].mean())
                ratio = float(view[:, :, ratio_i].mean())
                direct_rate = float(view[:, :, direct_i].sum())
                scatter_rate = float(view[:, :, scatter_i].sum())
                rows.append(
                    {
                        "sample_index": int(sample["sample_index"]),
                        "material": str(sample["material"]),
                        "split": str(sample["split"]),
                        "random_seed": int(sample["random_seed"]),
                        "sample_id": int(sample["sample_id"]),
                        "thickness_mm": thickness,
                        "source_id": source_id,
                        "energy_keV": float(source["energy_keV"]),
                        "source_variant": str(source["source_variant"]),
                        "incidence_angle_deg": float(source["incidence_angle_deg"]),
                        "detector_id": detector_id,
                        "path_length_mm": path_length,
                        "hit_rate_sum": hit_rate,
                        "calibrated_hit_ratio_mean": ratio,
                        "attenuation_mean": attenuation,
                        "path_norm_attenuation": attenuation / path_length,
                        "energy_mean_keV": float(view[:, :, energy_i].mean()),
                        "tail120_rate_sum": float(view[:, :, tail_i].sum()),
                        "direct_primary_rate_sum": direct_rate,
                        "scattered_primary_rate_sum": scatter_rate,
                        "scatter_to_direct": scatter_rate / max(direct_rate, EPS),
                        "detector_reliable": int(hit_rate > 0.0),
                    }
                )
    return pd.DataFrame(rows)


def add_response_features(view_summary: pd.DataFrame) -> pd.DataFrame:
    frame = view_summary.copy()
    frame = frame.sort_values(["sample_index", "detector_id", "source_variant", "energy_keV"])
    grouped = frame.groupby(["sample_index", "detector_id", "source_variant"], sort=False)
    frame["energy_slope_path_norm_attenuation"] = grouped["path_norm_attenuation"].transform(lambda s: s.diff().fillna(0.0))
    frame["thickness_response_path_norm_attenuation"] = frame.groupby(
        ["material", "split", "detector_id", "source_id"], sort=False
    )["path_norm_attenuation"].transform(lambda s: s.diff().fillna(0.0))

    baseline = frame[frame["source_variant"].eq(ANGLE_BASELINE_VARIANT)][
        ["sample_index", "detector_id", "energy_keV", "path_norm_attenuation"]
    ].rename(columns={"path_norm_attenuation": "baseline_path_norm_attenuation"})
    frame = frame.merge(baseline, on=["sample_index", "detector_id", "energy_keV"], how="left")
    frame["angle_sensitivity_path_norm_attenuation"] = (
        frame["path_norm_attenuation"] - frame["baseline_path_norm_attenuation"].fillna(frame["path_norm_attenuation"])
    )
    return frame


def hm_contrast_table(frame: pd.DataFrame) -> pd.DataFrame:
    hm = frame[frame["material"].isin(HM_PAIR)].copy()
    metrics = [
        "hit_rate_sum",
        "calibrated_hit_ratio_mean",
        "attenuation_mean",
        "path_norm_attenuation",
        "energy_slope_path_norm_attenuation",
        "angle_sensitivity_path_norm_attenuation",
        "scatter_to_direct",
        "tail120_rate_sum",
    ]
    rows = []
    keys = ["split", "source_id", "energy_keV", "source_variant", "incidence_angle_deg", "detector_id", "thickness_mm"]
    for key, part in hm.groupby(keys, dropna=False):
        values = dict(zip(keys, key))
        supports = part.groupby("material").size().to_dict()
        if not all(material in supports for material in HM_PAIR):
            continue
        row = {**values, "hematite_support": int(supports.get("Hematite", 0)), "magnetite_support": int(supports.get("Magnetite", 0))}
        for metric in metrics:
            means = part.groupby("material")[metric].mean().to_dict()
            stds = part.groupby("material")[metric].std(ddof=0).to_dict()
            h = float(means.get("Hematite", 0.0))
            m = float(means.get("Magnetite", 0.0))
            pooled = math.sqrt(max(float(stds.get("Hematite", 0.0)) ** 2 + float(stds.get("Magnetite", 0.0)) ** 2, EPS))
            row[f"{metric}_hematite_mean"] = h
            row[f"{metric}_magnetite_mean"] = m
            row[f"{metric}_delta_h_minus_m"] = h - m
            row[f"{metric}_effect_size"] = abs(h - m) / pooled
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v7B2 H/M physical-response feature tables from a measurement cube.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cube-dir", default="results/accuracy_v3/v7b2_hm_physics_dev")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    # Preserve extended UNC prefixes for Windows-side scientific Python.
    project_root = Path(args.project_root)
    cube_dir = project_root / args.cube_dir
    output_dir = project_root / (args.output_dir.strip() or args.cube_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cube, metadata, source_ids, detector_ids, channels, manifest = load_cube(cube_dir)
    if bool(manifest.get("shadow_or_final_used", False)):
        raise RuntimeError("Refusing to export v7B2 development features from a cube that used shadow/final data.")
    if set(metadata["material"].astype(str)) - set(HM_PAIR):
        raise RuntimeError("v7B2 Pilot feature export must be H/M-only.")

    view_summary = add_response_features(summarize_view(cube, source_ids, detector_ids, channels, metadata))
    contrast = hm_contrast_table(view_summary)
    view_summary.to_csv(output_dir / "v7b2_physical_view_features.csv", index=False, lineterminator="\n")
    contrast.to_csv(output_dir / "v7b2_hm_physical_contrast_by_view.csv", index=False, lineterminator="\n")
    feature_manifest = {
        "generated_by": "analysis/export_physics_features_v7b2.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_name": "v7b2_hm_physics_dev",
        "cube_dir": args.cube_dir,
        "output_dir": args.output_dir.strip() or args.cube_dir,
        "shadow_or_final_used": bool(manifest.get("shadow_or_final_used", False)),
        "samples": int(len(metadata)),
        "view_rows": int(len(view_summary)),
        "contrast_rows": int(len(contrast)),
        "materials": sorted(set(metadata["material"].astype(str))),
        "source_count": len(source_ids),
        "detectors": detector_ids,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    (output_dir / "v7b2_physical_feature_manifest.json").write_bytes(
        (json.dumps(feature_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    print(f"Wrote v7B2 physical features to {output_dir}")
    print(f"view_rows={len(view_summary)} contrast_rows={len(contrast)}")


if __name__ == "__main__":
    main()
