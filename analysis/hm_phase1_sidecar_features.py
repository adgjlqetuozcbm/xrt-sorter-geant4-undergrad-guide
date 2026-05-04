from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import train_v7b as v7b


HM_PAIR = ("Hematite", "Magnetite")
BASELINE_HM_MIN_RECALL = 0.6083333333333333
PASS_HM_MIN_RECALL = 0.68
PASS_IMPROVEMENT = 0.05
GRAY_HM_MIN_RECALL = 0.62
EPS = 1e-9
COMPACT_CHANNELS = [
    "hit_rate",
    "calibrated_hit_ratio",
    "attenuation",
    "energy_mean_keV",
    "tail120_rate",
    "tail120_fraction",
    "primary_rate",
    "direct_primary_rate",
    "scattered_primary_rate",
    "theta_mean_deg",
    "radius_mean_mm",
    "detector_total_rate",
]
DIAGNOSTIC_METRICS = [
    "attenuation_mean",
    "path_norm_attenuation",
    "log_transmission_per_path",
    "hit_rate_sum",
    "calibrated_hit_ratio_mean",
    "energy_mean_keV_mean",
    "tail120_rate_sum",
    "tail120_fraction_mean",
    "scatter_to_direct",
    "tail_to_direct",
]
METADATA_FEATURE_COLS = [
    "dataset",
    "sample_index",
    "material",
    "split",
    "random_seed",
    "sample_id",
    "thickness_mm",
]


def log_progress(message: str, *, start_time: float | None = None) -> None:
    elapsed = ""
    if start_time is not None:
        elapsed = f" elapsed={perf_counter() - start_time:.1f}s"
    print(f"[hm-phase1] {datetime.now().isoformat(timespec='seconds')} {message}{elapsed}", flush=True)


def require_sklearn() -> dict:
    sk = v7b.require_sklearn()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    sk.update(
        {
            "LinearSVC": LinearSVC,
            "LogisticRegression": LogisticRegression,
            "RandomForestClassifier": RandomForestClassifier,
            "StandardScaler": StandardScaler,
            "make_pipeline": make_pipeline,
            "precision_recall_fscore_support": precision_recall_fscore_support,
        }
    )
    return sk


def parse_source_id(source_id: str) -> dict:
    raw = source_id.removeprefix("mono_")
    energy_text, _, variant = raw.partition("kev")
    variant = variant.removeprefix("_") or "normal_narrow"
    try:
        energy = float(energy_text.replace("p", "."))
    except ValueError:
        energy = math.nan
    angle = 0.0
    angle_match = re.search(r"oblique_([0-9.]+)deg", variant)
    if angle_match:
        try:
            angle = float(angle_match.group(1))
        except ValueError:
            angle = math.nan
    return {
        "source_id": source_id,
        "energy_keV": energy,
        "source_variant": variant,
        "incidence_angle_deg": angle,
    }


def source_sort_key(source_id: str) -> tuple[float, int, str]:
    parsed = parse_source_id(source_id)
    rank = {
        "normal_narrow": 0,
        "normal_wide": 1,
        "oblique_10deg": 2,
        "oblique_20deg": 3,
        "oblique_30deg": 4,
        "oblique_40deg": 5,
    }.get(str(parsed["source_variant"]), 99)
    energy = float(parsed["energy_keV"]) if not math.isnan(float(parsed["energy_keV"])) else math.inf
    return energy, rank, source_id


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def finite_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def load_cube_bundle(cube_dir: Path, dataset: str) -> dict:
    data = np.load(cube_dir / "measurement_cube.npz", allow_pickle=True)
    cube = data["X"].astype(np.float32)
    metadata = pd.read_csv(cube_dir / "sample_metadata.csv")
    source_ids = [str(item) for item in data["source_ids"].tolist()]
    detector_ids = [str(item) for item in data["detector_ids"].tolist()]
    channels = [str(item) for item in data["channels"].tolist()]
    manifest_path = cube_dir / "measurement_cube_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if bool(manifest.get("shadow_or_final_used", False)):
        raise RuntimeError(f"Refusing {dataset}: manifest reports shadow/final use.")
    if set(metadata["split"].astype(str)) - {"train", "validation"}:
        raise RuntimeError(f"Refusing {dataset}: unsupported split values {sorted(set(metadata['split'].astype(str)))}")
    metadata = metadata.copy()
    metadata["dataset"] = dataset
    return {
        "dataset": dataset,
        "cube_dir": cube_dir,
        "cube": cube,
        "metadata": metadata,
        "source_ids": source_ids,
        "detector_ids": detector_ids,
        "channels": channels,
        "manifest": manifest,
    }


def selected_sources(cube_dir: Path, source_ids: list[str], top_n: int) -> list[str]:
    ranking_path = cube_dir / "v7b2_pilot_source_ranking.csv"
    if top_n <= 0 or not ranking_path.exists():
        return sorted(source_ids, key=source_sort_key)
    ranking = pd.read_csv(ranking_path)
    if "source_id" not in ranking.columns:
        return sorted(source_ids, key=source_sort_key)
    ordered = [str(item) for item in ranking["source_id"].tolist() if str(item) in set(source_ids)]
    if not ordered:
        return sorted(source_ids, key=source_sort_key)
    return ordered[: min(top_n, len(ordered))]


def channel_indices(channels: list[str]) -> dict[str, int]:
    return {channel: index for index, channel in enumerate(channels)}


def summarize_view(
    view: np.ndarray,
    channel_index: dict[str, int],
    path_length: float,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for channel in COMPACT_CHANNELS:
        if channel not in channel_index:
            continue
        grid = view[:, :, channel_index[channel]].astype(float)
        if channel.endswith("_rate") or channel in {"hit_rate", "primary_rate", "detector_total_rate"}:
            values[f"{channel}_sum"] = finite_float(float(np.sum(grid)))
        values[f"{channel}_mean"] = finite_float(float(np.mean(grid)))
        values[f"{channel}_std"] = finite_float(float(np.std(grid)))
    direct = values.get("direct_primary_rate_sum", 0.0)
    scatter = values.get("scattered_primary_rate_sum", 0.0)
    tail = values.get("tail120_rate_sum", 0.0)
    hit_ratio = max(values.get("calibrated_hit_ratio_mean", 0.0), EPS)
    attenuation = values.get("attenuation_mean", 0.0)
    values["path_norm_attenuation"] = finite_float(attenuation / max(path_length, EPS))
    values["log_transmission_per_path"] = finite_float(-math.log(hit_ratio) / max(path_length, EPS))
    values["scatter_to_direct"] = finite_float(scatter / max(direct, EPS))
    values["tail_to_direct"] = finite_float(tail / max(direct, EPS))
    return values


def add_feature(rows: list[dict], sample_index: int, feature: str, value: float) -> None:
    rows.append({"sample_index": int(sample_index), "feature": feature, "value": finite_float(float(value))})


def build_long_features(bundle: dict, source_top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cube = bundle["cube"]
    metadata = bundle["metadata"].reset_index(drop=True)
    source_ids = bundle["source_ids"]
    detector_ids = bundle["detector_ids"]
    channels = bundle["channels"]
    cidx = channel_indices(channels)
    chosen_sources = selected_sources(bundle["cube_dir"], source_ids, source_top_n)
    chosen_source_set = set(chosen_sources)
    source_lookup = {source_id: parse_source_id(source_id) for source_id in source_ids}
    source_index_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
    detector_index_lookup = {detector_id: index for index, detector_id in enumerate(detector_ids)}
    rows: list[dict] = []
    view_rows: list[dict] = []

    for sample_pos, sample in metadata.iterrows():
        sample_index = int(sample["sample_index"])
        thickness = max(float(sample["thickness_mm"]), EPS)
        add_feature(rows, sample_index, "metadata__thickness_mm", thickness)
        add_feature(rows, sample_index, "metadata__log_thickness_mm", math.log1p(thickness))
        add_feature(rows, sample_index, f"dataset__{safe_name(bundle['dataset'])}", 1.0)

        per_view: list[dict] = []
        for source_id in chosen_sources:
            source = source_lookup[source_id]
            source_index = source_index_lookup[source_id]
            angle = float(source["incidence_angle_deg"])
            angle_rad = math.radians(angle if not math.isnan(angle) else 0.0)
            path_length = thickness / max(math.cos(angle_rad), 0.15)
            for detector_id in detector_ids:
                detector_index = detector_index_lookup[detector_id]
                summary = summarize_view(cube[sample_pos, source_index, detector_index], cidx, path_length)
                base = f"src={source_id}|det={detector_id}"
                for metric, value in summary.items():
                    if metric in DIAGNOSTIC_METRICS:
                        add_feature(rows, sample_index, f"{base}|metric={metric}", value)
                    per_view.append(
                        {
                            "sample_index": sample_index,
                            "source_id": source_id,
                            "detector_id": detector_id,
                            "energy_keV": float(source["energy_keV"]),
                            "source_variant": str(source["source_variant"]),
                            "incidence_angle_deg": angle,
                            "path_length_mm": path_length,
                            **summary,
                        }
                    )
        view_frame = pd.DataFrame(per_view)
        view_rows.extend(per_view)
        if view_frame.empty:
            continue

        add_energy_features(rows, sample_index, view_frame)
        add_angle_features(rows, sample_index, view_frame)
        add_detector_features(rows, sample_index, view_frame)

    meta_cols = [col for col in METADATA_FEATURE_COLS if col in metadata.columns]
    meta = metadata[meta_cols].copy()
    long = pd.DataFrame(rows)
    audit = {
        "selected_source_ids": chosen_sources,
        "selected_source_count": len(chosen_sources),
        "available_source_count": len(source_ids),
        "detectors": detector_ids,
        "channels": channels,
        "view_rows": len(view_rows),
    }
    return meta, long, audit


def add_energy_features(rows: list[dict], sample_index: int, view_frame: pd.DataFrame) -> None:
    for detector_id in sorted(view_frame["detector_id"].astype(str).unique()):
        for variant in sorted(view_frame["source_variant"].astype(str).unique()):
            part = view_frame[
                view_frame["detector_id"].astype(str).eq(detector_id)
                & view_frame["source_variant"].astype(str).eq(variant)
            ].sort_values("energy_keV")
            if len(part) < 2:
                continue
            energies = part["energy_keV"].astype(float).to_numpy()
            for metric in DIAGNOSTIC_METRICS:
                if metric not in part.columns:
                    continue
                values = part[metric].astype(float).to_numpy()
                first_delta = values[-1] - values[0]
                denom = max(abs(values[0]), EPS)
                energy_span = max(float(energies[-1] - energies[0]), EPS)
                prefix = f"energy|det={detector_id}|variant={variant}|metric={metric}"
                add_feature(rows, sample_index, f"{prefix}|low_high_delta", first_delta)
                add_feature(rows, sample_index, f"{prefix}|low_high_ratio", values[-1] / denom)
                add_feature(rows, sample_index, f"{prefix}|slope", first_delta / energy_span)
                if len(values) >= 3:
                    second = float(np.mean(np.diff(values, n=2)))
                    add_feature(rows, sample_index, f"{prefix}|curvature_mean", second)


def add_angle_features(rows: list[dict], sample_index: int, view_frame: pd.DataFrame) -> None:
    for detector_id in sorted(view_frame["detector_id"].astype(str).unique()):
        for energy in sorted(view_frame["energy_keV"].astype(float).unique()):
            part = view_frame[
                view_frame["detector_id"].astype(str).eq(detector_id)
                & np.isclose(view_frame["energy_keV"].astype(float).to_numpy(), float(energy))
            ]
            if part.empty:
                continue
            baseline_rows = part[part["source_variant"].astype(str).eq("normal_narrow")]
            if baseline_rows.empty:
                baseline_rows = part[part["source_variant"].astype(str).eq("normal_wide")]
            if baseline_rows.empty:
                continue
            baseline = baseline_rows.iloc[0]
            for _, row in part.iterrows():
                variant = str(row["source_variant"])
                if variant in {"normal_narrow"}:
                    continue
                for metric in DIAGNOSTIC_METRICS:
                    if metric not in row or metric not in baseline:
                        continue
                    base_value = float(baseline[metric])
                    value = float(row[metric])
                    prefix = f"angle|det={detector_id}|energy={energy:g}|variant={variant}|metric={metric}"
                    add_feature(rows, sample_index, f"{prefix}|delta_vs_baseline", value - base_value)
                    add_feature(rows, sample_index, f"{prefix}|ratio_vs_baseline", value / max(abs(base_value), EPS))


def add_detector_features(rows: list[dict], sample_index: int, view_frame: pd.DataFrame) -> None:
    if not {"transmission", "side_scatter"}.issubset(set(view_frame["detector_id"].astype(str))):
        return
    for source_id in sorted(view_frame["source_id"].astype(str).unique(), key=source_sort_key):
        trans = view_frame[
            view_frame["source_id"].astype(str).eq(source_id)
            & view_frame["detector_id"].astype(str).eq("transmission")
        ]
        side = view_frame[
            view_frame["source_id"].astype(str).eq(source_id)
            & view_frame["detector_id"].astype(str).eq("side_scatter")
        ]
        if trans.empty or side.empty:
            continue
        t = trans.iloc[0]
        s = side.iloc[0]
        for metric in DIAGNOSTIC_METRICS:
            if metric not in t or metric not in s:
                continue
            t_value = float(t[metric])
            s_value = float(s[metric])
            prefix = f"detector_pair|src={source_id}|metric={metric}"
            add_feature(rows, sample_index, f"{prefix}|side_minus_trans", s_value - t_value)
            add_feature(rows, sample_index, f"{prefix}|side_over_trans", s_value / max(abs(t_value), EPS))


def pivot_features(meta: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    table = (
        long.pivot_table(index="sample_index", columns="feature", values="value", aggfunc="mean", fill_value=0.0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    merged = meta.merge(table, on="sample_index", how="left")
    feature_cols = [col for col in merged.columns if col not in METADATA_FEATURE_COLS]
    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged



def stack_metric_arrays(bundle: dict, chosen_sources: list[str]) -> tuple[dict[str, np.ndarray], list[dict]]:
    cube = bundle["cube"]
    metadata = bundle["metadata"].reset_index(drop=True)
    source_ids = bundle["source_ids"]
    channels = bundle["channels"]
    cidx = channel_indices(channels)
    source_index = {source_id: index for index, source_id in enumerate(source_ids)}
    selected_indices = [source_index[source_id] for source_id in chosen_sources]
    selected_cube = cube[:, selected_indices, :, :, :, :].astype(np.float32)
    source_meta = [parse_source_id(source_id) for source_id in chosen_sources]
    thickness = np.maximum(metadata["thickness_mm"].astype(float).to_numpy(dtype=np.float32), EPS)
    cosines = np.array(
        [max(math.cos(math.radians(float(item["incidence_angle_deg"]))), 0.15) for item in source_meta],
        dtype=np.float32,
    )
    path_lengths = thickness[:, None] / cosines[None, :]

    def channel_mean(name: str) -> np.ndarray:
        return selected_cube[:, :, :, :, :, cidx[name]].mean(axis=(3, 4), dtype=np.float64).astype(np.float32)

    def channel_sum(name: str) -> np.ndarray:
        return selected_cube[:, :, :, :, :, cidx[name]].sum(axis=(3, 4), dtype=np.float64).astype(np.float32)

    arrays: dict[str, np.ndarray] = {
        "hit_rate_sum": channel_sum("hit_rate"),
        "calibrated_hit_ratio_mean": channel_mean("calibrated_hit_ratio"),
        "attenuation_mean": channel_mean("attenuation"),
        "energy_mean_keV_mean": channel_mean("energy_mean_keV"),
        "tail120_rate_sum": channel_sum("tail120_rate"),
        "tail120_fraction_mean": channel_mean("tail120_fraction"),
        "direct_primary_rate_sum": channel_sum("direct_primary_rate"),
        "scattered_primary_rate_sum": channel_sum("scattered_primary_rate"),
    }
    arrays["path_norm_attenuation"] = arrays["attenuation_mean"] / np.maximum(path_lengths[:, :, None], EPS)
    arrays["log_transmission_per_path"] = -np.log(np.maximum(arrays["calibrated_hit_ratio_mean"], EPS)) / np.maximum(path_lengths[:, :, None], EPS)
    arrays["scatter_to_direct"] = arrays["scattered_primary_rate_sum"] / np.maximum(np.abs(arrays["direct_primary_rate_sum"]), EPS)
    arrays["tail_to_direct"] = arrays["tail120_rate_sum"] / np.maximum(np.abs(arrays["direct_primary_rate_sum"]), EPS)
    return {key: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) for key, value in arrays.items()}, source_meta


def add_column(columns: dict[str, np.ndarray], name: str, values: np.ndarray) -> None:
    columns[name] = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_wide_features(bundle: dict, source_top_n: int) -> tuple[pd.DataFrame, dict]:
    metadata = bundle["metadata"].reset_index(drop=True)
    source_ids = bundle["source_ids"]
    detector_ids = bundle["detector_ids"]
    chosen_sources = selected_sources(bundle["cube_dir"], source_ids, source_top_n)
    arrays, source_meta = stack_metric_arrays(bundle, chosen_sources)
    columns: dict[str, np.ndarray] = {}
    thickness = metadata["thickness_mm"].astype(float).to_numpy(dtype=np.float32)
    add_column(columns, "metadata__thickness_mm", thickness)
    add_column(columns, "metadata__log_thickness_mm", np.log1p(thickness))
    add_column(columns, f"dataset__{safe_name(bundle['dataset'])}", np.ones(len(metadata), dtype=np.float32))

    detector_index = {detector: index for index, detector in enumerate(detector_ids)}
    for source_pos, source_id in enumerate(chosen_sources):
        for detector_id, det_pos in detector_index.items():
            for metric in DIAGNOSTIC_METRICS:
                if metric in arrays:
                    add_column(columns, f"src={source_id}|det={detector_id}|metric={metric}", arrays[metric][:, source_pos, det_pos])

    variant_groups: dict[str, list[int]] = {}
    energy_groups: dict[float, list[int]] = {}
    for source_pos, meta in enumerate(source_meta):
        variant_groups.setdefault(str(meta["source_variant"]), []).append(source_pos)
        energy_groups.setdefault(float(meta["energy_keV"]), []).append(source_pos)

    for detector_id, det_pos in detector_index.items():
        for variant, positions in variant_groups.items():
            positions = sorted(positions, key=lambda pos: float(source_meta[pos]["energy_keV"]))
            if len(positions) < 2:
                continue
            energies = np.array([float(source_meta[pos]["energy_keV"]) for pos in positions], dtype=np.float32)
            span = max(float(energies[-1] - energies[0]), EPS)
            for metric in DIAGNOSTIC_METRICS:
                if metric not in arrays:
                    continue
                values = arrays[metric][:, positions, det_pos]
                low = values[:, 0]
                high = values[:, -1]
                prefix = f"energy|det={detector_id}|variant={variant}|metric={metric}"
                add_column(columns, f"{prefix}|low_high_delta", high - low)
                add_column(columns, f"{prefix}|low_high_ratio", high / np.maximum(np.abs(low), EPS))
                add_column(columns, f"{prefix}|slope", (high - low) / span)
                if values.shape[1] >= 3:
                    add_column(columns, f"{prefix}|curvature_mean", np.diff(values, n=2, axis=1).mean(axis=1))

        for energy, positions in energy_groups.items():
            by_variant = {str(source_meta[pos]["source_variant"]): pos for pos in positions}
            baseline_pos = by_variant.get("normal_narrow", by_variant.get("normal_wide"))
            if baseline_pos is None:
                continue
            for variant, pos in by_variant.items():
                if pos == baseline_pos:
                    continue
                for metric in DIAGNOSTIC_METRICS:
                    if metric not in arrays:
                        continue
                    base = arrays[metric][:, baseline_pos, det_pos]
                    value = arrays[metric][:, pos, det_pos]
                    prefix = f"angle|det={detector_id}|energy={energy:g}|variant={variant}|metric={metric}"
                    add_column(columns, f"{prefix}|delta_vs_baseline", value - base)
                    add_column(columns, f"{prefix}|ratio_vs_baseline", value / np.maximum(np.abs(base), EPS))

    if {"transmission", "side_scatter"}.issubset(detector_index):
        trans_pos = detector_index["transmission"]
        side_pos = detector_index["side_scatter"]
        for source_pos, source_id in enumerate(chosen_sources):
            for metric in DIAGNOSTIC_METRICS:
                if metric not in arrays:
                    continue
                trans = arrays[metric][:, source_pos, trans_pos]
                side = arrays[metric][:, source_pos, side_pos]
                prefix = f"detector_pair|src={source_id}|metric={metric}"
                add_column(columns, f"{prefix}|side_minus_trans", side - trans)
                add_column(columns, f"{prefix}|side_over_trans", side / np.maximum(np.abs(trans), EPS))

    meta_cols = [col for col in METADATA_FEATURE_COLS if col in metadata.columns]
    features = pd.concat([metadata[meta_cols].reset_index(drop=True), pd.DataFrame(columns)], axis=1)
    audit = {
        "selected_source_ids": chosen_sources,
        "selected_source_count": len(chosen_sources),
        "available_source_count": len(source_ids),
        "detectors": detector_ids,
        "channels": bundle["channels"],
        "samples": int(len(features)),
        "feature_count": int(len(columns)),
    }
    return features, audit

def build_dataset_features(bundle: dict, source_top_n: int) -> tuple[pd.DataFrame, dict]:
    return build_wide_features(bundle, source_top_n)


def align_feature_frames(frames: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[str]]:
    all_columns = sorted({col for frame in frames for col in frame.columns if col not in METADATA_FEATURE_COLS})
    aligned = []
    for frame in frames:
        meta = frame[METADATA_FEATURE_COLS].reset_index(drop=True)
        feature_frame = frame[[col for col in all_columns if col in frame.columns]].reset_index(drop=True)
        missing = [col for col in all_columns if col not in feature_frame.columns]
        if missing:
            zeros = pd.DataFrame(0.0, index=feature_frame.index, columns=missing)
            feature_frame = pd.concat([feature_frame, zeros], axis=1)
        feature_frame = feature_frame[all_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        aligned.append(pd.concat([meta, feature_frame], axis=1))
    return aligned, all_columns


def pair_recalls(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    rows = {}
    for material in HM_PAIR:
        mask = y_true == material
        rows[material] = float(np.mean(predictions[mask] == material)) if mask.any() else 0.0
    return rows


def evaluate_binary_predictions(
    dataset: str,
    method: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    feature_count: int,
) -> dict:
    recalls = pair_recalls(y_true.astype(str), predictions.astype(str))
    return {
        "dataset": dataset,
        "method": method,
        "support": int(len(y_true)),
        "feature_count": int(feature_count),
        "pair_accuracy": float(np.mean(y_true == predictions)) if len(y_true) else 0.0,
        "hematite_recall": recalls[HM_PAIR[0]],
        "magnetite_recall": recalls[HM_PAIR[1]],
        "hm_min_recall": float(min(recalls.values())) if recalls else 0.0,
    }


def make_models(sk: dict, n_jobs: int) -> dict:
    return {
        "ExtraTreesCompact": sk["ExtraTreesClassifier"](
            n_estimators=720,
            random_state=8101,
            n_jobs=n_jobs,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
        ),
        "RandomForestBalanced": sk["RandomForestClassifier"](
            n_estimators=520,
            random_state=8102,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
        ),
        "LogisticL2Balanced": sk["make_pipeline"](
            sk["StandardScaler"](),
            sk["LogisticRegression"](
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                random_state=8103,
            ),
        ),
        "LinearSVCBalanced": sk["make_pipeline"](
            sk["StandardScaler"](),
            sk["LinearSVC"](
                class_weight="balanced",
                max_iter=10000,
                random_state=8104,
            ),
        ),
    }


def train_eval_pair_models(
    features: pd.DataFrame,
    dataset: str,
    feature_cols: list[str],
    sk: dict,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict]:
    hm = features[features["material"].astype(str).isin(HM_PAIR)].copy()
    train = hm[hm["split"].astype(str).eq("train")]
    validation = hm[hm["split"].astype(str).eq("validation")]
    if train.empty or validation.empty:
        raise RuntimeError(f"{dataset} has no train/validation H/M samples.")
    x_train = train[feature_cols].to_numpy(dtype=np.float32)
    y_train = train["material"].astype(str).to_numpy()
    x_val = validation[feature_cols].to_numpy(dtype=np.float32)
    y_val = validation["material"].astype(str).to_numpy()
    rows = []
    payloads = {}
    for method, model in make_models(sk, n_jobs).items():
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_val)).astype(str)
        rows.append(evaluate_binary_predictions(dataset, method, y_val, predictions, len(feature_cols)))
        payloads[method] = (validation.reset_index(drop=True), predictions)
    table = pd.DataFrame(rows).sort_values(["hm_min_recall", "pair_accuracy"], ascending=[False, False])
    return table, payloads


def by_group_recall(validation_meta: pd.DataFrame, predictions: np.ndarray, group_cols: list[str]) -> pd.DataFrame:
    frame = validation_meta.reset_index(drop=True).copy()
    frame["prediction"] = predictions.astype(str)
    frame["is_correct"] = frame["material"].astype(str).eq(frame["prediction"].astype(str))
    rows = []
    for keys, part in frame.groupby(group_cols + ["material"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols + ["material"], keys))
        record["support"] = int(len(part))
        record["recall"] = float(part["is_correct"].mean()) if len(part) else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def choose_binary_model(model_table: pd.DataFrame) -> dict:
    ranked = model_table.sort_values(["hm_min_recall", "pair_accuracy", "method"], ascending=[False, False, True])
    return ranked.iloc[0].to_dict()


def train_best_pair_model(features: pd.DataFrame, feature_cols: list[str], sk: dict, n_jobs: int):
    model = sk["ExtraTreesClassifier"](
        n_estimators=720,
        random_state=9101,
        n_jobs=n_jobs,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )
    hm_train = features[
        features["split"].astype(str).eq("train")
        & features["material"].astype(str).isin(HM_PAIR)
    ]
    model.fit(hm_train[feature_cols].to_numpy(dtype=np.float32), hm_train["material"].astype(str).to_numpy())
    return model


def evaluate_full_predictions(y_true: np.ndarray, predictions: np.ndarray, sk: dict) -> dict:
    labels = [material for material in v7b.TARGET_MATERIALS if material in set(y_true.astype(str))]
    recalls = sk["recall_score"](y_true, predictions, labels=np.array(labels), average=None, zero_division=0)
    recall_map = {material: float(value) for material, value in zip(labels, recalls)}
    hm_recalls = {material: recall_map.get(material, 0.0) for material in HM_PAIR}
    return {
        "top1_accuracy": float(np.mean(y_true == predictions)),
        "macro_f1": float(sk["f1_score"](y_true, predictions, labels=np.array(labels), average="macro", zero_division=0)),
        "min_class_recall": float(np.min(recalls)) if len(recalls) else 0.0,
        "hematite_recall": hm_recalls[HM_PAIR[0]],
        "magnetite_recall": hm_recalls[HM_PAIR[1]],
        "hm_min_recall": float(min(hm_recalls.values())),
    }


def evaluate_v7b_sidecar(
    v7b_features: pd.DataFrame,
    feature_cols: list[str],
    v7b_dir: Path,
    sk: dict,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions_path = v7b_dir / "validation_decisions.csv"
    if not decisions_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    decisions = pd.read_csv(decisions_path)
    validation_features = v7b_features[v7b_features["split"].astype(str).eq("validation")].copy()
    key_cols = ["material", "thickness_mm", "random_seed", "sample_id", "split"]
    merged = validation_features.merge(decisions, on=key_cols, how="left", validate="one_to_one")
    if merged["predicted_material"].isna().any():
        missing = int(merged["predicted_material"].isna().sum())
        raise RuntimeError(f"Could not align v7B validation decisions to features; missing={missing}")
    model = train_best_pair_model(v7b_features, feature_cols, sk, n_jobs)
    sidecar_predictions = np.asarray(model.predict(merged[feature_cols].to_numpy(dtype=np.float32))).astype(str)
    top3 = merged["top3_candidates"].fillna("").astype(str)
    base_predictions = merged["predicted_material"].astype(str).to_numpy()
    truth = merged["material"].astype(str).to_numpy()
    policies = {
        "base_predicts_hm": np.isin(base_predictions, np.array(HM_PAIR)),
        "either_hm_top3": top3.str.contains(HM_PAIR[0], regex=False).to_numpy()
        | top3.str.contains(HM_PAIR[1], regex=False).to_numpy(),
        "both_hm_top3": top3.str.contains(HM_PAIR[0], regex=False).to_numpy()
        & top3.str.contains(HM_PAIR[1], regex=False).to_numpy(),
    }
    rows = []
    decision_frames = []
    base_metrics = evaluate_full_predictions(truth, base_predictions, sk)
    rows.append(
        {
            "policy": "baseline_no_sidecar",
            "triggered_samples": 0,
            "triggered_hm_samples": 0,
            **base_metrics,
        }
    )
    for policy, trigger in policies.items():
        updated = base_predictions.copy()
        updated[trigger] = sidecar_predictions[trigger]
        metrics = evaluate_full_predictions(truth, updated, sk)
        rows.append(
            {
                "policy": policy,
                "triggered_samples": int(trigger.sum()),
                "triggered_hm_samples": int(np.sum(trigger & np.isin(truth, np.array(HM_PAIR)))),
                **metrics,
            }
        )
        frame = merged[key_cols + ["predicted_material", "top3_candidates"]].copy()
        frame["policy"] = policy
        frame["sidecar_prediction"] = sidecar_predictions
        frame["triggered"] = trigger
        frame["final_prediction"] = updated
        frame["is_correct"] = truth == updated
        decision_frames.append(frame)
    return pd.DataFrame(rows).sort_values(["hm_min_recall", "top1_accuracy"], ascending=[False, False]), pd.concat(decision_frames, ignore_index=True)


def gate_decision(pair_models: pd.DataFrame, sidecar: pd.DataFrame, manifest: dict) -> dict:
    best_pair = choose_binary_model(pair_models).copy() if not pair_models.empty else {}
    best_sidecar = sidecar.iloc[0].to_dict() if not sidecar.empty else {}
    candidates = []
    if best_pair:
        candidates.append(("pair_model", float(best_pair["hm_min_recall"])))
    if best_sidecar:
        candidates.append(("v7b_sidecar", float(best_sidecar["hm_min_recall"])))
    best_kind, best_hm = max(candidates, key=lambda item: item[1]) if candidates else ("none", 0.0)
    improvement = best_hm - BASELINE_HM_MIN_RECALL
    if best_hm >= PASS_HM_MIN_RECALL and improvement >= PASS_IMPROVEMENT:
        decision = "continue_sidecar_optimization"
    elif best_hm >= GRAY_HM_MIN_RECALL:
        decision = "gray_zone_paper_supplement_only"
    else:
        decision = "stop_write_xrt_information_boundary"
    return {
        "generated_by": "analysis/hm_phase1_sidecar_features.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "hm_phase1_compact_xrt_sidecar",
        "development_only": True,
        "shadow_or_final_used": False,
        "baseline_hm_min_recall": BASELINE_HM_MIN_RECALL,
        "pass_hm_min_recall": PASS_HM_MIN_RECALL,
        "pass_improvement": PASS_IMPROVEMENT,
        "gray_hm_min_recall": GRAY_HM_MIN_RECALL,
        "best_candidate_kind": best_kind,
        "best_hm_min_recall": best_hm,
        "best_improvement_over_v7b": improvement,
        "decision": decision,
        "gate_passed": decision == "continue_sidecar_optimization",
        "best_pair_model": best_pair,
        "best_sidecar_policy": best_sidecar,
        "input_manifest": manifest,
        "stop_rule": "Do not run shadow/final or force full v7B2; if compact XRT features cannot improve H/M, treat this as an ordinary-XRT information boundary.",
    }


def write_report(output_dir: Path, gate: dict) -> None:
    text = f"""# H/M Phase 1 Compact XRT Sidecar

Generated: {gate['generated_at_utc']}

Development-only: `true`

Shadow/final used: `false`

Best candidate: `{gate['best_candidate_kind']}`

- Best H/M min recall: `{gate['best_hm_min_recall']:.4f}`
- v7B baseline H/M min recall: `{gate['baseline_hm_min_recall']:.4f}`
- Improvement over v7B: `{gate['best_improvement_over_v7b']:.4f}`
- Decision: `{gate['decision']}`

Interpretation:
The features remain ordinary XRT-derived attenuation, hit-rate, energy-response, angle-response, and scatter/direct summaries. A passing result would justify more sidecar optimization; a non-passing result supports the literature-based limitation that H/M phase ID normally needs diffraction or valence/local-structure observables.
"""
    (output_dir / "hm_phase1_report.md").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 H/M compact XRT feature and sidecar audit.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--v7b-cube-dir", default="results/accuracy_v3/v7b_hard_negative_dev")
    parser.add_argument("--v7b2-cube-dir", default="results/accuracy_v3/v7b2_hm_physics_dev")
    parser.add_argument("--output-dir", default="results/accuracy_v3/hm_phase1_sidecar_dev")
    parser.add_argument("--source-top-n", type=int, default=18)
    parser.add_argument("--n-jobs", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    args = parser.parse_args()

    start_time = perf_counter()
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = max(1, int(args.n_jobs))
    log_progress(f"start output_dir={output_dir} source_top_n={args.source_top_n} n_jobs={n_jobs}", start_time=start_time)

    sk = require_sklearn()
    v7b_bundle = load_cube_bundle(project_root / args.v7b_cube_dir, "v7b")
    v7b2_bundle = load_cube_bundle(project_root / args.v7b2_cube_dir, "v7b2")
    v7b_features, v7b_audit = build_dataset_features(v7b_bundle, source_top_n=0)
    log_progress(f"built v7b features samples={len(v7b_features)} cols={len(v7b_features.columns)}", start_time=start_time)
    v7b2_features, v7b2_audit = build_dataset_features(v7b2_bundle, source_top_n=args.source_top_n)
    log_progress(f"built v7b2 features samples={len(v7b2_features)} cols={len(v7b2_features.columns)}", start_time=start_time)

    (v7b_aligned, v7b2_aligned), feature_cols = align_feature_frames([v7b_features, v7b2_features])
    combined = pd.concat([v7b_aligned, v7b2_aligned], ignore_index=True)
    v7b_aligned.to_csv(output_dir / "hm_phase1_compact_features_v7b.csv", index=False, lineterminator="\n")
    v7b2_aligned.to_csv(output_dir / "hm_phase1_compact_features_v7b2.csv", index=False, lineterminator="\n")
    log_progress(f"aligned feature_count={len(feature_cols)} combined_samples={len(combined)}", start_time=start_time)

    pair_tables = []
    payloads = {}
    for dataset, frame in [("v7b", v7b_aligned), ("v7b2", v7b2_aligned), ("pooled", combined)]:
        table, model_payloads = train_eval_pair_models(frame, dataset, feature_cols, sk, n_jobs)
        pair_tables.append(table)
        payloads[dataset] = model_payloads
        log_progress(f"pair models dataset={dataset} best_hm={table.iloc[0]['hm_min_recall']:.4f}", start_time=start_time)
    pair_model_table = pd.concat(pair_tables, ignore_index=True).sort_values(["hm_min_recall", "pair_accuracy"], ascending=[False, False])

    best_rows = []
    for dataset, frame in [("v7b", v7b_aligned), ("v7b2", v7b2_aligned), ("pooled", combined)]:
        best = pair_model_table[pair_model_table["dataset"].eq(dataset)].iloc[0]
        validation_meta, predictions = payloads[dataset][str(best["method"])]
        thick = by_group_recall(validation_meta, predictions, ["dataset", "thickness_mm"])
        seed = by_group_recall(validation_meta, predictions, ["dataset", "random_seed"])
        thick["selected_method"] = best["method"]
        seed["selected_method"] = best["method"]
        best_rows.append((thick, seed))
    by_thickness = pd.concat([item[0] for item in best_rows], ignore_index=True)
    by_seed = pd.concat([item[1] for item in best_rows], ignore_index=True)

    sidecar_eval, sidecar_decisions = evaluate_v7b_sidecar(v7b_aligned, feature_cols, project_root / args.v7b_cube_dir, sk, n_jobs)
    log_progress(
        f"v7b sidecar best_hm={(sidecar_eval.iloc[0]['hm_min_recall'] if not sidecar_eval.empty else 0.0):.4f}",
        start_time=start_time,
    )

    manifest = {
        "project_root": project_root.as_posix(),
        "v7b_cube_dir": args.v7b_cube_dir,
        "v7b2_cube_dir": args.v7b2_cube_dir,
        "output_dir": args.output_dir,
        "source_top_n": int(args.source_top_n),
        "feature_count": int(len(feature_cols)),
        "v7b_audit": v7b_audit,
        "v7b2_audit": v7b2_audit,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    gate = gate_decision(pair_model_table, sidecar_eval, manifest)

    pair_model_table.to_csv(output_dir / "hm_phase1_model_selection.csv", index=False, lineterminator="\n")
    by_thickness.to_csv(output_dir / "hm_phase1_by_thickness_recall.csv", index=False, lineterminator="\n")
    by_seed.to_csv(output_dir / "hm_phase1_by_seed_recall.csv", index=False, lineterminator="\n")
    sidecar_eval.to_csv(output_dir / "hm_phase1_sidecar_policy_eval.csv", index=False, lineterminator="\n")
    if not sidecar_decisions.empty:
        sidecar_decisions.to_csv(output_dir / "hm_phase1_sidecar_decisions.csv", index=False, lineterminator="\n")
    (output_dir / "hm_phase1_feature_manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    (output_dir / "hm_phase1_gate.json").write_bytes(
        (json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    write_report(output_dir, gate)
    log_progress(f"wrote phase1 outputs gate_decision={gate['decision']} best_hm={gate['best_hm_min_recall']:.4f}", start_time=start_time)


if __name__ == "__main__":
    main()
