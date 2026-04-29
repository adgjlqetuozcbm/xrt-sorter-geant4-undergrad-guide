from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PHOTONS_PER_SAMPLE = 100
MATERIALS_FILE = Path("source_models/materials/material_catalog.csv")
TRAIN_SEED = 101
VALIDATION_SEED = 202
TEST_SEED = 303
TARGET_MATERIALS = [
    "Quartz",
    "Calcite",
    "Orthoclase",
    "Albite",
    "Dolomite",
    "Pyrite",
    "Hematite",
    "Magnetite",
    "Chalcopyrite",
    "Galena",
]
EXPECTED_THICKNESSES = [5.0, 10.0, 20.0]
EXPECTED_SOURCES = ["mono_60kev", "mono_100kev", "spectrum_120kv"]
EXPECTED_SEEDS = [TRAIN_SEED, VALIDATION_SEED, TEST_SEED]
ENERGY_EDGES = [0.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, math.inf]
DETECTOR_RESPONSE_SIGMA_KEV = [2.0, 5.0]
REVIEW_PROBABILITY_THRESHOLD = 0.65
REVIEW_MARGIN_THRESHOLD = 0.15
EPS = 0.5
THRESHOLD_PROBABILITY_GRID = [0.0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
THRESHOLD_MARGIN_QUANTILES = [0.0, 0.10, 0.20, 0.30, 0.40]
FEATURE_FAMILY_ORDER = [
    "raw_counts",
    "calibrated_transmission",
    "attenuation",
    "thickness_normalized_attenuation",
    "spectral_shape",
    "scatter_direct",
    "detector_response",
    "source_fusion",
    "dictionary_distance",
    "other_numeric",
]
HARD_EXCLUDED_COLUMNS = {
    "material",
    "true_material",
    "formula",
    "category",
    "density_g_cm3",
    "fine_label",
    "coarse_label",
    "group_label",
    "label_id",
    "fine_label_id",
    "coarse_label_id",
    "target_id",
    "held_out_material",
    "is_correct",
    "sample_key",
    "run_id",
    "experiment_label",
    "profile_name",
    "output_prefix",
    "config_path",
    "event_file",
    "hit_file",
    "metadata_path",
    "row_index",
    "sample_id",
    "event_id_min",
    "event_id_max",
    "random_seed",
    "split",
    "original_split",
    "run_role",
}
FORBIDDEN_FEATURE_FRAGMENTS = ["formula", "density", "group_label", "config_path", "output_prefix", "ore_primary_material"]


@dataclass(frozen=True)
class RunRecord:
    run_role: str
    material: str
    source_id: str
    thickness_mm: float
    random_seed: int
    run_id: str
    event_file: Path
    hit_file: Path
    metadata_file: Path


class CentroidModel:
    def __init__(self, method: str, feature_cols: list[str], standardize: bool) -> None:
        self.method = method
        self.feature_cols = feature_cols
        self.standardize = standardize
        self.classes_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.centroids_: dict[str, np.ndarray] = {}

    def fit(self, frame: pd.DataFrame) -> "CentroidModel":
        x = frame[self.feature_cols].to_numpy(dtype=float)
        self.center_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        z = self._transform(x)
        labels = frame["material"].astype(str).to_numpy()
        self.classes_ = np.array(sorted(set(labels)))
        self.centroids_ = {
            label: z[labels == label].mean(axis=0) for label in self.classes_
        }
        return self

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("CentroidModel must be fitted first.")
        if not self.standardize:
            return x
        return (x - self.center_) / self.scale_

    def score_matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.classes_ is None:
            raise RuntimeError("CentroidModel must be fitted first.")
        z = self._transform(frame[self.feature_cols].to_numpy(dtype=float))
        distances = []
        for label in self.classes_:
            distances.append(np.linalg.norm(z - self.centroids_[label], axis=1))
        distance_matrix = np.vstack(distances).T
        return -distance_matrix, self.classes_

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        scores, classes = self.score_matrix(frame)
        return classes[np.argmax(scores, axis=1)]


def require_sklearn():
    try:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, f1_score, recall_score
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "material_sorting_v2.py requires scikit-learn. Install with: pip install pandas scikit-learn"
        ) from exc

    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "MLPClassifier": MLPClassifier,
        "make_pipeline": make_pipeline,
        "recall_score": recall_score,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def source_id_from_metadata(meta: dict) -> str:
    mode = str(meta.get("source_mode", "unknown"))
    variant = str(meta.get("source_variant", "")).strip()
    if mode == "mono":
        energy = float(meta.get("mono_energy_keV", 0))
        energy_label = str(int(energy)) if energy.is_integer() else f"{energy:g}".replace(".", "p")
        base = f"mono_{energy_label}kev"
        return f"{base}_{variant}" if variant else base
    if mode == "spectrum":
        return f"spectrum_120kv_{variant}" if variant else "spectrum_120kv"
    return mode


def bin_labels() -> list[str]:
    labels = []
    for low, high in zip(ENERGY_EDGES[:-1], ENERGY_EDGES[1:]):
        if math.isinf(high):
            labels.append(f"e_{int(low):03d}_inf")
        else:
            labels.append(f"e_{int(low):03d}_{int(high):03d}")
    return labels


def response_bin_labels(sigma_kev: float) -> list[str]:
    tag = f"Rsig{int(round(sigma_kev * 10)):03d}"
    return [f"{tag}_{label}" for label in bin_labels()]


def normal_cdf(values: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(values / math.sqrt(2.0)))


def gaussian_response_counts(energies: np.ndarray, sigma_kev: float) -> np.ndarray:
    if energies.size == 0:
        return np.zeros(len(ENERGY_EDGES) - 1, dtype=float)
    counts = []
    for low, high in zip(ENERGY_EDGES[:-1], ENERGY_EDGES[1:]):
        lower = normal_cdf((low - energies) / sigma_kev)
        if math.isinf(high):
            upper = np.ones_like(lower)
        else:
            upper = normal_cdf((high - energies) / sigma_kev)
        counts.append(float(np.clip(upper - lower, 0.0, 1.0).sum()))
    return np.array(counts, dtype=float)


def finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.quantile(finite, quantile))


def histogram_entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_record_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    posix = raw_path.replace("\\", "/")
    marker = project_root.name + "/"
    if marker in posix:
        suffix = posix.split(marker, 1)[1]
        candidate = project_root / suffix
        if candidate.exists():
            return candidate
    return path


def discover_records(project_root: Path, raw_dir: Path) -> tuple[list[RunRecord], list[RunRecord]]:
    material_records: list[RunRecord] = []
    calibration_records: list[RunRecord] = []
    for meta_path in sorted(raw_dir.rglob("*_metadata.json")):
        meta = read_metadata(meta_path)
        run_role = str(meta.get("run_role", "")).strip().lower()
        ore_mode = str(meta.get("ore_material_mode", "")).strip().lower()
        material = str(meta.get("ore_primary_material", ""))
        if not run_role:
            run_role = "calibration" if ore_mode == "air_path" else "material"
        record = RunRecord(
            run_role=run_role,
            material="AIR_PATH" if run_role == "calibration" else material,
            source_id=source_id_from_metadata(meta),
            thickness_mm=float(meta.get("ore_thickness_mm", 0.0)),
            random_seed=int(meta.get("random_seed", -1)),
            run_id=str(meta.get("run_id", meta_path.stem)),
            event_file=resolve_record_path(project_root, str(meta["event_file"])),
            hit_file=resolve_record_path(project_root, str(meta["hit_file"])),
            metadata_file=meta_path,
        )
        if record.run_role == "calibration" or ore_mode == "air_path":
            calibration_records.append(record)
        elif record.material in TARGET_MATERIALS:
            material_records.append(record)
    return material_records, calibration_records


def records_inventory(material_records: list[RunRecord], calibration_records: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for record in [*material_records, *calibration_records]:
        rows.append(
            {
                "run_role": record.run_role,
                "material": record.material,
                "source_id": record.source_id,
                "thickness_mm": record.thickness_mm,
                "random_seed": record.random_seed,
                "run_id": record.run_id,
                "metadata_path": record.metadata_file.as_posix(),
                "event_file_exists": record.event_file.exists(),
                "hit_file_exists": record.hit_file.exists(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_run(record: RunRecord) -> pd.DataFrame:
    events = pd.read_csv(record.event_file).sort_values("event_id").reset_index(drop=True)
    if "transmission_gamma_entries" not in events.columns:
        events["transmission_gamma_entries"] = events["detector_gamma_entries"]
    if "transmission_primary_gamma_entries" not in events.columns:
        events["transmission_primary_gamma_entries"] = events["primary_gamma_entries"]
    if "side_scatter_gamma_entries" not in events.columns:
        events["side_scatter_gamma_entries"] = 0
    if "side_scatter_primary_gamma_entries" not in events.columns:
        events["side_scatter_primary_gamma_entries"] = 0
    complete = len(events) // PHOTONS_PER_SAMPLE
    events = events.iloc[: complete * PHOTONS_PER_SAMPLE].copy()
    if events.empty:
        raise ValueError(f"No complete {PHOTONS_PER_SAMPLE}-photon samples in {record.event_file}")
    events["sample_id"] = events["event_id"] // PHOTONS_PER_SAMPLE
    grouped = (
        events.groupby("sample_id")
        .agg(
            n_events=("event_id", "count"),
            detector_edep_sum=("detector_edep_keV", "sum"),
            detector_edep_mean=("detector_edep_keV", "mean"),
            detector_edep_std=("detector_edep_keV", "std"),
            detector_edep_q10=("detector_edep_keV", lambda value: float(value.quantile(0.10))),
            detector_edep_q50=("detector_edep_keV", lambda value: float(value.quantile(0.50))),
            detector_edep_q90=("detector_edep_keV", lambda value: float(value.quantile(0.90))),
            detector_edep_max=("detector_edep_keV", "max"),
            detector_edep_nonzero_fraction=("detector_edep_keV", lambda value: float((value > 0).mean())),
            detector_gamma_total_sum=("detector_gamma_entries", "sum"),
            primary_gamma_total_sum=("primary_gamma_entries", "sum"),
            detector_gamma_sum=("transmission_gamma_entries", "sum"),
            detector_gamma_nonzero_fraction=("transmission_gamma_entries", lambda value: float((value > 0).mean())),
            primary_gamma_sum=("transmission_primary_gamma_entries", "sum"),
            primary_gamma_zero_fraction=("transmission_primary_gamma_entries", lambda value: float((value <= 0).mean())),
            side_scatter_gamma_sum=("side_scatter_gamma_entries", "sum"),
            side_scatter_primary_gamma_sum=("side_scatter_primary_gamma_entries", "sum"),
        )
        .reset_index()
    )
    grouped["detector_edep_std"] = grouped["detector_edep_std"].fillna(0.0)
    grouped["detector_gamma_rate"] = grouped["detector_gamma_sum"] / grouped["n_events"]
    grouped["detector_gamma_total_rate"] = grouped["detector_gamma_total_sum"] / grouped["n_events"]
    grouped["primary_transmission_rate"] = grouped["primary_gamma_sum"] / grouped["n_events"]
    grouped["primary_total_rate"] = grouped["primary_gamma_total_sum"] / grouped["n_events"]
    grouped["side_scatter_gamma_rate"] = grouped["side_scatter_gamma_sum"] / grouped["n_events"]
    grouped["side_scatter_primary_rate"] = grouped["side_scatter_primary_gamma_sum"] / grouped["n_events"]
    grouped["scatter_to_transmission_gamma_ratio"] = grouped["side_scatter_gamma_sum"] / (
        grouped["detector_gamma_sum"] + 1.0
    )

    label_cols = bin_labels()
    for col in label_cols:
        grouped[f"I_{col}"] = 0.0
        grouped[f"side_I_{col}"] = 0.0
    response_cols = [(sigma, response_bin_labels(sigma)) for sigma in DETECTOR_RESPONSE_SIGMA_KEV]
    for _, cols in response_cols:
        for col in cols:
            grouped[col] = 0.0
    grouped["hit_count"] = 0.0
    grouped["direct_primary_count"] = 0.0
    grouped["scattered_primary_count"] = 0.0
    grouped["direct_hit_fraction"] = 0.0
    grouped["hit_energy_mean"] = 0.0
    grouped["hit_energy_std"] = 0.0
    grouped["hit_energy_q10"] = 0.0
    grouped["hit_energy_q50"] = 0.0
    grouped["hit_energy_q90"] = 0.0
    grouped["hit_energy_iqr"] = 0.0
    grouped["hit_energy_entropy"] = 0.0
    grouped["hit_y_mean"] = 0.0
    grouped["hit_y_std"] = 0.0
    grouped["hit_z_mean"] = 0.0
    grouped["hit_z_std"] = 0.0
    grouped["central_hit_fraction_r25"] = 0.0
    grouped["quadrant_balance_abs"] = 0.0
    grouped["direct_hit_energy_mean"] = 0.0
    grouped["scattered_hit_energy_mean"] = 0.0
    grouped["theta_mean"] = 0.0
    grouped["theta_std"] = 0.0
    grouped["theta_q50"] = 0.0
    grouped["theta_q90"] = 0.0
    grouped["r_mean"] = 0.0
    grouped["r_std"] = 0.0
    grouped["r_q50"] = 0.0
    grouped["r_q90"] = 0.0
    grouped["side_hit_count"] = 0.0
    grouped["side_primary_count"] = 0.0
    grouped["side_hit_energy_mean"] = 0.0
    grouped["side_hit_energy_std"] = 0.0
    grouped["side_hit_energy_entropy"] = 0.0
    grouped["side_theta_mean"] = 0.0
    grouped["side_theta_q50"] = 0.0
    grouped["side_theta_q90"] = 0.0
    grouped["side_r_mean"] = 0.0
    grouped["side_r_std"] = 0.0

    if record.hit_file.exists() and record.hit_file.stat().st_size > 100:
        hits = pd.read_csv(record.hit_file)
        if "detector_id" not in hits.columns:
            hits["detector_id"] = "transmission"
        if "x_mm" not in hits.columns:
            hits["x_mm"] = 0.0
        hits["detector_id"] = hits["detector_id"].fillna("transmission").astype(str)
        hits["sample_id"] = hits["event_id"] // PHOTONS_PER_SAMPLE
        hits["r_mm"] = np.sqrt(hits["y_mm"] ** 2 + hits["z_mm"] ** 2)
        by_sample = hits.groupby("sample_id")
        for sample_id, part in by_sample:
            mask = grouped["sample_id"] == sample_id
            transmission = part[part["detector_id"].eq("transmission")]
            side = part[part["detector_id"].eq("side_scatter")]
            energies = transmission["photon_energy_keV"].to_numpy(dtype=float)
            counts, _ = np.histogram(energies, bins=np.array(ENERGY_EDGES, dtype=float))
            for col, count in zip(label_cols, counts):
                grouped.loc[mask, f"I_{col}"] = float(count)
            for sigma, cols in response_cols:
                smoothed = gaussian_response_counts(energies, sigma)
                for col, count in zip(cols, smoothed):
                    grouped.loc[mask, col] = float(count)
            direct_mask = transmission["is_direct_primary"].astype(bool).to_numpy()
            scattered_mask = transmission["is_scattered_primary"].astype(bool).to_numpy()
            theta_values = transmission["theta_deg"].replace(-1.0, np.nan).dropna().to_numpy(dtype=float)
            radii = transmission["r_mm"].to_numpy(dtype=float)
            y_values = transmission["y_mm"].to_numpy(dtype=float)
            z_values = transmission["z_mm"].to_numpy(dtype=float)
            grouped.loc[mask, "hit_count"] = float(len(transmission))
            grouped.loc[mask, "direct_primary_count"] = float(transmission["is_direct_primary"].sum())
            grouped.loc[mask, "scattered_primary_count"] = float(transmission["is_scattered_primary"].sum())
            grouped.loc[mask, "direct_hit_fraction"] = float(transmission["is_direct_primary"].mean()) if len(transmission) else 0.0
            grouped.loc[mask, "hit_energy_mean"] = float(transmission["photon_energy_keV"].mean()) if len(transmission) else 0.0
            grouped.loc[mask, "hit_energy_std"] = float(transmission["photon_energy_keV"].std(ddof=0)) if len(transmission) else 0.0
            grouped.loc[mask, "hit_energy_q10"] = finite_quantile(energies, 0.10)
            grouped.loc[mask, "hit_energy_q50"] = finite_quantile(energies, 0.50)
            grouped.loc[mask, "hit_energy_q90"] = finite_quantile(energies, 0.90)
            grouped.loc[mask, "hit_energy_iqr"] = finite_quantile(energies, 0.75) - finite_quantile(energies, 0.25)
            grouped.loc[mask, "hit_energy_entropy"] = histogram_entropy(counts.astype(float))
            grouped.loc[mask, "hit_y_mean"] = float(y_values.mean()) if y_values.size else 0.0
            grouped.loc[mask, "hit_y_std"] = float(y_values.std(ddof=0)) if y_values.size else 0.0
            grouped.loc[mask, "hit_z_mean"] = float(z_values.mean()) if z_values.size else 0.0
            grouped.loc[mask, "hit_z_std"] = float(z_values.std(ddof=0)) if z_values.size else 0.0
            grouped.loc[mask, "central_hit_fraction_r25"] = float((radii <= 25.0).mean()) if radii.size else 0.0
            if y_values.size:
                q1 = np.mean((y_values >= 0) & (z_values >= 0))
                q2 = np.mean((y_values < 0) & (z_values >= 0))
                q3 = np.mean((y_values < 0) & (z_values < 0))
                q4 = np.mean((y_values >= 0) & (z_values < 0))
                grouped.loc[mask, "quadrant_balance_abs"] = float(max(q1, q2, q3, q4) - min(q1, q2, q3, q4))
            grouped.loc[mask, "direct_hit_energy_mean"] = float(energies[direct_mask].mean()) if direct_mask.any() else 0.0
            grouped.loc[mask, "scattered_hit_energy_mean"] = float(energies[scattered_mask].mean()) if scattered_mask.any() else 0.0
            grouped.loc[mask, "theta_mean"] = float(transmission["theta_deg"].replace(-1.0, np.nan).mean(skipna=True) or 0.0)
            grouped.loc[mask, "theta_std"] = float(transmission["theta_deg"].replace(-1.0, np.nan).std(ddof=0) or 0.0)
            grouped.loc[mask, "theta_q50"] = finite_quantile(theta_values, 0.50)
            grouped.loc[mask, "theta_q90"] = finite_quantile(theta_values, 0.90)
            grouped.loc[mask, "r_mean"] = float(transmission["r_mm"].mean()) if len(transmission) else 0.0
            grouped.loc[mask, "r_std"] = float(transmission["r_mm"].std(ddof=0)) if len(transmission) else 0.0
            grouped.loc[mask, "r_q50"] = finite_quantile(radii, 0.50)
            grouped.loc[mask, "r_q90"] = finite_quantile(radii, 0.90)
            side_energies = side["photon_energy_keV"].to_numpy(dtype=float)
            side_counts, _ = np.histogram(side_energies, bins=np.array(ENERGY_EDGES, dtype=float))
            for col, count in zip(label_cols, side_counts):
                grouped.loc[mask, f"side_I_{col}"] = float(count)
            side_theta = side["theta_deg"].replace(-1.0, np.nan).dropna().to_numpy(dtype=float)
            side_radii = np.sqrt(side["x_mm"].to_numpy(dtype=float) ** 2 + side["z_mm"].to_numpy(dtype=float) ** 2)
            grouped.loc[mask, "side_hit_count"] = float(len(side))
            grouped.loc[mask, "side_primary_count"] = float(side["is_primary"].sum()) if len(side) else 0.0
            grouped.loc[mask, "side_hit_energy_mean"] = float(side["photon_energy_keV"].mean()) if len(side) else 0.0
            grouped.loc[mask, "side_hit_energy_std"] = float(side["photon_energy_keV"].std(ddof=0)) if len(side) else 0.0
            grouped.loc[mask, "side_hit_energy_entropy"] = histogram_entropy(side_counts.astype(float))
            grouped.loc[mask, "side_theta_mean"] = float(side["theta_deg"].replace(-1.0, np.nan).mean(skipna=True) or 0.0) if len(side) else 0.0
            grouped.loc[mask, "side_theta_q50"] = finite_quantile(side_theta, 0.50)
            grouped.loc[mask, "side_theta_q90"] = finite_quantile(side_theta, 0.90)
            grouped.loc[mask, "side_r_mean"] = float(side_radii.mean()) if side_radii.size else 0.0
            grouped.loc[mask, "side_r_std"] = float(side_radii.std(ddof=0)) if side_radii.size else 0.0

    grouped["hit_absence"] = (grouped["hit_count"] <= 0).astype(float)
    grouped["log_hit_count"] = np.log1p(grouped["hit_count"])
    grouped["log_detector_edep_sum"] = np.log1p(grouped["detector_edep_sum"].clip(lower=0.0))

    grouped.insert(0, "run_id", record.run_id)
    grouped.insert(0, "random_seed", record.random_seed)
    grouped.insert(0, "thickness_mm", record.thickness_mm)
    grouped.insert(0, "source_id", record.source_id)
    grouped.insert(0, "material", record.material)
    grouped.insert(0, "run_role", record.run_role)
    return grouped.fillna(0.0)


def calibration_table(calibration_records: list[RunRecord]) -> pd.DataFrame:
    frames = [aggregate_run(record) for record in calibration_records]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    count_cols = [f"I_{label}" for label in bin_labels()]
    return (
        frame.groupby(["source_id", "random_seed"], as_index=False)[count_cols]
        .mean()
        .rename(columns={col: f"I0_{col[2:]}" for col in count_cols})
    )


def apply_calibration(samples: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    if calibration.empty:
        raise ValueError("No calibration runs found. Run calibration matrix rows first.")
    merged = samples.merge(calibration, on=["source_id", "random_seed"], how="left", validate="many_to_one")
    missing = merged[[f"I0_{label}" for label in bin_labels()]].isna().any(axis=1)
    if missing.any():
        missing_keys = (
            merged.loc[missing, ["source_id", "random_seed"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"Missing calibration for source/seed keys: {missing_keys}")

    edges_mid = []
    for low, high in zip(ENERGY_EDGES[:-1], ENERGY_EDGES[1:]):
        edges_mid.append(low + 10.0 if math.isinf(high) else (low + high) / 2.0)

    i_cols = [f"I_{label}" for label in bin_labels()]
    i0_cols = [f"I0_{label}" for label in bin_labels()]
    for label in bin_labels():
        i_col = f"I_{label}"
        i0_col = f"I0_{label}"
        t_col = f"T_{label}"
        a_col = f"A_{label}"
        an_col = f"A_per_mm_{label}"
        merged[t_col] = (merged[i_col] + EPS) / (merged[i0_col] + EPS)
        merged[a_col] = -np.log(np.clip(merged[t_col], 1e-6, 1e6))
        merged[an_col] = merged[a_col] / merged["thickness_mm"].clip(lower=1e-6)

    low_i = merged[[col for col in i_cols if not col.startswith("I_e_080") and not col.startswith("I_e_090") and not col.startswith("I_e_100") and not col.startswith("I_e_110") and not col.startswith("I_e_120")]].sum(axis=1)
    high_i = merged[[col for col in i_cols if col.startswith("I_e_080") or col.startswith("I_e_090") or col.startswith("I_e_100") or col.startswith("I_e_110") or col.startswith("I_e_120")]].sum(axis=1)
    low_i0 = merged[[col for col in i0_cols if not col.startswith("I0_e_080") and not col.startswith("I0_e_090") and not col.startswith("I0_e_100") and not col.startswith("I0_e_110") and not col.startswith("I0_e_120")]].sum(axis=1)
    high_i0 = merged[[col for col in i0_cols if col.startswith("I0_e_080") or col.startswith("I0_e_090") or col.startswith("I0_e_100") or col.startswith("I0_e_110") or col.startswith("I0_e_120")]].sum(axis=1)
    merged["T_low_0_80"] = (low_i + EPS) / (low_i0 + EPS)
    merged["T_high_80_inf"] = (high_i + EPS) / (high_i0 + EPS)
    merged["A_low_0_80"] = -np.log(np.clip(merged["T_low_0_80"], 1e-6, 1e6))
    merged["A_high_80_inf"] = -np.log(np.clip(merged["T_high_80_inf"], 1e-6, 1e6))
    merged["dual_log_ratio_low_high"] = np.log(
        np.clip(merged["T_low_0_80"], 1e-6, 1e6)
    ) - np.log(np.clip(merged["T_high_80_inf"], 1e-6, 1e6))

    counts = merged[i_cols].to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    mids = np.array(edges_mid, dtype=float)
    weighted_mean = np.divide(counts @ mids, totals, out=np.zeros(len(merged)), where=totals > 0)
    weighted_second = np.divide(counts @ (mids ** 2), totals, out=np.zeros(len(merged)), where=totals > 0)
    merged["spectrum_centroid_keV"] = weighted_mean
    merged["spectrum_std_keV"] = np.sqrt(np.clip(weighted_second - weighted_mean ** 2, 0.0, None))
    merged["direct_primary_fraction_stable"] = merged["direct_primary_count"] / (
        merged["direct_primary_count"] + merged["scattered_primary_count"] + 1.0
    )
    merged["scattered_primary_fraction_stable"] = merged["scattered_primary_count"] / (
        merged["direct_primary_count"] + merged["scattered_primary_count"] + 1.0
    )
    merged["log_scatter_minus_direct"] = np.log1p(merged["scattered_primary_count"]) - np.log1p(merged["direct_primary_count"])
    return merged.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for col in frame.columns:
        if col in HARD_EXCLUDED_COLUMNS or any(fragment in col for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) and float(frame[col].std()) > 1e-12:
            cols.append(col)
    return cols


def feature_family(col: str) -> str:
    base = col.split("__", 1)[1] if "__" in col else col
    if col.startswith("dict_"):
        return "dictionary_distance"
    if col.startswith("dual_source_") or col.startswith("dual_energy__"):
        return "source_fusion"
    if base.startswith("Rsig"):
        return "detector_response"
    if "direct_primary" in base or "scattered_primary" in base or "scatter" in base or base.startswith("side_"):
        return "scatter_direct"
    if base.startswith("T_") or base.startswith("primary_transmission") or base.startswith("detector_gamma_rate"):
        return "calibrated_transmission"
    if base.startswith("A_per_mm_"):
        return "thickness_normalized_attenuation"
    if base.startswith("A_") or "dual_log_ratio" in base:
        return "attenuation"
    if base.startswith("I_") or base.endswith("_count") or base.endswith("_sum"):
        return "raw_counts"
    if (
        "spectrum_" in base
        or "hit_energy" in base
        or base.startswith("theta_")
        or base.startswith("r_")
        or base.startswith("hit_y_")
        or base.startswith("hit_z_")
        or "central_hit_fraction" in base
        or "quadrant_balance" in base
    ):
        return "spectral_shape"
    return "other_numeric"


def feature_family_table(feature_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"feature": col, "family": feature_family(col)} for col in feature_cols]
    )


def columns_for_families(feature_cols: list[str], families: set[str]) -> list[str]:
    return [col for col in feature_cols if feature_family(col) in families]


def physics_feature_columns(feature_cols: list[str]) -> list[str]:
    families = {
        "calibrated_transmission",
        "attenuation",
        "thickness_normalized_attenuation",
        "spectral_shape",
        "scatter_direct",
        "detector_response",
        "source_fusion",
    }
    cols = columns_for_families(feature_cols, families)
    return cols or feature_cols


def dictionary_feature_columns(feature_cols: list[str]) -> list[str]:
    cols = columns_for_families(feature_cols, {"dictionary_distance"})
    return cols or feature_cols


def legacy_physics_dictionary_columns(feature_cols: list[str]) -> list[str]:
    families = {
        "calibrated_transmission",
        "attenuation",
        "thickness_normalized_attenuation",
        "source_fusion",
    }
    cols = columns_for_families(feature_cols, families)
    return cols or physics_feature_columns(feature_cols)


def source_energy_variant(source_id: str) -> tuple[float | None, str]:
    if not (source_id.startswith("mono_") and "kev" in source_id):
        return None, "default"
    raw = source_id.removeprefix("mono_")
    energy_text, _, variant = raw.partition("kev")
    if variant.startswith("_"):
        variant = variant[1:]
    try:
        energy = float(energy_text.replace("p", "."))
    except ValueError:
        return None, variant or "default"
    return energy, variant or "default"


def add_dual_energy_features(frame: pd.DataFrame) -> None:
    source_ids = sorted({col.split("__", 1)[0] for col in frame.columns if "__" in col})
    by_variant: dict[str, list[tuple[float, str]]] = {}
    for source_id in source_ids:
        energy, variant = source_energy_variant(source_id)
        if energy is None:
            continue
        by_variant.setdefault(variant, []).append((energy, source_id))

    def has(source_id: str, feature: str) -> bool:
        return f"{source_id}__{feature}" in frame.columns

    def values(source_id: str, feature: str) -> pd.Series:
        return frame[f"{source_id}__{feature}"].astype(float)

    for _, source_items in by_variant.items():
        source_items = sorted(source_items)
        for low_index, (low_energy, low_source) in enumerate(source_items):
            for high_energy, high_source in source_items[low_index + 1 :]:
                label = f"{low_source}__{high_source}"
                if has(low_source, "primary_transmission_rate") and has(high_source, "primary_transmission_rate"):
                    low_a = -np.log(np.clip(values(low_source, "primary_transmission_rate"), 1e-6, 1e6))
                    high_a = -np.log(np.clip(values(high_source, "primary_transmission_rate"), 1e-6, 1e6))
                    frame[f"dual_energy__R_primary__{label}"] = low_a / (high_a + EPS)
                    frame[f"dual_energy__I_primary__{label}"] = 0.5 * (
                        values(low_source, "primary_transmission_rate")
                        + values(high_source, "primary_transmission_rate")
                    )
                    frame[f"dual_energy__attenuation_slope_primary__{label}"] = (
                        high_a - low_a
                    ) / max(high_energy - low_energy, 1e-6)
                if has(low_source, "detector_gamma_rate") and has(high_source, "detector_gamma_rate"):
                    frame[f"dual_energy__log_gamma_ratio__{label}"] = (
                        np.log1p(values(high_source, "detector_gamma_rate").clip(lower=0.0))
                        - np.log1p(values(low_source, "detector_gamma_rate").clip(lower=0.0))
                    )
                for feature in ["Rsig050_e_120_inf", "Rsig020_e_120_inf", "side_scatter_gamma_rate"]:
                    if has(low_source, feature) and has(high_source, feature):
                        frame[f"dual_energy__tail_ratio__{feature}__{label}"] = (
                            np.log1p(values(high_source, feature).clip(lower=0.0))
                            - np.log1p(values(low_source, feature).clip(lower=0.0))
                        )


def fuse_sources(samples: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if samples["source_id"].nunique() < 2:
        return samples.copy(), "single_source"
    keys = ["material", "thickness_mm", "random_seed", "sample_id"]
    base_cols = numeric_feature_columns(samples)
    base_cols = [col for col in base_cols if col not in {"thickness_mm"}]
    pieces = []
    for source_id, part in samples.groupby("source_id"):
        renamed = part[keys + base_cols].copy()
        renamed = renamed.rename(columns={col: f"{source_id}__{col}" for col in base_cols})
        pieces.append(renamed)
    fused = pieces[0]
    for piece in pieces[1:]:
        fused = fused.merge(piece, on=keys, how="inner")
    if fused.empty:
        return samples.copy(), "single_source_unpaired"
    add_dual_energy_features(fused)
    if {
        "mono_60kev__primary_transmission_rate",
        "mono_100kev__primary_transmission_rate",
    }.issubset(fused.columns):
        fused["dual_source_log_transmission_ratio_60_100"] = np.log(
            (fused["mono_60kev__primary_transmission_rate"] + EPS)
            / (fused["mono_100kev__primary_transmission_rate"] + EPS)
        )
    return fused.replace([np.inf, -np.inf], np.nan).fillna(0.0), "multi_source_fused"


def matrix_status(material_records: list[RunRecord], calibration_records: list[RunRecord]) -> dict:
    material_keys = {
        (r.material, round(r.thickness_mm, 3), r.source_id, r.random_seed) for r in material_records
    }
    calibration_keys = {(r.source_id, r.random_seed) for r in calibration_records}
    expected_material_keys = {
        (material, thickness, source, seed)
        for material in TARGET_MATERIALS
        for thickness in EXPECTED_THICKNESSES
        for source in EXPECTED_SOURCES
        for seed in EXPECTED_SEEDS
    }
    expected_calibration_keys = {
        (source, seed) for source in EXPECTED_SOURCES for seed in EXPECTED_SEEDS
    }
    return {
        "material_metadata_found": len(material_records),
        "calibration_metadata_found": len(calibration_records),
        "materials_found": sorted({r.material for r in material_records}),
        "sources_found": sorted({r.source_id for r in material_records}),
        "thicknesses_found": sorted({r.thickness_mm for r in material_records}),
        "seeds_found": sorted({r.random_seed for r in material_records}),
        "missing_material_runs": len(expected_material_keys - material_keys),
        "missing_calibration_runs": len(expected_calibration_keys - calibration_keys),
        "complete_target_material_set": {r.material for r in material_records} == set(TARGET_MATERIALS),
        "complete_full_matrix": expected_material_keys.issubset(material_keys) and expected_calibration_keys.issubset(calibration_keys),
    }


def split_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame["random_seed"] == TRAIN_SEED].copy()
    validation = frame[frame["random_seed"] == VALIDATION_SEED].copy()
    test = frame[frame["random_seed"] == TEST_SEED].copy()
    return train, validation, test


def fit_dictionary(frame: pd.DataFrame, feature_cols: list[str]) -> dict:
    x = frame[feature_cols].to_numpy(dtype=float)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - center) / scale
    labels = frame["material"].astype(str).to_numpy()
    materials = sorted(set(labels))
    prototypes = {}
    for material in materials:
        part = z[labels == material]
        prototypes[material] = {
            "mean": part.mean(axis=0).tolist(),
            "std": part.std(axis=0).tolist(),
            "n_samples": int(len(part)),
        }
    return {
        "feature_columns": feature_cols,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "materials": materials,
        "prototypes": prototypes,
    }


def dictionary_distances(frame: pd.DataFrame, dictionary: dict) -> pd.DataFrame:
    feature_cols = dictionary["feature_columns"]
    center = np.array(dictionary["center"], dtype=float)
    scale = np.array(dictionary["scale"], dtype=float)
    x = (frame[feature_cols].to_numpy(dtype=float) - center) / scale
    result = pd.DataFrame(index=frame.index)
    distance_cols = []
    for material in dictionary["materials"]:
        centroid = np.array(dictionary["prototypes"][material]["mean"], dtype=float)
        col = f"dict_dist_{material}"
        result[col] = np.linalg.norm(x - centroid, axis=1)
        distance_cols.append(col)
    distances = result[distance_cols].to_numpy(dtype=float)
    order = np.argsort(distances, axis=1)
    materials = np.array(dictionary["materials"])
    result["dict_top1_distance"] = distances[np.arange(len(distances)), order[:, 0]]
    result["dict_top2_distance"] = distances[np.arange(len(distances)), order[:, 1]] if len(distance_cols) > 1 else np.nan
    result["dict_distance_margin"] = result["dict_top2_distance"] - result["dict_top1_distance"]
    result["dict_top1_material"] = materials[order[:, 0]]
    result["dict_top3_candidates"] = [";".join(materials[row[:3]]) for row in order]
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def append_dictionary_features(frame: pd.DataFrame, dictionary: dict) -> pd.DataFrame:
    dist = dictionary_distances(frame, dictionary)
    numeric_dist = dist.drop(columns=["dict_top1_material", "dict_top3_candidates"], errors="ignore")
    return pd.concat([frame.reset_index(drop=True), numeric_dist.reset_index(drop=True)], axis=1)


def load_material_catalog(project_root: Path) -> pd.DataFrame:
    cols = ["material_name", "formula", "density_g_cm3", "category", "group_label", "notes"]
    catalog = pd.read_csv(project_root / MATERIALS_FILE)
    catalog = catalog[catalog["material_name"].isin(TARGET_MATERIALS)].copy()
    for col in cols:
        if col not in catalog.columns:
            catalog[col] = ""
    return catalog[cols].rename(columns={"material_name": "material"})


def candidate_retrieval_frame(frame: pd.DataFrame, dictionary: dict, split: str) -> pd.DataFrame:
    dist = dictionary_distances(frame, dictionary)
    rows = frame[["material", "thickness_mm", "random_seed"]].reset_index(drop=True).copy()
    if "sample_id" in frame.columns:
        rows["sample_id"] = frame["sample_id"].reset_index(drop=True)
    rows["split"] = split
    rows["dict_top1_material"] = dist["dict_top1_material"].reset_index(drop=True)
    rows["dict_top3_candidates"] = dist["dict_top3_candidates"].reset_index(drop=True)
    rows["dict_top1_distance"] = dist["dict_top1_distance"].reset_index(drop=True)
    rows["dict_top2_distance"] = dist["dict_top2_distance"].reset_index(drop=True)
    rows["dict_distance_margin"] = dist["dict_distance_margin"].reset_index(drop=True)
    rows["dict_top1_correct"] = rows["material"].astype(str).eq(rows["dict_top1_material"].astype(str))
    rows["dict_top3_contains_true"] = [
        str(material) in str(candidates).split(";")
        for material, candidates in zip(rows["material"], rows["dict_top3_candidates"])
    ]
    return rows


def retrieval_summary(retrieval: pd.DataFrame, split: str) -> dict:
    if retrieval.empty:
        return {
            "split": split,
            "samples": 0,
            "dict_top1_accuracy": math.nan,
            "dict_top3_accuracy": math.nan,
            "mean_dict_distance_margin": math.nan,
        }
    return {
        "split": split,
        "samples": int(len(retrieval)),
        "dict_top1_accuracy": float(retrieval["dict_top1_correct"].mean()),
        "dict_top3_accuracy": float(retrieval["dict_top3_contains_true"].mean()),
        "mean_dict_distance_margin": float(retrieval["dict_distance_margin"].mean()),
    }


def feature_summary_for_material(frame: pd.DataFrame, material: str, feature_cols: list[str]) -> dict:
    part = frame[frame["material"] == material]
    summary = {}
    for family in FEATURE_FAMILY_ORDER:
        cols = columns_for_families(feature_cols, {family})
        if not cols:
            continue
        values = part[cols].to_numpy(dtype=float)
        summary[family] = {
            "feature_count": len(cols),
            "mean": float(np.mean(values)) if values.size else math.nan,
            "std": float(np.std(values)) if values.size else math.nan,
        }
    return summary


def stability_index(frame: pd.DataFrame, material: str, group_col: str, feature_cols: list[str]) -> dict:
    part = frame[frame["material"] == material]
    if group_col not in part.columns or part[group_col].nunique() < 2 or not feature_cols:
        return {"group_count": int(part[group_col].nunique()) if group_col in part.columns else 0, "mean_profile_std": math.nan}
    grouped = part.groupby(group_col)[feature_cols].mean()
    return {
        "group_count": int(grouped.shape[0]),
        "mean_profile_std": float(grouped.std(axis=0).mean()),
    }


def enriched_dictionary(
    dictionary: dict,
    catalog: pd.DataFrame,
    model_frame: pd.DataFrame,
    long_frame: pd.DataFrame,
    validation_retrieval: pd.DataFrame,
    confusion_graph: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    catalog_rows = {row["material"]: row for row in catalog.to_dict(orient="records")}
    feature_cols = dictionary["feature_columns"]
    entries = []
    table_rows = []
    for material in dictionary["materials"]:
        proto = dictionary["prototypes"][material]
        catalog_row = catalog_rows.get(material, {})
        retrieval_part = validation_retrieval[validation_retrieval["material"] == material]
        confused_as = confusion_graph[confusion_graph["true_material"] == material]["predicted_material"].tolist()
        model_part = model_frame[model_frame["material"] == material]
        long_part = long_frame[long_frame["material"] == material]
        thickness_stability = stability_index(model_frame, material, "thickness_mm", feature_cols)
        source_cols = [col for col in feature_cols if "__" in col]
        source_stability = {
            "group_count": len({col.split("__", 1)[0] for col in source_cols}),
            "mean_profile_std": float(model_part[source_cols].std(axis=0).mean()) if source_cols and not model_part.empty else math.nan,
        }
        entry = {
            "material": material,
            "catalog": {
                "formula": catalog_row.get("formula", ""),
                "density_g_cm3": catalog_row.get("density_g_cm3", ""),
                "category": catalog_row.get("category", ""),
                "group_label": catalog_row.get("group_label", ""),
                "notes": catalog_row.get("notes", ""),
            },
            "prototype": proto,
            "feature_family_summary": feature_summary_for_material(model_frame, material, feature_cols),
            "thickness_stability": thickness_stability,
            "source_stability": source_stability,
            "long_source_count": int(long_part["source_id"].nunique()) if "source_id" in long_part.columns else 0,
            "validation_dictionary_top1_accuracy": float(retrieval_part["dict_top1_correct"].mean()) if len(retrieval_part) else math.nan,
            "validation_dictionary_top3_accuracy": float(retrieval_part["dict_top3_contains_true"].mean()) if len(retrieval_part) else math.nan,
            "validation_confused_as": confused_as,
        }
        entries.append(entry)
        table_rows.append(
            {
                "material": material,
                "formula": catalog_row.get("formula", ""),
                "density_g_cm3": catalog_row.get("density_g_cm3", ""),
                "category": catalog_row.get("category", ""),
                "group_label": catalog_row.get("group_label", ""),
                "n_samples": int(proto["n_samples"]),
                "feature_count": len(feature_cols),
                "thickness_group_count": thickness_stability["group_count"],
                "thickness_mean_profile_std": thickness_stability["mean_profile_std"],
                "source_group_count": source_stability["group_count"],
                "source_mean_profile_std": source_stability["mean_profile_std"],
                "validation_dictionary_top1_accuracy": entry["validation_dictionary_top1_accuracy"],
                "validation_dictionary_top3_accuracy": entry["validation_dictionary_top3_accuracy"],
                "validation_confused_as": ";".join(confused_as),
            }
        )
    return {
        "feature_columns": feature_cols,
        "materials": dictionary["materials"],
        "entries": entries,
    }, pd.DataFrame(table_rows)


def build_sklearn_models(sk) -> dict[str, object]:
    return {
        "LogisticRegression": sk["make_pipeline"](
            sk["StandardScaler"](), sk["LogisticRegression"](max_iter=5000, class_weight="balanced")
        ),
        "SVM_RBF": sk["make_pipeline"](
            sk["StandardScaler"](), sk["SVC"](C=10.0, gamma="scale", probability=True, class_weight="balanced")
        ),
        "SVM_Linear": sk["make_pipeline"](
            sk["StandardScaler"](), sk["SVC"](C=1.0, kernel="linear", probability=True, class_weight="balanced")
        ),
        "RandomForest": sk["RandomForestClassifier"](n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"),
        "ExtraTrees": sk["ExtraTreesClassifier"](n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"),
        "HistGradientBoosting": sk["HistGradientBoostingClassifier"](random_state=42),
        "MLPClassifier": sk["make_pipeline"](
            sk["StandardScaler"](),
            sk["MLPClassifier"](
                hidden_layer_sizes=(64,),
                max_iter=250,
                tol=1e-3,
                random_state=42,
                early_stopping=False,
            ),
        ),
    }


def score_from_model(model: object, frame: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(model, CentroidModel):
        scores, classes = model.score_matrix(frame)
        predictions = classes[np.argmax(scores, axis=1)]
        return predictions, scores, classes
    x = frame[feature_cols]
    predictions = model.predict(x)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)
        classes = np.array(model.classes_)
    else:
        classes = np.array(sorted(set(predictions)))
        scores = np.zeros((len(frame), len(classes)))
        for idx, pred in enumerate(predictions):
            scores[idx, np.where(classes == pred)[0][0]] = 1.0
    return np.asarray(predictions), np.asarray(scores), classes


def topk_accuracy(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray, k: int) -> float:
    if scores.size == 0:
        return math.nan
    order = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    return float(np.mean([truth in classes[row] for truth, row in zip(y_true, order)]))


def evaluate_scores(
    method: str,
    frame: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sk,
) -> dict:
    labels = np.array(TARGET_MATERIALS)
    y_true = frame["material"].astype(str).to_numpy()
    recalls = sk["recall_score"](y_true, predictions, labels=labels, average=None, zero_division=0)
    return {
        "method": method,
        "samples": int(len(frame)),
        "top1_accuracy": float(np.mean(y_true == predictions)),
        "top3_accuracy": topk_accuracy(y_true, scores, classes, 3),
        "macro_f1": float(sk["f1_score"](y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "min_class_recall": float(np.min(recalls)),
    }


def train_and_score(method: str, train: pd.DataFrame, eval_frame: pd.DataFrame, feature_cols: list[str], sk):
    if method == "PhysicsDictionaryNN":
        model_cols = legacy_physics_dictionary_columns(feature_cols)
        model = CentroidModel(method, model_cols, standardize=True).fit(train)
        return model, *score_from_model(model, eval_frame, model_cols)
    if method == "PhysicsOnly":
        model_cols = physics_feature_columns(feature_cols)
        model = CentroidModel(method, model_cols, standardize=True).fit(train)
        return model, *score_from_model(model, eval_frame, model_cols)
    if method == "DictionaryOnly":
        model_cols = dictionary_feature_columns(feature_cols)
        model = CentroidModel(method, model_cols, standardize=True).fit(train)
        return model, *score_from_model(model, eval_frame, model_cols)
    if method == "PhysicsPlusDictionary":
        model_cols = physics_feature_columns(feature_cols) + dictionary_feature_columns(feature_cols)
        model_cols = list(dict.fromkeys(model_cols))
        model = CentroidModel(method, model_cols, standardize=True).fit(train)
        return model, *score_from_model(model, eval_frame, model_cols)
    if method == "MahalanobisCentroid":
        model = CentroidModel(method, feature_cols, standardize=True).fit(train)
        return model, *score_from_model(model, eval_frame, feature_cols)
    model = build_sklearn_models(sk)[method]
    model.fit(train[feature_cols], train["material"])
    return model, *score_from_model(model, eval_frame, feature_cols)


def decision_frame(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    probability_threshold: float = REVIEW_PROBABILITY_THRESHOLD,
    margin_threshold: float = REVIEW_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    order = np.argsort(scores, axis=1)[:, ::-1]
    top1_idx = order[:, 0]
    top2_idx = order[:, 1] if scores.shape[1] > 1 else order[:, 0]
    top1_score = scores[np.arange(len(scores)), top1_idx]
    top2_score = scores[np.arange(len(scores)), top2_idx]
    score_margin = top1_score - top2_score
    score_is_probability = np.all(scores >= -1e-9) and np.allclose(scores.sum(axis=1), 1.0, atol=1e-3)

    rows = []
    for idx, (_, sample) in enumerate(frame.reset_index(drop=True).iterrows()):
        reasons = []
        if score_is_probability and top1_score[idx] < probability_threshold:
            reasons.append("low_probability")
        if score_margin[idx] < margin_threshold:
            reasons.append("small_top1_top2_margin")
        row = {
            "material": sample["material"],
            "predicted_material": predictions[idx],
            "top1_score": float(top1_score[idx]),
            "top2_score": float(top2_score[idx]),
            "score_margin": float(score_margin[idx]),
            "decision": "auto_sort" if not reasons else "review_unknown_or_ambiguous",
            "review_reason": ";".join(reasons),
            "top3_candidates": ";".join(classes[order[idx, :3]]),
            "is_correct": bool(sample["material"] == predictions[idx]),
            "thickness_mm": float(sample["thickness_mm"]),
            "random_seed": int(sample["random_seed"]),
            "probability_threshold": float(probability_threshold),
            "margin_threshold": float(margin_threshold),
        }
        for col in ["dict_top1_material", "dict_top3_candidates", "dict_top1_distance", "dict_top2_distance", "dict_distance_margin"]:
            if col in sample.index:
                row[col] = sample[col]
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame | pd.Series, path: Path, *, index: bool = False) -> None:
    frame.to_csv(path, index=index, lineterminator="\n")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_manifest(path: Path, manifest: dict) -> None:
    content = json.dumps(json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_bytes(content.encode("utf-8"))


def json_energy_edges() -> list[float | str]:
    return ["inf" if math.isinf(value) else value for value in ENERGY_EDGES]


def split_assignment_table(frame: pd.DataFrame) -> pd.DataFrame:
    split = frame[["material", "thickness_mm", "random_seed", "sample_id"]].copy()
    split["split"] = np.select(
        [
            split["random_seed"].eq(TRAIN_SEED),
            split["random_seed"].eq(VALIDATION_SEED),
            split["random_seed"].eq(TEST_SEED),
        ],
        ["train", "validation", "test"],
        default="unused",
    )
    return split


def leakage_report(feature_cols: list[str], split_table: pd.DataFrame, status: dict) -> dict:
    exact_violations = sorted(set(feature_cols) & HARD_EXCLUDED_COLUMNS)
    fragment_violations = sorted(
        col for col in feature_cols if any(fragment in col for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
    )
    train_seed_set = set(split_table.loc[split_table["split"].eq("train"), "random_seed"].astype(int))
    validation_seed_set = set(split_table.loc[split_table["split"].eq("validation"), "random_seed"].astype(int))
    test_seed_set = set(split_table.loc[split_table["split"].eq("test"), "random_seed"].astype(int))
    return {
        "feature_exact_blocklist_violations": exact_violations,
        "feature_fragment_blocklist_violations": fragment_violations,
        "train_seeds": sorted(train_seed_set),
        "validation_seeds": sorted(validation_seed_set),
        "test_seeds": sorted(test_seed_set),
        "test_seed_in_train": TEST_SEED in train_seed_set,
        "calibration_rows_enter_supervised_table": False,
        "complete_full_matrix": bool(status["complete_full_matrix"]),
        "passes_leakage_checks": not exact_violations
        and not fragment_violations
        and TEST_SEED not in train_seed_set,
    }


def review_metrics(decisions: pd.DataFrame) -> dict:
    if decisions.empty:
        return {"auto_sort_precision": 0.0, "review_rate": 1.0, "auto_sort_coverage": 0.0}
    auto = decisions[decisions["decision"] == "auto_sort"]
    return {
        "auto_sort_precision": float(auto["is_correct"].mean()) if len(auto) else 0.0,
        "review_rate": float((decisions["decision"] != "auto_sort").mean()),
        "auto_sort_coverage": float((decisions["decision"] == "auto_sort").mean()),
    }


def threshold_candidates(scores: np.ndarray) -> tuple[list[float], list[float]]:
    order = np.argsort(scores, axis=1)[:, ::-1]
    top1 = scores[np.arange(len(scores)), order[:, 0]]
    top2 = scores[np.arange(len(scores)), order[:, 1]] if scores.shape[1] > 1 else top1
    margins = top1 - top2
    is_probability = np.all(scores >= -1e-9) and np.allclose(scores.sum(axis=1), 1.0, atol=1e-3)
    probability_grid = THRESHOLD_PROBABILITY_GRID if is_probability else [0.0]
    margin_grid = sorted(
        {
            0.0,
            REVIEW_MARGIN_THRESHOLD,
            *[float(np.quantile(margins, q)) for q in THRESHOLD_MARGIN_QUANTILES],
        }
    )
    return probability_grid, margin_grid


def select_review_thresholds(
    validation: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for probability_threshold in threshold_candidates(scores)[0]:
        for margin_threshold in threshold_candidates(scores)[1]:
            decisions = decision_frame(
                validation,
                predictions,
                scores,
                classes,
                probability_threshold=probability_threshold,
                margin_threshold=margin_threshold,
            )
            metrics = review_metrics(decisions)
            rows.append(
                {
                    "probability_threshold": probability_threshold,
                    "margin_threshold": margin_threshold,
                    "auto_sort_precision": metrics["auto_sort_precision"],
                    "review_rate": metrics["review_rate"],
                    "auto_sort_coverage": metrics["auto_sort_coverage"],
                    "auto_sort_samples": int((decisions["decision"] == "auto_sort").sum()),
                    "review_samples": int((decisions["decision"] != "auto_sort").sum()),
                }
            )
    table = pd.DataFrame(rows)
    preferred = table[table["auto_sort_precision"] >= 0.90].copy()
    if preferred.empty:
        ranked = table.sort_values(
            ["auto_sort_precision", "auto_sort_coverage", "review_rate"],
            ascending=[False, False, True],
        )
    else:
        ranked = preferred.sort_values(
            ["auto_sort_coverage", "auto_sort_precision", "review_rate"],
            ascending=[False, False, True],
        )
    selected = ranked.iloc[0].to_dict()
    selected["selected_on"] = "validation_seed_only"
    return selected, table.sort_values(["auto_sort_precision", "auto_sort_coverage"], ascending=[False, False])


def per_class_recall_table(frame: pd.DataFrame, predictions: np.ndarray, split: str, sk) -> pd.DataFrame:
    labels = np.array(TARGET_MATERIALS)
    recalls = sk["recall_score"](
        frame["material"].astype(str).to_numpy(),
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    support = frame["material"].value_counts().to_dict()
    return pd.DataFrame(
        [
            {
                "split": split,
                "material": material,
                "support": int(support.get(material, 0)),
                "recall": float(recall),
            }
            for material, recall in zip(labels, recalls)
        ]
    )


def confusion_graph_table(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    split: str,
) -> pd.DataFrame:
    decisions = decision_frame(frame, predictions, scores, classes, probability_threshold=0.0, margin_threshold=0.0)
    misses = decisions[~decisions["is_correct"]].copy()
    if misses.empty:
        return pd.DataFrame(columns=["split", "true_material", "predicted_material", "count", "mean_score_margin", "review_reason"])
    rows = []
    grouped = misses.groupby(["material", "predicted_material"], as_index=False)
    for _, part in grouped:
        rows.append(
            {
                "split": split,
                "true_material": part["material"].iloc[0],
                "predicted_material": part["predicted_material"].iloc[0],
                "count": int(len(part)),
                "mean_score_margin": float(part["score_margin"].mean()),
                "review_reason": "validation_confusion_pair",
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "mean_score_margin"], ascending=[False, True])


def feature_family_ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    sk,
) -> pd.DataFrame:
    rows = []
    families = [family for family in FEATURE_FAMILY_ORDER if columns_for_families(feature_cols, {family})]
    for family in families:
        cols = columns_for_families(feature_cols, {family})
        if len(cols) < 1:
            continue
        try:
            _, predictions, scores, classes = train_and_score("ExtraTrees", train, validation, cols, sk)
            metrics = evaluate_scores("ExtraTrees", validation, predictions, scores, classes, sk)
        except Exception as exc:  # noqa: BLE001 - diagnostics should record failed families.
            metrics = {
                "method": "ExtraTrees",
                "samples": int(len(validation)),
                "top1_accuracy": math.nan,
                "top3_accuracy": math.nan,
                "macro_f1": math.nan,
                "min_class_recall": math.nan,
                "error": str(exc),
            }
        metrics.update({"feature_family": family, "feature_count": len(cols)})
        rows.append(metrics)
    return pd.DataFrame(rows)


def run_pressure_thickness(frame: pd.DataFrame, feature_cols: list[str], method: str, sk) -> pd.DataFrame:
    rows = []
    for thickness in sorted(frame["thickness_mm"].unique()):
        train = frame[frame["thickness_mm"] != thickness].copy()
        test = frame[frame["thickness_mm"] == thickness].copy()
        if train["material"].nunique() < 2 or test.empty:
            continue
        dictionary = fit_dictionary(train, feature_cols)
        train_aug = append_dictionary_features(train, dictionary)
        test_aug = append_dictionary_features(test, dictionary)
        aug_cols = numeric_feature_columns(train_aug)
        _, predictions, scores, classes = train_and_score(method, train_aug, test_aug, aug_cols, sk)
        metrics = evaluate_scores(method, test_aug, predictions, scores, classes, sk)
        metrics["holdout_thickness_mm"] = float(thickness)
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Material sorting v2: calibrated fingerprints plus dictionary features.")
    parser.add_argument("--raw-dir", default="build/material_sorting_runs/full")
    parser.add_argument("--output-dir", default="results/material_sorting_v2")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allow-incomplete", action="store_true", help="Write diagnostics even when the full matrix is incomplete.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    raw_dir = project_root / args.raw_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_dir_label = raw_dir.relative_to(project_root).as_posix()
    except ValueError:
        raw_dir_label = str(raw_dir)

    material_records, calibration_records = discover_records(project_root, raw_dir)
    status = matrix_status(material_records, calibration_records)
    manifest_base = {
        "package": "xrt-sorter-geant4-undergrad-guide",
        "generated_by": "analysis/material_sorting_v2.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dir": raw_dir_label,
        "protocol": {
            "name": "material_sorting_v2_seed_holdout",
            "train_seed": TRAIN_SEED,
            "validation_seed": VALIDATION_SEED,
            "test_seed": TEST_SEED,
            "photons_per_sample": PHOTONS_PER_SAMPLE,
            "energy_edges_keV": json_energy_edges(),
            "acceptance_targets": {
                "top1_accuracy_ge_0_85": 0.85,
                "macro_f1_ge_0_80": 0.80,
                "min_class_recall_ge_0_70": 0.70,
            },
        },
        "matrix_status": status,
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    write_csv(records_inventory(material_records, calibration_records), output_dir / "material_raw_inventory_v2.csv")
    if not status["complete_full_matrix"] and not args.allow_incomplete:
        manifest_base["run_status"] = "incomplete_full_matrix"
        manifest_base["message"] = "Run the 270 material rows plus 9 calibration rows before final evaluation, or use --allow-incomplete for diagnostics."
        write_manifest(output_dir / "material_sorting_v2_manifest.json", manifest_base)
        raise SystemExit("Full material sorting matrix is incomplete; wrote manifest only.")

    if not material_records or not calibration_records:
        manifest_base["run_status"] = "no_material_or_calibration_records"
        write_manifest(output_dir / "material_sorting_v2_manifest.json", manifest_base)
        raise SystemExit("No material/calibration records available.")

    calibration = calibration_table(calibration_records)
    material_samples = pd.concat([aggregate_run(record) for record in material_records], ignore_index=True)
    calibrated = apply_calibration(material_samples, calibration)
    fused, table_mode = fuse_sources(calibrated)
    base_feature_cols = numeric_feature_columns(fused)
    write_csv(calibrated, output_dir / "material_fingerprint_samples_long.csv")
    write_csv(calibrated, output_dir / "material_samples_long_v2.csv")
    write_csv(fused, output_dir / "material_fingerprint_model_table.csv")
    write_csv(fused, output_dir / "material_model_table_v2.csv")
    write_csv(pd.Series(base_feature_cols, name="feature"), output_dir / "base_feature_columns.csv")
    write_csv(pd.Series(base_feature_cols, name="feature"), output_dir / "material_feature_columns_v2.csv")
    write_csv(feature_family_table(base_feature_cols), output_dir / "material_feature_families_v2.csv")
    write_csv(pd.Series(sorted(HARD_EXCLUDED_COLUMNS), name="excluded_column"), output_dir / "material_excluded_columns_v2.csv")
    write_csv(calibration, output_dir / "calibration_i0_table.csv")
    split_table = split_assignment_table(fused)
    write_csv(split_table, output_dir / "material_seed_split_assignments_v2.csv")
    write_manifest(output_dir / "material_leakage_report_v2.json", leakage_report(base_feature_cols, split_table, status))

    train, validation, test = split_frames(fused)
    if train["material"].nunique() < 2 or validation["material"].nunique() < 2 or test["material"].nunique() < 2:
        manifest_base.update(
            {
                "run_status": "diagnostic_only_not_enough_classes_or_seed_splits",
                "table_mode": table_mode,
                "sample_rows": len(fused),
                "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
                "feature_count": len(base_feature_cols),
            }
        )
        write_manifest(output_dir / "material_sorting_v2_manifest.json", manifest_base)
        print("Wrote fingerprint diagnostics; not enough classes or seed splits for model evaluation.")
        return

    sk = require_sklearn()
    dev_dictionary = fit_dictionary(train, base_feature_cols)
    write_manifest(output_dir / "material_dictionary_dev_train.json", dev_dictionary)
    dev_train = append_dictionary_features(train, dev_dictionary)
    dev_validation = append_dictionary_features(validation, dev_dictionary)
    dev_feature_cols = numeric_feature_columns(dev_train)
    write_csv(pd.Series(dev_feature_cols, name="feature"), output_dir / "model_feature_columns.csv")
    write_csv(feature_family_table(dev_feature_cols), output_dir / "model_feature_families.csv")

    methods = [
        "PhysicsDictionaryNN",
        "PhysicsOnly",
        "DictionaryOnly",
        "PhysicsPlusDictionary",
        "MahalanobisCentroid",
        "LogisticRegression",
        "SVM_RBF",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "MLPClassifier",
    ]
    validation_rows = []
    for method in methods:
        try:
            _, predictions, scores, classes = train_and_score(method, dev_train, dev_validation, dev_feature_cols, sk)
            validation_rows.append(evaluate_scores(method, dev_validation, predictions, scores, classes, sk))
        except Exception as exc:  # noqa: BLE001 - optional candidates should not abort final evaluation.
            validation_rows.append(
                {
                    "method": method,
                    "samples": int(len(dev_validation)),
                    "top1_accuracy": math.nan,
                    "top3_accuracy": math.nan,
                    "macro_f1": math.nan,
                    "min_class_recall": math.nan,
                    "error": str(exc),
                }
            )
    validation_summary = pd.DataFrame(validation_rows).sort_values(
        ["macro_f1", "top1_accuracy", "min_class_recall"], ascending=[False, False, False]
    )
    selectable = validation_summary.dropna(subset=["macro_f1", "top1_accuracy", "min_class_recall"])
    if selectable.empty:
        manifest_base["run_status"] = "model_selection_failed"
        manifest_base["model_selection_errors"] = validation_summary.to_dict(orient="records")
        write_manifest(output_dir / "material_sorting_v2_manifest.json", manifest_base)
        raise SystemExit("All validation candidate models failed; wrote manifest.")
    selected_method = str(selectable.iloc[0]["method"])
    write_csv(validation_summary, output_dir / "model_selection_validation.csv")
    write_csv(validation_summary, output_dir / "material_model_summary_v2.csv")
    write_csv(feature_family_ablation(dev_train, dev_validation, dev_feature_cols, sk), output_dir / "feature_family_ablation.csv")

    _, validation_predictions, validation_scores, validation_classes = train_and_score(
        selected_method, dev_train, dev_validation, dev_feature_cols, sk
    )
    selected_thresholds, threshold_table = select_review_thresholds(
        dev_validation, validation_predictions, validation_scores, validation_classes
    )
    write_csv(threshold_table, output_dir / "threshold_selection_validation.csv")
    validation_decisions = decision_frame(
        dev_validation,
        validation_predictions,
        validation_scores,
        validation_classes,
        probability_threshold=float(selected_thresholds["probability_threshold"]),
        margin_threshold=float(selected_thresholds["margin_threshold"]),
    )
    write_csv(validation_decisions, output_dir / "validation_decisions.csv")
    validation_retrieval = candidate_retrieval_frame(validation, dev_dictionary, "validation")
    write_csv(validation_retrieval, output_dir / "candidate_retrieval_validation.csv")
    validation_confusion = confusion_graph_table(
        dev_validation, validation_predictions, validation_scores, validation_classes, "validation"
    )
    write_csv(validation_confusion, output_dir / "material_confusion_graph.csv")
    write_csv(
        per_class_recall_table(dev_validation, validation_predictions, "validation", sk),
        output_dir / "per_class_recall_validation.csv",
    )

    final_train = pd.concat([train, validation], ignore_index=True)
    final_dictionary = fit_dictionary(final_train, base_feature_cols)
    write_manifest(output_dir / "material_dictionary.json", final_dictionary)
    final_validation_retrieval = candidate_retrieval_frame(validation, final_dictionary, "validation_refit")
    catalog = load_material_catalog(project_root)
    enriched_json, enriched_table = enriched_dictionary(
        final_dictionary,
        catalog,
        final_train,
        calibrated,
        final_validation_retrieval,
        validation_confusion,
    )
    write_manifest(output_dir / "material_dictionary_enriched.json", enriched_json)
    write_csv(enriched_table, output_dir / "material_dictionary_enriched.csv")
    dictionary_table = pd.DataFrame(
        [
            {
                "material": material,
                "n_samples": final_dictionary["prototypes"][material]["n_samples"],
            }
            for material in final_dictionary["materials"]
        ]
    )
    write_csv(dictionary_table, output_dir / "material_dictionary.csv")
    write_csv(dictionary_table, output_dir / "material_dictionary_v2.csv")

    final_train_aug = append_dictionary_features(final_train, final_dictionary)
    final_test_aug = append_dictionary_features(test, final_dictionary)
    final_test_retrieval = candidate_retrieval_frame(test, final_dictionary, "test")
    for col in ["dict_top1_material", "dict_top3_candidates"]:
        final_test_aug[col] = final_test_retrieval[col].to_numpy()
    final_feature_cols = numeric_feature_columns(final_train_aug)
    model, predictions, scores, classes = train_and_score(selected_method, final_train_aug, final_test_aug, final_feature_cols, sk)
    final_metrics = evaluate_scores(selected_method, final_test_aug, predictions, scores, classes, sk)
    final_summary = pd.DataFrame([final_metrics])
    write_csv(final_summary, output_dir / "final_test_summary.csv")

    decisions = decision_frame(
        final_test_aug,
        predictions,
        scores,
        classes,
        probability_threshold=float(selected_thresholds["probability_threshold"]),
        margin_threshold=float(selected_thresholds["margin_threshold"]),
    )
    write_csv(decisions, output_dir / "final_test_decisions.csv")
    write_csv(final_validation_retrieval, output_dir / "candidate_retrieval_validation_refit.csv")
    write_csv(final_test_retrieval, output_dir / "candidate_retrieval_final_test.csv")
    labels = np.array(TARGET_MATERIALS)
    cm = pd.DataFrame(
        sk["confusion_matrix"](final_test_aug["material"].astype(str), predictions, labels=labels),
        index=labels,
        columns=labels,
    )
    write_csv(cm, output_dir / f"final_test_confusion_{selected_method}.csv", index=True)
    write_csv(
        per_class_recall_table(final_test_aug, predictions, "test", sk),
        output_dir / "per_class_recall_final_test.csv",
    )
    pressure = run_pressure_thickness(fused, base_feature_cols, selected_method, sk)
    write_csv(pressure, output_dir / "pressure_leave_one_thickness_out.csv")

    auto = decisions[decisions["decision"] == "auto_sort"]
    auto_precision = float(auto["is_correct"].mean()) if len(auto) else 0.0
    review_rate = float((decisions["decision"] != "auto_sort").mean())
    acceptance = {
        "selected_method": selected_method,
        "final_test_metrics": final_metrics,
        "auto_sort_precision": auto_precision,
        "review_rate": review_rate,
        "review_thresholds_selected_on_validation": {
            "probability_threshold": float(selected_thresholds["probability_threshold"]),
            "margin_threshold": float(selected_thresholds["margin_threshold"]),
            "auto_sort_precision": float(selected_thresholds["auto_sort_precision"]),
            "review_rate": float(selected_thresholds["review_rate"]),
            "auto_sort_coverage": float(selected_thresholds["auto_sort_coverage"]),
        },
        "dictionary_retrieval": {
            "validation": retrieval_summary(validation_retrieval, "validation"),
            "final_test": retrieval_summary(final_test_retrieval, "test"),
        },
        "criteria": {
            "top1_accuracy_ge_0_85": final_metrics["top1_accuracy"] >= 0.85,
            "macro_f1_ge_0_80": final_metrics["macro_f1"] >= 0.80,
            "min_class_recall_ge_0_70": final_metrics["min_class_recall"] >= 0.70,
        },
    }
    acceptance["all_criteria_met"] = all(acceptance["criteria"].values())
    if acceptance["all_criteria_met"]:
        stage_conclusion = "automatic_ten_material_sorting_candidate"
    elif final_metrics["top3_accuracy"] >= 0.95:
        stage_conclusion = "top_k_dictionary_retrieval_with_review"
    else:
        stage_conclusion = "diagnostic_only_not_ready"

    manifest_base.update(
        {
            "run_status": "completed_evaluation",
            "table_mode": table_mode,
            "sample_rows": len(fused),
            "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
            "final_train_rows": len(final_train),
            "base_feature_count": len(base_feature_cols),
            "model_feature_count": len(final_feature_cols),
            "selected_by_validation": selected_method,
            "feature_family_count": int(feature_family_table(dev_feature_cols)["family"].nunique()),
            "feature_family_order": FEATURE_FAMILY_ORDER,
            "acceptance_status": acceptance,
            "stage_conclusion": stage_conclusion,
            "claim_boundary": [
                "The final test seed is never used for model selection or dictionary fitting before final evaluation.",
                "Review thresholds are selected on the validation seed before final test decisions are written.",
                "If acceptance criteria fail, this package must not claim complete ten-material automatic sorting.",
                "Dictionary distance features are fitted only from training or training+validation rows.",
                "Catalog fields such as formula, density, category, and group label are written for explanation only and are blocklisted from model features.",
            ],
        }
    )
    write_manifest(output_dir / "material_sorting_v2_manifest.json", manifest_base)
    print(final_summary.to_string(index=False))
    print(f"selected_method={selected_method}")
    print(f"stage_conclusion={stage_conclusion}")


if __name__ == "__main__":
    main()
