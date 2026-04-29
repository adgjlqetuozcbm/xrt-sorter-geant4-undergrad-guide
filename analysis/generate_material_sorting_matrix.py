from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


MATERIALS_FILE = Path("source_models/materials/material_catalog.csv")
OUTPUT_ROOT = Path("source_models/config/material_sorting_matrix")
THICKNESS_MM = [5.0, 10.0, 20.0]
SEEDS = [101, 202, 303]
SOURCES = [
    {"source_id": "mono_60kev", "source_mode": "mono", "mono_energy_keV": 60.0},
    {"source_id": "mono_100kev", "source_mode": "mono", "mono_energy_keV": 100.0},
    {
        "source_id": "spectrum_120kv",
        "source_mode": "spectrum",
        "mono_energy_keV": 80.0,
        "spectrum_file": "../../../spectra/w_target_120kV_1mmAl.csv",
    },
]
ENERGY_SCAN_SOURCES = [
    {"source_id": f"mono_{energy}kev", "source_mode": "mono", "mono_energy_keV": float(energy)}
    for energy in [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200]
]
ACCURACY_V3_LOW_ENERGY_SOURCES = [
    {"source_id": f"mono_{energy}kev", "source_mode": "mono", "mono_energy_keV": float(energy)}
    for energy in [15, 20, 25, 35]
]
SELECTED_REBUILD_SOURCE_IDS = ["mono_40kev", "mono_50kev", "mono_120kev"]
ACCURACY_V3_SOURCE_IDS = [
    "mono_30kev",
    "mono_40kev",
    "mono_50kev",
    "mono_70kev",
    "mono_90kev",
    "mono_110kev",
    "mono_120kev",
    "mono_150kev",
    "mono_200kev",
]
ACCURACY_V3_TRAIN_SEEDS = list(range(1201, 1221))
ACCURACY_V3_VALIDATION_SEEDS = list(range(1301, 1306))
ACCURACY_V3_FINAL_TEST_SEEDS = list(range(1401, 1406))
ACCURACY_V3_HM_MATERIALS = ["Hematite", "Magnetite", "Pyrite", "Chalcopyrite"]
PROFILE_EVENTS = {
    "pilot": 2000,
    "energy_scan": 5000,
    "full": 10000,
    "selected_rebuild": 10000,
    "accuracy_v3_hm": 10000,
    "accuracy_v3": 10000,
    "v6c_hm_source_design": 10000,
}
V6C_HM_MATERIALS = ["Hematite", "Magnetite"]
V6C_HM_ENERGIES_KEV = [50, 70, 90, 120, 150, 200]
V6C_HM_THICKNESS_MM = [5.0, 10.0, 15.0, 20.0, 30.0]
V6C_HM_SEEDS = [*range(2101, 2113), *range(2201, 2207), *range(2301, 2307)]
V6C_HM_SOURCE_VARIANTS = ["normal_narrow", "normal_wide", "oblique_10deg"]


def energy_slug(energy_keV: float) -> str:
    if float(energy_keV).is_integer():
        return str(int(energy_keV))
    return f"{energy_keV:g}".replace(".", "p")


def mono_source(energy_keV: float) -> dict:
    return {"source_id": f"mono_{energy_slug(energy_keV)}kev", "source_mode": "mono", "mono_energy_keV": float(energy_keV)}


def source_variant_config(variant: str) -> dict:
    if variant == "normal_narrow":
        return {
            "source_variant": variant,
            "beam_half_y_mm": 5.0,
            "beam_half_z_mm": 5.0,
            "source_y_mm": 0.0,
            "source_z_mm": 0.0,
            "dir_x": 1.0,
            "dir_y": 0.0,
            "dir_z": 0.0,
            "incidence_angle_deg": 0.0,
            "detector_layout": "transmission_plus_side_scatter",
        }
    if variant == "normal_wide":
        return {
            "source_variant": variant,
            "beam_half_y_mm": 20.0,
            "beam_half_z_mm": 20.0,
            "source_y_mm": 0.0,
            "source_z_mm": 0.0,
            "dir_x": 1.0,
            "dir_y": 0.0,
            "dir_z": 0.0,
            "incidence_angle_deg": 0.0,
            "detector_layout": "transmission_plus_side_scatter",
        }
    if variant == "oblique_10deg":
        angle_rad = math.radians(10.0)
        source_x_mm = -300.0
        source_y_mm = -abs(source_x_mm) * math.tan(angle_rad)
        return {
            "source_variant": variant,
            "beam_half_y_mm": 5.0,
            "beam_half_z_mm": 5.0,
            "source_y_mm": source_y_mm,
            "source_z_mm": 0.0,
            "dir_x": math.cos(angle_rad),
            "dir_y": math.sin(angle_rad),
            "dir_z": 0.0,
            "incidence_angle_deg": 10.0,
            "detector_layout": "transmission_plus_side_scatter",
        }
    raise ValueError(f"Unknown source variant: {variant}")


def v6c_source_id(energy_keV: float, variant: str) -> str:
    return f"mono_{energy_slug(energy_keV)}kev_{variant}"


def v6c_sources(
    energy_list_keV: list[float] | None = None,
    source_variants: list[str] | None = None,
) -> list[dict]:
    energies = energy_list_keV or [float(energy) for energy in V6C_HM_ENERGIES_KEV]
    variants = source_variants or V6C_HM_SOURCE_VARIANTS
    sources = []
    for energy in energies:
        for variant in variants:
            source = mono_source(float(energy))
            source.update(source_variant_config(variant))
            source["source_id"] = v6c_source_id(float(energy), variant)
            sources.append(source)
    return sources


def source_lookup() -> dict[str, dict]:
    return {source["source_id"]: source for source in [*SOURCES, *ENERGY_SCAN_SOURCES, *ACCURACY_V3_LOW_ENERGY_SOURCES]}


def profile_thicknesses(profile: str) -> list[float]:
    if profile == "energy_scan":
        return [10.0]
    if profile == "v6c_hm_source_design":
        return V6C_HM_THICKNESS_MM
    return THICKNESS_MM


def profile_seeds(profile: str) -> list[int]:
    if profile == "energy_scan":
        return [101, 202]
    if profile == "selected_rebuild":
        return [101, 202, 303, 404, 505]
    if profile in {"accuracy_v3_hm", "accuracy_v3"}:
        return [*ACCURACY_V3_TRAIN_SEEDS, *ACCURACY_V3_VALIDATION_SEEDS, *ACCURACY_V3_FINAL_TEST_SEEDS]
    if profile == "v6c_hm_source_design":
        return V6C_HM_SEEDS
    return SEEDS


def profile_sources(
    profile: str,
    selected_source_ids: list[str] | None = None,
    energy_list_keV: list[float] | None = None,
    source_variants: list[str] | None = None,
) -> list[dict]:
    if profile == "v6c_hm_source_design":
        return v6c_sources(energy_list_keV, source_variants)
    if energy_list_keV:
        return [mono_source(energy) for energy in energy_list_keV]
    if profile == "energy_scan":
        return ENERGY_SCAN_SOURCES
    if profile in {"selected_rebuild", "accuracy_v3_hm", "accuracy_v3"}:
        lookup = source_lookup()
        if profile == "selected_rebuild":
            source_ids = selected_source_ids or SELECTED_REBUILD_SOURCE_IDS
        else:
            source_ids = selected_source_ids or ACCURACY_V3_SOURCE_IDS
        missing = [source_id for source_id in source_ids if source_id not in lookup]
        if missing:
            raise ValueError(f"Unknown selected source ids: {missing}")
        return [lookup[source_id] for source_id in source_ids]
    return SOURCES


def profile_material_names(profile: str, material_names: list[str] | None = None) -> list[str] | None:
    if material_names:
        return material_names
    if profile == "accuracy_v3_hm":
        return ACCURACY_V3_HM_MATERIALS
    if profile == "v6c_hm_source_design":
        return V6C_HM_MATERIALS
    return None


@dataclass(frozen=True)
class MatrixRun:
    profile: str
    run_role: str
    material_name: str
    material_slug: str
    source_id: str
    source_mode: str
    source_variant: str
    detector_layout: str
    mono_energy_keV: float
    beam_half_y_mm: float
    beam_half_z_mm: float
    source_y_mm: float
    source_z_mm: float
    dir_x: float
    dir_y: float
    dir_z: float
    incidence_angle_deg: float
    thickness_mm: float
    random_seed: int
    config_path: Path
    output_prefix: str
    output_dir: str
    expected_events: int


@dataclass(frozen=True)
class MaterialRow:
    material_name: str


def slugify(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def load_materials(project_root: Path, material_names: list[str] | None = None) -> list[MaterialRow]:
    requested = set(material_names or [])
    enabled: list[MaterialRow] = []
    with (project_root / MATERIALS_FILE).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            is_enabled = str(row.get("enabled_for_undergrad", "")).strip().lower() in {"true", "1", "yes"}
            material_name = str(row.get("material_name", "")).strip()
            if not is_enabled or not material_name:
                continue
            if requested and material_name not in requested:
                continue
            enabled.append(MaterialRow(material_name=material_name))
    if requested:
        missing = sorted(requested - {row.material_name for row in enabled})
        if missing:
            raise ValueError(f"Requested materials are missing or not enabled: {missing}")
    if not enabled:
        raise ValueError("No enabled materials in material catalog.")
    return enabled


def render_config(run: MatrixRun, source: dict) -> str:
    spectrum_file = source.get("spectrum_file", "../../../spectra/w_target_120kV_1mmAl.csv")
    if run.run_role == "material":
        label = (
            f"material_sorting_{run.profile}_{run.material_slug}_{run.source_id}_"
            f"{int(run.thickness_mm)}mm_seed{run.random_seed}"
        )
    else:
        label = f"material_sorting_{run.profile}_calibration_{run.source_id}_seed{run.random_seed}"
    ore_material_mode = "air_path" if run.run_role == "calibration" else "single"
    ore_primary_material = "AIR_PATH" if run.run_role == "calibration" else run.material_name
    ore_thickness_mm = 1.0 if run.run_role == "calibration" else run.thickness_mm
    return f"""# Material sorting matrix config: {label}
run_id = {label}
experiment_label = {label}
output_prefix = {run.output_prefix}
output_dir = {run.output_dir}
benchmark_suite = material_sorting_v1
research_route = undergraduate_sorting_upgrade
prediction_stage = material_level
run_role = {run.run_role}
prep_profile = clean_dry_single_piece
feed_size_band = controlled_slab
feed_condition = simulated_single_material
sample_photons = 100
random_seed = {run.random_seed}

source_mode = {run.source_mode}
source_variant = {run.source_variant}
detector_layout = {run.detector_layout}
spectrum_file = {spectrum_file}
phase_space_file = ../../../phase_space/source_phase_space.csv
mono_energy_keV = {run.mono_energy_keV:.1f}
source_x_cm = -30.0
source_y_mm = {run.source_y_mm:.6f}
source_z_mm = {run.source_z_mm:.6f}
beam_half_y_mm = {run.beam_half_y_mm:.6f}
beam_half_z_mm = {run.beam_half_z_mm:.6f}
dir_x = {run.dir_x:.8f}
dir_y = {run.dir_y:.8f}
dir_z = {run.dir_z:.8f}
incidence_angle_deg = {run.incidence_angle_deg:.3f}

ore_material_mode = {ore_material_mode}
ore_primary_material = {ore_primary_material}
ore_secondary_material = Magnetite
ore_secondary_mass_fraction = 0.0

ore_shape = slab
ore_thickness_mm = {ore_thickness_mm:.1f}
ore_half_y_mm = 100.0
ore_half_z_mm = 100.0

heterogeneity_mode = none
inclusion_material = Magnetite
inclusion_shape = ellipsoid
inclusion_thickness_mm = 4.0
inclusion_radius_y_mm = 20.0
inclusion_radius_z_mm = 20.0
inclusion_offset_y_mm = 0.0
inclusion_offset_z_mm = 0.0

detector_thickness_mm = 5.0
detector_half_y_mm = 100.0
detector_half_z_mm = 100.0
detector_x_cm = 25.0
side_detector_thickness_mm = 5.0
side_detector_half_x_mm = 120.0
side_detector_half_z_mm = 100.0
side_detector_y_cm = 12.0

world_x_cm = 100.0
world_y_cm = 50.0
world_z_cm = 50.0
envelope_x_cm = 80.0
envelope_y_cm = 30.0
envelope_z_cm = 30.0
"""


def build_matrix(
    project_root: Path,
    profile: str,
    selected_source_ids: list[str] | None = None,
    profile_alias: str | None = None,
    seed_list: list[int] | None = None,
    events_per_run: int | None = None,
    material_names: list[str] | None = None,
    energy_list_keV: list[float] | None = None,
    thickness_list: list[float] | None = None,
    source_variants: list[str] | None = None,
) -> list[MatrixRun]:
    materials = load_materials(project_root, profile_material_names(profile, material_names))
    profile_name = profile_alias or profile
    expected_events = events_per_run or PROFILE_EVENTS[profile]
    out_dir = OUTPUT_ROOT / profile_name
    raw_output_dir = f"material_sorting_runs/{profile_name}"
    runs: list[MatrixRun] = []
    sources = profile_sources(profile, selected_source_ids, energy_list_keV, source_variants)
    thicknesses = thickness_list or profile_thicknesses(profile)
    seeds = seed_list or profile_seeds(profile)
    for source in sources:
        for seed in seeds:
            output_prefix = f"ms_{profile_name}_calibration_{source['source_id']}_seed{seed}"
            config_path = out_dir / f"{output_prefix}.txt"
            runs.append(
                MatrixRun(
                    profile=profile_name,
                    run_role="calibration",
                    material_name="AIR_PATH",
                    material_slug="calibration",
                    source_id=source["source_id"],
                    source_mode=source["source_mode"],
                    source_variant=source.get("source_variant", "normal_narrow"),
                    detector_layout=source.get("detector_layout", "transmission"),
                    mono_energy_keV=float(source["mono_energy_keV"]),
                    beam_half_y_mm=float(source.get("beam_half_y_mm", 5.0)),
                    beam_half_z_mm=float(source.get("beam_half_z_mm", 5.0)),
                    source_y_mm=float(source.get("source_y_mm", 0.0)),
                    source_z_mm=float(source.get("source_z_mm", 0.0)),
                    dir_x=float(source.get("dir_x", 1.0)),
                    dir_y=float(source.get("dir_y", 0.0)),
                    dir_z=float(source.get("dir_z", 0.0)),
                    incidence_angle_deg=float(source.get("incidence_angle_deg", 0.0)),
                    thickness_mm=1.0,
                    random_seed=seed,
                    config_path=config_path,
                    output_prefix=output_prefix,
                    output_dir=raw_output_dir,
                    expected_events=expected_events,
                )
            )
    for row in materials:
        material_slug = slugify(str(row.material_name))
        for thickness in thicknesses:
            for source in sources:
                for seed in seeds:
                    output_prefix = (
                        f"ms_{profile_name}_{material_slug}_{source['source_id']}_"
                        f"{int(thickness)}mm_seed{seed}"
                    )
                    config_path = out_dir / f"{output_prefix}.txt"
                    runs.append(
                        MatrixRun(
                            profile=profile_name,
                            run_role="material",
                            material_name=str(row.material_name),
                            material_slug=material_slug,
                            source_id=source["source_id"],
                            source_mode=source["source_mode"],
                            source_variant=source.get("source_variant", "normal_narrow"),
                            detector_layout=source.get("detector_layout", "transmission"),
                            mono_energy_keV=float(source["mono_energy_keV"]),
                            beam_half_y_mm=float(source.get("beam_half_y_mm", 5.0)),
                            beam_half_z_mm=float(source.get("beam_half_z_mm", 5.0)),
                            source_y_mm=float(source.get("source_y_mm", 0.0)),
                            source_z_mm=float(source.get("source_z_mm", 0.0)),
                            dir_x=float(source.get("dir_x", 1.0)),
                            dir_y=float(source.get("dir_y", 0.0)),
                            dir_z=float(source.get("dir_z", 0.0)),
                            incidence_angle_deg=float(source.get("incidence_angle_deg", 0.0)),
                            thickness_mm=thickness,
                            random_seed=seed,
                            config_path=config_path,
                            output_prefix=output_prefix,
                            output_dir=raw_output_dir,
                            expected_events=expected_events,
                        )
                    )
    return runs


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_matrix(
    project_root: Path,
    profile: str,
    selected_source_ids: list[str] | None = None,
    profile_alias: str | None = None,
    seed_list: list[int] | None = None,
    events_per_run: int | None = None,
    material_names: list[str] | None = None,
    energy_list_keV: list[float] | None = None,
    thickness_list: list[float] | None = None,
    source_variants: list[str] | None = None,
) -> Path:
    profile_name = profile_alias or profile
    runs = build_matrix(
        project_root,
        profile,
        selected_source_ids,
        profile_name,
        seed_list,
        events_per_run,
        material_names,
        energy_list_keV,
        thickness_list,
        source_variants,
    )
    out_dir = project_root / OUTPUT_ROOT / profile_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    by_key = {
        (source["source_id"], source["source_mode"]): source
        for source in profile_sources(profile, selected_source_ids, energy_list_keV, source_variants)
    }
    for run in runs:
        source = by_key[(run.source_id, run.source_mode)]
        with (project_root / run.config_path).open("w", encoding="utf-8", newline="\n") as f:
            f.write(render_config(run, source))
        rows.append(
            {
                "profile": run.profile,
                "run_role": run.run_role,
                "material": run.material_name,
                "source_id": run.source_id,
                "source_mode": run.source_mode,
                "source_variant": run.source_variant,
                "detector_layout": run.detector_layout,
                "mono_energy_keV": run.mono_energy_keV,
                "incidence_angle_deg": run.incidence_angle_deg,
                "beam_half_y_mm": run.beam_half_y_mm,
                "beam_half_z_mm": run.beam_half_z_mm,
                "thickness_mm": run.thickness_mm,
                "random_seed": run.random_seed,
                "expected_events": run.expected_events,
                "config_path": run.config_path.as_posix(),
                "output_prefix": run.output_prefix,
                "output_dir": run.output_dir,
            }
        )

    matrix_path = out_dir / "material_sorting_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return matrix_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate material-level sorting configs.")
    parser.add_argument("--profile", choices=sorted(PROFILE_EVENTS), default="pilot")
    parser.add_argument(
        "--selected-source-ids",
        default="",
        help="Comma-separated source ids for selected_rebuild or accuracy_v3 profiles. Defaults are profile-specific.",
    )
    parser.add_argument(
        "--energy-list-kev",
        default="",
        help="Comma-separated mono energies in keV. Overrides selected source ids and built-in profile source defaults.",
    )
    parser.add_argument(
        "--source-variant-list",
        default="",
        help="Comma-separated v6c source variants. Defaults to normal_narrow,normal_wide,oblique_10deg for v6c.",
    )
    parser.add_argument(
        "--thickness-list",
        default="",
        help="Comma-separated thickness values in mm. Overrides profile thickness defaults.",
    )
    parser.add_argument(
        "--profile-alias",
        default="",
        help="Output profile name. Use this for locked reruns such as selected_rebuild_r2 without overwriting prior evidence.",
    )
    parser.add_argument(
        "--seed-list",
        default="",
        help="Comma-separated random seeds. Defaults to the selected profile's standard seeds.",
    )
    parser.add_argument(
        "--events-per-run",
        type=int,
        default=0,
        help="Override macro/event expectation recorded in the matrix. The runner macro must still beamOn the same count.",
    )
    parser.add_argument(
        "--material-list",
        default="",
        help="Comma-separated material names or 'all'. accuracy_v3_hm defaults to Hematite/Magnetite/Pyrite/Chalcopyrite.",
    )
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    profile_alias = args.profile_alias.strip() or None
    seed_list = parse_int_list(args.seed_list) if args.seed_list.strip() else None
    events_per_run = args.events_per_run or None
    selected_source_ids = [item.strip() for item in str(args.selected_source_ids).split(",") if item.strip()] or None
    energy_list_keV = parse_float_list(args.energy_list_kev) if args.energy_list_kev.strip() else None
    thickness_list = parse_float_list(args.thickness_list) if args.thickness_list.strip() else None
    source_variants = [item.strip() for item in args.source_variant_list.split(",") if item.strip()] or None
    material_names = None
    if args.material_list.strip() and args.material_list.strip().lower() != "all":
        material_names = [item.strip() for item in args.material_list.split(",") if item.strip()]
    matrix_path = write_matrix(
        project_root,
        args.profile,
        selected_source_ids if args.profile in {"selected_rebuild", "accuracy_v3_hm", "accuracy_v3"} else None,
        profile_alias=profile_alias,
        seed_list=seed_list,
        events_per_run=events_per_run,
        material_names=material_names,
        energy_list_keV=energy_list_keV,
        thickness_list=thickness_list,
        source_variants=source_variants,
    )
    runs = list(csv.DictReader(matrix_path.open(encoding="utf-8")))
    profile_name = profile_alias or args.profile
    print(f"Wrote {len(runs)} {profile_name} configs to {matrix_path}")
    sources = profile_sources(
        args.profile,
        selected_source_ids if args.profile in {"selected_rebuild", "accuracy_v3_hm", "accuracy_v3"} else None,
        energy_list_keV,
        source_variants,
    )
    thicknesses = thickness_list or profile_thicknesses(args.profile)
    seeds = seed_list or profile_seeds(args.profile)
    material_count = len(load_materials(project_root, profile_material_names(args.profile, material_names)))
    print(
        "Expected material matrix: "
        f"{material_count} materials x {len(thicknesses)} thicknesses x {len(sources)} sources x {len(seeds)} seeds "
        f"= {material_count * len(thicknesses) * len(sources) * len(seeds)} runs"
    )
    print(f"Expected calibration matrix: {len(sources)} sources x {len(seeds)} seeds = {len(sources) * len(seeds)} runs")


if __name__ == "__main__":
    main()
