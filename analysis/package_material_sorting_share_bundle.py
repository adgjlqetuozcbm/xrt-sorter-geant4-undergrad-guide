from __future__ import annotations

import argparse
import shutil
import tarfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOC_FILES = [
    "docs/G4_XRT_MATERIAL_SORTING_RESULTS_INDEX_zh.md",
    "docs/MATERIAL_SORTING_SHARE_GUIDE_zh.md",
    "docs/MATERIAL_SORTING_SHARE_PACKAGE_README_zh.md",
]

DOC_DIRS = [
    "docs/material_sorting_results_manifest",
]

ANALYSIS_FILES = [
    "analysis/build_v8a_ten_material_peak_manifest_from_rruff.py",
    "analysis/build_v8a_twenty_material_peak_manifest_from_rruff.py",
    "analysis/generate_v8a_multiclass_context_v6_physical_robust_matrix.py",
    "analysis/generate_v8a_multiclass_context_v7_twenty_material_scalability_scout_matrix.py",
    "analysis/generate_v8a_multiclass_context_v8_hard_negative_robust_matrix.py",
    "analysis/audit_v8a_multiclass_context_v6_training_data_final.py",
    "analysis/audit_v8a_multiclass_context_v6_skeptical_shortcut.py",
    "analysis/audit_v8a_multiclass_context_v7_hard_negative_diagnostics.py",
    "analysis/audit_v8a_multiclass_context_v7_twenty_material_scalability_scout_skeptical_shortcut.py",
    "analysis/audit_v8a_multiclass_context_v8_hard_negative_robust_diagnostics.py",
    "analysis/audit_v8a_multiclass_context_v8_hard_negative_robust_skeptical_shortcut.py",
    "analysis/audit_v8a_multiclass_context_v8_reporting_grade_extension_gate.py",
    "analysis/train_v8a_multiclass_context_v6_physical_robust.py",
    "analysis/train_v8a_multiclass_context_v7_twenty_material_scalability_scout.py",
    "analysis/train_v8a_multiclass_context_v8_hard_negative_robust.py",
    "analysis/train_v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate.py",
    "analysis/probe_v8_reporting_model_sweep.py",
    "analysis/probe_v8_ensemble_relaxed.py",
    "analysis/package_material_sorting_share_bundle.py",
]

CONFIG_FILES = [
    "analysis/configs/v8a_multiclass_context_v6_physical_robust_config.json",
    "analysis/configs/v8a_multiclass_context_v7_twenty_material_scalability_scout_config.json",
    "analysis/configs/v8a_multiclass_context_v8_hard_negative_robust_config.json",
    "source_models/config/diffraction_peak_tables/ten_material_powder_peaks_cif_or_literature_v8a_manifest.json",
    "source_models/config/diffraction_peak_tables/twenty_material_powder_peaks_rruff_v8a_manifest.json",
]

MATRIX_DIRS = [
    "source_models/config/material_sorting_matrix/v8a_multiclass_context_v6_physical_robust",
    "source_models/config/material_sorting_matrix/v8a_multiclass_context_v7_twenty_material_scalability_scout",
    "source_models/config/material_sorting_matrix/v8a_multiclass_context_v8_hard_negative_robust",
]

RESULT_DIRS = [
    "results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_event_to_feature",
    "results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_model",
    "results/accuracy_v3/v8a_multiclass_context_v6_skeptical_shortcut_audit",
    "results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_event_to_feature",
    "results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_model",
    "results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_hard_negative_diagnostics",
    "results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_skeptical_shortcut_audit",
    "results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_event_to_feature",
    "results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_model",
    "results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_hard_negative_diagnostics",
    "results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_skeptical_shortcut_audit",
    "results/accuracy_v3/v8a_multiclass_context_v8_reporting_grade_extension_gate",
    "results/accuracy_v3/v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate",
]

RUN_STATUS_FILES = [
    "results/material_sorting/run_status_v8a_multiclass_context_v6_physical_robust.csv",
    "results/material_sorting/run_status_v8a_multiclass_context_v7_twenty_material_scalability_scout.csv",
    "results/material_sorting/run_status_v8a_multiclass_context_v8_hard_negative_robust.csv",
]

SKIP_SUFFIXES = {
    ".pdparams",
    ".pkl",
    ".joblib",
}


def copy_file(rel: str, bundle_root: Path) -> None:
    src = ROOT / rel
    if not src.exists():
        return
    dst = bundle_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(rel: str, bundle_root: Path, include_all: bool = True) -> None:
    src = ROOT / rel
    if not src.exists():
        return
    dst = bundle_root / rel
    if dst.exists():
        shutil.rmtree(dst)
    if include_all:
        shutil.copytree(src, dst, ignore=ignore_generated_binaries)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("matrix_manifest.json", "material_sorting_matrix.csv"):
        item = src / name
        if item.exists():
            shutil.copy2(item, dst / name)


def ignore_generated_binaries(_: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if Path(name).suffix in SKIP_SUFFIXES:
            ignored.add(name)
    return ignored


def write_package_readme(bundle_root: Path) -> None:
    src = ROOT / "docs" / "MATERIAL_SORTING_SHARE_PACKAGE_README_zh.md"
    dst = bundle_root / "README_SHARE_PACKAGE_zh.md"
    if src.exists():
        shutil.copy2(src, dst)


def build_archive(bundle_root: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(bundle_root, arcname=bundle_root.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a shareable material-sorting result bundle.")
    parser.add_argument("--output-dir", default="share_packages")
    parser.add_argument("--name", default="")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = args.name or f"g4_xrt_material_sorting_share_{stamp}"
    out_dir = ROOT / args.output_dir
    bundle_root = out_dir / package_name
    archive_path = out_dir / f"{package_name}.tar.gz"

    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    write_package_readme(bundle_root)
    for rel in DOC_FILES + ANALYSIS_FILES + CONFIG_FILES + RUN_STATUS_FILES:
        copy_file(rel, bundle_root)
    for rel in DOC_DIRS:
        copy_tree(rel, bundle_root)
    for rel in MATRIX_DIRS:
        copy_tree(rel, bundle_root, include_all=False)
    for rel in RESULT_DIRS:
        copy_tree(rel, bundle_root)

    if not args.no_archive:
        build_archive(bundle_root, archive_path)
        print(archive_path)
    else:
        print(bundle_root)


if __name__ == "__main__":
    main()
