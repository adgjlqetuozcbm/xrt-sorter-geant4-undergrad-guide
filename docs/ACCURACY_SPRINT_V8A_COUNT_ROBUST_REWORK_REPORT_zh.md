# Accuracy Sprint v8A: count-robust rework and matrix-admission report

Date: 2026-05-06

## 1. Editorial decision

The long-plan implementation reached a clear stop point. The new diagnostics
and gates confirm that the current development evidence is not yet ready for an
H/M development matrix large run preregistration.

The important result is not a failure of the diffraction-aware route. The main
count-robust model still separates H/M strongly in development data. The blocker
is that the ordinary development distribution still lets total-count-only
controls perform too well, and the strict count-balanced retest now also trips a
shuffled-label control.

Decision:

- continue feature/data-distribution rework;
- do not start H/M development matrix large run;
- do not open shadow/final;
- do not start full ten-material v8A matrix;
- do not claim product accuracy, hardware validation, or manuscript-grade
  powder-XRD simulation.

## 2. Implemented artifacts

New source-controlled scripts and config:

- `analysis/diagnose_v8a_total_count_anatomy.py`
- `analysis/build_v8a_count_robust_features.py`
- `analysis/audit_v8a_count_robust_stability.py`
- `analysis/audit_v8a_development_matrix_admission.py`
- `analysis/configs/v8a_count_robust_v1_stress_gate_config.json`

New generated development evidence directories:

- `results/accuracy_v3/v8a_medium_plus_count_overlap_total_count_anatomy/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_event_to_feature/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_event_training/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_event_feature_stress_gate/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_development_model/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_count_matched_rework/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_count_balance_sensitivity/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v1_count_balanced_retest/`
- `results/accuracy_v3/v8a_development_matrix_admission/`

Generated outputs remain untracked by default.

## 3. Gate results

| Gate | Result | Interpretation |
| --- | --- | --- |
| Total-count anatomy | `feature_rework_needed`; Phase 4 total-count-only H/M `0.7344`; max standardized total-count gap `1.8258`; max main-feature correlation `0.6469` | Confounding is real and is partly a data-distribution issue. |
| Count-robust feature transform | passed schema; `1536` samples; `41` robust main features; `20` absolute count-like source features removed | New feature directory is usable by existing gates. |
| Count-robust baseline training | passed; main H/M `1.0`; source-off H/M `0.25` | The main signal remains visible after feature rework. |
| Count-robust stress gate | passed; worst main `1.0`; overlap-only `0.5104`; source-off `0.2778` | Stress behavior is acceptable at this stage. |
| Count-robust ordinary Phase 4 | failed; main validation/stress `1.0/1.0`; total-count-only `0.7344` | Ordinary gate remains blocked by total-count-only control. |
| Count-robust count-matched rework | failed; pairs `280/176/137`; total-count-only max `0.7273` | Matching alone still does not remove the shortcut. |
| Count-robust count-balance sensitivity | passed; `7` supported strategies | Strict balancing remains feasible. |
| Count-robust count-balanced retest | failed; main `1.0/1.0`; total-count-only `0.5745`; shuffled-label `0.5745` | Main and total-count checks are acceptable, but shuffled-label sanity blocks acceptance. |
| Development matrix admission | failed; preregistration unlocked `false` | Large H/M development matrix is not allowed yet. |

## 4. What this means

The new `v8a_count_robust_v1` feature family did something useful: it preserved
the main H/M signal and reduced source-off behavior. However, it did not solve
the ordinary total-count confounder. The same total-count-only control remains
too predictive in the ordinary Phase 4 validation setting.

This strongly suggests the next rework should address both feature design and
matrix distribution, not only feature naming. In practical terms, the next
candidate should reduce count separability before the model stage, not rely on
the model to ignore it.

## 5. Next rework: v8A count-robust v2

The next phase should not enlarge the matrix yet. It should build a second
development-only rework with stricter count distribution handling:

- create a train/validation/stress count-stratified source-on sampling layer for
  the ordinary gate, not only the count-balanced retest;
- remove `control_high_angle_primary_norm`, `control_direct_primary_norm`, and
  `control_scattered_primary_norm` from total-count-only controls only if a
  separate preregistered diagnostic proves they are not total-count proxies;
- add stricter shuffled-label repetition across at least three shuffle seeds;
- compare two candidate transforms:
  - `v8a_count_robust_v2_proportion_only`;
  - `v8a_count_robust_v2_train_residualized_without_absolute_windows`;
- require ordinary Phase 4 and count-balanced retest to pass before stability
  audit;
- run development matrix admission audit only after both gates pass.

## 6. Stop rules

Stop before any development matrix large run if any of the following holds:

- ordinary Phase 4 fails;
- count-balanced retest fails;
- shuffled-label H/M min recall is `>= 0.55`;
- total-count-only H/M min recall is `>= 0.60`;
- source-off H/M min recall is `>= 0.60`;
- lineage-like main features appear;
- shadow/final or full ten-material flags appear.

## 7. Claim boundary

Allowed wording:

- "count-robust v1 preserved development-only H/M sidecar signal";
- "ordinary Phase 4 remains blocked by total-count-only control";
- "count-balanced retest is not accepted after v1 because shuffled-label sanity failed";
- "development matrix preregistration remains locked."

Forbidden wording:

- "ordinary XRT solves H/M";
- "development matrix large run is unlocked";
- "shadow/final can be opened";
- "product accuracy";
- "hardware validation";
- "publishable powder-XRD simulation."

## 8. Plain-language summary

We built the next gate system and tried the first count-robust feature version.
The good news is that the real H/M signal did not disappear. The bad news is
that the old shortcut did not disappear either: total count alone still performs
too well in the ordinary gate.

So we are not allowed to start the large development matrix yet. The next move
is a v2 rework that changes the sampling/distribution logic as well as the
features.
