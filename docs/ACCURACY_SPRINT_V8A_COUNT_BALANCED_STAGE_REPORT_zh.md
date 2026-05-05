# Accuracy Sprint v8A: count-balanced development stage report

Date: 2026-05-06

## 1. Editorial decision

The v8A count-overlap extension and the formal count-balanced retest support a
narrow positive conclusion:

- H/M diffraction-aware sidecar observability remains visible in development
  data after a strict total-count balancing retest.

They do not support a broader promotion:

- ordinary XRT does not solve H/M;
- the ordinary medium-plus-extension Phase 4 model gate has not passed;
- shadow/final remains sealed;
- full ten-material v8A matrix remains locked;
- no product accuracy, hardware validation, or manuscript-grade powder-XRD
  claim is supported.

The correct next action is feature rework against total-count shortcutting, not
shadow/final validation or full-matrix expansion.

## 2. Evidence ledger

| Evidence item | Result | Interpretation |
| --- | --- | --- |
| Count-overlap extension run | `672/672` completed, `0` failed | The preregistered development-only source-on extension is available. |
| Extension event-to-feature | `672` samples, `137990` sidecar rows | Feature extraction works; extension-only gate fails as expected because the extension has no source-off controls. |
| Medium-plus-extension combiner | `1536` samples | Existing medium source-off controls were preserved while adding extension source-on rows. |
| Combined stress gate | passed; worst main H/M min recall `1.0`; worst total-count control `0.5104`; worst overlap-only `0.5104`; worst source-off `0.4722` | The stressed development feature table remains usable for controlled retesting. |
| Ordinary combined Phase 4 | failed; main validation/stress H/M `1.0/1.0`; total-count-only H/M `0.7344` | The broad ordinary model is still confounded by total-count shortcutting. |
| Count-matched rework | failed; pairs `280/176/137`; total-count-only max `0.7273` | More support was obtained, but the ordinary matched gate still does not isolate the diffraction signal. |
| Count-balance sensitivity | passed; `7` supported strategies | A strict count-balanced retest is supportable in the expanded development data. |
| Formal count-balanced retest | passed for `fixed_bin_width_0p003`; pairs `116/94/77`; main H/M `1.0/1.0`; total-count-only H/M `0.5745` | Accepted as the current strongest development-only positive result. |

Key generated evidence paths:

- `results/material_sorting/run_status_v8a_hm_count_overlap_extension_cif_literature.csv`
- `results/accuracy_v3/v8a_count_overlap_extension_event_to_feature/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_event_to_feature/v8a_event_schema_gate.json`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_event_feature_stress_gate/v8a_event_feature_stress_gate.json`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_development_model/v8a_medium_development_model_gate.json`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_matched_rework/v8a_medium_count_matched_rework_gate.json`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_balance_sensitivity/v8a_count_balance_sensitivity_gate.json`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_balanced_retest/v8a_count_balanced_retest_gate.json`

Generated evidence remains untracked unless a later packaging step explicitly
selects compact report artifacts for release.

## 3. What passed

The extension achieved its narrow purpose. The previous count-balanced support
problem was real, and the extension supplied enough development-only rows to
run a strict selected balancing strategy with useful support:

- train matched pairs: `116`;
- validation matched pairs: `94`;
- stress-holdout matched pairs: `77`;
- selected strategy: `fixed_bin_width_0p003`.

Under that strict retest, the main diffraction-aware model kept H/M min recall
at `1.0` on both validation and stress-holdout, while the total-count-only
control remained below the `<0.60` ceiling at `0.5745`.

This means the signal is not only a trivial total-count artifact under the
selected count-balanced development retest.

## 4. What did not pass

The ordinary medium-plus-extension Phase 4 gate still failed. Its main model
looked strong, but the control model using only total-count information reached
H/M min recall `0.7344`, above the allowed `<0.60` ceiling.

The count-matched rework also failed. Even after improved matched support, the
total-count-only max H/M min recall remained `0.7273`, and the main-minus-total
margin was not acceptable.

This is the key scientific blocker: the ordinary feature table still contains
enough count-correlated structure that a reviewer could argue the classifier is
using a shortcut rather than robust diffraction-aware observability.

## 5. Why we cannot expand the run yet

Starting shadow/final or a full ten-material v8A matrix now would burn sealed
evidence before the development controls are clean. The ordinary gate has a
known confounder, and the accepted positive result is conditional on a strict
count-balanced retest.

The current result is valuable, but it is not yet a product validation result.
It is closer to a development checkpoint saying:

- the route is still alive;
- the signal survives a serious balancing test;
- the broader feature design must be cleaned before promotion.

## 6. Next rework plan

### R1. Total-count anatomy review

Create a development-only diagnostic that explains where the total-count
shortcut enters the ordinary feature table.

Required outputs:

- per-material total-count distributions by split, thickness, pose, source mode,
  and stress label;
- feature-to-total-count correlation summary for every `diffraction_*` feature;
- ranked feature list showing which main features are most count-like;
- report decision: `feature_rework_needed`, `feature_rework_not_needed`, or
  `stop_sidecar_claim_too_count_coupled`.

No model promotion is allowed from this diagnostic alone.

### R2. Split-safe feature rework

Implement count-robust candidate feature families without using validation or
stress-holdout labels for fitting transformations.

Allowed candidates:

- within-window normalized peak proportions;
- within-material-agnostic peak ratio features;
- source-on minus source-off-style residual features only where lineage is
  explicit and source-off controls remain sealed from main fitting;
- train-split-only total-count residualization;
- removal or down-weighting of absolute count-like peak totals from the main
  feature family.

Disallowed candidates:

- using material labels to fit residualization outside the train split;
- using validation/stress rows to choose feature transformations;
- adding `material`, `source_id`, `sample_id`, `seed`, `thickness`, `pose`,
  `split`, row path, or run lineage to the main features;
- hiding total count inside renamed features.

### R3. Ordinary Phase 4 retest

Rerun the ordinary medium-plus-extension Phase 4 gate on the reworked feature
table.

Acceptance thresholds:

- validation H/M min recall `>= 0.95`;
- stress-holdout H/M min recall `>= 0.90`;
- worst-thickness H/M min recall `>= 0.90`;
- worst-pose H/M min recall `>= 0.90`;
- total-count-only H/M min recall `< 0.60`;
- overlap-only H/M min recall `< 0.60`;
- source-off H/M min recall `< 0.60`;
- shuffled-label H/M min recall `< 0.55`;
- main-minus-source-off margin `>= 0.35`.

The count-balanced retest must also be rerun after feature rework. Passing only
the ordinary gate or only the count-balanced gate is not enough for promotion.

### R4. Promotion decision

Only if both gates pass:

- ordinary reworked Phase 4 gate;
- strict count-balanced development retest.

Then the next allowed planning step is a larger H/M development-only matrix or
a frozen preregistration for the next development matrix. Shadow/final remains
sealed until the development evidence is stable and preregistered.

If either gate fails, the route stays in rework. The report should say the
sidecar signal is promising but still not clean enough to support promotion.

## 7. Claim boundary for the next report

Allowed wording:

- "development-only diffraction-aware sidecar observability";
- "strict count-balanced H/M retest passed in development data";
- "ordinary medium-plus-extension Phase 4 remains blocked by total-count
  confounding";
- "the next step is feature rework and renewed development controls."

Forbidden wording:

- "ordinary XRT solves H/M";
- "product accuracy";
- "hardware validation";
- "shadow/final validated";
- "full ten-material v8A success";
- "publishable powder-XRD simulation";
- "Geant4 process-level powder diffraction has been validated."

## 8. Plain-language summary

The extension helped. It gave us enough extra development data to ask a harder
question: "If H/M are balanced by total count, does the diffraction-aware signal
still work?" The answer was yes.

But the broader ordinary model is still not clean. A model that sees only total
count can still separate H/M too well. That means a reviewer can still say:
"Your model may be using a shortcut." Because of that, we cannot responsibly
start shadow/final or a big matrix run yet.

The next job is to redesign the features so the main model cannot lean on total
count, then rerun both gates: the ordinary Phase 4 gate and the strict
count-balanced retest.
