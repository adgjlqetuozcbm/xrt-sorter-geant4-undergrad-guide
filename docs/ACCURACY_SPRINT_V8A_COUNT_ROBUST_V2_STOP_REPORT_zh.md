# Accuracy Sprint v8A: count-robust v2 stop report

Date: 2026-05-06

## 1. Editorial decision

本阶段按长期计划实施了 `v8a_count_robust_v2`，但没有继续推进到
ordinary Phase 4、count-balanced retest、stability replication 或开发矩阵
大跑 preregistration。

原因不是主 H/M 信号消失。两个 v2 候选在 count-stratified development
view 中仍然给出很强的主模型 H/M 可观测性。真正的止损点是
shuffled-label sanity：假标签模型在 stress gate 和多种子复核中仍能超过
预设 ceiling。这意味着当前证据还不能排除 shortcut 或 null-audit protocol
问题，不能把结果升级为可审稿的稳健 development evidence。

Decision:

- stop v2 before Phase 4;
- do not run count-balanced retest from v2;
- do not run stability replication from v2;
- do not start H/M development matrix large run;
- do not open shadow/final;
- do not start full ten-material v8A matrix;
- do not claim product accuracy, hardware validation, or manuscript-grade
  powder-XRD simulation.

## 2. Implemented artifacts

New source-controlled scripts and configs:

- `analysis/build_v8a_count_robust_v2_features.py`
- `analysis/build_v8a_count_stratified_ordinary_view.py`
- `analysis/audit_v8a_multiseed_shuffled_label.py`
- `analysis/audit_v8a_count_robust_v2_candidates.py`
- `analysis/configs/v8a_count_robust_v2_proportion_only_count_stratified_stress_gate_config.json`
- `analysis/configs/v8a_count_robust_v2_residualized_no_absolute_windows_count_stratified_stress_gate_config.json`

Generated development evidence, left untracked by default:

- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_event_to_feature/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_count_stratified_event_to_feature/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_count_stratified_event_training/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_count_stratified_event_feature_stress_gate/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_multiseed_shuffled_label/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_event_to_feature/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_count_stratified_event_to_feature/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_count_stratified_event_training/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_count_stratified_event_feature_stress_gate/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_residualized_no_absolute_windows_multiseed_shuffled_label/`
- `results/accuracy_v3/v8a_count_robust_v2_candidate_review/`

## 3. Candidate results

| Candidate | Count-stratified support | Count gap | Main signal | Total-count control | Stress shuffled-label | Multi-seed shuffled-label | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `v2_proportion_only` | pairs `116/94/77` | max standardized gap `0.0163` | training main `1.0`, stress main `1.0` | stress worst `0.4681` | worst `0.5745` | validation/stress max `0.7234/0.6623` | stop before Phase 4 |
| `v2_residualized_no_absolute_windows` | pairs `116/94/77` | max standardized gap `0.0163` | training main `1.0`, stress main `1.0` | stress worst `0.4681` | worst `0.9362` | validation/stress max `0.6702/0.6234` | stop before Phase 4 |

Candidate review gate:

- file: `results/accuracy_v3/v8a_count_robust_v2_candidate_review/v8a_count_robust_v2_candidate_review_gate.json`
- decision: `stop_v8a_count_robust_v2_before_phase4`
- `phase4_unlocked`: `false`
- passed candidates: `0`
- stop reasons:
  - `no_v2_candidate_passed_pre_phase4_gates`
  - `multiseed_shuffled_label_sanity_failed`
  - `stress_gate_shuffled_label_sanity_failed`

## 4. Interpretation

The v2 attempt did make one thing cleaner: the count-stratified view reduced
the H/M total-count distribution gap to near zero while preserving support.
That part of the direction is useful.

However, a clean count gap is not enough. A shuffled-label model should behave
near chance. In this run, at least one shuffled seed and several stressed
variants performed above the accepted null ceiling. This is exactly the kind of
artifact a reviewer would challenge, because it can make a real-looking main
result less trustworthy.

The right interpretation is therefore:

- the diffraction-aware sidecar route remains alive;
- v2 did not generate promotion-grade development evidence;
- the next phase must diagnose null behavior and shortcut structure before any
  additional training expansion.

## 5. Next diagnostic phase

Do not start another feature-training loop blindly. The next phase should answer
which of these is happening:

- the features still contain shortcut structure that survives count
  stratification;
- the count-stratified subset has row-order, source, stress, or binning patterns
  that a tree model can exploit even with shuffled labels;
- the shuffled-label audit is too permissive because threshold/model selection
  is partly tuned on the same validation split;
- stress perturbations are creating artificial class-correlated artifacts in the
  null model path.

Recommended next artifacts:

- `diagnose_v8a_shuffled_label_null_behavior.py`
  - compare fixed threshold `0.5` versus validation-selected threshold;
  - run label permutation at row level and within split/thickness/pose/count-bin
    strata;
  - report null distribution, not only max seed;
  - compare Logistic versus ExtraTrees null behavior.
- `audit_v8a_feature_shortcut_structure.py`
  - inspect feature correlations with count bin, stress label, source id,
    thickness, pose, and combined feature origin;
  - run simple unsupervised cluster/source-origin separability checks;
  - identify feature families most associated with null-model success.
- `v8a_count_robust_v3_prereg`
  - only after the null diagnostic names a fixable mechanism.

## 6. Stop rules carried forward

Stop before Phase 4 if any candidate has:

- stress shuffled-label H/M min recall `>= 0.55`;
- multi-seed shuffled-label max H/M min recall `>= 0.55`;
- total-count-only H/M min recall `>= 0.60`;
- source-off H/M min recall `>= 0.60`;
- lineage-like main features;
- shadow/final, full ten-material, or existing XRT cube flags.

Only if a successor candidate passes pre-Phase-4 stress and multi-seed
shuffled-label checks should ordinary Phase 4 be rerun.

## 7. Plain-language summary

这轮我们没有白跑。我们把总计数分布压得很干净，主模型也还是能把 H/M
分开。

但问题是：把训练标签打乱以后，有些模型居然还能分得像样。这在审稿里是
危险信号，说明模型可能抓到了某种不该抓的规律，或者我们的假标签检查方式
本身太容易误报。

所以现在不能继续大跑，也不能继续 Phase 4。下一步不是加数据，而是先查清楚
“假标签为什么还能好”。查明白以后，再决定做 v3 特征还是改 gate protocol。
