# Accuracy Sprint v8A: crystal-clean stop report

Date: 2026-05-06

## 1. Editorial decision

本阶段没有进入任何训练扩展。我们先尝试把当前 v2 feature/view 清理成“只让模型学习晶体差异”的
development-only 视图，并把训练准入改成硬门：

- 先过滤掉人工 stress 和 source-off，只保留 `custom_diffraction_on + default`；
- 在 split 内按 H/M 做精确配对；
- source/origin/count/thickness/pose 只作为 nuisance control，不作为主特征；
- `diffraction_*` 主特征被转换成 train-split-only nuisance residual；
- 只有 shortcut gate 和 null gate 同时通过，才允许 training diagnostics。

当前结论：`training_unlocked=false`。不允许进入 Phase 4、v3、count-balanced retest、
stability replication、H/M development matrix large run、shadow/final 或 full ten-material v8A matrix。

## 2. Implemented artifacts

New source-controlled scripts:

- `analysis/build_v8a_crystal_clean_view.py`
- `analysis/audit_v8a_crystal_clean_admission.py`

Updated audit:

- `analysis/diagnose_v8a_shuffled_label_null_behavior.py`

The null audit now records effective shuffle fraction and ignores shuffle modes that do not actually perturb enough
training labels. This matters because exact-paired clean views can make overly fine within-strata shuffles nearly
identity transformations.

## 3. Clean-view attempts

| View | Pair support | Shortcut result | Null result | Decision |
| --- | --- | --- | --- | --- |
| `strict_source_origin_thickness_pose_count` | train `13`, validation `9`, stress `10` | not run | not run | support too small |
| `count_only` | train `58`, validation `45`, stress `38` | failed, max non-material BA `0.8762` | failed, null max `1.0` | origin shortcut remains |
| `origin_count` | train `45`, validation `23`, stress `28` | passed, max non-material BA `0.5940` | failed after null fix, fixed max `0.7857`, fixed p95 `0.7435` | stop before training |
| `origin_thickness_pose` | train `41`, validation `29`, stress `28` | passed, max non-material BA `0.5263` | failed, fixed max `0.7857`, fixed p95 `0.6821`, selected p95 `0.6397` | stop before training |

The best cleaned views now pass the visible non-material shortcut audit, but shuffled-label/null behavior is still too
high. Therefore the current generated data still cannot be treated as clean enough for model training.

## 4. Interpretation

This is a hard stop, not a model-tuning problem.

The previous root cause was clear stress/origin shortcuting. The new crystal-clean views reduce that visible shortcut:
for `origin_thickness_pose`, max non-material balanced accuracy drops to `0.5263`. That is good.

However, the null gate still fails. Even after using effective-shuffle-aware null auditing, row-level null performance
can reach H/M min recall `0.7857`. This means the current view still contains learnable structure that should not exist
under fake labels, or the support/paired structure is too small and too regular for a trustworthy training claim.

Therefore we should not “start training and see.” Training on this view would be statistically unreliable and
editorially unsafe.

## 5. Next sampling requirements

The next allowed step is not a model upgrade. It is a new clean-data prereg/matrix design with enough support per
nuisance cell:

- keep development-only H/M scope;
- no shadow/final;
- no full ten-material matrix;
- generate source-on/default rows with balanced H/M within:
  - split,
  - origin/source family,
  - thickness,
  - pose,
  - count bin,
  - seed block;
- avoid mixing medium and extension origins unless both classes have equal support in every relevant cell;
- reserve enough rows so strict source/origin/thickness/pose/count matching still leaves at least:
  - train pairs `>= 100`,
  - validation pairs `>= 50`,
  - stress-holdout pairs `>= 50`;
- rerun shortcut and null gates before any training.

## 6. Plain-language summary

这次我们确实开始“洗数据”了，而且没有把脏数据直接丢给模型。

结果是：明显的 stress/origin 旁门已经能压下去，但假标签测试还是没干净。也就是说，现在这批数据就算
清洗过，也还不能放心训练。继续训练会变成模型继续抓某种隐藏结构，而不是只学 Hematite/Magnetite 的
晶体差异。

下一步要回到采样设计：重新生成一批从一开始就平衡好的 development-only H/M 数据，让每个厚度、姿态、
来源、计数区间里 H/M 都成对出现。只有这样，机器才有资格去学真正的晶体差异。
