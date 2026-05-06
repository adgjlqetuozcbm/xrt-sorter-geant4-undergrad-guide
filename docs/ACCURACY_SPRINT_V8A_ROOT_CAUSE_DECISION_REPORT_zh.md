# Accuracy Sprint v8A: root-cause decision report

Date: 2026-05-06

## 1. Editorial decision

本阶段没有进入 v3 特征、ordinary Phase 4、count-balanced retest、stability
replication 或开发矩阵大跑。我们先执行了 root-cause diagnosis，目标是回答
一个更基础的问题：为什么 shuffled-label/null sanity 会失败。

当前结论比上一阶段更具体：

- 不是简单的 total-count gap 问题；
- 不是 validation-selected threshold 单独放水；
- 不是“高级模型还没试所以不能判断”；
- 主要根因是 `sampling_or_origin_shortcut_found`。

最直接证据是：v2 proportion-only 的 main `diffraction_*` 特征可以用来
完美预测 `stress_label`，balanced accuracy `1.0`。这说明当前 feature/view
里携带了非材料结构信号。假标签模型可以借这种结构取得异常表现，因此不能把
主 H/M `1.0` 解释为稳健可审稿 evidence。

Decision:

- stop before v3 preregistration;
- do not run Phase 4 from current v2;
- do not run count-balanced retest or stability replication from current v2;
- block advanced-model feature sufficiency probe as promotion evidence;
- do not start H/M development matrix large run;
- keep shadow/final and full ten-material v8A matrix sealed.

## 2. Implemented artifacts

New source-controlled diagnostic scripts:

- `analysis/diagnose_v8a_shuffled_label_null_behavior.py`
- `analysis/audit_v8a_feature_shortcut_structure.py`
- `analysis/audit_v8a_stress_null_path.py`
- `analysis/probe_v8a_feature_sufficiency_models.py`
- `analysis/decide_v8a_root_cause.py`

Generated development evidence, left untracked by default:

- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_null_behavior/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_feature_shortcut/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_stress_null_path/`
- `results/accuracy_v3/v8a_medium_plus_count_overlap_count_robust_v2_proportion_only_feature_sufficiency_probe/`
- `results/accuracy_v3/v8a_root_cause_decision/`

## 3. Root-cause evidence

| Audit | Decision | Key evidence | Interpretation |
| --- | --- | --- | --- |
| Shuffled-label null behavior | `null_behavior_root_cause_needed` | fixed-threshold null max `1.0`; selected-threshold null max `1.0`; `tree_null_overfit_suspected=false` | This is not only threshold-selection artifact and not merely ExtraTrees-vs-Logistic overfit. |
| Feature shortcut structure | `feature_or_sampling_shortcut_found` | main features predict `audit_stress_label` with balanced accuracy `1.0` | `diffraction_*` carries non-material stress/origin structure. |
| Stress null path | `stress_null_path_artifact_found` | max selected null H/M `1.0`; max material-correlated stress delta gap `10.8273` | Stress scenarios can create or amplify null-path artifacts. |
| Feature sufficiency probe | `feature_sufficiency_probe_blocked_until_root_cause_clean` | probe blocked by failed root-cause gates | Advanced models would currently test shortcut-capturing power, not clean feature sufficiency. |
| Root-cause decision | `sampling_or_origin_shortcut_found` | main features predict non-material audit targets above ceiling | v3 prereg remains locked. |

Root-cause gate:

- file: `results/accuracy_v3/v8a_root_cause_decision/v8a_root_cause_decision_gate.json`
- decision: `sampling_or_origin_shortcut_found`
- `v3_prereg_unlocked`: `false`

## 4. What this means

之前我们只知道“假标签也能好”。现在更进一步：当前 v2 feature/view 不是只
编码 H/M 差异，也编码了 `stress_label` 这样的非材料结构。这样一来，模型即使
训练标签被打乱，也可能通过数据结构、stress/default 分布、或采样来源模式获得
看似不错的表现。

这解释了为什么总计数已经平衡后，shuffled-label 仍然失败。问题已经从
`total-count shortcut` 变成了更高一级的 `sampling/stress/origin shortcut`。

## 5. Advanced-model decision

高级模型需要考虑，但当前不能作为 promotion evidence。

原因很简单：如果 `diffraction_*` 现在能完美预测 `stress_label`，那么 CNN、
MLP、Transformer 这种更强的模型更可能更快抓到这个捷径。此时高级模型跑出
更高 real-label recall，并不能证明特征足够；它只能证明更强模型更会利用当前
结构。

因此本阶段只实现了 guarded sufficiency probe。由于 root-cause gates 没过，
probe 被正确阻断：

- decision: `feature_sufficiency_probe_blocked_until_root_cause_clean`

只有当下一版 feature/view 不再预测 non-material audit targets，且 null audit
干净后，高级模型才有解释价值。

## 6. Next stage

下一阶段不应继续调模型，而应修复采样和 stress protocol：

- 重新定义 stress/default 分布，让 `stress_label` 不能被 main features 轻松预测。
- 在 train/validation/stress 中平衡或隔离 stress/default、medium/extension、
  origin、source-id 结构。
- 将 stress gate 拆成两层：
  - main signal gate 只评估真实 label；
  - null stress gate 单独要求 stress 不抬高 shuffled-label performance。
- 下一版 feature candidate 必须先通过：
  - non-material target predictability ceiling；
  - fixed-threshold and selected-threshold null distribution；
  - stress null path audit。

只有这些通过，才允许写 `v8a_count_robust_v3_prereg`。

## 7. Plain-language summary

现在终于不是只知道“假标签高”，而是找到了一个很像真根因的东西：

我们现在的 `diffraction_*` 特征能完美判断一行数据是不是 stress/default。
这不是材料本身，而是数据生成/采样/扰动方式留下的结构痕迹。

所以模型可能并不是单纯在学 Hematite 和 Magnetite 的晶相差异，而是在部分
利用“这行数据来自哪种 stress/来源结构”这样的旁门信息。标签打乱以后它还
能表现好，就是这个原因。

因此高级模型现在不能救场。高级模型会更擅长抓这种旁门结构。下一步要先把
stress 和采样结构修干净，再让高级模型验证真正的特征上限。
