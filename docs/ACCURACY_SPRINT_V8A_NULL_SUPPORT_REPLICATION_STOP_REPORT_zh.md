# v8A Null-Support Replication Stop Report

## 一句话结论

null-support replication 已经完成 Geant4 development-only run，并跑完 event-to-feature、clean-design-cell view、shortcut gate、paired-clean null gate 和 admission。结果比上一轮明显更好，但仍差一点没有过：paired-clean null primary p95 从 `0.5972` 降到 `0.5569`，但仍高于 `0.55` ceiling；all-mode p95 是 `0.5628`。因此训练继续锁住。

## 本轮做了什么

本轮使用 profile：

- `v8a_hm_clean_nullrep_cif_lit`

运行结果：

- Geant4 rows：`1008/1008`
- failed：`0`
- source mode：source-on only
- stress label：default only
- H/M-only
- development-only
- no shadow/final
- no full ten-material matrix

preflight 结果：

- row count：`1008`
- strict pairs：train `216`，validation `144`，stress-holdout `144`
- `development_run_prereg_unlocked=true`
- `training_unlocked=false`

event-to-feature：

- samples：`1008`
- sidecar rows：`179125`
- schema/control gate：passed
- tiny training gate：仍按设计 blocked

clean-design-cell view：

- matched pairs：train `216`，validation `144`，stress-holdout `144`
- view gate：passed

shortcut gate：

- `gate_passed=true`
- max non-material balanced accuracy：`0.4080`

paired-clean null gate：

- `gate_passed=false`
- primary fixed/selected p95：`0.5569`
- primary single-seed max：`0.6319`
- all-mode fixed/selected p95：`0.5628`
- all-mode single-seed max：`0.6319`
- stop reasons：
  - `fixed_threshold_null_p95_exceeded_ceiling`
  - `selected_threshold_null_p95_exceeded_ceiling`
  - `all_modes_fixed_threshold_null_p95_exceeded_ceiling`
  - `all_modes_selected_threshold_null_p95_exceeded_ceiling`

admission：

- `gate_passed=false`
- `training_unlocked=false`
- admission gate p95 value：`0.5569`
- true single-seed max：`0.6319`

## 如何解释

这次结果支持两个判断：

1. 源头 clean sampling 方向是对的。non-material shortcut gate 继续很低，visible shortcut 没有回来。
2. 上一轮 null 偏高确实有一部分来自支撑量不足。把 pairs 从 `108/72/72` 提到 `216/144/144` 后，primary p95 从 `0.5972` 降到 `0.5569`。

但结果还没有达到准入标准。`0.5569` 和 `0.5628` 虽然很接近 `0.55`，但不能四舍五入当作通过。训练、ExtraTrees/Logistic baseline、高级模型、development robustness、shadow/final、full ten-material matrix 都继续锁住。

## 下一步

下一步不应该盲目扩大到开发矩阵大跑，也不应该为了过门调低阈值。已新增 null-tail anatomy，先定位尾部来自哪里。

null-tail anatomy 初步结果：

- tail rows above `0.55`：`44/960`
- max tail H/M min recall：`0.6319`
- top seed share：`0.0909`
- top mode share：`0.5909`
- top split share：`0.5000`
- mode 分布：primary `26`，secondary `18`
- split 分布：validation `22`，stress-holdout `22`
- model 分布：Logistic `32`，ExtraTrees `12`

解释：tail 不是集中在某一个 seed、某一个 split 或某一个 null mode 上，而是一个很弱但广泛存在的残余尾部；Logistic 占比更高，提示下一步要查线性方向上的特征族/低维组合，而不是继续盲目加大矩阵。

下一步应该做：

- 检查 tail seed 中 Logistic 权重最大的 feature family；
- 检查这些 feature 是否集中在 ratio、window、unique、prop_peak 或某类峰窗口；
- 检查 tail 是否与 thickness、pose、count-bin、seed-block 的交互有关；
- 如果尾部来自少数 feature family，则做 preregistered blacklist/rebuild；
- 如果尾部仍是广泛小幅随机波动，再做更窄的 validation/stress replication；
- 如果 admission 真正写出 `training_unlocked=true`，才允许恢复 development-only baseline training。

只有 paired-clean null p95 真正低于 `0.55`，admission 写出 `training_unlocked=true`，才允许恢复 development-only baseline training。

## 大白话

这轮就像我们把假标签考试的题量翻倍了。好消息是，模型靠假标签乱猜的高分明显少了，说明之前确实有一部分是题量少、偶然性大造成的。坏消息是，它还是比及格线高了一点点。

所以现在不能说“已经干净，可以训练”。但也不是路线错了。更准确地说：方向是对的，数据比之前干净很多，已经接近过线；下一步要查最后这点假标签尾巴到底从哪里来。我们不能为了赶进度把标准降下来，也不能把真实晶体差异洗掉。
