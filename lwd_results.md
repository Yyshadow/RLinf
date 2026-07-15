# LWD / pi0.5 阶段性实验结果

更新时间：2026-07-13

本文档记录当前 RoboTwin `beat_block_hammer` 上已经完成的阶段性结果，主要包含两部分：

1. pi0.5 SFT actor 的闭环评估结果。
2. LWD critic 的训练指标和离线曲线诊断结果。

## 1. 当前主要产物

### pi0.5 actor

当前主要使用的 pi0.5 SFT checkpoint：

```text
/data/wam_codebase/RLinf/checkpoints/rlinf_pi05_sft_10000/global_step_10000
```

对应权重：

```text
/data/wam_codebase/RLinf/checkpoints/rlinf_pi05_sft_10000/global_step_10000/actor/model_state_dict/full_weights.pt
```

训练配置目标：

- 使用 50 条 `beat_block_hammer` 成功 demo 做 pi0.5 full finetuning。
- 训练 10000 step。
- 当前云端配置每 500 step 保存一次 checkpoint，便于机器重启后自动 resume。
- 当前评估主要看 `global_step_10000`。

### LWD critic

当前主要使用的 critic checkpoint：

```text
/data/wam_codebase/RLinf/checkpoints/robotwin_lwd_critic_train_8a100/checkpoints/global_step_8000
```

对应权重：

```text
/data/wam_codebase/RLinf/checkpoints/robotwin_lwd_critic_train_8a100/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
```

保存节点：

```text
global_step_2000
global_step_4000
global_step_6000
global_step_8000
```

## 2. pi0.5 actor 闭环评估结果

### 2.1 `global_step_500` 早期结果

早期 `global_step_500` 的闭环效果较差：

| Eval 目录 | 轨迹数 | 成功率 |
| --- | ---: | ---: |
| `outputs/eval_pi05_hammer50_step500_quick` | 4 | 0/4 |
| `outputs/eval_pi05_hammer50_step500_train4_noise05` | 4 | 0/4 |

结论：`global_step_500` 基本还没有形成稳定闭环能力，后续改为从 base package 重新训练 10000 step，而不是继续沿用这个 500-step run。

### 2.2 `global_step_10000` 小规模 eval

`global_step_10000` 相比 step500 有明显提升，但仍不稳定。

| Eval 目录 | 轨迹数 | 成功率 | 备注 |
| --- | ---: | ---: | --- |
| `outputs/eval_pi05_hammer50_step10000_metrics_eval4_clean` | 4 | 0/4 | eval seeds 小批量失败 |
| `outputs/eval_pi05_hammer50_step10000_metrics_train4` | 4 | 2/4 | train seeds 上能完成部分闭环 |

结论：模型不是完全没有学会任务，而是对初始状态、末端对准和扰动仍然敏感。训练 seed 上出现成功，说明已经学到一部分 hammer 行为；eval seed 上不稳定，说明泛化和闭环鲁棒性还不足。

### 2.3 不同 `action_exec_horizon` 的 30 次 eval

使用 `global_step_10000` 权重，对同一任务测试不同执行步长：

| 执行方式 | Eval 目录 | 轨迹数 | 成功数 | 成功率 |
| --- | --- | ---: | ---: | ---: |
| `action_exec_horizon=20` | `outputs/eval_pi05_hammer50_gs10000_exec20_n30` | 30 | 2 | 6.67% |
| `action_exec_horizon=30` | `outputs/eval_pi05_hammer50_gs10000_exec30_n30` | 30 | 11 | 36.67% |
| `action_exec_horizon=50` | `outputs/eval_pi05_hammer50_gs10000_exec50_n30` | 30 | 2 | 6.67% |

解释：

- 模型仍然一次预测 50-step action chunk。
- `action_exec_horizon=30` 表示只执行前 30 个 step，然后重新观测和重新规划。
- `action_exec_horizon=50` 是完整执行整个 chunk，重规划频率最低。
- `action_exec_horizon=20` 重规划更频繁，但当前效果也不好，可能是动作 chunk 前段尚未充分完成抓取/挥锤动作，过早重规划反而破坏时序。

阶段性结论：

```text
当前 pi0.5 hammer actor 最合理的 eval 设置是 action_exec_horizon=30。
```

这说明当前模型受益于 receding-horizon 执行，但不能过短，也不能完整执行 50 步。

### 2.4 `exec_horizon=30` 视频诊断批次

为了具体观察成功和失败行为，又使用 `action_exec_horizon=30` 跑了一批带内部相机录像的 eval：

```text
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_gs10000_exec30_video_pick20
```

结果：

| 轨迹数 | 成功数 | 失败数 | 成功率 |
| ---: | ---: | ---: | ---: |
| 20 | 5 | 15 | 25% |

已经从中整理出 5 个成功视频和 5 个失败视频：

成功视频：

```text
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_gs10000_exec30_selected_videos/success
```

失败视频：

```text
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_gs10000_exec30_selected_videos/failure
```

对应关系说明：

```text
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_gs10000_exec30_selected_videos/README.txt
```

5 个成功视频：

| 文件 | seed | env_id |
| --- | ---: | ---: |
| `success_01_seed_100112514_env_1.mp4` | 100112514 | 1 |
| `success_02_seed_100175033_env_2.mp4` | 100175033 | 2 |
| `success_03_seed_100100065_env_0.mp4` | 100100065 | 0 |
| `success_04_seed_100162521_env_0.mp4` | 100162521 | 0 |
| `success_05_seed_100162537_env_1.mp4` | 100162537 | 1 |

5 个失败视频：

| 文件 | seed | env_id |
| --- | ---: | ---: |
| `failure_01_seed_100137506_env_0.mp4` | 100137506 | 0 |
| `failure_02_seed_100187555_env_3.mp4` | 100187555 | 3 |
| `failure_03_seed_100150010_env_4.mp4` | 100150010 | 4 |
| `failure_04_seed_100125012_env_1.mp4` | 100125012 | 1 |
| `failure_05_seed_100162556_env_2.mp4` | 100162556 | 2 |

当前观察倾向：

- 成功率还不够高，但模型已经能完成一部分抓取、接近和敲击过程。
- 失败不一定是完全不会做，很多更可能是 near-miss，例如抓取、挥锤或末端对准差一点。
- 需要继续逐条看失败视频，把失败分成“抓锤失败 / 接近但没敲准 / 时序问题 / 末端姿态问题”等类别。

## 3. LWD critic 训练结果

### 3.1 训练指标

critic 的主要 TensorBoard 目录：

```text
/data/server/tensorboard_7.4
```

训练 8000 step 后，关键指标如下：

| 指标 | step 0 | step 7999 | 趋势 |
| --- | ---: | ---: | --- |
| `train/q_loss` | 0.2506 | 0.0318 | 明显下降 |
| `train/value_loss` | 5.7347 | 3.7984 | 下降 |
| `train/q_head_gap` | 0.1071 | 0.0104 | 明显收敛 |
| `train/grad_norm` | 87.57 | 8.86 | 明显变稳 |
| `train/q_min_mean` | 0.0074 | 0.4487 | 上升到合理区间 |
| `train/target_q_mean` | 0.5090 | 0.4453 | 保持在中间价值范围 |

解释：

- `q_loss` 下降说明 action-conditioned Q 分支在拟合 bootstrap target。
- `value_loss` 下降说明 distributional value head 在学习 target distribution。
- `q_head_gap` 下降说明 double-Q 两个 head 的差距在缩小，训练更稳定。
- `grad_norm` 从很高降下来，说明训练早期的大梯度逐渐稳定。

### 3.2 critic eval 指标

step 7999 的关键 eval 指标：

| Eval 数据 | `q_min_mean` | `target_q_mean` | 备注 |
| --- | ---: | ---: | --- |
| success | 0.6157 | 0.7351 | 成功轨迹价值最高 |
| failed | 0.3665 | 0.2896 | 失败轨迹较低 |
| nearmiss | 0.3387 | 0.2705 | near-miss 也较低 |

success 内部更细指标：

| 指标 | 数值 |
| --- | ---: |
| `eval/beat_block_success/success/q_min_mean` | 0.7580 |
| `eval/beat_block_success/other/q_min_mean` | 0.5523 |

阶段性结论：

```text
critic 已经能把 success 和 failed/nearmiss 拉开。
```

这比只看训练 loss 更重要，因为它说明 critic 不是单纯拟合数值，而是在不同类型 episode 上产生了可解释的价值差异。

## 4. LWD critic 离线曲线诊断

离线曲线输出目录：

```text
/data/wam_codebase/RLinf/outputs/lwd_critic_value_curves
```

主要文件：

```text
robotwin_lwd_critic_episode_values.pdf
robotwin_lwd_critic_episode_values.png
robotwin_lwd_critic_episode_values.csv
robotwin_lwd_critic_episode_values_summary.json
```

对应视频：

```text
outputs/lwd_critic_value_curves/videos/success/
outputs/lwd_critic_value_curves/videos/failed/
outputs/lwd_critic_value_curves/videos/nearmiss/
```

summary 中记录的第一帧和最后一帧价值：

| 类型 | V 首帧 | V 末帧 | Q 首帧 | Q 末帧 |
| --- | ---: | ---: | ---: | ---: |
| success | 0.536 | 0.932 | 0.601 | 0.942 |
| failed | 0.464 | 0.062 | 0.501 | 0.049 |
| nearmiss | 0.530 | 0.044 | 0.593 | -0.008 |

解释：

- success episode 后期 V/Q 明显升高，符合“接近任务完成”的趋势。
- failed 和 nearmiss 后期 V/Q 明显降低，说明 critic 能识别出后续成功概率变低。
- nearmiss 的末尾 Q 甚至低于 failed，可能是因为该轨迹后续 action chunk 对完成任务更不利。

需要注意：

```text
LWD critic 不是 collision detector，也不是逐帧失败分类器。
```

它学到的是“当前状态和动作 chunk 往后是否更可能完成任务”的进展信号。因此价值不一定在某个 collision 或失误帧突然断崖式下降，而可能表现为一段趋势上的平台化或逐渐下降。

## 5. 2026-07-13：allstats SFT/QAM 的 ODE 闭环评估

本轮评估统一改为 OpenPI ODE 推理口径：

```yaml
rollout:
  model:
    openpi:
      noise_method: flow_ode
      noise_level: 0.0
```

`flow_ode` 表示 denoise 过程中不额外加入 SDE 噪声；但 OpenPI flow 的初始
Gaussian noise 仍由 policy forward 时采样。因此同一个环境 seed 固定的是物体、
机器人和 reset 初始场景，不等价于固定每次 policy 的 flow initial noise。

本轮同时把 hammer pi0.5 eval 配置默认改为 ODE：

```text
evaluations/robotwin/robotwin_beat_block_hammer_openpi_pi05_eval.yaml
```

### 5.1 评估设置

共同设置：

```text
env.eval.action_exec_horizon = 50
env.eval.total_num_envs      = 3
env.eval.rollout_epoch       = 10
env.eval.use_fixed_reset_state_ids = true
```

因此这不是 30 个完全不同的初始场景，而是 3 个固定 reset seed 各重复 10 次：

```text
100112514
100137506
100175033
```

SFT checkpoint：

```text
/data/wam_codebase/RLinf/checkpoints/rlinf_pi05_sft_allstats/global_step_11000
```

QAM checkpoint：

```text
/data/wam_codebase/RLinf/checkpoints/rlinf_lwd_qam_allstats/robotwin_beat_block_hammer_lwd_qam_openpi_pi05_allstats_probe_lq005_gc005/checkpoints/global_step_300
```

### 5.2 ODE 结果

| 模型 | Eval 目录 | 成功数 | 成功率 |
| --- | --- | ---: | ---: |
| SFT allstats `global_step_11000` | `outputs/eval_pi05_hammer50_allstats_gs11000_exec50_n30_ode` | 13/30 | 43.33% |
| QAM allstats `lq005_gc005 global_step_300` | `outputs/eval_qam_allstats_probe_lq005_gc005_gs300_exec50_n30_ode` | 17/30 | 56.67% |

补充：2026-07-15 额外测试了同一个最新 SFT checkpoint，只把
`action_exec_horizon` 从 50 改成 30：

| 模型 | Eval 目录 | 成功数 | 成功率 |
| --- | --- | ---: | ---: |
| SFT allstats `global_step_11000`, `exec30` | `outputs/eval_pi05_hammer50_allstats_gs11000_exec30_n30_ode` | 5/30 | 16.67% |

按 seed 拆分：`100112514: 0/10`，`100137506: 4/10`，`100175033: 1/10`。
因此旧版 `global_step_10000` 上观察到的 `exec30` 优势没有在最新 allstats SFT
上复现；当前最新 SFT 在同一 ODE/fixed-reset 口径下反而是 `exec50` 明显更好
（13/30 vs 5/30）。这说明执行步长和 checkpoint/动作时序强相关，不能直接沿用旧
SFT 的 `exec30` 结论。

按 seed 拆分：

| seed | SFT | QAM | 变化 |
| ---: | ---: | ---: | ---: |
| 100112514 | 6/10 | 7/10 | +1 |
| 100137506 | 1/10 | 5/10 | +4 |
| 100175033 | 6/10 | 5/10 | -1 |

阶段性结论：

```text
QAM 在 ODE 评估下仍然从 13/30 提升到 17/30，说明提升不是依赖 flow_sde/noise_level=0.5
带来的采样噪声。主要收益来自 hard seed 100137506。
```

同时，QAM 不是无损增强：`100175033` 从 6/10 降到 5/10。这说明当前 QAM 更像是
在 SFT 附近做有净收益的小局部修正，而不是对所有 fixed seeds 都稳定改进。

### 5.3 对比口径

后续汇报建议使用更准确的说法：

```text
same-seed / same-reset-scene comparison
```

不要把它过度解释成严格的 paired counterfactual。当前对齐的是同一个环境 seed，
即同一个初始场景；没有显式固定同一个 policy flow initial noise，也没有对齐中间
状态轨迹。因此本轮结果能说明：

```text
在同一批固定初始场景上，QAM 的闭环成功率更高。
```

但不能单独证明：

```text
在完全相同 flow initial noise 和完全相同中间状态下，QAM 每一步都优于 SFT。
```

如果后续需要更严格的可控实验，需要显式记录或复用 policy 的初始 flow noise，
例如按 `env_seed + rollout_epoch + env_id + chunk_idx` 构造固定 generator，或者
直接把相同 noise tensor 分别传给 SFT 和 QAM。

### 5.4 A/B/C 后续验证结果

按“先 checkpoint sweep，再扩大 fixed seeds，再看 unique seeds”的顺序继续验证后，
结论从“QAM 有稳定提升”修正为“当前 QAM 不稳定，不能作为改进模型继续放大”。

#### A. `global_step_100/200/300` checkpoint sweep

固定 ODE、`action_exec_horizon=50`、3 个 fixed seeds 各重复 10 次：

| 模型 | 成功数 | 成功率 | 100112514 | 100137506 | 100175033 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT `global_step_11000` | 13/30 | 43.33% | 6/10 | 1/10 | 6/10 |
| QAM `global_step_100` | 12/30 | 40.00% | 8/10 | 3/10 | 1/10 |
| QAM `global_step_200` | 12/30 | 40.00% | 5/10 | 2/10 | 5/10 |
| QAM `global_step_300` | 17/30 | 56.67% | 7/10 | 5/10 | 5/10 |

`global_step_300` 是 30 次 fixed-seed eval 下的最好 checkpoint，但 step100/200
都没有超过 SFT，说明 QAM 训练过程不是单调提升。

#### B. 最优 checkpoint 的 60 次 fixed-seed 验证

把 `global_step_300` 扩大到 3 个 fixed seeds 各重复 20 次：

| 模型 | 成功数 | 成功率 | 100112514 | 100137506 | 100175033 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT `global_step_11000` | 28/60 | 46.67% | 16/20 | 2/20 | 10/20 |
| QAM `global_step_300` | 25/60 | 41.67% | 15/20 | 5/20 | 5/20 |

扩大样本后，QAM 不再优于 SFT。它确实改善了 hard seed `100137506`
（2/20 -> 5/20），但同时明显损伤 `100175033`（10/20 -> 5/20），总分变低。

#### C. unique seeds 验证

设置 `env.eval.use_fixed_reset_state_ids=false`，ODE、`action_exec_horizon=50`，
各跑 30 条不同 reset 场景：

| 模型 | 成功数 | 成功率 |
| --- | ---: | ---: |
| SFT `global_step_11000` | 6/30 | 20.00% |
| QAM `global_step_300` | 6/30 | 20.00% |

unique seeds 上 QAM 没有提升，也没有明显退化。

#### 阶段性结论

```text
当前 lq005_gc005 QAM checkpoint 不是稳定改进。
它主要把 hard seed 100137506 往好方向推了一点，但会牺牲部分原本较稳的 seed。
30 次 fixed-seed eval 的 17/30 是乐观小样本结果，60 次 fixed-seed 后净收益消失。
```

因此，原计划中的 D：

```text
如果提升稳定，再训练 lq005_gc003 和 lq01_gc005
```

前提不成立，本轮不应继续盲目训练新的 QAM ablation。下一步应先回到诊断层面：

1. 固定同一个 flow initial noise，做更严格的 SFT/QAM counterfactual 对比。
2. 对 `100137506` 和 `100175033` 分别输出视频，确认 QAM 是如何改善 hard seed、又如何破坏原本较稳 seed。
3. 优先加“保守门控/约束”而不是继续放大 QAM，例如只在 critic advantage 明显为正时更新，或加入 per-state trust region。

### 5.5 fixed flow initial noise 复验

为排除 OpenPI flow 初始 Gaussian noise 的影响，新增 eval-only 配置
`rollout.model.openpi.fixed_eval_noise_seed`。本轮设置：

```text
noise_method: flow_ode
noise_level: 0.0
fixed_eval_noise_seed: 20260713
total_num_envs: 3
rollout_epoch: 20
action_exec_horizon: 50
use_fixed_reset_state_ids: true
```

输出：

```text
outputs/eval_pi05_hammer50_allstats_gs11000_exec50_fixednoise_n60_ode
outputs/eval_qam_allstats_probe_lq005_gc005_gs300_exec50_fixednoise_n60_ode
```

| 模型 | 成功数 | 成功率 | 100112514 | 100137506 | 100175033 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT `global_step_11000` | 29/60 | 48.33% | 16/20 | 4/20 | 9/20 |
| QAM `global_step_300` | 28/60 | 46.67% | 15/20 | 3/20 | 10/20 |

按同一个 `(rollout_epoch, env_id, seed)` 配对后：

| 类型 | 数量 |
| --- | ---: |
| SFT 成功、QAM 成功 | 19 |
| SFT 失败、QAM 失败 | 22 |
| SFT 成功、QAM 失败 | 10 |
| SFT 失败、QAM 成功 | 9 |

结论：在固定 flow initial noise 后，QAM 仍没有稳定超过 SFT。SFT-only
和 QAM-only 的翻转几乎抵消，说明当前 QAM checkpoint 更像是在 SFT 附近改变动作分布，
而不是形成稳定的价值改进方向。

### 5.6 critic gradient direct-edit 诊断

为了先判断 critic 的 action gradient 是否有闭环意义，做了一个不训练 actor 的
direct-edit probe：

```text
outputs/lwd_critic_grad_edit_diag_n10
```

每个 seed 从相同 reset 场景出发，同时比较 4 条闭环轨迹：

| 变体 | 含义 |
| --- | --- |
| `base` | 原始 SFT action |
| `plus` | 对执行前缀沿 `+grad_A Q(s,A)` 做小步编辑 |
| `minus` | 沿 `-grad_A Q(s,A)` 做同等幅度编辑 |
| `random` | 随机方向、同等幅度编辑 |

设置：

```text
num_cases: 10
epsilon: 0.01
action_exec_horizon: 20
```

闭环成功率：

| 变体 | 成功数 | 成功率 |
| --- | ---: | ---: |
| `base` | 0/10 | 0.00% |
| `plus` | 0/10 | 0.00% |
| `minus` | 0/10 | 0.00% |
| `random` | 0/10 | 0.00% |

即时 critic 预测的 `Q(after edit) - Q(before edit)` 统计：

| 范围 | `plus` 正 delta 比例 | `plus > base` | `plus > minus` | `plus > random` |
| --- | ---: | ---: | ---: | ---: |
| all chunks | 52.0% | 69.0% | 72.0% | 59.0% |
| chunks 2-9 | 63.8% | 76.2% | 83.8% | 66.2% |
| chunks 5-9 | 90.0% | 82.0% | 86.0% | 78.0% |

阶段性结论：

```text
critic 梯度在 critic 自己的局部 Q 预测里有一定方向性，尤其 episode 后半段；
但这 10 个 hard cases 中没有任何变体成功，所以还不能证明该梯度已经能稳定带来
真实闭环成功率提升。
```

更准确的判断是：

```text
grad_A Q 有局部自洽信号，但信号幅度和闭环物理效果之间还没有被验证打通。
```

本次进程在写完 summary 后退出清理阶段出现 `free(): invalid pointer`，但
`critic_grad_edit_episode_metrics.jsonl` 和 `critic_grad_edit_summary.json`
均已完整落盘，因此不影响本次统计。

### 5.7 `seed100175033/env2` 成功触发几何 trace

为验证 paired video 里 QAM 的成功到底是稳定对准，还是短暂进入成功阈值，新增：

```text
examples/lwd/trace_qam_success_geometry.py
outputs/paired_sft_qam_geometry_trace_seed100175033_evalapi_v2
```

该脚本 monkey-patch `beat_block_hammer.check_success()`，在每次 success 判断时记录：

```text
hammer functional point xy
block target functional point xy
dx / dy / xy_linf / xy_l2
contact
success
```

本次使用与视频一致的正式 eval action API、`flow_ode`、`fixed_eval_noise_seed=20260713`、
`action_exec_horizon=50`、3 env batch 顺序：

```text
env0: 100137506
env1: 100112514
env2: 100175033
```

复现结果和视频一致：

| 模型 | env0 `100137506` | env1 `100112514` | env2 `100175033` |
| --- | --- | --- | --- |
| SFT | fail | success | fail |
| QAM | fail | success | success |

目标视频 `seed100175033/env2` 的关键几何量：

| 模型 | success check 数 | within 2cm check 数 | near 3cm check 数 | min `xy_linf` | min `xy_l2` | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SFT | 0 | 0 | 98 | 2.090 cm | 2.636 cm | 一直差一点，未进 2cm 矩形阈值 |
| QAM | 1 | 1 | 44 | 1.995 cm | 2.238 cm | 只在最后一次判断贴边进入阈值 |

QAM 成功前后的 trace：

```text
check 1416: dx=-0.897cm, dy=-2.051cm, linf=2.051cm, contact=True, success=False
check 1417: dx=-0.942cm, dy=-2.032cm, linf=2.032cm, contact=True, success=False
check 1418: dx=-0.977cm, dy=-2.023cm, linf=2.023cm, contact=True, success=False
check 1419: dx=-1.008cm, dy=-2.013cm, linf=2.013cm, contact=True, success=False
check 1420: dx=-1.044cm, dy=-1.995cm, linf=1.995cm, contact=True, success=True
```

因此，这个案例不能解释为“QAM 已经稳定学会把 hammer 放到目标中心”。更准确的解释是：

```text
QAM 把原本接近阈值的动作推到了成功矩形阈值内，但成功非常贴边；
由于 RoboTwin 在 success=True 后立即 termination，无法从这个 episode 判断它会不会继续稳定保持。
```

这和前面 60 次 fixed-noise eval 的结论一致：QAM 确实能改变末端对准，并能 rescue
部分 near-threshold case；但当前 checkpoint 的收益更像局部/贴边修正，还不是稳定的
闭环技能提升。

### 5.8 same-state action ranking 诊断

为直接回答“同一个状态下，critic 对多个 action 的排序是否符合真实环境结果”，新增：

```text
examples/lwd/diagnose_same_state_action_ranking.py
outputs/lwd_same_state_action_ranking_pilot
```

运行命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/data/wam_codebase/RLinf:/data/wam_codebase/RoboTwin_RLinf \
ROBOTWIN_PATH=/data/wam_codebase/RoboTwin_RLinf \
ROBOTWIN_ASSETS_PATH=/data/wam_codebase/RoboTwin_RLinf \
EMBODIED_PATH=/data/wam_codebase/RLinf/examples/embodiment \
REPO_PATH=/data/wam_codebase/RLinf \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
/data/wam_codebase/RLinf/.venv-openpi/bin/python \
examples/lwd/diagnose_same_state_action_ranking.py \
  --mode pilot \
  --num-states 4 \
  --out-dir outputs/lwd_same_state_action_ranking_pilot
```

实现口径：

- canonical state 使用 `same seed + SFT prefix replay` 恢复，不使用底层 Sapien snapshot API。
- 4 个 state 的 replay 复现测试全部通过：hammer/block/robot 最大误差为 0，observation hash 完全一致。
- 候选包含 `sft`、`qam_same_noise`、`qam_mirror`、`grad_plus`、`grad_minus`、`random_plus`、`random_minus`。
- 当前可执行 action 是 qpos chunk；没有安全接入 Cartesian IK/controller 转换，因此 `x/y/z ±2mm` 候选跳过，未用 joint 加减伪装 Cartesian 位移。
- critic 主排序使用当前 QAM 配置一致的 `q_mean`，同时保存 `q1/q2/q_min`。

输出文件：

```text
config.json
states_manifest.jsonl
candidate_scores.jsonl
rollout_results.jsonl
per_state_summary.csv
pairwise_comparisons.csv
summary.json
```

总体结果：

| 指标 | 值 |
| --- | ---: |
| valid states | 4/4 |
| success pairwise accuracy | 0.800 |
| margin pairwise accuracy | 0.449 |
| margin accuracy 95% CI | [0.171, 0.692] |
| median Spearman(Q, margin) | 0.054 |
| Spearman 95% CI | [-0.652, 0.429] |
| QAM-vs-SFT sign agreement | 0.250 |
| QAM-vs-mirror sign agreement | 0.333 |
| grad_plus better than grad_minus | 0.333 |
| script decision | `INCONCLUSIVE` |

关键解读：

```text
critic 对非常粗的 success/fail pair 有一定排序信号；
但对连续 margin、QAM 方向和局部 gradient 方向都不可靠。
```

例如：

- `seed100112514_q2`：SFT margin `+1.05cm`，QAM margin `+0.038cm`，但 critic 给 QAM 更高 Q。
- `seed100175033_q2`：SFT 成功且 margin `+0.004cm`，QAM 失败且 margin `-1.19cm`，critic 给 QAM 略低 Q；但 `grad_plus` 被 critic 明显抬高，却真实失败且 margin `-0.90cm`。
- `seed100187555_q2`：QAM 相对 SFT 的 Q 略升，但真实 margin 从 `-0.16cm` 降到 `-1.76cm`。

因此，这个 pilot 不能证明 critic 完全随机；但它已经足够说明：

```text
当前 critic 不满足 QAM actor 长训所需的 reliable same-state action-gradient 条件。
```

工程决策上，本轮结果更接近 `NO_GO for QAM gradient`，而不是 `CRITIC_PASS`。下一步不应继续主要扫描
`lambda_q / qam_grad_clip`，而应优先改 critic 数据和目标：加入 same-state action contrast、
几何 margin/contact/grasp stability 辅助监督，或先尝试 candidate ranking / best-of-K，而不是直接用
`grad_A Q` 更新 actor。

本次运行在结果完整写盘后仍出现 RoboTwin/Sapien 清理阶段 `free(): invalid pointer`，不影响上述文件统计。

## 6. 当前阶段性判断

### pi0.5 actor

当前 pi0.5 actor 已经学到部分任务行为，但闭环成功率还不稳定：

- `exec_horizon=30` 下 30 次 eval 达到 36.67%。
- 带视频的 20 次 eval 达到 25%。
- 说明模型不是完全不可用，但距离稳定策略还有差距。

更可能的问题：

1. 50 条 success demo 的覆盖范围偏小。
2. 末端姿态和敲击对准误差对成功率影响很大。
3. 完整执行 50 步 chunk 太长，容易积累误差。
4. 过短执行 20 步也不好，可能破坏动作时序。

当前推荐 eval 设置：

```yaml
env:
  eval:
    action_exec_horizon: 30
```

### LWD critic

critic 的阶段性结果更积极：

- 训练 loss 正常下降。
- eval 上 success 的 Q 明显高于 failed/nearmiss。
- 离线曲线中 success 后期上升，failed/nearmiss 后期下降。

当前判断：

```text
critic 已经具备一定区分能力，可以作为后续策略优化或数据筛选的候选信号。
```

但仍需要注意：

- 当前只是离线诊断和 supervised/TD critic 训练闭环。
- 还没有真正接入 actor 更新。
- 后续要验证 critic 指导策略更新时是否能提升真实闭环 success rate。

## 7. 下一步建议

1. 逐条观看 `exec_horizon=30` 的 5 个成功和 5 个失败视频，人工标注失败类型。
2. 如果失败主要是 near-miss，可以优先补充更多末端对准和敲击角度多样的数据。
3. 如果失败主要是抓锤失败，需要补充更多抓取阶段覆盖。
4. 继续保留 `action_exec_horizon=30` 作为当前主 eval 设置。
5. 用 critic 对成功、失败、near-miss 轨迹做更大规模离线打分，确认排序是否稳定。
6. critic 排序稳定后，再考虑把 critic 接入后续 actor 更新。
