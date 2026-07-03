# LWD-style Critic 接入说明

本文档说明当前类 LWD critic 的代码边界、pi0.5 对齐方式、训练数据要求和后续使用方向。

## 目标

当前实现的重点不是单独复现一个轨迹分类器，而是搭建后续 Actor-Critic 式策略后训需要的 critic 基础组件。它要能同时回答两个问题：

```text
V(s): 当前状态是否接近任务成功，整体价值是多少
Q(s, a_chunk): 在当前状态执行某段 action chunk 的质量如何
```

这样后续在策略优化时，critic 可以为 actor 提供比 success/fail 二值标签更细的价值评估信号。

## 代码位置

```text
rlinf/models/embodiment/lwd_critic/
  lwd_critic_model.py   # distributional V(s) + action-conditioned double Q
  lwd_loss.py           # chunk TD loss、distributional value loss、EMA target update
  __init__.py           # model_type=lwd_critic 的 get_model 入口

rlinf/data/datasets/lwd/
  chunk_dataset.py      # 从 LeRobot-Aloha 数据构造 pi0.5 对齐的 chunk transition
  __init__.py

rlinf/workers/sft/
  fsdp_lwd_critic_worker.py
                      # 使用 RLinf SFTRunner/FSDPModelManager 训练 LWD critic

examples/lwd/
  train_lwd_critic.py
  run_lwd_critic.sh
  config/robotwin_lwd_critic.yaml
  config/model/lwd_critic.yaml
  config/training_backend/fsdp.yaml
                      # LWD 自包含的模型和 FSDP 配置片段
```

## 模型结构

当前 critic 参考了 pi0.6* 相关开源实现中的 value model 思路，并复用现有 ReCap/value model 的视觉语言编码链路：

```text
多视角图像
  + pi0.5 风格文本 prompt:
    "Task: ..., State: <discrete_state_bins>;\nAction: "
        |
        v
SigLIP2 Vision Encoder
        |
        v
Gemma3 Backbone
        |
        v
Gemma-style Critic Expert readout token
        |
        v
state feature z_t
        |----------------------|
        v                      v
Distributional V head      ActionChunkEncoder
V(s) distribution          a_t:t+H -> action feature
        |                      |
        |----------------------|
                 v
          Double-Q head
          Q1(s, a_chunk), Q2(s, a_chunk)
```

模型 base 是：

```text
SigLIP2 Vision Encoder
+ Gemma3-270M Backbone
+ Gemma-style Critic Expert
+ distributional value head
+ action chunk encoder
+ double-Q head
```

## 输入输出

单个训练样本来自一个 chunk transition：

```text
obs_t
action_chunk = a_t:t+H
reward_chunk = r_t:t+H
next_obs = obs_t+H
done
success
task / episode_id / frame_idx / source
```

collator 输出给模型的是：

```text
observation:
  images
  image_masks
  tokenized_prompt
  tokenized_prompt_mask

next_observation:
  images
  image_masks
  tokenized_prompt
  tokenized_prompt_mask

action_chunk:
  [batch, horizon, action_dim]

reward_chunk:
  [batch, horizon]

done:
  [batch]
```

模型输出包括：

```text
value_logits       # V(s) 在 201 个 value atoms 上的 logits
value_probs        # softmax 后的 value distribution
value_mean         # distributional mean
value_quantile     # 默认 tau=0.6 的 value quantile
q_values           # [q1, q2]
q_min              # min(q1, q2)
state_features     # critic readout token hidden state z_t
action_features    # action chunk 编码后的特征
```

## pi0.5 对齐方式

当前版本把 state 和 action 都显式对齐到 pi0.5 SFT 使用的语义空间，避免 critic 和 actor 学到不同的坐标系。

### State

pi0.5 不使用单独的连续 StateProjector。OpenPI 源码中 pi0.5 会先把 state 归一化到近似 `[-1, 1]`，再离散成 256 个 bin，并拼进 prompt：

```text
Task: <task>, State: <bin_0> <bin_1> ... <bin_n>;
Action:
```

当前 `LWDChunkDataset` 也采用这个方式：先用 Aloha/pi0.5 的 state 变换和 norm stats 归一化，再把离散 state 写入 prompt。因此当前 critic 的 `V(s)` 不是只看图像和任务文本，而是也能看到低维关节状态。

### Action

action 采用和 RoboTwin pi0.5 SFT dataconfig 一致的 delta 语义：

```text
关节维度: target_qpos - current_state_qpos
夹爪维度: absolute gripper target
```

对应 mask 是：

```text
[6 个左臂关节 delta, 左夹爪 absolute,
 6 个右臂关节 delta, 右夹爪 absolute]
```

`ActionChunkEncoder` 对 `[H, action_dim]` 的 action chunk 做逐步 MLP 编码，并加入 learned timestep embedding，再通过 temporal attention pooling 得到一个 chunk-level action feature。这样 double-Q head 评估的是整段候选动作序列，而不是单步动作。

### Norm Stats

critic 训练依赖一份和 pi0.5 SFT 相同口径的统计量：

```yaml
data:
  norm_stats_path: /data/wam_codebase/RLinf/datasets/robotwin_aloha/pi05_norm_stats.json
  use_quantile_norm: true
  adapt_to_pi: true
```

这份统计量必须在 `AlohaInputs -> DeltaActions -> Normalize` 同一语义下计算，至少包含：

```text
state:   mean / std / q01 / q99
actions: mean / std / q01 / q99
```

如果训练时报 `norm_stats_path` 不存在，说明还没有把 pi0.5 口径的统计量生成并放到该路径。可以用 `toolkits/lerobot/calculate_norm_stats.py` 按对应 OpenPI dataconfig 生成，再把输出的 `norm_stats.json` 放到配置指定位置。

## Loss 逻辑

当前训练使用 online critic 和 EMA target critic：

```text
target <- (1 - tau) * target + tau * online
```

chunk TD target：

```text
reward_sum = sum_i gamma^i r_{t+i}
target_q = reward_sum + gamma^H * (1 - done) * Quantile(V_target(s_{t+H}))
```

Q loss：

```text
L_Q = MSE(Q1(s_t, a_t:t+H), target_q)
    + MSE(Q2(s_t, a_t:t+H), target_q)
```

V loss：

```text
target_v = min(Q_target1(s_t, a_t:t+H), Q_target2(s_t, a_t:t+H))
L_V = CE(project_to_value_atoms(target_v), V_logits(s_t))
```

默认关键超参：

```yaml
action_horizon: 50
gamma: 0.9999
ema_tau: 0.005
num_bins: 201
v_min: -0.1
v_max: 1.1
quantile_tau: 0.6
```

`v_min=-0.1, v_max=1.1` 给二值成功奖励附近留出少量边界，避免 value target 贴在 0/1 边缘时全部被硬截断。

## RLinf-native 训练入口

配置文件：

```text
examples/lwd/config/robotwin_lwd_critic.yaml
```

当前训练入口已经对齐 RLinf 的 SFT runner 路径：

```text
Hydra config
  -> Cluster
  -> HybridComponentPlacement
  -> FSDPLWDCriticWorker
  -> SFTRunner
```

训练生命周期由 RLinf 管理：

```text
MetricLogger
FSDP wrapping
gradient accumulation
mixed precision
checkpoint / resume
periodic eval
```

运行前需要把模型路径、数据路径和 norm stats 路径改成真实路径：

```yaml
actor:
  model:
    siglip_path: /path/to/siglip2-so400m-patch14-224
    gemma3_path: /path/to/gemma-3-270m
    tokenizer_path: /path/to/gemma-3-270m

data:
  train_data_paths:
    - dataset_path: /path/to/success_lerobot_dataset
    - dataset_path: /path/to/failed_lerobot_dataset
  eval_data_paths:
    - dataset_path: /path/to/eval_lerobot_dataset
  norm_stats_path: /path/to/pi05_norm_stats.json
```

启动示例：

```bash
cd /data/wam_codebase/RLinf
bash examples/lwd/run_lwd_critic.sh robotwin_lwd_critic
```

### 云端 beat_block 训练

云端优先使用已经按 episode 切好的数据：

```text
datasets/robotwin_aloha_lwd_split/
  beat_block_hammer_success_train
  beat_block_hammer_success_eval
  beat_block_hammer_failed_train
  beat_block_hammer_failed_eval
  beat_block_hammer_nearmiss_train
  beat_block_hammer_nearmiss_eval
  pi05_norm_stats.json
```

对应配置文件：

```text
examples/lwd/config/robotwin_lwd_critic_cloud_beat_block.yaml
examples/lwd/config/robotwin_lwd_critic_cloud_beat_block_smoke.yaml
```

这些 cloud 配置只依赖 `examples/lwd/config` 下的本地配置组，不需要额外
Hydra search path 指向 `examples/sft/config`。

默认云端路径通过环境变量控制：

```bash
export REPO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/RLinf
export RLINF_LWD_DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/datasets/rl_data/robotwin_aloha_lwd_split
export RLINF_LWD_LOG_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/checkpoints/rlinf_lwd
export RLINF_SIGLIP_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/weights/pretrained/siglip2-so400m-patch14-224
export RLINF_GEMMA3_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/weights/pretrained/gemma-3-270m
export RLINF_TOKENIZER_PATH=$RLINF_GEMMA3_PATH
```

提交文件：

```text
examples/lwd/hope/robotwin_lwd_critic_smoke_8a100.hope
examples/lwd/hope/robotwin_lwd_critic_train_8a100.hope
```

建议先跑 smoke 文件，确认 import、dataloader、FSDP 初始化和一次 eval 都能跑通；
再提交正式训练文件。正式配置默认使用 8 卡：

```yaml
runner:
  max_steps: 8000
  val_check_interval: 500
  save_interval: 2000
actor:
  micro_batch_size: 4
  global_batch_size: 64
  optim:
    lr: 5.0e-5
    value_lr: 1.0e-4
    lr_warmup_steps: 500
```

这对应每张卡一次处理 4 条，8 卡合计 32 条，每 2 个 micro-batch
做一次 optimizer update。后续如果显存和吞吐都稳定，可以在提交命令中覆盖为：

```bash
actor.micro_batch_size=8 actor.global_batch_size=64
```

TensorBoard 是离线本地日志。当前 RLinf `MetricLogger` 会把事件文件写到：

```text
${RLINF_LWD_LOG_ROOT}/tensorboard/
```

checkpoint 会按实验名写在：

```text
${RLINF_LWD_LOG_ROOT}/<experiment_name>/
```

恢复训练时把 `runner.resume_dir` 指向 checkpoint 的 `global_step_*` 目录，例如：

```bash
runner.resume_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/checkpoints/rlinf_lwd/robotwin_lwd_critic_train_8a100/checkpoints/global_step_2000
```

### 当前 FSDP 边界

LWD critic 有 EMA target critic。为了避免 FSDP flat/sharded
parameters 和 target model 的参数语义错位，当前版本显式使用：

```yaml
actor:
  fsdp_config:
    strategy: fsdp
    sharding_strategy: no_shard
    use_orig_params: true
```

这条路径支持单卡和多卡数据并行。后续如果需要 `full_shard`，应单独实现
sharded EMA target，而不是直接把当前配置切过去。

## 本次解决的核心问题

1. 补齐了 critic 的 action-conditioned Q 分支

   原有 ReCap/value model 更接近 `V(s)`，当前版本新增 `Q(s, a_chunk)`，使 critic 能评估候选动作序列质量，这是后续 Actor-Critic 策略优化的基础。

2. 修正了 state 没有进入 critic 语义输入的问题

   当前 state 按 pi0.5 离散 state prompt 进入 tokenizer，critic 不再只依赖图像和任务语言。

3. 对齐了 critic 和 pi0.5 actor 的 action 空间

   action chunk 从原始 qpos target 转成 pi0.5 使用的 delta joint + absolute gripper 表达，避免后续用 critic 指导 actor 时出现动作语义错位。

4. 显式化了 normalization 依赖

   训练配置现在要求提供 pi0.5 口径的 `norm_stats_path`，state/action 的归一化不再隐含在代码外部，后续排查数据问题会更直接。

5. 对齐了 LWD 风格 value 支撑区间和 chunk TD 训练方式

   使用 201-bin distributional value head、`[-0.1, 1.1]` value support、`gamma=0.9999`、EMA target critic 和 chunk bootstrap。

6. 清理并保留了必要配置字段

   删除了 LWD 内部没有使用的冗余输出字段；同时保留 `is_lora: false` 这类 RLinf 公共建模入口需要的字段，避免配置看似精简但运行时报缺字段。

## 当前边界

当前版本是 critic 训练闭环，还没有实现：

```text
actor policy update
online rollout 后训闭环
自适应 action proposal / action refinement
更复杂的 advantage 或 policy improvement 逻辑
```

训练期间 `FSDPLWDCriticWorker.run_eval()` 会记录基础 LWD 指标。仍建议再做
更深入的离线诊断，验证 critic 本身是否有区分能力：

```text
成功轨迹的 V/Q 整体高于失败轨迹；
near-miss 轨迹在失误附近 V/Q 明显下降；
同一状态下 expert action chunk 的 Q 高于扰动 action chunk；
不同任务之间 value 曲线不出现明显尺度崩坏。
```

critic 质量稳定后，再把它接入后续 actor 更新和策略优化探索。
