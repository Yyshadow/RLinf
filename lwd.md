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

rlinf/models/embodiment/openpi/
  __init__.py         # OpenPI/pi0.5 模型加载入口，支持显式 norm_stats_path
  openpi_action_model.py
                      # OpenPi0Config，包含 train_expert_only / norm_stats_path 等字段

examples/lwd/
  train_lwd_critic.py
  run_lwd_critic.sh
  config/robotwin_lwd_critic.yaml
  config/model/lwd_critic.yaml
  config/training_backend/fsdp.yaml
                      # LWD 自包含的模型和 FSDP 配置片段

examples/sft/
  config/robotwin_sft_openpi_pi05_hammer50_cloud.yaml
  hope/robotwin_pi05_hammer50_smoke_8a100.hope
  hope/robotwin_pi05_hammer50_train_8a100.hope
                      # pi0.5 hammer50 quick SFT 的云端配置和提交文件
```

## Robotwin pi0.5 / LWD 数据闭环

Robotwin 数据闭环的目标是先把 RoboTwin/RLinf 采集结果整理成 OpenPI pi0.5
能直接读取的 LeRobot-Aloha 格式，再从同一份数据派生 LWD/BPG critic 需要的
chunk 索引。这里不改 RLinf 训练框架，也不改通用 `CollectEpisode`。

核心工具：

```text
toolkits/robotwin/prepare_lerobot_aloha.py
toolkits/robotwin/validate_lerobot_aloha.py
toolkits/robotwin/build_lwd_chunks.py
```

### 数据主格式

pi0.5 Robotwin 训练使用 OpenPI Aloha dataconfig，核心字段是：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state
action
task
```

其中 `observation.state` 和 `action` 都是 14 维 Aloha 关节/夹爪向量。
`action` 保存 absolute action，不提前转 delta；`pi05_aloha_robotwin`
配置会在 dataconfig 里处理 delta action。

当前 Robotwin 数据集采用 LeRobot 的 `image` 存法，不是 `video` 存法，所以
`meta/info.json` 里看到下面内容是正常的：

```text
total_videos: 0
video_path: null
```

训练 pi0.5 只需要 LeRobot parquet 里的图像字节和特征 schema，不要求一定有
单独 mp4。如果只是想看视频，额外导出可视化即可，不必把训练数据本身改成
video 格式。

如果环境不能写 `~/.cache`，可以把缓存放到 `/tmp`：

```bash
export HF_HOME=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export MPLCONFIGDIR=/tmp/matplotlib
```

### 1. 转换数据

RoboTwin 原生 hdf5 转 OpenPI Aloha LeRobot：

```bash
python toolkits/robotwin/prepare_lerobot_aloha.py \
  --source /data/wam_codebase/RoboTwin_RLinf/demo_videos/_work/beat_block_hammer/data \
  --output /data/wam_codebase/RLinf/datasets/robotwin_aloha/beat_block_hammer \
  --task beat_block_hammer \
  --source-label expert_success
```

如果输入是 RLinf `CollectEpisode` 产出的通用 LeRobot 数据，也可以直接传数据集根目录：

```bash
python toolkits/robotwin/prepare_lerobot_aloha.py \
  --source /path/to/generic_lerobot_dataset \
  --output /path/to/robotwin_aloha_dataset \
  --source-type lerobot
```

转换脚本会额外写：

```text
meta/robotwin_episode_metadata.jsonl
```

里面保存 episode 级信息，例如 `source`、`success`、`return`、`num_steps`。
这些字段不是 pi0.5 SFT 必需，但后续筛数据和训练 critic 会用。

对于 RoboTwin 原生 hdf5，逐帧 `success` 只在最后一帧为 True；episode 是否
成功看 `meta/robotwin_episode_metadata.jsonl`。这样可以避免把整条轨迹每一步
都误当成成功状态。

如果转换失败轨迹，显式传 `--no-success --source-label failed_policy`；如果输入
数据本身已经有 `success` / `is_success` 字段，脚本会优先保留原字段。

### 2. 校验数据

训练前先检查字段、shape、OpenPI transform：

```bash
python toolkits/robotwin/validate_lerobot_aloha.py \
  --dataset /data/wam_codebase/RLinf/datasets/robotwin_aloha/beat_block_hammer \
  --config-name pi05_aloha_robotwin
```

这个脚本会检查：

```text
三路相机字段是否存在
state/action 是否为 14 维
gripper 是否在 [0,1]
OpenPI dataconfig 是否能读出 state/actions/images/prompt
```

### 3. 计算 pi0.5 归一化统计

校验通过后，用现有工具计算 norm stats：

```bash
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_aloha_robotwin \
  --repo-id /data/wam_codebase/RLinf/datasets/robotwin_aloha/beat_block_hammer
```

本地路径作为 `--repo-id` 时，`norm_stats.json` 会写到这个数据集根目录下。
训练时同一个 repo id / 本地路径会读到这份统计。

### 4. 训练 pi0.5 SFT

修改或覆盖 `examples/sft/config/robotwin_sft_openpi_pi05.yaml`：

```yaml
data:
  train_data_paths: /data/wam_codebase/RLinf/datasets/robotwin_aloha/beat_block_hammer

actor:
  model:
    model_path: /path/to/pi05_base_or_checkpoint
    openpi:
      config_name: pi05_aloha_robotwin
```

然后运行：

```bash
bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pi05
```

pi0.5 SFT 只应该使用成功 expert 或修正后的 near-miss 数据。普通失败轨迹不要
直接作为 BC 正样本。

### 5. 生成 LWD/BPG chunk 索引

同一份 LeRobot-Aloha 数据可以派生 critic 训练用 chunk：

```bash
python toolkits/robotwin/build_lwd_chunks.py \
  --dataset /data/wam_codebase/RLinf/datasets/robotwin_aloha/beat_block_hammer \
  --output /data/wam_codebase/RLinf/datasets/robotwin_chunks/beat_block_hammer_H10.parquet \
  --horizon 10 \
  --stride 1
```

chunk 文件包含：

```text
episode_id
frame_idx
next_frame_idx
task
source
success
done
state
next_state
action_chunk
reward_chunk
reward_sum
```

图像不复制进 chunk 文件，只保留 episode/frame 索引；critic dataloader 后续按索引
回读 LeRobot 数据，避免数据膨胀。

### 可视化和 eval 视频

如果只想快速看一个离线 episode，推荐导出成一个视频加一个 json：

```bash
python toolkits/lerobot/visualize_lerobot_dataset.py \
  --dataset-path /data/wam_codebase/RLinf/datasets/robotwin_aloha/beat_block_hammer \
  --output-dir /data/wam_codebase/RLinf/datasets/robotwin_vis/beat_block_hammer \
  --mode video \
  --camera-key observation.images.cam_high
```

输出会是：

```text
episode_000000/episode_000000_cam_high.mp4
episode_000000/episode_000000.json
```

pi0.5 eval 时不要为了录像把 action chunk 拆开执行。正式评估应保持
`num_action_chunks=50` 的原始语义：模型一次输出 `[B, 50, 14]`，RoboTwin
对完整 chunk 做 TOPP 规划并执行。

如果要验证更高频重规划，可以在 eval 配置里设置：

```yaml
env:
  eval:
    action_exec_horizon: 20
```

这表示模型仍然一次预测 50 步 action chunk，但 RoboTwin 只执行前 20 步，
然后重新观测、重新推理下一段 50 步。`max_steps_per_rollout_epoch=200` 时，
`action_exec_horizon=20` 会产生 10 次模型推理。默认不设置该字段时沿用旧逻辑：
按 `max_steps_per_rollout_epoch // num_action_chunks` 做完整 chunk 执行；在当前
`200/50` 配置下就是 4 次推理、完整 chunk 执行。这个参数只影响 eval，不改变
SFT 训练数据的 action chunk 长度。

如果需要看完整过程，使用 RoboTwin 内部相机录制：

```yaml
env:
  eval:
    video_cfg:
      save_video: true
      record_internal_camera: true
      camera: head_camera
      fps: 25
      write_every_n_sim_steps: 10
```

这一路只在已有 `scene.step()` 后读取相机 RGB 并写 mp4，不新增仿真步进、
不改变 TOPP 输入，也不改变 reward / success 判断。只做批量指标时可以关掉
`save_video` 以节省渲染和编码时间。

### Smoke test

已用现有样例
`/data/wam_codebase/RoboTwin_RLinf/demo_videos/_work/beat_block_hammer/data/episode0.hdf5`
跑通：

```text
prepare_lerobot_aloha.py -> 1 episode / 64 frames
validate_lerobot_aloha.py -> OpenPI pi05_aloha_robotwin transform passed
calculate_norm_stats.py -> 写出 norm_stats.json
build_lwd_chunks.py --horizon 10 -> 54 chunks
visualize_lerobot_dataset.py --mode video -> 1 mp4 + 1 json
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
examples/lwd/scripts/train_lwd_critic_cloud.sh
```

hope 文件只负责云端资源、docker、failover 等平台配置；真正的环境变量、
离线 cache、import 检查、训练命令和自动 resume 逻辑都在
`train_lwd_critic_cloud.sh` 里。建议先跑 smoke 文件，确认 import、dataloader、
FSDP 初始化和一次 eval 都能跑通；再提交正式训练文件。正式配置默认使用 8 卡：

```yaml
runner:
  max_steps: 8000
  val_check_interval: 500
  save_interval: 1000
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

正式 train 模式会自动扫描最新完整 checkpoint 并追加 `runner.resume_dir`。LWD
critic 的完整 checkpoint 需要同时包含：

```text
actor/dcp_checkpoint/.metadata
actor/model_state_dict/full_weights.pt
actor/target_model.pt
```

如果云端自动重启，同一个 hope 会再次执行 cloud 脚本，并从最新完整的
`global_step_*` 继续训练。若要强制从头开始，提交前设置：

```bash
export RLINF_FORCE_RESTART=1
```

### pi0.5 hammer50 50 条数据 SFT

这一部分训练的是 pi0.5 actor，不是 LWD critic，所以入口放在 `examples/sft`。
它用于用少量 hammer 成功数据先做 actor 过拟合诊断，确认训练链路、数据统计量、
闭环评估和视频诊断都可靠，再和 critic 训练结果一起做后续策略优化实验。

当前 SFT 使用 50 条 `beat_block_hammer` 成功 episode：

```text
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/datasets/rl_data/robotwin_aloha_pi05_quick/
  beat_block_hammer_success_50_train/
    data/
    meta/
    norm_stats.json
  beat_block_hammer_success_20_eval/
```

训练权重目录默认是：

```text
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/weights/rlinf_pi05_pytorch/pi05_base_hammer50
```

这里的 `model.safetensors` 仍然是 pi0.5 base 权重；`norm_stats.json` 是由
50 条训练数据按 `pi05_aloha_robotwin` 口径计算出来的数据统计量。当前
OpenPI loader 已经支持显式配置：

```yaml
actor:
  model:
    openpi:
      norm_stats_path: /path/to/beat_block_hammer_success_50_train/norm_stats.json
```

如果设置了 `norm_stats_path`，模型 wrapper 和 OpenPI SFT dataloader 都会读取
这份训练集统计量；如果不设置，仍然兼容旧逻辑，从
`model_path/<asset_id>/norm_stats.json` 读取。这样可以避免把数据集统计量强行塞进
base model 目录。

对应配置文件：

```text
examples/sft/config/robotwin_sft_openpi_pi05_hammer50_cloud.yaml
```

默认关键参数：

```yaml
runner:
  max_steps: 10000
  val_check_interval: -1
  save_interval: 500
  logger:
    experiment_name: pi05_hammer50_overfit50_10k_v1

actor:
  micro_batch_size: 4
  global_batch_size: 64
  model:
    num_action_chunks: 50
    action_dim: 14
    openpi:
      config_name: pi05_aloha_robotwin
      train_expert_only: false
      noise_level: 0.5
      num_images_in_input: 3
      detach_critic_input: true
  optim:
    lr: 2.5e-5
    min_lr: 2.5e-6
    weight_decay: 1.0e-10
    lr_warmup_steps: 500
```

这里采用 `train_expert_only: false`，也就是 full finetuning。原因是当前
RLinf + OpenPI pi0.5 + FSDP SFT 的 expert-only 路径会触发 Gemma expert
view/inplace autograd 报错；在没有代码级修复前，full finetuning 是稳定方案。
8 卡下 `micro_batch_size=4, global_batch_size=64` 表示每张卡一次处理 4 条，
全局累计到 64 条后做一次 optimizer update。

这次不是从 `global_step_500` resume，而是从 `pi05_base_hammer50` base package
重新启动新实验。旧 500-step run 的学习率和 cosine schedule 都是短训练设置，
直接 resume 会继承旧 optimizer/scheduler 状态，不适合作为 10k overfit 实验起点。

默认云端路径通过环境变量控制：

```bash
export REPO_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/RLinf
export EMBODIED_PATH=$REPO_PATH/examples/sft
export RLINF_PI05_DATA_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/datasets/rl_data/robotwin_aloha_pi05_quick
export RLINF_PI05_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/weights/rlinf_pi05_pytorch/pi05_base_hammer50
export RLINF_PI05_NORM_STATS_PATH=$RLINF_PI05_DATA_ROOT/beat_block_hammer_success_50_train/norm_stats.json
export RLINF_PI05_LOG_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/checkpoints/rlinf_pi05_sft
export OPENPI_DATA_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/weights
```

OpenPI 的 pi0/pi0.5 文本 tokenizer 不是 HuggingFace tokenizer，而是 Big Vision
的 PaliGemma SentencePiece 文件。OpenPI 默认会从：

```text
gs://big_vision/paligemma_tokenizer.model
```

下载并缓存到：

```text
${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model
```

云端离线训练前必须提前准备这个文件。它是一个 tokenizer asset，不是新的 pi0.5
模型权重。可以在有网络的环境先下载一次，再上传到云端对应路径；也可以如果之前
OpenPI 训练已经用同一个 `OPENPI_DATA_HOME` 跑通过，直接复用已有缓存。当前
hammer50 smoke/train hope 会在启动训练前检查这个文件是否存在，缺失时直接报错，
避免训练时再尝试访问 Google Cloud。

提交文件：

```text
examples/sft/hope/robotwin_pi05_hammer50_smoke_8a100.hope
examples/sft/hope/robotwin_pi05_hammer50_train_8a100.hope
examples/sft/scripts/train_pi05_hammer50_cloud.sh
```

hope 文件只保留云端资源、docker、failover 等平台配置；conda 环境、离线缓存、
OpenPI tokenizer 检查、训练命令和自动 resume 逻辑都在
`train_pi05_hammer50_cloud.sh` 里。

建议先提交 smoke：

```bash
hope run examples/sft/hope/robotwin_pi05_hammer50_smoke_8a100.hope
```

确认 import、OpenPI dataloader、FSDP 初始化和 checkpoint 保存都通过后，再提交正式
10k-step overfit SFT：

```bash
hope run examples/sft/hope/robotwin_pi05_hammer50_train_8a100.hope
```

checkpoint 会保存到：

```text
${RLINF_PI05_LOG_ROOT}/<experiment_name>/checkpoints/global_step_<N>/actor
```

正式 train 模式会自动扫描最新完整 checkpoint 并追加 `runner.resume_dir`。pi0.5
SFT 的完整 checkpoint 需要同时包含：

```text
actor/dcp_checkpoint/.metadata
actor/model_state_dict/full_weights.pt
```

如果云端自动重启，同一个 hope 会再次执行 cloud 脚本，并从最新完整的
`global_step_*` 继续训练。若要强制从头开始，提交前设置：

```bash
export RLINF_FORCE_RESTART=1
```

当前 embodied SFT worker 没有实现单独 eval，所以 SFT 训练中主要看
`train/loss` 是否继续下降、checkpoint 是否每 500 step 正常保存。闭环效果需要
后续用 rollout/eval 脚本验证；建议先用训练集 seeds 做 `total_num_envs<=4` 的快速
评估，再对候选 checkpoint 开启 `record_internal_camera=true` 生成完整视频。

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

7. 解耦了 pi0.5 base 权重和数据集统计量

   OpenPI 模型配置新增 `openpi.norm_stats_path`，模型 wrapper 和 SFT dataloader 会使用同一份训练集 `norm_stats.json`。base model 目录不再必须承担数据集统计量职责，云端切换数据集时只需要改配置或环境变量。

8. 补齐了 pi0.5 hammer50 quick SFT 云端入口

   新增 `robotwin_sft_openpi_pi05_hammer50_cloud.yaml` 和对应 8A100 smoke/train hope 文件，用 50 条 hammer success demo 做 pi0.5 full finetuning，便于在 critic 后续实验前快速得到一个 actor baseline。

9. 统一了 RoboTwin eval 视频语义

   评估视频改为 RoboTwin 内部相机录制：模型仍一次输出 `[B, 50, 14]`，环境仍按完整 chunk 做 TOPP 执行，录像只在已有 physics step 后读取相机帧，不再使用会改变执行语义的拆 chunk 视频路径。

10. 合并了 Robotwin 数据闭环说明

   数据转换、校验、norm stats、pi0.5 SFT、LWD chunk 索引和视频建议已经统一放进本文档，删除单独说明文件，避免后续两份文档不一致。

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
