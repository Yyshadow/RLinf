# pi0.5 Hammer50 SFT 调试记录

## 2026-07-06：pi0.5 SFT expert-only 的 view-inplace 报错

### 现象

云端运行 `robotwin_sft_openpi_pi05_hammer50_cloud.yaml` 时，环境、离线 tokenizer、OpenPI import、RLinf import 都已经通过，训练在第一步 forward/backward 前后失败。

关键日志形态：

```text
pi05 sft imports ok
Forcing gradient checkpointing to be enabled for Gemma expert model
RuntimeError: Output 0 of ViewBackward0 is a view and its base or another view of its base has been modified inplace.
```

报错栈进入了 OpenPI 的 Gemma expert：

```text
FSDPVlaSftWorker.run_training
-> OpenPi0ForRLActionPrediction.sft_forward
-> PI0Pytorch.forward
-> paligemma_with_expert.forward
-> openpi/models_pytorch/gemma_pytorch.py
```

所以这次不是数据路径、norm_stats、tokenizer、离线缓存的问题，而是 pi0.5 SFT 的模型前向路径触发了 PyTorch autograd 的 view/inplace 限制。

### 根因判断

当前失败配置里 `actor.model.openpi.train_expert_only: true`，也就是只训练 action expert、冻结 PaliGemma/VLM。

RLinf 现有配置中已经记录过同类问题：

```yaml
# model/pi0_5 defaults train_expert_only to True, which freezes
# PaliGemma and breaks FSDP+gradient-checkpointing with a
# view-inplace error. SFT here trains the full model.
train_expert_only: False
```

对应文件：

```text
examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml
```

也就是说，这不是 hammer50 数据特有问题，而是当前 RLinf + OpenPI pi0.5 + FSDP SFT 路径下的已知组合风险：

```text
pi0.5 SFT
+ train_expert_only=True
+ FSDP
+ OpenPI 内部 Gemma expert gradient checkpointing
=> view-inplace autograd error
```

即使 RLinf 配置里写了：

```yaml
actor.fsdp_config.gradient_checkpointing: false
```

日志里仍然出现：

```text
Forcing gradient checkpointing to be enabled for Gemma expert model
```

说明 OpenPI 内部仍会在 Gemma expert 侧强制启用相关 checkpoint 路径，最终与 FSDP/冻结参数组合产生冲突。

### 为什么 PPO/GRPO 默认可以 expert-only

PPO/GRPO 的 OpenPI 配置多数继承：

```text
examples/embodiment/config/model/pi0_5.yaml
```

其中默认：

```yaml
train_expert_only: True
```

这表示 RL 后训练通常冻结 VLM，只训练 action expert；如果 `add_value_head=True`，还会训练 value head。

它和 SFT 的关键区别是 forward 路径不同：

```text
SFT:
  sft_forward()
  -> super().forward(observation, actions)
  -> OpenPI 原生 PI0Pytorch.forward

PPO/GRPO:
  default_forward()
  -> get_log_prob_value()
  -> 对 rollout 保存的 denoise chain 重算 logprob/value
```

所以 PPO/GRPO 默认 expert-only 不等于 SFT expert-only 一定可跑。当前出错的是 SFT 的 `super().forward()` 路径。

### 当前处理方案

为了先把 hammer50 的 pi0.5 SFT 云端训练链路跑通，当前采用 full finetuning：

```yaml
actor.model.openpi.train_expert_only: false
```

最初用于验证链路的 500-step 配置比较保守：

```yaml
actor.micro_batch_size: 2
actor.global_batch_size: 16
actor.optim.lr: 7.5e-6
actor.optim.min_lr: 7.5e-7
runner.max_steps: 500
runner.save_interval: 100
```

smoke 作业进一步覆盖为更小配置：

```yaml
actor.micro_batch_size: 1
actor.global_batch_size: 8
runner.max_steps: 10
runner.save_interval: 10
```

推荐运行顺序：

1. 先跑 `examples/sft/hope/robotwin_pi05_hammer50_allstats_smoke_8a100.hope`，确认 10 step 能完整 forward/backward/save。
2. smoke 通过后，再跑 `examples/sft/hope/robotwin_pi05_hammer50_allstats_train_8a100.hope`。
3. 如果 full finetune 显存稳定，正式训练使用下面的 10k overfit 配置；如果 OOM，再把 `micro_batch_size` 从 4 降到 2。

### 后续如果必须只训 expert

只训 expert 不是理论上不可行，但当前不能只靠 YAML 稳定解决。需要代码级处理 OpenPI 内部强制启用 Gemma expert gradient checkpointing 与 FSDP/view 的冲突，例如：

- 禁止 expert-only SFT 时 Gemma expert 进入强制 checkpoint 路径；
- 或在 OpenPI Gemma expert 的 view/inplace 敏感路径上做更细的 out-of-place/clone 处理；
- 或改用不触发该 SFT `super().forward()` 的训练路径。

在没有完成代码级修复前，hammer50 SFT 的稳定方案是 full finetuning。

## 2026-07-06：hammer50 50 条数据的 10k overfit 方案

step500 checkpoint 已经能保存并 eval，但闭环效果不理想：普通 eval seed 失败，
训练集 seed 上的快速闭环检查也没有成功。这说明问题不只是泛化，至少当前 500
step 还没有把 50 条成功数据拟合到足够好。

参考 RLinf 自带的 RobotWin pi0.5 SFT 配置：

```text
examples/sft/config/robotwin_sft_openpi_pi05.yaml
```

官方风格的关键参数是：

```yaml
actor.micro_batch_size: 32
actor.global_batch_size: 64
actor.optim.lr: 2.5e-5
actor.optim.weight_decay: 1.0e-10
actor.optim.lr_warmup_steps: 1000
actor.model.openpi.train_expert_only: false
```

hammer50 只有 50 条成功 demo，目标先不是泛化，而是确认训练链路能否把训练集
overfit 到闭环可用。因此当前正式云端配置改为：

```yaml
runner.max_steps: 10000
runner.save_interval: 500
runner.logger.experiment_name: pi05_hammer50_overfit50_10k_v1

actor.micro_batch_size: 4
actor.global_batch_size: 64
actor.model.openpi.train_expert_only: false
actor.model.openpi.noise_level: 0.5

actor.optim.lr: 2.5e-5
actor.optim.min_lr: 2.5e-6
actor.optim.weight_decay: 1.0e-10
actor.optim.lr_warmup_steps: 500
actor.optim.total_training_steps: ${runner.max_steps}
```

这里不从 `global_step_500` 继续 resume。原因是旧 500-step run 用的是更小学习率和
短 cosine schedule，末尾学习率已经衰减很多；继续 resume 会继承旧 optimizer 和
scheduler 状态，不适合新的 10k overfit 实验。应该从
`pi05_base_hammer50` base package 重新启动一个新实验。

后续验证建议：

1. 每 500 step 保存一次 checkpoint。
2. 优先看 `train/loss` 是否继续降到更低，至少先接近 `0.00x` 量级。
3. 每个 checkpoint 用训练集 seeds 做闭环 eval，`total_num_envs` 建议先不超过 4，
   因为 RoboTwin 多相机环境曾在 10 env 下出现 `cannot create buffer`。
4. 对最好的 checkpoint 开启 `record_internal_camera=true` 生成完整视频，确认失败
   是模型动作问题还是评估/录像误判。这个录像路径不拆 action chunk。

## 2026-07-07：RoboTwin pi0.5 eval 内部相机视频逻辑

### 现象

最开始用 `evaluations/robotwin/robotwin_beat_block_hammer_openpi_pi05_eval.yaml`
做 step500 checkpoint 可视化时，视频可以生成，但只有很少几帧：

```text
200 env steps
num_action_chunks = 50
=> 只有约 4 个 chunk 末尾观测 + reset/final 相关帧
=> mp4 大约 5-6 帧，无法判断中间抓锤子、移动、敲击过程
```

这个不是 mp4 编码坏了，也不是 TensorBoard 或 `ffmpeg` 问题，而是
RoboTwin eval 的 step 接口返回粒度太粗。

### 为什么外层 RecordVideo 会很短

RLinf 的视频保存由 `RecordVideo` wrapper 负责：

```text
rlinf/workers/env/env_worker.py
  if env_cfg.video_cfg.save_video:
      env = RecordVideo(env, env_cfg.video_cfg)

rlinf/envs/wrappers/record_video.py
  RecordVideo.chunk_step()
  -> env.chunk_step()
  -> record_video_in_result(result)
  -> 从 obs_list 里取图并写 mp4
```

也就是说，`RecordVideo` 能保存多少帧，取决于 `RoboTwinEnv.chunk_step()`
返回的 `obs_list` 里有多少个观测。

原来的 `RoboTwinEnv.chunk_step()` 是一次性把完整 action chunk 交给
RoboTwin：

```text
chunk_actions: [B, 50, 14]
self.venv.step(chunk_actions)
```

然后只调用一次 `_extract_obs_image(raw_obs)`，所以 `obs_list` 里只有
一个 chunk 末尾观测。对于 `max_episode_steps=200`、`chunk=50` 的 eval，
整条 episode 只有 4 次 `chunk_step()`，因此视频天然只有几帧。

注意：只改 YAML 里的 `render_freq: 1` 不足以解决这个问题。RLinf 当前
视频不是直接读取 RoboTwin 内部 render 日志，而是读取 env step 返回的
observation；只要 `chunk_step()` 只返回 chunk 末尾观测，`RecordVideo`
就没有中间帧可以写。

### 当前解决方案

正式方案改为 RoboTwin 内部相机录制，不再拆 action chunk。RLinf 只把
`video_cfg` 传入 RoboTwin，正式执行路径仍然保持：

```text
OpenPI output: [B, 50, 14]
RoboTwinEnv.chunk_step()
  -> self.venv.step(chunk_actions)
  -> RoboTwin gen_sparse_reward_data(chunk_actions)
  -> TOPP 对完整 50-step chunk 规划
  -> 内部 scene.step() 执行
```

新增配置：

```yaml
env:
  eval:
    video_cfg:
      save_video: true
      record_internal_camera: true
      camera: head_camera
      fps: 25
      write_every_n_sim_steps: 10
      video_base_dir: ${runner.logger.log_path}/video/internal
```

核心代码改动：

```text
rlinf/envs/robotwin/robotwin_env.py
  _init_env()
  -> task_config["video_cfg"] = cfg.video_cfg

rlinf/workers/env/env_worker.py
  record_internal_camera=true 时跳过外层 RecordVideo wrapper

/data/wam_codebase/RoboTwin_RLinf/robotwin/envs/vector_env.py
  VectorEnv 读取 video_cfg
  -> eval_video_log / eval_video_base_dir / fps / stride / camera

/data/wam_codebase/RoboTwin_RLinf/envs/_base_task.py
  gen_sparse_reward_data()
  -> 原有 scene.step()
  -> 原有 _update_render()
  -> 每隔 write_every_n_sim_steps 只读取 head_camera RGB 并写入 ffmpeg
```

这个方案只增加相机读取和视频写入，不新增 `scene.step()`，不改变 TOPP 输入，
不改变 joint target，不改变 reward / success 计算。

### 为什么移除旧拆 chunk 录像开关

之前的临时方案是在 RLinf wrapper 里把 `[B, 50, 14]` 拆成 50 次
`[B, 1, 14]` 执行。它能让外层 `RecordVideo` 得到更多帧，但会改变
RoboTwin 的执行语义：

```text
完整 chunk:
  TOPP 对 50 个路点整体规划

拆 chunk:
  TOPP 每次只看 1 个路点，重复局部规划 50 次
```

因此旧拆 chunk 录像开关和对应函数已移除。
正式视频现在只能走内部相机录制，避免把诊断视频误当成正式执行效果。

### 这个方案解决了什么

1. 解决了 `chunk=50` eval 视频只有 5-6 帧的问题。
2. 保留 pi0.5 policy 一次输出 50 个 action 的真实推理方式。
3. 保留 RoboTwin/TOPP 对完整 chunk 的原始执行方式。
4. 录像不会改变仿真步进、动作执行、reward 或 success 判断。
5. 多 env 录像目录按 `task/seed/env_id` 隔离，避免并行 eval 覆盖视频。

### 推荐使用方式

生成单条正式语义视频：

```bash
PATH=/data/wam_codebase/RLinf/.venv-openpi/bin:$PATH \
ROBOTWIN_PATH=/data/wam_codebase/RoboTwin_RLinf \
ROBOTWIN_ASSETS_PATH=/data/wam_codebase/RoboTwin_RLinf \
PYTHONPATH=/data/wam_codebase/RLinf:/data/wam_codebase/RoboTwin_RLinf \
HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp CUDA_VISIBLE_DEVICES=0 \
bash evaluations/run_eval.sh robotwin robotwin_beat_block_hammer_openpi_pi05_eval \
  env.eval.total_num_envs=1 \
  runner.logger.log_path=/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_internal_video
```

如果只做批量指标评估、不需要视频，可以关掉内部相机录制以节省时间：

```bash
bash evaluations/run_eval.sh robotwin robotwin_beat_block_hammer_openpi_pi05_eval \
  env.eval.video_cfg.save_video=false \
  env.eval.video_cfg.record_internal_camera=false \
  env.eval.total_num_envs=4 \
  runner.logger.log_path=/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_metrics
```

## 2026-07-07：视频与数据闭环文档最终清理

本次进一步把临时方案和重复文档收紧：

1. 移除旧拆 chunk 录像配置和对应代码路径。RoboTwin eval 现在只有内部相机录制
   这一条正式视频路径，不再保留会改变 TOPP 执行语义的调试开关。
2. `RoboTwinEnv.chunk_step()` 永远把完整 `[B, 50, 14]` chunk 交给
   `venv.step(chunk_actions)`，避免视频诊断和正式 success rate 使用不同执行方式。
3. 内部相机录制开启时，`EnvWorker` 会跳过 RLinf 外层 `RecordVideo` wrapper，
   避免同时生成一份只有 chunk 末尾帧的旧式视频。
4. 删除 `RoboTwinEnv` 中没有任何指标产出的 `fail_once` 缓存，以及未使用且引用
   未定义 `self.horizon` 的 `sample_action_space()`。
5. 将单独的数据闭环说明合并进 `lwd.md` 的 “Robotwin pi0.5 / LWD 数据闭环”
   章节，并删除原来的独立文档，避免两份说明后续不一致。

## 2026-07-07：pi0.5 eval 支持 receding-horizon 执行

本次给 RoboTwin / OpenPI eval 增加了 `env.eval.action_exec_horizon`：

```yaml
env:
  eval:
    max_steps_per_rollout_epoch: 200
    action_exec_horizon: 20

rollout:
  model:
    num_action_chunks: 50
```

语义是：模型仍然一次预测 50 步 action chunk，但环境只执行前 20 步，
然后重新观测并重新预测下一段 50 步。这样可以测试 “短执行视野 + 高频重规划”
是否比一次完整执行 50 步更适合 hammer 任务。

这次同步改了两侧循环：

1. `EnvWorker` 按 `ceil(max_steps_per_rollout_epoch / action_exec_horizon)` 计算
   eval 交互轮数，并在送入环境前截断 action chunk。
2. HuggingFace rollout worker 使用同样的交互轮数，保证 env 需要多少轮 action，
   rollout 就预测多少轮 action，避免评估中途等待。
3. `robotwin_beat_block_hammer_openpi_pi05_eval.yaml` 默认设置
   `action_exec_horizon: 20`。如果不设置该字段，则直接沿用旧逻辑：
   `max_steps_per_rollout_epoch // num_action_chunks` 个完整 chunk，旧配置保持
   完整 chunk 执行语义。

这个改动只影响 eval，不改变 SFT 训练的 action chunk 长度，也不改变内部相机
录制逻辑。

## 2026-07-07：云端 hope 脚本化和自动 resume

### 背景

云端训练有时会因为机器异常而被平台重启。原来的 hope 文件把 conda 环境、
路径、离线 cache、import 检查和训练命令全部写在 `worker.script` 里，而且
`runner.resume_dir` 默认是 `null`。如果平台重启后重新执行同一个 hope，训练会
从头开始，浪费已经完成的 step。

### 当前改法

把云端训练逻辑从 hope 文件下沉到两个 repo 内脚本：

```text
examples/sft/scripts/train_pi05_hammer50_allstats_cloud.sh
examples/lwd/scripts/train_lwd_critic_cloud.sh
```

四个 hope 文件现在只负责申请资源、指定 docker、打开 failover，并调用脚本：

```text
examples/sft/hope/robotwin_pi05_hammer50_allstats_smoke_8a100.hope
examples/sft/hope/robotwin_pi05_hammer50_allstats_train_8a100.hope
examples/lwd/hope/robotwin_lwd_critic_smoke_8a100.hope
examples/lwd/hope/robotwin_lwd_critic_train_8a100.hope
```

train 模式会自动扫描最新完整 checkpoint，并把它作为
`runner.resume_dir=<global_step_dir>` 传给 RLinf。smoke 模式默认不 resume，
避免调试 smoke 时接到旧的 smoke checkpoint。

### 完整 checkpoint 判断

pi0.5 SFT 需要：

```text
actor/dcp_checkpoint/.metadata
actor/model_state_dict/full_weights.pt
```

LWD critic 额外需要：

```text
actor/target_model.pt
```

这样可以避免机器刚好在保存 checkpoint 时挂掉，留下一个目录存在但文件不完整的
`global_step_*`，导致下次启动加载坏 checkpoint。

### 强制从头开始

如果要重新开一个干净实验，可以在提交前设置：

```bash
export RLINF_FORCE_RESTART=1
```

设置后脚本会跳过自动 resume。否则 train 模式会优先从最新完整 checkpoint 继续。

### 本次解决的问题

1. hope 文件不再维护几十行 shell，降低路径、cache、Hydra 参数改错的概率。
2. 云端自动重启后，同一个 hope 会自动从最新完整 checkpoint 接着训练。
3. pi0.5 和 LWD critic 的 smoke/train 共用同一套环境初始化和离线检查逻辑。
4. 打开 `afo.app.support.engine.failover = true`，平台重启和 RLinf resume 配合使用。
5. 正式 pi0.5 保存间隔改为 500 step，LWD critic 保存间隔改为 1000 step，
   减少机器异常时损失的训练进度。

### 2026-07-07 补充：`CONDA_ENV=smoke` 云端变量冲突

第一次提交拆分后的 smoke hope 时，stderr 里只有：

```text
EnvironmentNameNotFound: Could not find conda environment: smoke
```

原因不是缺少名为 `smoke` 的 conda 环境，而是 cloud 脚本内部变量原来叫
`CONDA_ENV`，这个名字在云端运行环境里可能已经被平台或提交参数污染成了
`smoke`。脚本执行到 `conda activate "${CONDA_ENV}"` 时就变成了
`conda activate smoke`。

第一版修复尝试改用了 `RLINF_CONDA_ENV_NAME`，但云端仍可能通过兼容变量继续污染
环境名。最终修法是只按固定环境路径激活，不再按环境名激活：

```bash
ENV_PREFIX="${CLOUD_ROOT}/Miniforge/envs/rlinf_lwd"
source "${CLOUD_ROOT}/Miniforge/bin/activate" "${ENV_PREFIX}"
python examples/lwd/train_lwd_critic.py ...
```

后续 cloud 脚本不再依赖 `CONDA_ENV`、`RLINF_CONDA_ENV` 或
`RLINF_CONDA_ENV_NAME`。如果 stderr 里还出现 `EnvironmentNameNotFound: smoke`，
就说明云端运行的不是当前脚本版本，需要检查是否已经 `git push`、云端是否已经
`git pull`，以及新增的 `examples/*/scripts/*.sh` 是否真的进入了仓库。

进一步收敛后，hope 文件也不再在 `worker.script` 尾部传 `smoke` 或 `train`
参数。当前入口是：

```text
train:
  worker.script = bash .../train_lwd_critic_cloud.sh
smoke:
  worker.script = bash .../smoke_lwd_critic_cloud.sh
```

这样云端平台的命令解析层不会再看到裸的 `smoke` 参数。`smoke` 只在 wrapper
脚本内部作为 `RLINF_RUN_MODE=smoke` 使用，用来选择短步数配置。

## 2026-07-07：第一版 LWD QAM policy extraction 接入

### 背景

此前代码已经有三块基础能力：

```text
pi0.5 SFT actor
LWD distributional V + chunk Q critic
RoboTwin eval / 视频诊断
```

但还缺少 LWD 论文里真正用 critic 更新 VLA 的策略提取部分。直接最大化
`Q(s, a_chunk)` 会把梯度穿过完整 denoise 过程，计算贵且容易不稳定；只做
advantage-weighted BC 又不适合 OpenPI 这种 flow action generator。因此这次接入
第一版 QAM。

### 当时实现

新增文件：

```text
rlinf/algorithms/lwd/qam.py
rlinf/data/datasets/lwd/qam_dataset.py
rlinf/workers/sft/fsdp_lwd_qam_worker.py
examples/lwd/train_lwd_qam.py
examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05.yaml
examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05_smoke.yaml
examples/lwd/scripts/train_lwd_qam_cloud.sh
examples/lwd/scripts/smoke_lwd_qam_cloud.sh
examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_train_8a100.hope
examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_smoke_8a100.hope
```

训练时显式区分三类模型：

```text
actor.model.model_path
  当前要训练的 OpenPI/pi0.5 actor 初始化权重

algorithm.reference_model_path
  固定 reference actor，通常和 SFT checkpoint 一致

critic.model.model_path
  固定 LWD critic checkpoint，通常指向 .../checkpoints/global_step_xxx/actor
```

这样做是为了避免把 base package、SFT checkpoint、critic checkpoint 混成一个
`model_path`，导致 OpenPI loader、critic loader 和 resume 语义互相污染。

### 当时 QAM 训练逻辑

每个 batch：

```text
1. LWDQAMDataCollator 同时产出 critic_observation 和 policy_inputs；
2. reference actor 从 Gaussian noise 做 flow_ode rollout；
3. LWD critic 在 reference endpoint 上计算 Q(s, a)；
4. 对 endpoint 求 action gradient；
5. 用 reference flow transition 的 VJP 反向解 adjoint；
6. 当前 actor 用 ForwardType.NFT 在同一批中间 x_t/t 上预测 velocity；
7. 优化 QAM local regression loss + 小权重 anchor loss + 小权重 BC flow loss。
```

这属于 2026-07-07 的第一版接入方案。后续实验证明 mixed replay 上的 BC 会把
failed/nearmiss action 也当作专家动作模仿，因此 2026-07-09 起默认训练目标已改为
strict QAM：`anchor_weight=0.0`、`bc_weight=0.0`，只保留 QAM local regression
作为主优化信号。

第一版固定 critic，不和 actor 一起更新。这样 smoke 和早期实验更容易定位问题：
如果 QAM 后闭环变差，优先排查 policy extraction 和 critic gradient，而不是
critic/value 同时漂移。

### 需要重点看的指标

```text
train/qam_loss
train/anchor_loss
train/bc_loss
train/q_mean
train/action_grad_norm
train/adjoint_norm
train/qam_delta_norm
train/qam_delta_clip_frac
train/grad_norm
```

判断方式：

```text
qam_loss 不应快速 NaN；
action_grad_norm 不能长期为 0，否则 critic 对 action 没有有效梯度；
qam_delta_clip_frac 如果长期接近 1，说明 Q 引导过强或 lambda/clip 太激进；
q_mean 上升只能作为参考，最终仍要看 RoboTwin success rate 和视频。
```

### 当前边界

这一版不是完整在线 LWD：

```text
没有在 QAM step 中继续训练 critic/value；
没有接 QAM-FQL；
没有接 edit policy；
没有 online replay 和 autonomous rollout 混合采样；
reference actor 固定，不做周期性更新。
```

下一步建议先跑 smoke，确认三模型加载、QAM forward/backward/save 都通过；再跑
短训 500-2000 step，并用 `action_exec_horizon=20/30/50` 做闭环 eval 对比 SFT
checkpoint。

## 2026-07-08：QAM 第一性原理修正和当前待办

### 已修正的问题

1. QAM 更新方向按 OpenPI `flow_ode` 重新对齐

   OpenPI 的 deterministic flow ODE 等价于：

   ```text
   x_next = x - dt * v
   ```

   因此 critic 希望最终 action 往高 Q 方向移动时，velocity 的修正方向和 action
   endpoint 的移动方向相反。当前 `qam_vector_field_loss()` 使用
   `v_beta - v_theta` 构造 residual，已经把这个时间方向修正进去。新增的单元测试
   会检查：在简单一维 Q 梯度下，正确方向的 velocity loss 更低，并且 ODE step
   确实把 endpoint 推向更高 Q 的方向。

2. critic action gradient 支持 `mean|min`

   当前配置新增：

   ```yaml
   algorithm:
     qam_critic_grad_mode: mean
   ```

   `mean` 使用所有 Q heads 的均值求 action gradient，默认采用它，因为更平滑，也
   对齐 QAM 源码里 ensemble mean 的做法。`min` 保留为 clipped double-Q 风格的
   保守 ablation。训练日志会同时记录 `q_mean`、`q_min` 和 `q_head_gap`，用于判断
   多个 Q heads 是否分歧过大。

3. action clamp 契约显式化

   当前配置新增：

   ```yaml
   algorithm:
     qam_clip_action_for_critic: false
   ```

   QAM 源码里 clamp 是可选工程开关，LWD 论文公式本身没有强制要求查询 critic 前
   clamp action。默认不 clamp，可以避免 endpoint 超出 `[-1, 1]` 后由于 clamp
   饱和导致 action gradient 变成 0。若后续想做保守实验，可以打开该开关。训练日志
   会记录 `endpoint_min/max`、`endpoint_saturation_frac` 和
   `critic_action_min/max`，用于判断 critic 实际看到的 action 范围。

4. QAM dataloader 可以跨 epoch 继续训练

   之前 QAM worker 直接 `next(data_iter)`，如果 `runner.max_steps` 大于 dataloader
   一个 epoch 的长度，训练会在 `StopIteration` 处中断。现在 worker 内部封装
   `_next_batch()`，一个 epoch 取完后会更新 sampler epoch 并重新创建 iterator，
   因此可以跑完整的多 epoch QAM 训练。

### 当前仍建议补的诊断

这些不是 QAM 本体必须条件，所以本次先不改 critic 算法，只作为后续 TODO：

```text
value_target_clamp_frac
  统计 C51 projection 里有多少 scalar target 被 clip 到 value support 边界。
  如果长期很高，说明 support 范围或 reward/return 尺度需要重新检查。

value_entropy
  统计 V 分布是否过早塌成尖峰，帮助判断 distributional value 是否还在表达不确定性。

success/nearmiss/failed ranking
  离线比较成功、near-miss、失败轨迹的 Q/V 排序，确认 critic 是否真的具备区分能力。
```

QAM 下一轮实验建议先固定：

```text
qam_critic_grad_mode: mean
qam_clip_action_for_critic: false
```

跑通 smoke 后，再用同一个 critic 分别做 `mean` vs `min`、`clip=false` vs
`clip=true` 的小规模 ablation。最终判断仍以 RoboTwin success rate 和完整视频为准，
不要只看 `q_mean` 是否上升。

## 2026-07-08：QAM 云端训练入口路径固化

### 背景

本轮要验证的是 frozen LWD critic 是否能通过 QAM 改进已经 SFT 过的 pi0.5 actor。
云端实际可用 checkpoint 是：

```text
SFT actor:
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/checkpoints/rlinf_pi05_sft_all_stats/pi05_hammer50_overfit50_10k_allstats_v1/checkpoints/global_step_11000

LWD critic:
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/checkpoints/rlinf_lwd_critic/robotwin_lwd_critic_train_8a100/checkpoints/global_step_8000/actor
```

之前 QAM cloud 脚本里的 actor 默认路径仍指向旧的
`checkpoints/rlinf_pi05_sft/...`，容易导致云端提交后加载不到正确 SFT actor。

### 本次修改

1. `examples/lwd/scripts/train_lwd_qam_cloud.sh`

   默认 `RLINF_QAM_ACTOR_MODEL_PATH` 改为新的
   `checkpoints/rlinf_pi05_sft_all_stats/pi05_hammer50_overfit50_10k_allstats_v1/checkpoints/global_step_11000`。
   `RLINF_QAM_REFERENCE_MODEL_PATH` 仍默认等于 actor path，也就是 QAM 从 SFT actor
   初始化，并用同一个 SFT actor 作为固定 reference flow。

2. `examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05.yaml`

   同步更新 actor 和 reference 的默认路径，保证不用 cloud wrapper 直接启动
   Hydra config 时也不会回到旧 checkpoint。

3. cloud 脚本启动前新增最小路径检查

   检查项包括：

   ```text
   actor/actor/model_state_dict/full_weights.pt
   reference/actor/model_state_dict/full_weights.pt
   critic/model_state_dict/full_weights.pt
   LWD replay norm_stats.json
   OpenPI norm_stats.json
   big_vision/paligemma_tokenizer.model
   ```

   这些检查只用于尽早暴露路径错误，不改变训练算法。

### 当前验证路线

先跑：

```text
hope run examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_smoke_8a100.hope
```

smoke 通过后再跑：

```text
hope run examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_train_8a100.hope
```

第一轮仍保持：

```text
qam_critic_grad_mode: mean
qam_clip_action_for_critic: false
train_expert_only: false
```

也就是先验证 LWD offline 风格的全量 actor QAM。训练输出保存到
旧版目录 `RLINF_QAM_LOG_ROOT/robotwin_beat_block_hammer_lwd_qam_openpi_pi05/checkpoints`。
2026-07-09 strict 版本已改用
`RLINF_QAM_LOG_ROOT/robotwin_beat_block_hammer_lwd_qam_openpi_pi05_strict/checkpoints`，
默认每 100 step 保存一次，用于避免自动 resume 到旧的 mixed-BC checkpoint。

## 2026-07-08：QAM smoke actor.model 缺少 is_lora

### 现象

QAM smoke 已经通过了路径检查，日志里可以看到：

```text
lwd qam imports ok
QAM actor/reference/critic 路径均解析到预期 checkpoint
```

但 worker 初始化 actor 时失败：

```text
omegaconf.errors.ConfigAttributeError: Key 'is_lora' is not in struct
full_key: actor.model.is_lora
```

### 原因

`rlinf.models.get_model()` 是所有模型共用入口。模型实例化完成后，它会读取
`cfg.is_lora` 判断是否接 LoRA。critic 的 config 已经有 `is_lora: false`，但
QAM 新增的 OpenPI actor config 漏了这个 RLinf 公共字段，所以 Hydra struct 模式下
访问 `actor.model.is_lora` 会直接报错。

### 修复

在 `examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05.yaml`
的 `actor.model` 下补充：

```yaml
is_lora: false
```

这个字段不改变 QAM 算法，也不会启用 LoRA，只是显式告诉 RLinf 模型工厂当前 actor
走普通全量参数路径。修复后重新提交 smoke 即可继续验证后续模型加载和
forward/backward。

### 进一步根治

后续 `stdout.20260708161034` 里打印出的 Hydra 配置仍然没有
`actor.model.is_lora`，说明云端实际运行的配置还没有包含本地这次 YAML 修复。
为了避免任何旧 config 或新增 config 再触发同类问题，模型工厂入口也改成：

```python
cfg.get("is_lora", False)
```

也就是：只有显式写了 `is_lora: true` 才启用 LoRA；没有写时默认全量模型路径。
这和 RLinf 的模型配置语义更一致，也能避免 Hydra struct 模式下访问缺失字段报错。

## 2026-07-08：QAM smoke 后续隐患排查

### 已确认不是问题的部分

1. hope 和 conda 激活

   `stdout.20260708144145` 里已经打印出：

   ```text
   Using python: .../Miniforge/envs/rlinf_lwd/bin/python
   lwd qam imports ok
   ```

   说明当前 smoke 已经进入云端 Python 环境和 RLinf import 阶段。当前报错不是
   `worker.script` 后面是否带 `smoke`、也不是 conda 激活失败。

2. actor/reference/critic 路径形态

   OpenPI loader 支持：

   ```text
   global_step_xxx/actor/model_state_dict/full_weights.pt
   model_state_dict/full_weights.pt
   ```

   因此 SFT actor 使用 `global_step_10000`，critic 使用 `global_step_8000/actor`
   是符合当前 loader 契约的。cloud script 也会提前检查这些文件，避免路径写错后
   进入很晚才失败的模型构建阶段。

3. QAM policy input 和 critic input 的空间

   QAM dataset 给 OpenPI 的 policy view 是原始 `images/state/prompt`，会走
   `AlohaInputs -> Normalize -> model_transforms`；给 critic 的 action chunk 则是
   pi0.5 归一化 action。两者都使用同一套 pi05 norm stats，因此当前没有发现
   action 空间或 state 空间错位。

### 本次补充的代码检查

1. batch 并行契约

   QAM worker 现在显式检查：

   ```text
   actor.global_batch_size % (actor.micro_batch_size * world_size) == 0
   ```

   当前 smoke/train 配置是 `global_batch_size=8, micro_batch_size=1, world_size=8`，
   所以每步正好一个 micro batch。这个检查是为了避免以后改 GPU 数或 batch size
   时，`gradient_accumulation` 变成错误值，导致训练循环后面才出现难读的报错。

2. QAM 数学契约

   `algorithm.qam_num_denoise_steps` 必须至少为 2，因为当前实现会跳过 OpenPI flow
   的端点 `t=1`，只在内部 denoise state 上做 QAM local regression。`lambda_q`
   也必须大于 0，否则 `adjoint = -grad_Q / lambda_q` 没有数学意义。

### 仍需关注的真实风险

1. 显存风险

   QAM worker 会同时驻留三套模型：

   ```text
   current actor: 需要训练和保存 optimizer
   reference actor: 冻结，但需要保留 x_t -> x_next 的伴随梯度图
   frozen critic: 需要对 action 输入求 ∇a Q
   ```

   当前 FSDP 配置是 `sharding_strategy: no_shard`，所以每张 GPU 都会放完整模型。
   8x80G 大概率可以做 smoke，但正式 QAM 仍会比普通 SFT 更吃
   显存。如果下一次日志变成 CUDA OOM，优先把正式训练的
   `algorithm.qam_num_denoise_steps` 从当前 5 降到 3 或 4 做验证，而不是先改算法方向。

2. 训练有效性风险

   2026-07-09 之后当前 QAM 默认使用：

   ```yaml
   qam_critic_grad_mode: mean
   qam_clip_action_for_critic: false
   qam_loss_weight: 1.0
   anchor_weight: 0.0
   bc_weight: 0.0
   ```

   这是为了先验证 frozen critic 是否能通过纯 QAM residual 给 actor 提供有用方向，
   避免 mixed replay action BC 干扰。是否真正提升策略，仍然要看后续 RoboTwin eval
   success rate 和视频，而不能只看 `q_mean` 是否上升。

## 2026-07-08：QAM smoke critic checkpoint loader 路径类型问题

### 现象

`stdout.20260708163627` 里，`actor.model.is_lora` 已经出现在 Hydra 配置中，
actor FSDP 也初始化完成，说明上一轮 `is_lora` 问题已经解决。新的失败发生在
加载 frozen LWD critic：

```text
AttributeError: 'str' object has no attribute 'is_file'
```

位置是：

```text
rlinf/models/embodiment/value_model/checkpoint_utils.py
load_state_dict_from_checkpoint()
```

### 原因

QAM config 里的 `critic.model.model_path` 是字符串路径：

```text
.../checkpoints/global_step_8000/actor
```

但 `load_state_dict_from_checkpoint()` 原来只按 `pathlib.Path` 使用它，直接调用
`checkpoint_path.is_file()`。此外，即使先转成 `Path`，这个函数也只会找目录下
直接存在的 `*.pt` 或 `*.safetensors`，而 RLinf FSDP 保存的完整权重在：

```text
actor/model_state_dict/full_weights.pt
```

也就是 checkpoint 目录的子目录里。

### 修复

`load_state_dict_from_checkpoint()` 现在支持：

```text
str 或 pathlib.Path
model_state_dict/full_weights.pt
actor/model_state_dict/full_weights.pt
目录下直接的 *.pt / *.pth / *.safetensors
单文件 *.pt / *.pth / *.safetensors
```

这样 QAM 传入 `global_step_8000/actor` 时，会自动解析到
`global_step_8000/actor/model_state_dict/full_weights.pt`。这属于 checkpoint
加载契约修复，不改变 critic 模型结构或 QAM 算法。

## 2026-07-08：QAM smoke OpenPI NFT token 设备不一致

### 现象

`stdout.20260708164833` 已经进入 `run_training()`，说明 actor 初始化、reference
初始化、critic checkpoint 加载都已经通过。新的失败发生在 QAM 第一个 batch 的
reference flow rollout：

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
```

traceback 位于：

```text
FSDPLWDQAMWorker.compute_qam_loss()
  -> _rollout_reference_flow()
  -> _policy_velocity()
  -> OpenPI nft_forward()
  -> _build_prefix_cache()
  -> embed_language_tokens()
```

### 原因

OpenPI wrapper 的 `nft_forward()` 会先调用 `input_transform()`，这个 transform
生成的是 CPU tensor。原实现只把：

```text
images
image masks
state
```

搬到模型所在 GPU，但漏了：

```text
lang_tokens
lang_masks
```

因此在 Gemma embedding 里出现：

```text
embedding weight: cuda
token ids: cpu
```

QAM 使用 `ForwardType.NFT` 显式查询 OpenPI velocity field，所以这个设备契约问题
被第一轮训练立刻触发。

### 修复

在 `OpenPi0ForRLActionPrediction` 中增加统一 helper，把 policy forward 所需的
五类张量一起移动到模型设备：

```text
images, img_masks, lang_tokens, lang_masks, state
```

并在以下入口使用：

```text
default_forward()
nft_forward()
sample_actions()
```

这不是 QAM 数学问题，而是 OpenPI wrapper 的设备一致性修复。修复后 QAM smoke
应该继续推进到 critic action gradient 或 backward 阶段。

## 2026-07-08：QAM smoke OpenPI 内部 action dim 与 critic action dim 不一致

### 现象

`stdout.20260708171824` 已经通过 token 设备一致性检查，进入 OpenPI `embed_suffix()`。
新的错误是：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (50x14 and 32x1024)
```

traceback 位于：

```text
OpenPI nft_forward()
  -> get_velocity()
  -> get_suffix_out()
  -> embed_suffix()
  -> action_in_proj(noisy_actions)
```

### 原因

RoboTwin/Aloha 环境动作是 14 维：

```text
left arm 6 + left gripper 1 + right arm 6 + right gripper 1
```

LWD critic 也是在这个 14 维 pi0.5 归一化 action chunk 上训练的。但 pi0.5
OpenPI action expert 内部使用 padded action space，`action_in_proj` 期待的是
32 维输入。普通 rollout 最终只把前 14 维通过 output transform 给环境，但 flow
denoise 过程本身是在 32 维 action space 上运行。

QAM 之前直接用 replay action 的 `[B, 50, 14]` 作为 reference flow 的 `x_t`，
所以传到 `action_in_proj` 时和 `32 -> 1024` 的线性层不匹配。

### 修复

QAM worker 现在显式区分两个 action space：

```text
model action space: 32 维，用于 OpenPI flow / NFT velocity / QAM adjoint
env critic action space: 14 维，用于 LWD critic 评分
```

具体做法：

1. reference flow rollout 从 `action_in_proj.in_features` 读取内部 action dim，并用 `[B, 50, 32]` 的 internal action。
2. replay BC target 从 14 维 pad 到 32 维。
3. critic 查询前从 internal endpoint 切前 14 维。
4. critic action gradient 仍然对 32 维 endpoint 求导，后 18 维自然是 0。

这样 QAM 对接的是 OpenPI 真正的 flow 内部空间，同时 critic 仍然只看自己训练过的
14 维动作空间。

## 2026-07-09：QAM strict 版本优化和训练坏掉的根因修正

### 背景

第一轮正式 QAM checkpoint 闭环表现明显差于 SFT baseline。结合视频和
TensorBoard 指标看，主要不是“训练步数不够”，而是优化目标本身被旧版工程项带偏：

```text
QAM loss contribution   约 13%
anchor contribution     约 5%
BC contribution         约 82%
qam_delta_clip_frac     长期接近 1.0
```

其中最大的问题是旧版 QAM 数据混用了 success/failed/nearmiss，并且
`bc_weight=0.1`。这会把 failed 和 nearmiss 的 replay action 也当成专家动作去模仿，
而 QAM 本应使用这些数据里的状态分布和 critic action gradient，不应默认模仿失败动作。

### 第一性原理修正

当前实现改为 strict QAM policy extraction：

```yaml
algorithm:
  qam_loss_weight: 1.0
  anchor_weight: 0.0
  bc_weight: 0.0
  qam_delta_clip: 5.0
  qam_clip_action_for_critic: false
```

训练目标退回到 QAM 源码和论文公式对应的局部 vector-field matching：

```text
L = L_QAM
L_QAM = || 2(f_theta - f_beta) / sigma + sigma * g ||^2
```

需要注意的是，上式里的 `f` 是 noise-to-action 方向的 flow field；OpenPI/pi0.5
内部训练的是反方向 velocity：

```text
x_t = t * noise + (1 - t) * action
v_openpi = noise - action
x_next = x - dt * v_openpi
```

所以在代码的 OpenPI velocity 坐标中，等价 residual 是：

```text
2(v_beta - v_theta) / sigma + sigma * adjoint
```

这里 `v_beta` 是 frozen SFT reference actor，`v_theta` 是当前训练 actor，
`adjoint = -grad_a Q / lambda_q`，和 QAM 源码里 `adj = -grad_fn(...) * inv_temp`
保持同一方向。因此这不是偏离公式，而是把 QAM 公式换到 OpenPI 的反向 velocity
坐标后得到的等价形式。

### 数据策略

strict 主实验默认只使用：

```yaml
data:
  train_data_paths:
    - dataset_path: beat_block_hammer_success_train
      weight: 1.0
    - dataset_path: beat_block_hammer_nearmiss_train
      weight: 1.0
```

pure failed 状态先不放进主实验。原因是当前 critic 虽然能在离线诊断中区分
success/failed/nearmiss，但 QAM 是直接用 critic action gradient 改 actor；在 critic
还没有通过更多闭环验证前，先用 success + nearmiss 状态更稳。failed 状态后续可以
作为 ablation 以较小权重加入，例如 `success:nearmiss:failed = 1:1:0.25`。

### 代码变更

1. `rlinf/algorithms/lwd/qam.py`

   明确注释 QAM 原始公式和 OpenPI reverse-time velocity 的等价关系，避免后续把
   `v_beta - v_theta` 误改成符号相反的形式。

2. `rlinf/workers/sft/fsdp_lwd_qam_worker.py`

   默认权重改为 strict QAM：

   ```text
   anchor_weight default: 0.0
   bc_weight default: 0.0
   qam_delta_clip default: 5.0
   ```

   同时当 `bc_weight <= 0` 时不再计算 `_bc_flow_loss()`，避免无意义的额外 forward。
   训练日志新增 action 分布诊断：

   ```text
   train/action_grad_clip_frac
   train/endpoint_abs_p95
   train/critic_action_abs_p95
   train/replay_action_min
   train/replay_action_max
   train/replay_action_abs_p95
   ```

3. `examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05.yaml`

   默认实验名改为：

   ```text
   robotwin_beat_block_hammer_lwd_qam_openpi_pi05_strict
   ```

   这样云端自动 resume 不会误接到旧的坏 checkpoint。

4. `examples/lwd/scripts/train_lwd_qam_cloud.sh`

   train 默认跑 500 step、每 100 step 保存一次，并自动从 strict 实验目录里最新完整
   checkpoint resume。如果要完全重开，设置：

   ```bash
   export RLINF_FORCE_RESTART=1
   ```

### 下一步验证

先跑 smoke，确认导入、三模型加载、QAM forward/backward/save 全部通过；再跑 strict
train 500 step。完成后不要只看 `q_mean`，至少需要：

```text
1. 对比 SFT baseline 和 strict QAM checkpoint 的 RoboTwin success rate；
2. 固定 action_exec_horizon=30 做 30 次 eval；
3. 保存成功/失败视频，重点看 endpoint 是否仍然大幅偏移；
4. 检查 qam_delta_clip_frac 是否显著低于旧版长期 1.0 的状态；
5. 检查 replay_action_abs_p95 和 critic_action_abs_p95 是否处在同一量级。
```

如果 strict QAM 仍然退化，下一步优先调小 Q 引导强度：

```text
lambda_q: 4.0 或 8.0
qam_grad_clip: 0.02
lr: 1.0e-6
```

而不是重新打开 mixed replay BC。

## 2026-07-09：QAM step500 结果复盘和 qstrong_lq05 版本

### 当前结果

strict QAM 500-step checkpoint 已经不再像旧版 QAM 那样把策略训练坏，但闭环结果还
没有超过 SFT baseline。在相同 RoboTwin eval 设置下：

```text
SFT baseline                 11 / 30 = 36.67%
strict QAM global_step_500   11 / 30 = 36.67%
旧版 QAM global_step_2000      0 / 30 = 0.00%
```

这个结果说明前一轮 strict 修正解决了“QAM 明显退化”的问题，但还没有证明 critic
guidance 真的带来了策略提升。

需要注意，当前 `3 env x 10 epoch` 的 eval 在
`env.eval.use_fixed_reset_state_ids=true` 时，本质上是 3 个固定 reset seed 各重复
10 次，不是 30 个完全不同的初始场景。后续要同时保留 hard seed repeat 评估和
`env.eval.use_fixed_reset_state_ids=false` 的多 seed 评估。

### TensorBoard 观察

从最新 strict QAM 日志看，主要信号如下：

```text
bc_loss                 0
qam_delta_clip_frac     0
action_grad_clip_frac   约 0.35
action_grad_norm        约 0.044
adjoint_norm            约 0.0014
lr                      最后约 4e-10
```

因此当前主要矛盾不是 BC 混入失败动作，也不是 `qam_delta_clip_frac` 长期饱和；
真正的问题是 QAM 信号到 actor 端偏弱，并且 500-step cosine scheduler 在后期把学习率
衰减到几乎为 0。旧日志里的 `q_mean/q_min` 仍是在 reference flow endpoint 上算的，
不能直接回答 current actor 是否比 frozen reference actor 更好。

### 本轮代码和配置修正

1. 增加 current-vs-reference 成对 Q 诊断

   `FSDPLWDQAMWorker.compute_qam_loss()` 现在会按 `algorithm.qam_compare_interval`
   低频执行纯诊断分支：同一个 observation/noise 下，分别 rollout frozen reference
   actor 和 current actor，并用同一个 frozen critic 计算 endpoint Q。

   新增日志：

   ```text
   train/q_ref_mean
   train/q_cur_mean
   train/q_cur_minus_ref
   train/q_ref_min
   train/q_cur_min
   train/q_cur_minus_ref_min
   train/cur_ref_endpoint_l2
   train/cur_endpoint_abs_p95
   train/cur_endpoint_saturation_frac
   ```

   这些指标不参与 loss，只用于判断 QAM 是否真的把 current actor endpoint 推向更高
   critic value。如果 `q_cur_minus_ref` 长期接近 0，说明 actor 基本没动；如果上升但
   eval 不升，优先排查 critic 梯度和真实闭环收益是否一致；如果下降，则优先排查
   QAM 符号、action 归一化或 critic 查询契约。

2. 保持 QAM 公式不变，但增强价值引导强度

   QAM loss 仍然是 OpenPI 坐标下的等价形式：

   ```text
   2(v_beta - v_theta) / sigma + sigma * adjoint
   adjoint = -grad_a Q / lambda_q
   ```

   新配置把 `lambda_q` 从 2.0 改为 0.5，相当于把同一 critic action gradient 对 flow
   的引导强度放大 4 倍。`qam_grad_clip` 暂时仍为 0.05，先只动一个主旋钮，避免
   同时改变太多变量。

3. 改成更适合短程验证的学习率调度

   500-step cosine run 后期学习率衰减太快，最后几十步几乎没有更新。因此新正式
   配置改为：

   ```yaml
   runner:
     max_steps: 1500
     save_interval: 500

   actor:
     optim:
       lr: 2.0e-6
       lr_scheduler: constant
       lr_warmup_steps: 50
   ```

   这不是把学习率做大，而是避免短训练后半段学习率过早归零。

4. 新实验目录隔离 resume

   新实验名：

   ```text
   robotwin_beat_block_hammer_lwd_qam_openpi_pi05_qstrong_lq05
   ```

   smoke 实验名：

   ```text
   robotwin_beat_block_hammer_lwd_qam_openpi_pi05_qstrong_lq05_smoke
   ```

   这样云端自动 resume 不会接到 strict 500-step 的旧 checkpoint。smoke 配置把
   `qam_compare_interval` 设为 1，用来覆盖新增诊断分支。

### 下一步验证

先跑 smoke，确认新增 current/reference 对比路径能通过；再跑正式 1500 step。每个
500 step checkpoint 至少做两类 eval：

```text
1. hard seed repeat：保留当前 3 个固定 seed，各重复 10 次，看原失败 seed 是否改善；
2. unique seed eval：设置 env.eval.use_fixed_reset_state_ids=false，看泛化成功率。
```

如果 `q_cur_minus_ref` 上升且 `cur_ref_endpoint_l2` 有合理增大，但闭环成功率不涨，
下一步优先做 critic 梯度可信度诊断，而不是继续盲目放大 QAM。若
`q_cur_minus_ref` 仍接近 0，则优先考虑继续增强引导强度，例如适度提高
`qam_grad_clip` 或继续降低 `lambda_q`，但每次只改一个主变量。

## 2026-07-09：QAM qstrong_lq05 结果和 probe_lq01_gc01 短实验

### qstrong_lq05 结果

`global_step_1500` 的 QAM 训练已经稳定跑完，checkpoint 结构完整：

```text
global_step_1500/actor/dcp_checkpoint/.metadata
global_step_1500/actor/model_state_dict/full_weights.pt
```

关键 TensorBoard 指标：

```text
bc_loss                     0
lr last                     2e-6
adjoint_norm mean           约 0.00586
qam_delta_clip_frac mean    约 4.17e-05
cur_ref_endpoint_l2 mean    约 0.099
q_cur_minus_ref mean        约 -6.4e-05
q_cur_minus_ref positive    约 42.7%
```

这说明 `lambda_q=0.5` 和 constant lr 都生效了，QAM 训练没有坏掉；但 current actor
endpoint 只小幅偏离 frozen reference，而且 critic 并没有稳定认为 current endpoint
更好。第一性原理上，这一轮卡在：

```text
critic gradient 存在
-> QAM adjoint 存在
-> actor endpoint 小幅变化
-> 但 Q(current endpoint) - Q(reference endpoint) 没有稳定变正
```

因此 qstrong_lq05 不是失败动作 BC、delta clip、学习率归零或 action 尺度明显错位
的问题，而是 policy improvement 信号仍未有效转化为更高 Q endpoint。

### 本次代码变更

为避免直接把正式训练配置改得过激，新增一个独立 probe 入口：

```text
examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05_probe.yaml
examples/lwd/scripts/probe_lwd_qam_cloud.sh
examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_probe_8a100.hope
```

`train_lwd_qam_cloud.sh` 新增 `RLINF_RUN_MODE=probe`，对应配置：

```yaml
runner:
  logger:
    experiment_name: robotwin_beat_block_hammer_lwd_qam_openpi_pi05_probe_lq01_gc01
  max_steps: 300
  save_interval: 100

algorithm:
  lambda_q: 0.1
  qam_grad_clip: 0.1
  qam_compare_interval: 5
```

probe 仍然继承 strict QAM 契约：

```yaml
anchor_weight: 0.0
bc_weight: 0.0
qam_clip_action_for_critic: false
qam_critic_grad_mode: mean
qam_delta_clip: 5.0
```

也就是说，这不是重新引入 BC/anchor，而是只做“更强价值引导”的短程 stress test。

### 运行方式

云端直接运行：

```bash
hope run examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_probe_8a100.hope
```

输出目录：

```text
RLINF_QAM_LOG_ROOT/robotwin_beat_block_hammer_lwd_qam_openpi_pi05_probe_lq01_gc01
```

### 判断标准

probe 不先追求闭环成功率，先看训练日志是否满足：

```text
q_cur_minus_ref 明显 > 0
cur_ref_endpoint_l2 从约 0.1 提升到 0.3~0.8
qam_delta_clip_frac 不长期接近 1
endpoint_saturation_frac 不明显爆炸
```

如果 probe 后 `q_cur_minus_ref` 明显变正，说明之前主要是 QAM 信号偏弱，可以再做
300/500-step checkpoint 的 RoboTwin eval。如果 probe 后 `q_cur_minus_ref` 仍接近
0，则说明单纯增强 QAM 不够，下一步应优先做 critic gradient finite-difference
诊断，确认 frozen critic 在 actor endpoint 附近的局部 action gradient 是否可靠。

## 2026-07-13：QAM allstats ODE eval 和 same-seed 对比口径

### ODE 评估结论

为了和原版 OpenPI 推理口径对齐，本轮重新评估 SFT allstats baseline 和当前最优 QAM
checkpoint 时显式使用：

```yaml
rollout:
  model:
    openpi:
      noise_method: flow_ode
      noise_level: 0.0
```

同时已把 hammer pi0.5 eval 配置默认改成 ODE：

```text
evaluations/robotwin/robotwin_beat_block_hammer_openpi_pi05_eval.yaml
```

共同 eval 设置：

```text
action_exec_horizon = 50
total_num_envs      = 3
rollout_epoch       = 10
fixed reset seeds   = true
```

结果如下：

| 模型 | checkpoint | 成功数 | 成功率 |
| --- | --- | ---: | ---: |
| SFT allstats | `rlinf_pi05_sft_allstats/global_step_11000` | 13/30 | 43.33% |
| QAM allstats | `probe_lq005_gc005/global_step_300` | 17/30 | 56.67% |

按 seed 拆分：

| seed | SFT | QAM | 变化 |
| ---: | ---: | ---: | ---: |
| 100112514 | 6/10 | 7/10 | +1 |
| 100137506 | 1/10 | 5/10 | +4 |
| 100175033 | 6/10 | 5/10 | -1 |

输出文件：

```text
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_allstats_gs11000_exec50_n30_ode/eval_episode_metrics.jsonl
/data/wam_codebase/RLinf/outputs/eval_qam_allstats_probe_lq005_gc005_gs300_exec50_n30_ode/eval_episode_metrics.jsonl
```

结论：QAM 在 ODE 评估下仍然有净提升，说明之前看到的提升不是只依赖
`flow_sde/noise_level=0.5` 的采样噪声。收益主要来自 hard seed `100137506`，但
`100175033` 有轻微退化，所以当前 QAM 是有净收益的小局部修正，不是无损增强。

### flow initial noise 的来源

同一个 RoboTwin seed 固定的是环境初始场景：

```text
物体初始位置/姿态
机器人 reset 状态
相机初始观测
任务 reset 随机量
```

但它不直接固定 OpenPI policy 每次 flow 采样的初始 Gaussian noise。当前 RLinf
OpenPI eval 的调用链是：

```text
MultiStepRolloutWorker.predict()
-> OpenPi0ForRLActionPrediction.predict_action_batch()
-> OpenPi0ForRLActionPrediction.sample_actions()
-> if noise is None: self.sample_noise(actions_shape, device)
```

`sample_noise()` 来自 OpenPI PyTorch 父类：

```python
torch.normal(
    mean=0.0,
    std=1.0,
    size=shape,
    dtype=torch.float32,
    device=device,
)
```

在当前 hammer pi0.5 eval 中，每次 policy forward 会采：

```text
[batch_size, action_horizon, action_dim] = [3, 50, 14]
```

因此 `flow_ode` 的含义是 denoise 过程中不额外加入 SDE 噪声；但初始 `x_1`
仍是 policy forward 时从 PyTorch RNG 采样的 Gaussian noise。给定同一个初始
noise 后，ODE 积分过程是确定的；但 eval 重复之间的初始 noise 不一定相同。

### 对比口径修正

后续不要把当前结果称为严格的 paired counterfactual。更准确的说法是：

```text
same-seed / same-reset-scene comparison
```

当前对齐的是：

```text
同一个环境 seed
同一个初始 reset 场景
同一批 fixed seeds 上的重复评估
```

当前没有对齐：

```text
同一个 flow initial noise
同一个 action chunk
同一个中间状态轨迹
```

因此本轮结果可以客观说明：

```text
在同一批固定初始场景上，QAM 的闭环成功率高于 SFT。
```

但不能过度解释为：

```text
在完全相同的 flow initial noise 和中间状态下，QAM 每一步都优于 SFT。
```

如果后续要做更严格的 counterfactual 诊断，应显式固定 policy initial noise，例如
按 `env_seed + rollout_epoch + env_id + chunk_idx` 构造固定 `torch.Generator`，或者
直接把同一个 noise tensor 同时传给 SFT 和 QAM，然后再比较第一段 action 或闭环结果。

## 2026-07-13：A/B/C 复验后，暂缓新的 QAM ablation

### A. checkpoint sweep

对 `lq005_gc005` 的 `global_step_100/200/300` 做 ODE、fixed-seed 30 次 eval：

| 模型 | 成功数 | 成功率 | 100112514 | 100137506 | 100175033 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT `global_step_11000` | 13/30 | 43.33% | 6/10 | 1/10 | 6/10 |
| QAM `global_step_100` | 12/30 | 40.00% | 8/10 | 3/10 | 1/10 |
| QAM `global_step_200` | 12/30 | 40.00% | 5/10 | 2/10 | 5/10 |
| QAM `global_step_300` | 17/30 | 56.67% | 7/10 | 5/10 | 5/10 |

step300 是 30 次 eval 中最优，但 step100/200 不超过 SFT，说明当前 QAM 训练不是
单调策略改进。

### B. step300 的 60 次 fixed-seed 复验

扩大到 3 个 fixed seeds 各 20 次后：

| 模型 | 成功数 | 成功率 | 100112514 | 100137506 | 100175033 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT `global_step_11000` | 28/60 | 46.67% | 16/20 | 2/20 | 10/20 |
| QAM `global_step_300` | 25/60 | 41.67% | 15/20 | 5/20 | 5/20 |

这说明 30 次 eval 的 `17/30` 是乐观小样本结果。QAM 对 hard seed `100137506`
有帮助，但显著破坏 `100175033`，整体低于 SFT。

### C. unique seeds

`env.eval.use_fixed_reset_state_ids=false`、ODE、`action_exec_horizon=50`、30 条不同
reset 场景：

| 模型 | 成功数 | 成功率 |
| --- | ---: | ---: |
| SFT `global_step_11000` | 6/30 | 20.00% |
| QAM `global_step_300` | 6/30 | 20.00% |

unique seeds 上 QAM 没有泛化提升。

### 当前判断

原计划 D 是：

```text
如果提升稳定，再训练 lq005_gc003 和 lq01_gc005
```

但 A/B/C 之后，“提升稳定”这个前提不成立。因此本轮暂缓新的 QAM ablation，不继续
盲目提交 `lq005_gc003` / `lq01_gc005` 云端训练。

下一步应该先做诊断，而不是继续调强度：

1. 固定 policy initial flow noise，做真正的 SFT/QAM counterfactual 对比。
2. 给 `100137506` 和 `100175033` 输出 SFT/QAM 对比视频，确认 QAM 改善/破坏的具体阶段。
3. 设计保守门控：只有当 critic 明确认为 current endpoint 高于 reference endpoint 时才允许更新，或者对 current/reference endpoint L2 做更严格 trust region。
4. 如果再训练 QAM，优先做“少破坏”的约束，而不是单纯调 `lambda_q` / `qam_grad_clip`。

## 2026-07-13：fixed flow initial noise counterfactual 复验

### 改动

在 `OpenPi0Config` 中新增 eval-only 字段：

```text
fixed_eval_noise_seed: null
```

默认 `null` 时保持原始 `sample_noise()` 行为。设置为整数时，OpenPI action sampling
使用固定 `torch.Generator` 生成 flow 初始 Gaussian noise。这样在 SFT/QAM 两次 eval
配置完全一致时，同一个 `(rollout_epoch, env_id, seed)` 会消费同一串初始 noise。

本轮固定：

```text
noise_method: flow_ode
noise_level: 0.0
fixed_eval_noise_seed: 20260713
total_num_envs: 3
rollout_epoch: 20
action_exec_horizon: 50
use_fixed_reset_state_ids: true
```

### 结果

| 模型 | 成功数 | 成功率 | 100112514 | 100137506 | 100175033 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT `global_step_11000` | 29/60 | 48.33% | 16/20 | 4/20 | 9/20 |
| QAM `global_step_300` | 28/60 | 46.67% | 15/20 | 3/20 | 10/20 |

配对翻转：

| 类型 | 数量 |
| --- | ---: |
| both success | 19 |
| both fail | 22 |
| SFT only | 10 |
| QAM only | 9 |

### 判断

固定 flow initial noise 后，QAM 仍没有超过 SFT。之前非 fixed-noise 的 n60 里
QAM 对 `100137506` 的改善更明显，但 fixed-noise 后这个改善消失，说明之前的一部分
差异来自 policy 初始 Gaussian noise 与闭环随机性的交互。

当前更可靠的结论是：

```text
QAM lq005_gc005 global_step_300 没有形成稳定的策略改进。
它会制造一些 SFT fail -> QAM success 的翻转，但也几乎同等数量地制造
SFT success -> QAM fail 的翻转。
```

因此接下来不应继续靠统一增大或减小 `lambda_q / qam_grad_clip` 来碰运气。更优先的方向是：

1. 对 fixed-noise 的 SFT-only / QAM-only episode 输出视频和 action diff。
2. 看 QAM 改动是否集中在关键接触阶段，还是整段 action chunk 漂移。
3. 训练侧优先做门控/约束，例如只有 critic 明确给出正 advantage 时允许偏离 reference，
   或者对 current/reference endpoint L2 加 per-state trust region。

## 2026-07-13：critic action-gradient direct-edit 诊断

### 目的

为了确认问题是否来自 QAM 引导链路，先不训练 actor，而是直接测试 critic 的
`grad_A Q(s,A)` 是否能在闭环中产生正向效果。

脚本：

```text
examples/lwd/diagnose_critic_gradient_edit.py
```

输出：

```text
outputs/lwd_critic_grad_edit_diag_n10
```

### 方法

每个 seed 从同一个 reset 场景出发，同时跑 4 个变体：

| 变体 | 含义 |
| --- | --- |
| `base` | 原始 SFT action |
| `plus` | 沿 `+grad_A Q(s,A)` 编辑执行前缀 |
| `minus` | 沿 `-grad_A Q(s,A)` 编辑执行前缀 |
| `random` | 同等幅度随机方向编辑 |

本轮设置：

```text
num_cases: 10
epsilon: 0.01
action_exec_horizon: 20
```

### 结果

闭环成功率全部为 0：

| 变体 | 成功数 |
| --- | ---: |
| `base` | 0/10 |
| `plus` | 0/10 |
| `minus` | 0/10 |
| `random` | 0/10 |

即时 critic 预测里，`plus` 方向有一定自洽性：

| 范围 | `plus` 正 delta 比例 | `plus > base` | `plus > minus` | `plus > random` |
| --- | ---: | ---: | ---: | ---: |
| all chunks | 52.0% | 69.0% | 72.0% | 59.0% |
| chunks 2-9 | 63.8% | 76.2% | 83.8% | 66.2% |
| chunks 5-9 | 90.0% | 82.0% | 86.0% | 78.0% |

### 判断

这次结果不能说明 critic 梯度完全无意义。更准确地说：

```text
critic 梯度能在 critic 自己的局部 Q 预测上产生方向性，
但在这 10 个 hard cases 中还没有转化为真实闭环成功。
```

因此下一步不建议直接长训 QAM 或继续盲目扫 `lambda_q / qam_grad_clip`。
更优先的是做两个更有判别力的实验：

1. 选择包含 SFT 成功和失败的 seed，重复 direct-edit probe，避免全是 hard failure。
2. 对 `plus` 有明显提高 predicted Q 的片段输出 action diff 和视频，看编辑是否发生在
   抓锤、接触、敲击等关键阶段，还是只是让 critic 数值变高但物理动作无效。

本次运行在写完结果后退出清理阶段出现 `free(): invalid pointer`。结果文件已经完整落盘，
该报错更像 RoboTwin/底层库析构问题，不影响本次诊断统计。

## 2026-07-14：same-state action ranking 诊断

### 目的

回答一个更上游的问题：

```text
在完全相同状态 s 下，critic 给多个 action chunk 的排序，
是否和真实环境执行后的 success / margin 一致？
```

如果这项不成立，继续调 `lambda_q / qam_grad_clip` 没有第一性依据。

脚本：

```text
examples/lwd/diagnose_same_state_action_ranking.py
```

输出：

```text
outputs/lwd_same_state_action_ranking_pilot
```

### 实现口径

- 使用 `same seed + SFT prefix replay` 恢复 canonical state。
- 4 个 state 的 replay 复现测试全部通过：hammer/block/robot 最大误差为 0，observation hash 一致。
- 候选：`sft`、`qam_same_noise`、`qam_mirror`、`grad_plus`、`grad_minus`、`random_plus`、`random_minus`。
- 当前 action 是 qpos chunk，未安全接入 Cartesian IK/controller，因此跳过 `x/y/z ±2mm`，没有用 joint offset 冒充 Cartesian offset。
- critic 主排序用当前 QAM 配置一致的 `q_mean`，同时保存 `q1/q2/q_min`。

### 运行

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

### 结果

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

### 判断

critic 对粗粒度 success/fail pair 有一定排序信号，但对 QAM 最需要的连续 margin、
QAM 改动方向、以及 `grad_plus` vs `grad_minus` 不可靠。

典型例子：

- `seed100112514_q2`：SFT margin `+1.05cm`，QAM margin `+0.038cm`，但 critic 给 QAM 更高 Q。
- `seed100175033_q2`：SFT 成功且 margin `+0.004cm`，QAM 失败且 margin `-1.19cm`；`grad_plus` 被 critic 明显抬高，但真实失败且 margin `-0.90cm`。
- `seed100187555_q2`：QAM 相对 SFT 的 Q 略升，但真实 margin 从 `-0.16cm` 降到 `-1.76cm`。

因此，本轮不是 `CRITIC_PASS`。工程结论更接近：

```text
NO_GO for QAM gradient.
```

也就是说，当前 critic 也许能做一些 coarse trajectory/state ranking，但还不能作为可靠的
same-state action-gradient provider。下一步应优先改 critic 数据和监督目标，例如加入
same-state action contrast、xy margin/contact/grasp stability 辅助目标，或先验证 best-of-K
candidate ranking，而不是继续放大 QAM actor training。

本次结果完整写盘后仍出现 `free(): invalid pointer`，判断为 RoboTwin/Sapien 清理阶段问题，
不影响 `summary.json`、`rollout_results.jsonl` 等统计文件。

## 2026-07-15：QAM 接入新 critic ablation 的代码适配

### 背景

新的 critic ablation 中有两类值得进入 QAM probe：

| critic | 含义 | 使用原因 |
| --- | --- | --- |
| `s1_f1_n1_h50_tau09` | action horizon 仍为 50，IQL expectile/quantile 更偏高价值动作 | 和当前 QAM action chunk 长度完全一致，可直接验证高 tau critic 是否更适合引导 |
| `s1_f1_n1_h30_tau06` | action horizon 缩短到 30，其他比例保持 baseline | source-paired 诊断里 cross/action-branch 指标更好，但需要 QAM 侧支持 H30 critic |

之前 QAM worker 默认把 actor 生成的完整 50 步 endpoint 直接交给 critic。这个逻辑
只适配 H50 critic；如果换成 H30 critic，就会把 50 步 action 输入到一个按 30 步训练
的 action encoder，语义和 tensor shape 都不对。

### 本次代码改动

1. `FSDPLWDQAMWorker` 现在根据 `critic.model.action_horizon` 裁剪 critic 输入：

```text
actor/reference endpoint: [B, 50, action_dim]
H50 critic input:        [B, 50, env_action_dim]
H30 critic input:        [B, 30, env_action_dim]
```

actor、reference flow 和 QAM loss 仍然按 50 步工作。H30 critic 只决定 critic 看前
30 步 action；梯度会通过同一个 50 步 endpoint 回传，未被 critic 直接使用的后 20 步
不会被错误送进 H30 action encoder。

2. 新增两个 QAM probe config：

```text
examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05_probe_h50_tau09_lq005_gc005.yaml
examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05_probe_h30_tau06_lq005_gc005.yaml
```

两者都沿用当前较稳的 probe 强度：

```yaml
lambda_q: 0.05
qam_grad_clip: 0.05
max_steps: 300
save_interval: 100
```

区别只在 critic checkpoint、`critic.model.action_horizon` 和 `critic.model.quantile_tau`。

3. `examples/lwd/scripts/train_lwd_qam_cloud.sh` 新增两个 run mode：

```bash
RLINF_RUN_MODE=probe_h50_tau09_lq005_gc005
RLINF_RUN_MODE=probe_h30_tau06_lq005_gc005
```

脚本会为不同 mode 设置对应的默认 critic checkpoint 路径和不同的
`experiment_name`。所有结果仍放在同一个 log root：

```text
${RLINF_QAM_LOG_ROOT}/${experiment_name}/checkpoints
```

因此不同任务不会互相覆盖 checkpoint，也不需要反复手改 bash 文件。

同时新增两个可直接提交的 hope：

```text
examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_probe_h50_tau09_lq005_gc005_8a100.hope
examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_probe_h30_tau06_lq005_gc005_8a100.hope
```

它们分别调用两个很薄的 wrapper：

```text
examples/lwd/scripts/probe_h50_tau09_lq005_gc005_lwd_qam_cloud.sh
examples/lwd/scripts/probe_h30_tau06_lq005_gc005_lwd_qam_cloud.sh
```

wrapper 只固定 `RLINF_RUN_MODE`，公共环境初始化和训练逻辑仍然复用
`train_lwd_qam_cloud.sh`。这样 hope 的入口是分开的，但不会复制两份完整训练脚本。

### 当前解决的核心问题

1. 解决了 H30 critic 不能正确接入 QAM 的适配问题。
2. 保持 actor/QAM 训练仍为 50 步，避免把 SFT policy 的 action chunk 语义一起改掉。
3. 把不同 QAM probe 固化为 run mode 和独立 experiment_name，降低云端 failover/resume
   时读到被修改 bash 的风险。
4. 代码里没有新增额外的 algorithm horizon 参数，直接复用已有
   `critic.model.action_horizon`，避免同一含义在多个位置重复配置。

### 建议提交的下一步实验

先同时提交两个 300-step QAM probe：

```text
examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_probe_h50_tau09_lq005_gc005_8a100.hope
examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_probe_h30_tau06_lq005_gc005_8a100.hope
```

训练后优先评估 `global_step_100/200/300`，用 fixed reset state 和 fixed ODE noise
与 SFT 做 paired evaluation。判断标准仍然不是训练曲线里的 `q_mean`，而是：

```text
QAM 是否稳定增加成功数；
QAM-only 是否明显多于 SFT-only；
视频中 QAM 是否改善抓取/对齐/敲击，而不是只刚好擦过 2cm 阈值。
```
