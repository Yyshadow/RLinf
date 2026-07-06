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

同时把 batch 和学习率调保守，避免 50 条 demo 全量微调过猛：

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

1. 先跑 `examples/sft/hope/robotwin_pi05_hammer50_smoke_8a100.hope`，确认 10 step 能完整 forward/backward/save。
2. smoke 通过后，再跑 `examples/sft/hope/robotwin_pi05_hammer50_train_8a100.hope`。
3. 如果 full finetune 显存稳定，可以再尝试把 `micro_batch_size` 从 2 增到 4；如果 OOM，则保持 1 或 2。

### 后续如果必须只训 expert

只训 expert 不是理论上不可行，但当前不能只靠 YAML 稳定解决。需要代码级处理 OpenPI 内部强制启用 Gemma expert gradient checkpointing 与 FSDP/view 的冲突，例如：

- 禁止 expert-only SFT 时 Gemma expert 进入强制 checkpoint 路径；
- 或在 OpenPI Gemma expert 的 view/inplace 敏感路径上做更细的 out-of-place/clone 处理；
- 或改用不触发该 SFT `super().forward()` 的训练路径。

在没有完成代码级修复前，hammer50 SFT 的稳定方案是 full finetuning。
