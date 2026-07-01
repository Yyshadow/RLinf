# Robotwin pi0.5 / LWD 数据闭环说明

这次改动只补 Robotwin 数据闭环的核心工具，不改 RLinf 的训练框架，也不改通用 `CollectEpisode`。目标是先把数据从 RoboTwin/RLinf 采集结果整理成 OpenPI pi0.5 能直接读取的 LeRobot-Aloha 格式，再从同一份数据派生 LWD/BPG critic 需要的 chunk 索引。

## 新增文件

```text
toolkits/robotwin/prepare_lerobot_aloha.py
toolkits/robotwin/validate_lerobot_aloha.py
toolkits/robotwin/build_lwd_chunks.py
```

## 数据主格式

pi0.5 Robotwin 训练使用 OpenPI Aloha dataconfig，核心字段是：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state
action
task
```

其中 `observation.state` 和 `action` 都是 14 维 Aloha 关节/夹爪向量。`action` 保存 absolute action，不提前转 delta；`pi05_aloha_robotwin` 配置会在 dataconfig 里自己处理 delta action。

这份 Robotwin 数据集当前采用的是 LeRobot 的 `image` 存法，不是 `video` 存法，所以 `meta/info.json` 里会看到：

```text
total_videos: 0
video_path: null
```

这不是缺东西，而是正常的落盘方式。训练 pi0.5 只需要 LeRobot parquet 里的图像字节和特征 schema，不要求一定有单独 mp4。

如果只是想看视频，建议额外导出可视化，不必把训练数据本身改成 video 格式。

如果当前环境不能写 `~/.cache`，可以先把缓存放到 `/tmp`：

```bash
export HF_HOME=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export MPLCONFIGDIR=/tmp/matplotlib
```

## 1. 转换数据

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

里面保存 episode 级信息，例如 `source`、`success`、`return`、`num_steps`。这些字段不是 pi0.5 SFT 必需，但后续筛数据和训练 critic 会用。

对于 RoboTwin 原生 hdf5，逐帧 `success` 只在最后一帧为 True；episode 是否成功看 `meta/robotwin_episode_metadata.jsonl`。这样可以避免把整条轨迹每一步都误当成成功状态。

如果转换失败轨迹，显式传 `--no-success --source-label failed_policy`；如果输入数据本身已经有 `success` / `is_success` 字段，脚本会优先保留原字段。

## 2. 校验数据

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

## 3. 计算 pi0.5 归一化统计

校验通过后，用现有工具计算 norm stats：

```bash
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_aloha_robotwin \
  --repo-id /data/wam_codebase/RLinf/datasets/robotwin_aloha/beat_block_hammer
```

本地路径作为 `--repo-id` 时，`norm_stats.json` 会写到这个数据集根目录下。训练时同一个 repo id / 本地路径会读到这份统计。

## 4. 训练 pi0.5 SFT

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

pi0.5 SFT 只应该使用成功 expert 或修正后的 near-miss 数据。普通失败轨迹不要直接作为 BC 正样本。

## 5. 生成 LWD/BPG chunk 索引

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

图像不复制进 chunk 文件，只保留 episode/frame 索引；critic dataloader 后续按索引回读 LeRobot 数据，避免数据膨胀。

## 可视化建议

如果你只想快速看一个 episode，推荐导出成一个视频加一个 json：

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

## 当前边界

第一阶段只搭数据闭环，不新增 LWD critic 模型和 worker。等以下链路稳定后再接 critic：

```text
RoboTwin hdf5 / RLinf LeRobot
  -> prepare_lerobot_aloha.py
  -> validate_lerobot_aloha.py
  -> calculate_norm_stats.py
  -> pi0.5 SFT
  -> build_lwd_chunks.py
```

这样做的好处是数据格式和 pi0.5 训练先闭环，后面失败轨迹、LWD critic、BPG-Flow 都可以基于同一份数据继续扩展。

## 本次 smoke test

已用现有样例 `/data/wam_codebase/RoboTwin_RLinf/demo_videos/_work/beat_block_hammer/data/episode0.hdf5` 跑通：

```text
prepare_lerobot_aloha.py -> 1 episode / 64 frames
validate_lerobot_aloha.py -> OpenPI pi05_aloha_robotwin transform passed
calculate_norm_stats.py -> 写出 norm_stats.json
build_lwd_chunks.py --horizon 10 -> 54 chunks
visualize_lerobot_dataset.py --mode video -> 1 mp4 + 1 json
```
