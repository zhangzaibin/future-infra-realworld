# VLA — Atlas Bimanual Robot

## Table of Contents
- [Data Collection via Bimanual Teleoperation](#data-collection-via-bimanual-teleoperation)
- [Inference & Evaluation](#inference--evaluation)

---

## Data Collection via Bimanual Teleoperation

### Step 1 — Connect Hardware
Plug in all power cables and USB connections for both leader and follower arms.
```

### Step 3 — Grant USB Permissions （给USB端口权限）

```bash
sudo chmod 666 /dev/ttyACM*
```

### 查看USB设备
```bash
# 这块usb拆入电脑顺序，会有ttyACM0 ttyACM1等等，从而自己就知道那个ttyACM是哪个arm。
ls -l /dev/ttyACM*
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
```

### Step 4 — Discover Wrist Cameras （查看摄像头）
List all connected OpenCV-compatible cameras (typically the two wrist cameras):
这块就能看到不同的相机是哪个arm了
```bash
lerobot-find-cameras opencv
```

### Step 5 — Start Recording （采集脚本，非初次采集要加 --resume=true）
enter启动/结束一轮episode录制，录制过程中按‘r'丢弃该条数据
Update `index_or_path` for `left` and `right` cameras based on the indices found above,
then adjust `repo_id`, `single_task`, and other parameters as needed:
```bash
HF_HUB_OFFLINE=1 lerobot-record \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/ttyACM0 \
  --robot.right_arm_port=/dev/ttyACM2 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    right:      {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: "YUYV"},
    left:       {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: "YUYV"},
    orbbec_rgb: {type: opencv, index_or_path: 4, width: 1920, height: 1080, fps: 30, fourcc: "YUYV"}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/ttyACM1 \
  --teleop.right_arm_port=/dev/ttyACM3 \
  --teleop.id=bimanual_leader \
  --display_data=false \
  --dataset.repo_id=robodata/teleoperation_data \
  --dataset.root=./datasets/teleoperation_data \
  --dataset.num_episodes=200 \
  --dataset.single_task="have fun." \
  --dataset.push_to_hub=false \
  --dataset.manual_episode_control=true \
  --resume=true
```

### 删除末尾episode
用 bash wrapper（自动找 conda 环境）
bash scripts/delete_episodes.sh --dataset ./datasets/pick_and_place --last 1

或直接用 Python
python scripts/delete_last_episodes.py --dataset ./datasets/pick_and_place --last 1

加 --yes 跳过确认
bash scripts/delete_episodes.sh --dataset ./datasets/pick_and_place --last 1 --yes


---

## Inference & Evaluation

硬件与相机配置与上方数据采集脚本对齐（`bi_so100_follower`，VLA 服务 `http://10.8.131.51:31707`）：
```bash
 
```

相机索引变化时可通过参数覆盖，例如：
```bash
bash scripts/run_bimanual_so100_vla_infer.sh \
  --task "Stack the bowls" \
  --vla-server http://10.8.131.51:31707 \
  --orbbec-index 8 \
  --left-wrist-index 2 \
  --right-wrist-index 0
```

### 推理中人工接管（纠错数据写入 dataset）

VLA 执行过程中如果人看出错了，按 **Space**（或 **T**）切到主臂遥操。接管后的数据格式与 `lerobot-record` 相同，Enter 保存、R 丢弃。保存后按 Enter 继续 VLA。

主臂默认 `left=/dev/ttyACM1`、`right=/dev/ttyACM3`，可用 `--left-leader-port` / `--right-leader-port` 覆盖。已有 dataset 时默认 `--resume` 追加写入。

```bash
HF_HUB_OFFLINE=1 bash scripts/run_bimanual_so100_vla_infer.sh \
  --task "fold the towels" \
  --vla-host 10.8.131.51 \
  --vla-port 31608 \
  --left-arm-index 0 \
  --right-arm-index 2 \
  --right-wrist-index 8 \
  --left-wrist-index 6 \
  --orbbec-index 4 \
  --enable-takeover \
  --enable-leader-align \
  --dataset-repo-id robodata/dagger1013 \
  --dataset-root ./datasets/dagger1013
```


## 从臂对齐主臂（一次性）

将主臂摆到目标姿态后运行，默认配对为 **ttyACM0 从臂 ← ttyACM1 主臂**、
**ttyACM2 从臂 ← ttyACM3 主臂**，包含各臂全部 6 个关节（含夹爪）。
先停止占用这些串口的采集/推理程序，确保四个臂已按下方命令完成校准，并留出运动空间。

```bash
# 只读取并打印当前/目标位置，不写入电机
bash scripts/align_followers_to_leaders.sh --dry-run

# 读取主臂姿态快照，用约 5 秒平滑对齐，从臂到位后保持
bash scripts/align_followers_to_leaders.sh

# 放慢移动；限幅或负载可能使实际到位时间超过 duration
bash scripts/align_followers_to_leaders.sh --duration 10 --timeout 30
```

脚本复用 `bimanual_follower_left/right` 和 `bimanual_leader_left/right` 的校准。
复制的是与遥操一致的**校准后归一化位置**（普通关节 -100～100、夹爪 0～100），
不是原始编码器数值或角度；姿态一致依赖正确的主从臂校准。
这是一次性对齐，运行中移动主臂不会更新目标。

默认每周期最大位置增量 `--max-step 2`，控制频率 `--fps 30`，到位容差 `--tolerance 2`，
超过 `--timeout 20` 秒仍未到位会报错退出。
成功退出时从臂**仍有扭矩并保持目标姿态**；加 `--release-on-exit` 可在成功后释放。
Ctrl+C 或运行异常会尝试释放已启用的从臂扭矩，机械臂可能下落，请做好支撑。
主臂全程只读，保持其原有扭矩状态。
串口和校准 ID 可通过 `--left-follower-port`、`--right-follower-port`、
`--left-leader-port`、`--right-leader-port`、`--robot-id`、`--teleop-id` 覆盖；
其余参数见 `bash scripts/align_followers_to_leaders.sh --help`。

## Calibration
```
# 1. 从臂左 (ttyACM0)
lerobot-calibrate --robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=bimanual_follower_left

# 2. 从臂右 (ttyACM2)
lerobot-calibrate --robot.type=so100_follower --robot.port=/dev/ttyACM2 --robot.id=bimanual_follower_right

# 3. 主臂左 (ttyACM1)
lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM1 --teleop.id=bimanual_leader_left

# 4. 主臂右 (ttyACM3)
lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM3 --teleop.id=bimanual_leader_right
```


## 从臂复位
bash scripts/align_followers_to_leaders.sh --dry-run
bash scripts/align_followers_to_leaders.sh