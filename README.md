# VLA — Atlas Bimanual Robot

## Table of Contents
- [Data Collection via Bimanual Teleoperation](#data-collection-via-bimanual-teleoperation)
- [Inference & Evaluation](#inference--evaluation)

---

## Data Collection via Bimanual Teleoperation

### Step 1 — Connect Hardware
Plug in all power cables and USB connections for both leader and follower arms.

### Step 2 — Start the Orbbec Camera
Launch the Orbbec Astra camera bridge to stream RGB-D frames via shared memory:
```bash
~/AstraSDK/samples/lerobot_bridge/start_orbbec_capture.sh
```

### Step 3 — Grant USB Permissions （给USB端口权限）

```bash
sudo chmod 666 /dev/ttyACM*
```

### 查看USB设备
```bash
ls -l /dev/ttyACM*
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
```

### Step 4 — Discover Wrist Cameras （查看摄像头）
List all connected OpenCV-compatible cameras (typically the two wrist cameras):
```bash
lerobot-find-cameras opencv
```

### Step 5 — Start Recording （采集脚本，非初次采集要加 --resume=true）
enter启动/结束一轮episode录制，录制过程中按‘r'丢弃该条数据
Update `index_or_path` for `left` and `right` cameras based on the indices found above,
then adjust `repo_id`, `single_task`, and other parameters as needed:
```bash
lerobot-record \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/ttyACM0 \
  --robot.right_arm_port=/dev/ttyACM2 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    right:      {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"},
    left:       {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"},
    orbbec_rgb: {type: orbbec_astra, width: 640, height: 480, fps: 30, output_mode: "color"}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/ttyACM1 \
  --teleop.right_arm_port=/dev/ttyACM3 \
  --teleop.id=bimanual_leader \
  --display_data=true \
  --dataset.repo_id=robodata/pass_judy \
  --dataset.num_episodes=50 \
  --dataset.single_task="Hand over the doll and place it into the basket." \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10
```

```bash
HF_HUB_OFFLINE=1 lerobot-record \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/ttyACM0 \
  --robot.right_arm_port=/dev/ttyACM2 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    right:      {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "YUYV"},
    left:       {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "YUYV"},
    orbbec_rgb: {type: opencv, index_or_path: 8, width: 1920, height: 1080, fps: 30, fourcc: "YUYV"}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/ttyACM1 \
  --teleop.right_arm_port=/dev/ttyACM3 \
  --teleop.id=bimanual_leader \
  --display_data=false \
  --dataset.repo_id=robodata/fold_towel \
  --dataset.root=./datasets/fold_towel \
  --dataset.num_episodes=200 \
  --dataset.single_task="foldthetowel." \
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
bash scripts/run_bimanual_so100_vla_infer.sh --task "Stack the bowls"
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
