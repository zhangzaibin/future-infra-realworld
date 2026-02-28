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

### Step 3 — Grant USB Permissions
```bash
sudo chmod 666 /dev/ttyACM*
```

### Step 4 — Discover Wrist Cameras
List all connected OpenCV-compatible cameras (typically the two wrist cameras):
```bash
lerobot-find-cameras opencv
```

### Step 5 — Start Recording
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

---

## Inference & Evaluation

Run the bimanual VLA inference script with a custom task description:
```bash
bash scripts/run_bimanual_vla_infer.sh --task "Stack the bowls"
```
