# Bimanual Pi0 VLA Evaluation Guide

双臂 Pi0 VLA 真机评测指南。

## 硬件配置


| 设备 | 路径/地址 | 说明 |
|---|---|---|
| 左臂 (SO101) | `/dev/ttyACM0` | Feetech STS3215 x6, 1M baud |
| 右臂 (SO101) | `/dev/ttyACM1` | Feetech STS3215 x6, 1M baud |
| 全局相机 | Orbbec Astra (device_index=0) | RGB 模式, 640x480@30fps |
| 左手腕相机 | `/dev/video2` | OpenCV UVC, 640x480@30fps |
| 右手腕相机 | `/dev/video4` | OpenCV UVC, 640x480@30fps |

## 服务端 (Pi0 VLA Server)

Pi0 模型部署在远程集群上，通过 HTTP API 通信。

- **URL**: `http://10.8.131.51:31771`（在 `run_bimanual_vla_infer.sh` 中配置）
- **输入**: 12 维关节状态 (left 6 + right 6) + 3 张图像 (global / left_wrist / right_wrist) + task 文本
- **输出**: 32 维动作向量，客户端截取前 12 维 (left 6 + right 6)

## 前置准备

### 1. 确认环境

```bash
conda activate lerobot
```

### 2. 确认硬件连接

```bash
# 检查机械臂串口
ls -l /dev/ttyACM0 /dev/ttyACM1

# 检查wrist相机
lerobot-find-cameras opencv
# 在outputs/captured_images路径下可以看到video字样的图片，通过图片确定好左右相机。

```

### 3. 启动 Orbbec 共享内存采集程序

Orbbec Astra 相机需要先启动共享内存采集进程（在另一个终端）：

```bash
# 打开另一个终端运行，这就是一直运行着奥比相机
~/AstraSDK/samples/lerobot_bridge/start_orbbec_capture.sh

#运行成功后，就可以使用了
```

### 4. 确认 VLA 服务器可达

```bash

#这块
curl http://10.8.131.51:31771/health
# 应返回: {"status": "healthy", "model_loaded": true}
```

## 运行评测

### 快速启动（使用默认配置）

```bash
cd /home/zzb/projects/so101/lerobot
bash scripts/run_bimanual_vla_infer.sh
```

### 指定任务

```bash
bash scripts/run_bimanual_vla_infer.sh --task "stack the bowls"
```

### 指定 VLA 服务器

```bash
bash scripts/run_bimanual_vla_infer.sh --vla-server http://10.8.131.51:31771 --task "pick up the cup"
```

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--task` | `"完成操作任务"` | 任务文本描述 |
| `--vla-server` | `http://10.8.131.51:31771` | VLA 服务器地址 |
| `--fps` | `30` | 控制频率 |
| `--duration` | `6000` | 最长运行时间（秒） |
| `--chunk-horizon` | `150` | 每个 action chunk 执行的步数 |
| `--left-arm-port` | `/dev/ttyACM0` | 左臂串口 |
| `--right-arm-port` | `/dev/ttyACM1` | 右臂串口 |
| `--action-dim` | `12` | 从 32 维输出截取的维度数 |
| `--no-limit` | (默认已关闭) | 关闭动作限幅 |
| `--enable-limit` | - | 启用动作限幅（慎用，chunk 模式下会严重限制运动幅度） |

### 用 Python 直接调用

```bash
python -m lerobot.scripts.infer_bimanual_vla \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/ttyACM0 \
    --robot.right_arm_port=/dev/ttyACM1 \
    --robot.id=bimanual_follower \
    --robot.cameras="{ global: {type: orbbec_astra, device_index: 0, width: 640, height: 480, fps: 30, output_mode: color}, left_wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
    --vla_server_url=http://10.8.131.51:31771 \
    --task="stack the bowls" \
    --fps=30 \
    --duration=6000 \
    --use_action_chunk=true \
    --chunk_execution_horizon=150 \
    --enable_action_limit=false
```

## 数据流

```
BiSO101Follower.get_observation()
  ├── left_shoulder_pan.pos ... left_gripper.pos  (6 floats)
  ├── right_shoulder_pan.pos ... right_gripper.pos (6 floats)
  ├── global         (Orbbec RGB 640x480)
  ├── left_wrist     (OpenCV 640x480)
  └── right_wrist    (OpenCV 640x480)
        │
        ▼
BimanualVLAObservationAdapter.convert()
  ├── observation/state: (12,) = [left_6, right_6]
  ├── observation/image: global camera (BGR→RGB)
  ├── observation/wrist_image: left wrist (BGR→RGB)
  ├── observation/right_wrist_image: right wrist (BGR→RGB)
  └── task: str
        │
        ▼
VLAClient  ──HTTP POST──▶  Pi0 Server (10.8.131.51:31771)
                              │
                              ▼
                         actions (chunk_size, 32)
        │
        ▼
BimanualVLAActionAdapter.convert()
  取前 12 维 → {left_xxx.pos: float × 6, right_xxx.pos: float × 6}
        │
        ▼
BiSO101Follower.send_action()
  ├── 左臂执行 left_xxx.pos
  └── 右臂执行 right_xxx.pos
```

## 停止评测

按 `Ctrl+C` 即可安全停止。脚本会自动断开机器人连接并释放力矩。

## 故障排查

### 机械臂连接失败

```bash
# 扫描电机（不会移动机器人）
conda activate lerobot
python3 -c "
from lerobot.motors.feetech import FeetechMotorsBus
for port in ['/dev/ttyACM0', '/dev/ttyACM1']:
    print(f'\n== {port}')
    print(FeetechMotorsBus.scan_port(port))
"
```

- 如果某个端口返回 `{}`：检查该臂的电源、TTL 线连接
- 如果端口不存在：检查 USB 连接，运行 `ls /dev/ttyACM*`

### VLA 服务器超时

- 默认 HTTP 超时 120 秒，可通过 `--vla_timeout=300` 加长
- 检查网络：`ping 10.8.131.51`

### 相机无法打开

```bash
# 确认相机设备
python -m lerobot.scripts.lerobot_find_cameras opencv

# 确认 Orbbec 共享内存进程在运行
ps aux | grep orbbec_shm
```

## 文件结构

```
scripts/
  └── run_bimanual_vla_infer.sh          # 启动脚本（硬件配置在此修改）

src/lerobot/
  ├── robots/bi_so101_follower/          # 双臂 SO101 机器人驱动
  │   ├── config_bi_so101_follower.py
  │   └── bi_so101_follower.py
  ├── scripts/
  │   └── infer_bimanual_vla.py          # 双臂 VLA 推理主程序
  └── vla_client/
      ├── client.py                       # HTTP 客户端
      ├── observation_adapter.py          # 观察格式转换 (含 BimanualVLAObservationAdapter)
      └── action_adapter.py              # 动作格式转换 (含 BimanualVLAActionAdapter)
```

