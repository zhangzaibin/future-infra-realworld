#!/bin/bash
# =============================================================================
# 使用远程 VLA (Vision-Language-Action) API 进行双臂真机推理的启动脚本
# Pi0 返回 32 维动作，取前 12 维 (left 6 + right 6)
# =============================================================================

# 默认配置（可以通过环境变量或命令行参数覆盖）
ROBOT_TYPE="${ROBOT_TYPE:-bi_so101_follower}"
LEFT_ARM_PORT="${LEFT_ARM_PORT:-/dev/ttyACM0}"
RIGHT_ARM_PORT="${RIGHT_ARM_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-bimanual_follower}"

# VLA 服务器配置
VLA_SERVER_URL="${VLA_SERVER_URL:-http://10.8.131.51:31771}"

# 任务配置
TASK="${TASK:-完成操作任务}"

# 相机配置
# global 相机: orbbec_rgb (Orbbec Astra shared-memory capture, output_mode=color)
# 左手腕相机: OpenCV /dev/video2
# 右手腕相机: OpenCV /dev/video4
GLOBAL_CAMERA_KEY="${GLOBAL_CAMERA_KEY:-global}"
GLOBAL_CAMERA_DEVICE_INDEX="${GLOBAL_CAMERA_DEVICE_INDEX:-0}"
LEFT_WRIST_CAMERA_KEY="${LEFT_WRIST_CAMERA_KEY:-left_wrist}"
LEFT_WRIST_CAMERA_INDEX="${LEFT_WRIST_CAMERA_INDEX:-2}"
RIGHT_WRIST_CAMERA_KEY="${RIGHT_WRIST_CAMERA_KEY:-right_wrist}"
RIGHT_WRIST_CAMERA_INDEX="${RIGHT_WRIST_CAMERA_INDEX:-4}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

# 推理配置
FPS="${FPS:-30}"
DURATION="${DURATION:-6000}"

# 动作维度 (从 32 维截取前 12 维)
ACTION_DIM="${ACTION_DIM:-12}"

# Action Chunk 配置
USE_ACTION_CHUNK="${USE_ACTION_CHUNK:-true}"
CHUNK_EXECUTION_HORIZON="${CHUNK_EXECUTION_HORIZON:-150}"

# 动作限幅配置
ENABLE_ACTION_LIMIT="${ENABLE_ACTION_LIMIT:-false}"
MAX_ACTION_DELTA="${MAX_ACTION_DELTA:-5.0}"

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "双臂 VLA 推理启动脚本"
    echo ""
    echo "选项:"
    echo "  --left-arm-port PORT        左臂端口 (默认: /dev/ttyACM0)"
    echo "  --right-arm-port PORT       右臂端口 (默认: /dev/ttyACM1)"
    echo "  --robot-id ID               机器人 ID (默认: bimanual_follower)"
    echo "  --vla-server URL            VLA 服务器 URL (默认: http://10.8.131.8:20011)"
    echo "  --task TASK                 任务描述 (默认: 完成操作任务)"
    echo "  --global-device-index IDX   Orbbec 设备索引/ID (默认: 0)"
    echo "  --left-wrist-index IDX      左手腕相机索引 (默认: 2)"
    echo "  --right-wrist-index IDX     右手腕相机索引 (默认: 4)"
    echo "  --fps FPS                   控制频率 (默认: 30)"
    echo "  --duration SECONDS          运行时长 (默认: 60)"
    echo "  --action-dim DIM            动作维度 (默认: 12, 从 32 维截取)"
    echo "  --use-action-chunk          使用 Action Chunk 模式 (默认: true)"
    echo "  --no-action-chunk           不使用 Action Chunk 模式"
    echo "  --chunk-horizon N           每个 chunk 执行多少步 (默认: 50)"
    echo "  --enable-limit              启用动作限幅 (默认: true)"
    echo "  --no-limit                  不启用动作限幅"
    echo "  --max-delta DELTA           最大动作变化量 (默认: 5.0)"
    echo "  --help                      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --task \"pick up the cup with both hands\""
    echo ""
    echo "  $0 --vla-server http://10.8.131.8:20011 \\"
    echo "     --left-arm-port /dev/ttyACM0 \\"
    echo "     --right-arm-port /dev/ttyACM1 \\"
    echo "     --task \"stack the bowls\""
    echo ""
    echo "硬件配置:"
    echo "  左臂:         /dev/ttyACM0"
    echo "  右臂:         /dev/ttyACM1"
    echo "  全局相机:     orbbec_astra (key: ${GLOBAL_CAMERA_KEY})"
    echo "  左手腕相机:   /dev/video2"
    echo "  右手腕相机:   /dev/video4"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --left-arm-port)
            LEFT_ARM_PORT="$2"
            shift 2
            ;;
        --right-arm-port)
            RIGHT_ARM_PORT="$2"
            shift 2
            ;;
        --robot-id)
            ROBOT_ID="$2"
            shift 2
            ;;
        --vla-server)
            VLA_SERVER_URL="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --global-device-index)
            GLOBAL_CAMERA_DEVICE_INDEX="$2"
            shift 2
            ;;
        --left-wrist-index)
            LEFT_WRIST_CAMERA_INDEX="$2"
            shift 2
            ;;
        --right-wrist-index)
            RIGHT_WRIST_CAMERA_INDEX="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --action-dim)
            ACTION_DIM="$2"
            shift 2
            ;;
        --use-action-chunk)
            USE_ACTION_CHUNK="true"
            shift
            ;;
        --no-action-chunk)
            USE_ACTION_CHUNK="false"
            shift
            ;;
        --chunk-horizon)
            CHUNK_EXECUTION_HORIZON="$2"
            shift 2
            ;;
        --enable-limit)
            ENABLE_ACTION_LIMIT="true"
            shift
            ;;
        --no-limit)
            ENABLE_ACTION_LIMIT="false"
            shift
            ;;
        --max-delta)
            MAX_ACTION_DELTA="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 打印配置信息
echo "=============================================="
echo "双臂 VLA 远程推理配置"
echo "=============================================="
echo "机器人类型:       $ROBOT_TYPE"
echo "左臂端口:         $LEFT_ARM_PORT"
echo "右臂端口:         $RIGHT_ARM_PORT"
echo "机器人 ID:        $ROBOT_ID"
echo "VLA 服务器:       $VLA_SERVER_URL"
echo "任务:             $TASK"
echo "全局相机:         $GLOBAL_CAMERA_KEY (type: orbbec_astra, device_index: $GLOBAL_CAMERA_DEVICE_INDEX, output_mode: color)"
echo "左手腕相机:       $LEFT_WRIST_CAMERA_KEY (index: $LEFT_WRIST_CAMERA_INDEX)"
echo "右手腕相机:       $RIGHT_WRIST_CAMERA_KEY (index: $RIGHT_WRIST_CAMERA_INDEX)"
echo "动作维度:         $ACTION_DIM (从 32 维截取)"
echo "FPS:              $FPS"
echo "运行时长:         $DURATION 秒"
echo "Action Chunk:     $USE_ACTION_CHUNK (horizon: $CHUNK_EXECUTION_HORIZON)"
echo "动作限幅:         $ENABLE_ACTION_LIMIT (max_delta: $MAX_ACTION_DELTA)"
echo "=============================================="
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 构建三摄像头配置
# - global: Orbbec Astra RGB (from shared memory; ensure orbbec_shm_capture is running)
# - left/right wrist: OpenCV (UVC cameras)
CAMERA_CONFIG="{ ${GLOBAL_CAMERA_KEY}: {type: orbbec_astra, device_index: ${GLOBAL_CAMERA_DEVICE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, output_mode: color}, ${LEFT_WRIST_CAMERA_KEY}: {type: opencv, index_or_path: ${LEFT_WRIST_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: MJPG}, ${RIGHT_WRIST_CAMERA_KEY}: {type: opencv, index_or_path: ${RIGHT_WRIST_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: MJPG}}"

# 运行双臂推理脚本
python -m lerobot.scripts.infer_bimanual_vla \
    --robot.type="$ROBOT_TYPE" \
    --robot.left_arm_port="$LEFT_ARM_PORT" \
    --robot.right_arm_port="$RIGHT_ARM_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.cameras="$CAMERA_CONFIG" \
    --vla_server_url="$VLA_SERVER_URL" \
    --task="$TASK" \
    --global_camera_key="$GLOBAL_CAMERA_KEY" \
    --left_wrist_camera_key="$LEFT_WRIST_CAMERA_KEY" \
    --right_wrist_camera_key="$RIGHT_WRIST_CAMERA_KEY" \
    --action_dim="$ACTION_DIM" \
    --fps="$FPS" \
    --duration="$DURATION" \
    --use_action_chunk="$USE_ACTION_CHUNK" \
    --chunk_execution_horizon="$CHUNK_EXECUTION_HORIZON" \
    --enable_action_limit="$ENABLE_ACTION_LIMIT" \
    --max_action_delta="$MAX_ACTION_DELTA"

