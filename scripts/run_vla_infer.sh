#!/bin/bash
# =============================================================================
# 使用远程 VLA (Vision-Language-Action) API 进行真机推理的启动脚本
# =============================================================================

# 默认配置（可以通过环境变量或命令行参数覆盖）
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_robot}"

# VLA 服务器配置
VLA_SERVER_URL="${VLA_SERVER_URL:-http://localhost:20019}"

# 任务配置
TASK="${TASK:-完成操作任务}"

# 相机配置
CAMERA_KEY="${CAMERA_KEY:-global}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
WRIST_CAMERA_KEY="${WRIST_CAMERA_KEY:-wrist}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-2}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

# 推理配置
FPS="${FPS:-30}"
DURATION="${DURATION:-60}"

# Action Chunk 配置
USE_ACTION_CHUNK="${USE_ACTION_CHUNK:-true}"
CHUNK_EXECUTION_HORIZON="${CHUNK_EXECUTION_HORIZON:-50}"

# 动作限幅配置
ENABLE_ACTION_LIMIT="${ENABLE_ACTION_LIMIT:-true}"
MAX_ACTION_DELTA="${MAX_ACTION_DELTA:-5.0}"

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --robot-type TYPE         机器人类型 (默认: so101_follower)"
    echo "  --robot-port PORT         机器人端口 (默认: /dev/ttyACM0)"
    echo "  --robot-id ID             机器人 ID (默认: my_robot)"
    echo "  --vla-server URL          VLA 服务器 URL (默认: http://localhost:20019)"
    echo "  --task TASK               任务描述 (默认: 完成操作任务)"
    echo "  --camera-key KEY          主相机名称 (默认: global)"
    echo "  --camera-index INDEX      主相机索引 (默认: 0)"
    echo "  --wrist-camera KEY        腕部相机名称 (默认: wrist)"
    echo "  --wrist-camera-index IDX  腕部相机索引 (默认: 2)"
    echo "  --fps FPS                 控制频率 (默认: 30)"
    echo "  --duration SECONDS        运行时长 (默认: 60)"
    echo "  --use-action-chunk        使用 Action Chunk 模式 (默认: true)"
    echo "  --no-action-chunk         不使用 Action Chunk 模式"
    echo "  --chunk-horizon N         每个 chunk 执行多少步 (默认: 50)"
    echo "  --enable-limit            启用动作限幅 (默认: true)"
    echo "  --no-limit                不启用动作限幅"
    echo "  --max-delta DELTA         最大动作变化量 (默认: 5.0)"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 默认双摄像头 (global + wrist)"
    echo "  $0 --vla-server http://192.168.1.100:20019 --task \"pick up the cup\""
    echo ""
    echo "  # 自定义相机索引"
    echo "  $0 --vla-server http://192.168.1.100:20019 \\"
    echo "     --camera-key global --camera-index 0 \\"
    echo "     --wrist-camera wrist --wrist-camera-index 2 \\"
    echo "     --task \"pick up the cup and place it in the basket\""
    echo ""
    echo "环境变量:"
    echo "  ROBOT_TYPE, ROBOT_PORT, ROBOT_ID, VLA_SERVER_URL, TASK,"
    echo "  CAMERA_KEY, CAMERA_INDEX, WRIST_CAMERA_KEY, WRIST_CAMERA_INDEX,"
    echo "  CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, FPS, DURATION,"
    echo "  USE_ACTION_CHUNK, CHUNK_EXECUTION_HORIZON, ENABLE_ACTION_LIMIT, MAX_ACTION_DELTA"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --robot-type)
            ROBOT_TYPE="$2"
            shift 2
            ;;
        --robot-port)
            ROBOT_PORT="$2"
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
        --camera-key)
            CAMERA_KEY="$2"
            shift 2
            ;;
        --camera-index)
            CAMERA_INDEX="$2"
            shift 2
            ;;
        --wrist-camera)
            WRIST_CAMERA_KEY="$2"
            shift 2
            ;;
        --wrist-camera-index)
            WRIST_CAMERA_INDEX="$2"
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
echo "VLA 远程推理配置"
echo "=============================================="
echo "机器人类型:       $ROBOT_TYPE"
echo "机器人端口:       $ROBOT_PORT"
echo "机器人 ID:        $ROBOT_ID"
echo "VLA 服务器:       $VLA_SERVER_URL"
echo "任务:             $TASK"
echo "相机:             $CAMERA_KEY (index: $CAMERA_INDEX)"
if [[ -n "$WRIST_CAMERA_KEY" ]]; then
    echo "腕部相机:         $WRIST_CAMERA_KEY (index: $WRIST_CAMERA_INDEX)"
fi
echo "FPS:              $FPS"
echo "运行时长:         $DURATION 秒"
echo "Action Chunk:     $USE_ACTION_CHUNK (horizon: $CHUNK_EXECUTION_HORIZON)"
echo "动作限幅:         $ENABLE_ACTION_LIMIT (max_delta: $MAX_ACTION_DELTA)"
echo "=============================================="
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 构建相机配置 JSON
if [[ -n "$WRIST_CAMERA_KEY" ]]; then
    # 双摄像头配置
    CAMERA_CONFIG="{ ${CAMERA_KEY}: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: MJPG}, ${WRIST_CAMERA_KEY}: {type: opencv, index_or_path: ${WRIST_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: MJPG}}"
    WRIST_CAMERA_ARG="--wrist_camera_key=${WRIST_CAMERA_KEY}"
else
    # 单摄像头配置
    CAMERA_CONFIG="{ ${CAMERA_KEY}: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: MJPG}}"
    WRIST_CAMERA_ARG=""
fi

# 运行推理脚本
python -m lerobot.scripts.infer_with_vla_api \
    --robot.type="$ROBOT_TYPE" \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.cameras="$CAMERA_CONFIG" \
    --vla_server_url="$VLA_SERVER_URL" \
    --task="$TASK" \
    --camera_key="$CAMERA_KEY" \
    $WRIST_CAMERA_ARG \
    --fps="$FPS" \
    --duration="$DURATION" \
    --use_action_chunk="$USE_ACTION_CHUNK" \
    --chunk_execution_horizon="$CHUNK_EXECUTION_HORIZON" \
    --enable_action_limit="$ENABLE_ACTION_LIMIT" \
    --max_action_delta="$MAX_ACTION_DELTA"

