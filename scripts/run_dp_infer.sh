#!/bin/bash
# =============================================================================
# 使用远程 Diffusion Policy API 进行真机推理的启动脚本
# =============================================================================

# 默认配置（可以通过环境变量或命令行参数覆盖）
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_robot}"

# DP 服务器配置
DP_SERVER_URL="${DP_SERVER_URL:-http://localhost:20020}"
AGENT_ID="${AGENT_ID:-0}"

# 相机配置
CAMERA_KEY="${CAMERA_KEY:-front}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

# 推理配置
FPS="${FPS:-30}"
DURATION="${DURATION:-60}"

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --robot-type TYPE       机器人类型 (默认: so101_follower)"
    echo "  --robot-port PORT       机器人端口 (默认: /dev/ttyACM0)"
    echo "  --robot-id ID           机器人 ID (默认: my_robot)"
    echo "  --dp-server URL         DP 服务器 URL (默认: http://localhost:20020)"
    echo "  --agent-id ID           Agent ID (默认: 0)"
    echo "  --camera-key KEY        相机名称 (默认: front)"
    echo "  --camera-index INDEX    相机索引 (默认: 0)"
    echo "  --fps FPS               控制频率 (默认: 30)"
    echo "  --duration SECONDS      运行时长 (默认: 60)"
    echo "  --help                  显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --dp-server http://192.168.1.100:20020 --duration 120"
    echo ""
    echo "环境变量:"
    echo "  ROBOT_TYPE, ROBOT_PORT, ROBOT_ID, DP_SERVER_URL, AGENT_ID,"
    echo "  CAMERA_KEY, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,"
    echo "  CAMERA_FPS, FPS, DURATION"
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
        --dp-server)
            DP_SERVER_URL="$2"
            shift 2
            ;;
        --agent-id)
            AGENT_ID="$2"
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
        --fps)
            FPS="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
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
echo "DP 远程推理配置"
echo "=============================================="
echo "机器人类型: $ROBOT_TYPE"
echo "机器人端口: $ROBOT_PORT"
echo "机器人 ID:  $ROBOT_ID"
echo "DP 服务器:  $DP_SERVER_URL"
echo "Agent ID:   $AGENT_ID"
echo "相机:       $CAMERA_KEY (index: $CAMERA_INDEX)"
echo "FPS:        $FPS"
echo "运行时长:   $DURATION 秒"
echo "=============================================="
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 构建相机配置 JSON
CAMERA_CONFIG="{ ${CAMERA_KEY}: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: MJPG}}"

# 运行推理脚本
python -m lerobot.scripts.infer_with_dp_api \
    --robot.type="$ROBOT_TYPE" \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.cameras="$CAMERA_CONFIG" \
    --dp_server_url="$DP_SERVER_URL" \
    --agent_id="$AGENT_ID" \
    --camera_key="$CAMERA_KEY" \
    --fps="$FPS" \
    --duration="$DURATION"





