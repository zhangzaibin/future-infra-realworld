#!/bin/bash
# =============================================================================
# 使用远程 Diffusion Policy API 进行真机推理的启动脚本 (双相机版本)
# 
# 此脚本配置与 lerobot-record 命令类似，支持 front 和 side 双相机
# =============================================================================

set -e

# 默认配置
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_awesome_follower_arm}"

# DP 服务器配置
DP_SERVER_URL="${DP_SERVER_URL:-http://localhost:20020}"
AGENT_ID="${AGENT_ID:-0}"

# 相机配置 (使用哪个相机发送给 DP 服务器)
CAMERA_KEY="${CAMERA_KEY:-front}"
CAMERA_KEY_IN_DP="${CAMERA_KEY_IN_DP:-head_cam}"

# 推理配置
FPS="${FPS:-30}"
DURATION="${DURATION:-60}"
USE_ACTION_CHUNK="${USE_ACTION_CHUNK:-false}"
CHUNK_HORIZON="${CHUNK_HORIZON:-8}"

# 打印配置信息
echo "=============================================="
echo "DP 远程推理配置 (双相机版本)"
echo "=============================================="
echo "机器人类型: $ROBOT_TYPE"
echo "机器人端口: $ROBOT_PORT"
echo "机器人 ID:  $ROBOT_ID"
echo "DP 服务器:  $DP_SERVER_URL"
echo "Agent ID:   $AGENT_ID"
echo "使用相机:   $CAMERA_KEY -> $CAMERA_KEY_IN_DP"
echo "FPS:        $FPS"
echo "运行时长:   $DURATION 秒"
echo "Action Chunk: $USE_ACTION_CHUNK (horizon: $CHUNK_HORIZON)"
echo "=============================================="
echo ""

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# 双相机配置 (front + side)
CAMERA_CONFIG='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}'

# 构建命令
CMD="python -m lerobot.scripts.infer_with_dp_api \
    --robot.type=$ROBOT_TYPE \
    --robot.port=$ROBOT_PORT \
    --robot.id=$ROBOT_ID \
    --robot.cameras=\"$CAMERA_CONFIG\" \
    --dp_server_url=$DP_SERVER_URL \
    --agent_id=$AGENT_ID \
    --camera_key=$CAMERA_KEY \
    --camera_key_in_dp=$CAMERA_KEY_IN_DP \
    --fps=$FPS \
    --duration=$DURATION"

# 添加 action chunk 参数
if [ "$USE_ACTION_CHUNK" = "true" ]; then
    CMD="$CMD --use_action_chunk=true --chunk_execution_horizon=$CHUNK_HORIZON"
fi

# 执行命令
echo "执行命令:"
echo "$CMD"
echo ""

eval $CMD





