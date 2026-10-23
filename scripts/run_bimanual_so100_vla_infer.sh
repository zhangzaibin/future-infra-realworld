#!/bin/bash
# =============================================================================
# 双臂 SO100 VLA 真机推理启动脚本
# 硬件/相机配置与数据采集脚本对齐（README 采集命令）：
#   left_arm=/dev/ttyACM0  right_arm=/dev/ttyACM2
#   right 腕部相机:  OpenCV index 0, 640x480, YUYV
#   left  腕部相机:  OpenCV index 2, 640x480, YUYV
#   orbbec_rgb:      OpenCV index 8, 1920x1080, YUYV
# Pi0 返回 32 维动作，取前 12 维 (left 6 + right 6)
# =============================================================================

# 默认配置（可以通过环境变量或命令行参数覆盖）
ROBOT_TYPE="${ROBOT_TYPE:-bi_so100_follower}"
LEFT_ARM_PORT="${LEFT_ARM_PORT:-/dev/ttyACM0}"
RIGHT_ARM_PORT="${RIGHT_ARM_PORT:-/dev/ttyACM2}"
ROBOT_ID="${ROBOT_ID:-bimanual_follower}"

# VLA 服务器配置（--vla-host/--vla-port 分开指定，或用 --vla-server 直接给完整 URL）
VLA_HOST="${VLA_HOST:-10.8.131.51}"
VLA_PORT="${VLA_PORT:-31806}"
VLA_SERVER_URL="${VLA_SERVER_URL:-}"

# 任务配置
TASK="${TASK:-Stack the bowls}"

# 相机配置（key 必须与采集数据集一致，才能正确喂给 VLA）
# 全局相机: orbbec_rgb (OpenCV /dev/video8, 1920x1080)
# 左手腕相机: left  (OpenCV /dev/video2, 640x480)
# 右手腕相机: right (OpenCV /dev/video0, 640x480)
GLOBAL_CAMERA_KEY="${GLOBAL_CAMERA_KEY:-orbbec_rgb}"
GLOBAL_CAMERA_INDEX="${GLOBAL_CAMERA_INDEX:-8}"
GLOBAL_CAMERA_WIDTH="${GLOBAL_CAMERA_WIDTH:-1920}"
GLOBAL_CAMERA_HEIGHT="${GLOBAL_CAMERA_HEIGHT:-1080}"
LEFT_WRIST_CAMERA_KEY="${LEFT_WRIST_CAMERA_KEY:-left}"
LEFT_WRIST_CAMERA_INDEX="${LEFT_WRIST_CAMERA_INDEX:-2}"
RIGHT_WRIST_CAMERA_KEY="${RIGHT_WRIST_CAMERA_KEY:-right}"
RIGHT_WRIST_CAMERA_INDEX="${RIGHT_WRIST_CAMERA_INDEX:-0}"
WRIST_CAMERA_WIDTH="${WRIST_CAMERA_WIDTH:-640}"
WRIST_CAMERA_HEIGHT="${WRIST_CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CAMERA_FOURCC="${CAMERA_FOURCC:-YUYV}"

# 推理配置
FPS="${FPS:-30}"
DURATION="${DURATION:-6000}"

# 动作维度 (从 32 维截取前 12 维)
ACTION_DIM="${ACTION_DIM:-12}"

# Action Chunk 配置
USE_ACTION_CHUNK="${USE_ACTION_CHUNK:-true}"
CHUNK_EXECUTION_HORIZON="${CHUNK_EXECUTION_HORIZON:-150}"

# 动作限幅配置（chunk 模式下默认关闭，避免严重限制运动幅度）
ENABLE_ACTION_LIMIT="${ENABLE_ACTION_LIMIT:-false}"
MAX_ACTION_DELTA="${MAX_ACTION_DELTA:-5.0}"

# 人工接管（VLA 出错后按 Space 切到主臂遥操，数据按 lerobot-record 格式写入 dataset）
ENABLE_TAKEOVER="${ENABLE_TAKEOVER:-false}"
LEFT_LEADER_PORT="${LEFT_LEADER_PORT:-/dev/ttyACM1}"
RIGHT_LEADER_PORT="${RIGHT_LEADER_PORT:-/dev/ttyACM3}"
TELEOP_ID="${TELEOP_ID:-bimanual_leader}"
DATASET_REPO_ID="${DATASET_REPO_ID:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
DATASET_NUM_EPISODES="${DATASET_NUM_EPISODES:-200}"
RESUME="${RESUME:-true}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
ENABLE_LEADER_ALIGN="${ENABLE_LEADER_ALIGN:-false}"

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "双臂 SO100 VLA 推理启动脚本（硬件与数据采集脚本对齐）"
    echo ""
    echo "选项:"
    echo "  --left-arm-port PORT        左臂端口完整路径 (默认: /dev/ttyACM0)"
    echo "  --left-arm-index N          左臂 ACM 序号 (默认: 0, 等价于 --left-arm-port /dev/ttyACM<N>)"
    echo "  --right-arm-port PORT       右臂端口完整路径 (默认: /dev/ttyACM2)"
    echo "  --right-arm-index N         右臂 ACM 序号 (默认: 2, 等价于 --right-arm-port /dev/ttyACM<N>)"
    echo "  --robot-id ID               机器人 ID (默认: bimanual_follower)"
    echo "  --vla-host HOST             VLA 服务器 IP/主机名 (默认: 10.8.131.51)"
    echo "  --vla-port PORT             VLA 服务器端口 (默认: 31806)"
    echo "  --vla-server URL            直接指定 VLA 服务器完整 URL (覆盖 --vla-host/--vla-port)"
    echo "  --task TASK                 任务描述 (默认: Stack the bowls)"
    echo "  --orbbec-index IDX          全局相机 OpenCV 索引 (默认: 8)"
    echo "  --left-wrist-index IDX      左手腕相机索引 (默认: 2)"
    echo "  --right-wrist-index IDX     右手腕相机索引 (默认: 0)"
    echo "  --fps FPS                   控制频率 (默认: 30)"
    echo "  --duration SECONDS          运行时长 (默认: 6000)"
    echo "  --action-dim DIM            动作维度 (默认: 12, 从 32 维截取)"
    echo "  --use-action-chunk          使用 Action Chunk 模式 (默认: true)"
    echo "  --no-action-chunk           不使用 Action Chunk 模式"
    echo "  --chunk-horizon N           每个 chunk 执行多少步 (默认: 150)"
    echo "  --enable-limit              启用动作限幅"
    echo "  --no-limit                  不启用动作限幅 (默认)"
    echo "  --max-delta DELTA           最大动作变化量 (默认: 5.0)"
    echo "  --enable-takeover           启用人工接管：VLA 出错后按 Space/T 切到主臂遥操并写入 dataset"
    echo "  --left-leader-port PORT     左主臂端口 (默认: /dev/ttyACM1)"
    echo "  --right-leader-port PORT    右主臂端口 (默认: /dev/ttyACM3)"
    echo "  --teleop-id ID              主臂 ID (默认: bimanual_leader)"
    echo "  --dataset-repo-id REPO      接管数据的 repo_id，例如 robodata/fun"
    echo "  --dataset-root DIR          接管数据本地目录 (默认: ./datasets/<repo名>)"
    echo "  --num-episodes N            最多保存多少条接管 episode (默认: 200)"
    echo "  --resume                    已有 dataset 时追加写入 (接管模式默认开启)"
    echo "  --no-resume                 禁止追加；若目录已存在则报错"
    echo "  --display-data              用 rerun 显示相机画面"
    echo "  --enable-leader-align       接管暂停后按 A 让主臂自动对齐从臂 (默认关)"
    echo "  --help                      显示此帮助信息"
    echo ""
    echo "接管按键:"
    echo "  Space 或 T    VLA 运行中切换到人工遥操并开始录制"
    echo "  Enter         保存当前接管 episode；保存后再次 Enter 继续 VLA"
    echo "  R             丢弃当前接管 episode"
    echo "  A             接管暂停时让主臂对齐从臂 (需 --enable-leader-align)"
    echo "  Esc           结束整个会话"
    echo ""
    echo "示例:"
    echo "  $0 --task \"Stack the bowls\""
    echo ""
    echo "  $0 --vla-host 10.8.131.51 --vla-port 31806 --task \"fold the towels\""
    echo ""
    echo "  $0 --task \"fold the towels\" \\"
    echo "     --vla-host 10.8.131.51 --vla-port 31806 \\"
    echo "     --left-arm-index 0 --right-arm-index 2 \\"
    echo "     --orbbec-index 9 --left-wrist-index 2 --right-wrist-index 0 \\"
    echo "     --enable-takeover --enable-leader-align \\"
    echo "     --dataset-repo-id robodata/dagger0916 \\"
    echo "     --dataset-root ./datasets/dagger0916"
    echo ""
    echo "硬件配置 (与采集脚本一致):"
    echo "  从臂左/右:    /dev/ttyACM0  /dev/ttyACM2"
    echo "  主臂左/右:    /dev/ttyACM1  /dev/ttyACM3"
    echo "  全局相机:     orbbec_rgb  /dev/video8  1920x1080 YUYV"
    echo "  左手腕相机:   left        /dev/video2  640x480   YUYV"
    echo "  右手腕相机:   right       /dev/video0  640x480   YUYV"
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
        --left-arm-index)
            LEFT_ARM_PORT="/dev/ttyACM$2"
            shift 2
            ;;
        --right-arm-index)
            RIGHT_ARM_PORT="/dev/ttyACM$2"
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
        --vla-host)
            VLA_HOST="$2"
            shift 2
            ;;
        --vla-port)
            VLA_PORT="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --orbbec-index)
            GLOBAL_CAMERA_INDEX="$2"
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
        --enable-takeover)
            ENABLE_TAKEOVER="true"
            shift
            ;;
        --left-leader-port)
            LEFT_LEADER_PORT="$2"
            shift 2
            ;;
        --right-leader-port)
            RIGHT_LEADER_PORT="$2"
            shift 2
            ;;
        --teleop-id)
            TELEOP_ID="$2"
            shift 2
            ;;
        --dataset-repo-id)
            DATASET_REPO_ID="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --num-episodes)
            DATASET_NUM_EPISODES="$2"
            shift 2
            ;;
        --resume)
            RESUME="true"
            shift
            ;;
        --no-resume)
            RESUME="false"
            shift
            ;;
        --display-data)
            DISPLAY_DATA="true"
            shift
            ;;
        --enable-leader-align)
            ENABLE_LEADER_ALIGN="true"
            shift
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

# 未通过 --vla-server 直接指定完整 URL 时，用 host + port 拼装
if [[ -z "$VLA_SERVER_URL" ]]; then
    VLA_SERVER_URL="http://${VLA_HOST}:${VLA_PORT}"
fi

if [[ "$ENABLE_TAKEOVER" == "true" ]]; then
    if [[ -z "$DATASET_REPO_ID" ]]; then
        echo "错误: --enable-takeover 需要同时指定 --dataset-repo-id，例如 robodata/fun"
        exit 1
    fi
    if [[ -z "$DATASET_ROOT" ]]; then
        DATASET_NAME="${DATASET_REPO_ID#*/}"
        DATASET_ROOT="./datasets/${DATASET_NAME}"
    fi
fi

# 打印配置信息
echo "=============================================="
echo "双臂 SO100 VLA 远程推理配置"
echo "=============================================="
echo "机器人类型:       $ROBOT_TYPE"
echo "左臂端口:         $LEFT_ARM_PORT"
echo "右臂端口:         $RIGHT_ARM_PORT"
echo "机器人 ID:        $ROBOT_ID"
echo "VLA 服务器:       $VLA_SERVER_URL"
echo "任务:             $TASK"
echo "全局相机:         $GLOBAL_CAMERA_KEY (opencv index: $GLOBAL_CAMERA_INDEX, ${GLOBAL_CAMERA_WIDTH}x${GLOBAL_CAMERA_HEIGHT}, $CAMERA_FOURCC)"
echo "左手腕相机:       $LEFT_WRIST_CAMERA_KEY (opencv index: $LEFT_WRIST_CAMERA_INDEX, ${WRIST_CAMERA_WIDTH}x${WRIST_CAMERA_HEIGHT}, $CAMERA_FOURCC)"
echo "右手腕相机:       $RIGHT_WRIST_CAMERA_KEY (opencv index: $RIGHT_WRIST_CAMERA_INDEX, ${WRIST_CAMERA_WIDTH}x${WRIST_CAMERA_HEIGHT}, $CAMERA_FOURCC)"
echo "动作维度:         $ACTION_DIM (从 32 维截取)"
echo "FPS:              $FPS"
echo "运行时长:         $DURATION 秒"
echo "Action Chunk:     $USE_ACTION_CHUNK (horizon: $CHUNK_EXECUTION_HORIZON)"
echo "动作限幅:         $ENABLE_ACTION_LIMIT (max_delta: $MAX_ACTION_DELTA)"
echo "人工接管:         $ENABLE_TAKEOVER"
if [[ "$ENABLE_TAKEOVER" == "true" ]]; then
    echo "左主臂端口:       $LEFT_LEADER_PORT"
    echo "右主臂端口:       $RIGHT_LEADER_PORT"
    echo "主臂 ID:          $TELEOP_ID"
    echo "数据集 repo:      $DATASET_REPO_ID"
    echo "数据集目录:       $DATASET_ROOT"
    echo "最大接管条数:     $DATASET_NUM_EPISODES"
    echo "追加写入:         $RESUME"
    echo "主臂自动对齐:     $ENABLE_LEADER_ALIGN"
    echo "按键: Space/T 接管, Enter 保存, R 丢弃, Esc 退出"
fi
echo "=============================================="
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 构建三摄像头配置（与采集脚本相机 key / 分辨率 / fourcc 一致）
CAMERA_CONFIG="{ ${RIGHT_WRIST_CAMERA_KEY}: {type: opencv, index_or_path: ${RIGHT_WRIST_CAMERA_INDEX}, width: ${WRIST_CAMERA_WIDTH}, height: ${WRIST_CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}}, ${LEFT_WRIST_CAMERA_KEY}: {type: opencv, index_or_path: ${LEFT_WRIST_CAMERA_INDEX}, width: ${WRIST_CAMERA_WIDTH}, height: ${WRIST_CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}}, ${GLOBAL_CAMERA_KEY}: {type: opencv, index_or_path: ${GLOBAL_CAMERA_INDEX}, width: ${GLOBAL_CAMERA_WIDTH}, height: ${GLOBAL_CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: ${CAMERA_FOURCC}}}"

TAKEOVER_ARGS=()
if [[ "$ENABLE_TAKEOVER" == "true" ]]; then
    TAKEOVER_ARGS+=(
        --teleop.type=bi_so100_leader
        --teleop.left_arm_port="$LEFT_LEADER_PORT"
        --teleop.right_arm_port="$RIGHT_LEADER_PORT"
        --teleop.id="$TELEOP_ID"
        --dataset.repo_id="$DATASET_REPO_ID"
        --dataset.root="$DATASET_ROOT"
        --dataset.single_task="$TASK"
        --dataset.num_episodes="$DATASET_NUM_EPISODES"
        --dataset.push_to_hub=false
        --resume="$RESUME"
        --display_data="$DISPLAY_DATA"
    )
fi

# 运行双臂推理脚本
HF_HUB_OFFLINE=1 python -m lerobot.scripts.infer_bimanual_vla \
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
    --max_action_delta="$MAX_ACTION_DELTA" \
    --enable_leader_align="$ENABLE_LEADER_ALIGN" \
    "${TAKEOVER_ARGS[@]}"
