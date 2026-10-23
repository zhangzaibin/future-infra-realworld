#!/usr/bin/env bash
# ==============================================================================
# delete_episodes.sh — 删除 LeRobot 数据集末尾的 N 条 episode
#
# 用法:
#   bash scripts/delete_episodes.sh --dataset ./datasets/pick_and_place --last 1
#   bash scripts/delete_episodes.sh --dataset ./datasets/pick_and_place --last 3 --yes
#
# 注意:
#   - 只支持删除「末尾连续」的 episode
#   - 从中间删除请使用: python -m lerobot.scripts.lerobot_edit_dataset
#   - 添加 --yes 跳过确认
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/delete_last_episodes.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "错误: 找不到 $PY_SCRIPT"
    exit 1
fi

# 找可用的 Python
PYTHON=""
for candidate in ~/anaconda3/envs/atlas_lerobot/bin/python python3 python; do
    if "$candidate" -c "import pyarrow" 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "错误: 找不到可用的 Python（需要 pyarrow）"
    exit 1
fi

exec "$PYTHON" "$PY_SCRIPT" "$@"
