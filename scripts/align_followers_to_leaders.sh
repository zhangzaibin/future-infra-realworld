#!/usr/bin/env bash
# 从臂 ttyACM0 ← 主臂 ttyACM1；从臂 ttyACM2 ← 主臂 ttyACM3。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/align_followers_to_leaders.py" "$@"
elif [[ -x "$HOME/anaconda3/envs/atlas_lerobot/bin/python" ]]; then
    exec "$HOME/anaconda3/envs/atlas_lerobot/bin/python" "$SCRIPT_DIR/align_followers_to_leaders.py" "$@"
else
    exec python3 "$SCRIPT_DIR/align_followers_to_leaders.py" "$@"
fi
