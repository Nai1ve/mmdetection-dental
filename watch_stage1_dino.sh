#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${1:-/root/autodl-tmp/work_dirs/stage1_dino_private_53}"
LAUNCH_ROOT="${2:-/root/autodl-tmp/teeth_launches/stage1_dino_private_53}"

LATEST_RUN_DIR="$(find "$LAUNCH_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
LATEST_MMENGINE_LOG="$(find "$WORK_DIR" -mindepth 2 -maxdepth 3 -type f \( -name '*.log' -o -name '*.log.json' \) 2>/dev/null | sort | tail -n 1 || true)"

echo "Work dir:   $WORK_DIR"
echo "Launch dir: $LATEST_RUN_DIR"

if [ -n "$LATEST_RUN_DIR" ] && [ -f "$LATEST_RUN_DIR/train.pid" ]; then
    PID="$(cat "$LATEST_RUN_DIR/train.pid")"
    if ps -p "$PID" >/dev/null 2>&1; then
        echo "PID:        $PID (running)"
        ps -p "$PID" -o pid=,ppid=,etime=,cmd=
    else
        echo "PID:        $PID (not running)"
    fi
fi

if [ -n "$LATEST_MMENGINE_LOG" ]; then
    echo "Tailing:    $LATEST_MMENGINE_LOG"
    tail -F "$LATEST_MMENGINE_LOG"
elif [ -n "$LATEST_RUN_DIR" ] && [ -f "$LATEST_RUN_DIR/launcher.log" ]; then
    echo "Tailing:    $LATEST_RUN_DIR/launcher.log"
    tail -F "$LATEST_RUN_DIR/launcher.log"
else
    echo "No log file found yet."
    exit 1
fi
