#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-/root/mmdetection-dental}
BASE=${BASE:-/root/autodl-tmp}
export REPO BASE
export RUN=${RUN:-$BASE/experiments/yolo_structure_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN" || { echo 'RUN exists; refusing to overwrite'; exit 1; }
cd "$REPO"
nohup setsid bash tools/run_yolo_structure.sh > "$RUN/launcher.log" 2>&1 < /dev/null &
echo $! > "$RUN/launcher.pid"
for ((i=0; i<20; i++)); do
  [[ -f "$RUN/runner.pid" ]] && break
  sleep 1
done
[[ -f "$RUN/runner.pid" ]] || { echo "Controller did not start; inspect $RUN/launcher.log"; exit 1; }
PID=$(cat "$RUN/runner.pid")
kill -0 "$PID" || { echo "Controller exited; inspect $RUN/launcher.log"; exit 1; }
printf 'Controller PID=%s; records=%s\n' "$PID" "$RUN"
