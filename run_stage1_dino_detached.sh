#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

CONFIG="${1:-configs/_teeth_/dino-4scale_r50_8xb2-24e_coco_teeth_stage1.py}"
WORK_DIR="${2:-/root/autodl-tmp/work_dirs/stage1_dino_private_53}"
GPUS="${3:-${GPUS:-1}}"
LAUNCH_ROOT="${LAUNCH_ROOT:-/root/autodl-tmp/teeth_launches/stage1_dino_private_53}"
RECORD_FILE="${RECORD_FILE:-$REPO_DIR/records/stage1_dino_experiments.md}"

if ! [[ "$GPUS" =~ ^[0-9]+$ ]]; then
    echo "GPUS must be an integer, got: $GPUS" >&2
    exit 1
fi

EXTRA_ARGS=()
if [ "$#" -gt 3 ]; then
    EXTRA_ARGS=("${@:4}")
fi

RUN_ID="$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="$LAUNCH_ROOT/$RUN_ID"
RUN_SCRIPT="$RUN_DIR/run_training.sh"
LAUNCH_LOG="$RUN_DIR/launcher.log"
PID_FILE="$RUN_DIR/train.pid"
META_FILE="$RUN_DIR/run.meta"

mkdir -p "$RUN_DIR" "$WORK_DIR" "$(dirname "$RECORD_FILE")"

EXTRA_ARGS_QUOTED=""
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    printf -v EXTRA_ARGS_QUOTED '%q ' "${EXTRA_ARGS[@]}"
fi

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_DIR"

if [ -f /etc/network_turbo ]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:\${PYTHONPATH:-}"

if [ "$GPUS" -gt 1 ]; then
    exec bash tools/dist_train.sh "$CONFIG" "$GPUS" --work-dir "$WORK_DIR" ${EXTRA_ARGS_QUOTED}
else
    exec python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" ${EXTRA_ARGS_QUOTED}
fi
EOF
chmod +x "$RUN_SCRIPT"

cat > "$META_FILE" <<EOF
run_id=$RUN_ID
config=$CONFIG
work_dir=$WORK_DIR
gpus=$GPUS
extra_args=${EXTRA_ARGS[*]:-}
launcher_log=$LAUNCH_LOG
pid_file=$PID_FILE
EOF

if [ ! -f "$RECORD_FILE" ]; then
    cat > "$RECORD_FILE" <<'EOF'
# Stage 1 DINO experiments

Keep one row per run. Update final metrics after validation.

| Start | Config | Work dir | Launch log | PID | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
EOF
fi

nohup setsid bash "$RUN_SCRIPT" > "$LAUNCH_LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

printf '| %s | `%s` | `%s` | `%s` | `%s` | running | detached launch, gpus=%s |\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$CONFIG" \
    "$WORK_DIR" \
    "$LAUNCH_LOG" \
    "$PID" \
    "$GPUS" >> "$RECORD_FILE"

cat <<EOF
Launched detached training.
  PID:        $PID
  Run dir:    $RUN_DIR
  Launch log: $LAUNCH_LOG
  Work dir:   $WORK_DIR
  Record:     $RECORD_FILE

Monitor:
  tail -f "$LAUNCH_LOG"
  find "$WORK_DIR" -mindepth 2 -maxdepth 3 -type f \( -name '*.log' -o -name '*.log.json' \) | sort | tail -n 1
EOF
