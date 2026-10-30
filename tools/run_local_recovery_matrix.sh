#!/usr/bin/env bash
set -euo pipefail
cd "${REPO:-/root/mmdetection-dental}"
BASE=${BASE:-/root/autodl-tmp}
PY=${PY:-$BASE/envs/ultralytics/bin/python}
DATA=${DATA:-$BASE/datasets/private_only}
CHECKPOINT=${CHECKPOINT:?Set the frozen original single-class YOLO checkpoint}
RUN=${RUN:?Set a fresh output directory}
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
echo $$ > "$RUN/runner.pid"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
git rev-parse HEAD > "$RUN/commit.txt"
sha256sum "$CHECKPOINT" tools/dental_local_recovery.py tools/run_dental_local_recovery.py \
  configs/dental_local_recovery/*.json > "$RUN/source_sha256.txt"
"$PY" tools/run_dental_local_recovery.py --data-root "$DATA" --fit-prior "$RUN/train_prior.json"
printf 'config,state\n' > "$RUN/status.csv"
# Development on val only. No automatic test-set sweep or power operation.
for config in full uniform spatial uniform_rotation spatial_rotation; do
  printf '%s,RUNNING\n' "$config" >> "$RUN/status.csv"
  if "$PY" tools/run_dental_local_recovery.py --data-root "$DATA" \
      --checkpoint "$CHECKPOINT" --prior "$RUN/train_prior.json" \
      --config "configs/dental_local_recovery/$config.json" \
      --output "$RUN/$config" --split val > "$RUN/$config.log" 2>&1; then
    printf '%s,SUCCESS\n' "$config" >> "$RUN/status.csv"
  else
    printf '%s,FAILED\n' "$config" >> "$RUN/status.csv"
    exit 1
  fi
done
date --iso-8601=seconds > "$RUN/completed.txt"
