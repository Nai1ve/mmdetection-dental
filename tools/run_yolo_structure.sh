#!/usr/bin/env bash
# Launch via launch_yolo_structure.sh so the controller survives SSH disconnection.
set -euo pipefail
REPO=${REPO:-/root/mmdetection-dental}
BASE=${BASE:-/root/autodl-tmp}
PY=${PY:-$BASE/envs/ultralytics/bin/python}
DATA=${DATA:-$BASE/datasets/private_only}
CONVERTED=${CONVERTED:-$BASE/datasets/private_only_yolo_tooth1_v1}
RUN=${RUN:?Set a new run directory}
GROUP=${GROUP:-core}
SEEDS=${SEEDS:-42 43 44}
cd "$REPO"
mkdir "$RUN/controller.lock" || { echo 'Controller already started; use a new RUN'; exit 1; }
echo $$ > "$RUN/runner.pid"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
case "$GROUP" in
  core) variants='baseline control direction direction_balance' ;;
  secondary) variants='relation quality direction_relation direction_quality' ;;
  *) echo 'GROUP must be core or secondary'; exit 2 ;;
esac
git rev-parse HEAD > "$RUN/commit.txt"
sha256sum tools/run_modern_dental.py tools/dental_yolo_structure.py tools/dental_yolo_ablation.py \
  tools/run_yolo_structure.sh tools/validate_yolo_structure_run.py \
  configs/dental_yolo_structure/*.json > "$RUN/source_sha256.txt"
printf 'job,seed,state,time,work_dir\n' > "$RUN/status.csv"
"$PY" -c 'import torch,ultralytics; assert torch.cuda.is_available(); assert ultralytics.__version__=="8.4.191"; print(torch.__version__,ultralytics.__version__,torch.cuda.get_device_name())' > "$RUN/environment.txt"
for variant in $variants; do
  for seed in $SEEDS; do
    job=${variant}_s${seed}
    work=$BASE/work_dirs/$(basename "$RUN")_$job
    printf '%s,%s,RUNNING,%s,%s\n' "$job" "$seed" "$(date --iso-8601=seconds)" "$work" >> "$RUN/status.csv"
    common=(--model yolo11m --view tooth1 --data-root "$DATA" --converted "$CONVERTED"
      --work-dir "$work" --imgsz 960 --batch 8 --epochs 24 --seed "$seed"
      --structure-config "configs/dental_yolo_structure/$variant.json")
    if "$PY" tools/run_modern_dental.py "${common[@]}" > "$RUN/$job.log" 2>&1 &&
       "$PY" tools/run_modern_dental.py "${common[@]}" --evaluate-only --nms-iou .85 --eval-suffix _nms085 >> "$RUN/$job.log" 2>&1 &&
       "$PY" tools/validate_yolo_structure_run.py "$work" >> "$RUN/$job.log" 2>&1; then
      printf '%s,%s,SUCCESS,%s,%s\n' "$job" "$seed" "$(date --iso-8601=seconds)" "$work" >> "$RUN/status.csv"
    else
      printf '%s,%s,FAILED,%s,%s\n' "$job" "$seed" "$(date --iso-8601=seconds)" "$work" >> "$RUN/status.csv"
      echo "Stopped on failure: $job; inspect $RUN/$job.log" >&2
      exit 1
    fi
  done
done
date --iso-8601=seconds > "$RUN/completed.txt"
# No automatic shutdown: this batch has not received its own shutdown authorization.
