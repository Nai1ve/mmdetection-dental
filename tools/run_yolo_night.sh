#!/usr/bin/env bash
set -uo pipefail
cd /root/mmdetection-dental
BASE=/root/autodl-tmp
RUN=$BASE/experiments/yolo_ablation_20261030
PY=$BASE/envs/ultralytics/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$RUN"
echo $$ > "$RUN/runner.pid"
git rev-parse HEAD > "$RUN/controller_commit.txt"
sha256sum tools/dental_yolo_ablation.py tools/run_modern_dental.py tools/run_yolo_night.sh > "$RUN/source_sha256.txt"
printf 'job,variant,seed,end,status,work_dir\n' > "$RUN/status.csv"
failures=0
validate_completion() {
    "$PY" -c 'import csv,json,sys,pathlib;p=pathlib.Path(sys.argv[1]);r=list(csv.DictReader((p/"results.csv").open()));assert len(r)==24 and float(r[-1]["epoch"])==24;d=json.loads((p/"evaluation_summary.json").read_text());assert {x["split"] for x in d}=={"val","internal_test","external_test"}' "$1"
}
evaluate085() {
    "$PY" tools/run_modern_dental.py --model yolo11m --data-root "$BASE/datasets/private_only" \
        --converted "$BASE/datasets/private_only_yolo_tooth1_v1" --view tooth1 --work-dir "$1" \
        --batch 4 --evaluate-only --nms-iou .85 --eval-suffix _nms085
}
# This standard batch-eight control was launched while the code was prepared.
while ps -p 11672 -o args= | grep -q 'run_modern_dental_eval_v2.py.*yolo_night_b0_s42'; do sleep 10; done
work=$BASE/work_dirs/yolo_night_b0_s42
if validate_completion "$work" && evaluate085 "$work" > "$RUN/b0_s42_nms085.log" 2>&1; then
    code=0
else code=1; failures=$((failures+1)); fi
printf 'b0_s42,standard,42,%s,%s,%s\n' "$(date --iso-8601=seconds)" "$code" "$work" >> "$RUN/status.csv"
if [[ $code -eq 0 ]]; then
    "$PY" tools/evaluate_yolo_soft_nms.py --checkpoint "$work/weights/best.pt" --output "$RUN/soft_nms_val.json" > "$RUN/soft_nms.log" 2>&1
    code=$?
    printf 'soft_nms_val,postprocess,42,%s,%s,%s\n' "$(date --iso-8601=seconds)" "$code" "$work" >> "$RUN/status.csv"
    [[ $code -eq 0 ]] || failures=$((failures+1))
fi
for spec in 'balance 42' 'hard 42' 'combined 42' 'baseline 43' 'balance 43' 'baseline 44' 'balance 44'; do
    read -r variant seed <<< "$spec"
    job=${variant}_s${seed}
    work=$BASE/work_dirs/yolo_night_$job
    echo "START $job $(date --iso-8601=seconds)"
    printf '%s,%s,%s,,RUNNING,%s\n' "$job" "$variant" "$seed" "$work" >> "$RUN/status.csv"
    "$PY" tools/run_modern_dental.py --model yolo11m --data-root "$BASE/datasets/private_only" \
        --converted "$BASE/datasets/private_only_yolo_tooth1_v1" --view tooth1 --work-dir "$work" \
        --batch 8 --epochs 24 --imgsz 960 --loss-variant "$variant" --seed "$seed" > "$RUN/$job.log" 2>&1
    code=$?
    if [[ $code -eq 0 ]]; then
        validate_completion "$work" && evaluate085 "$work" > "$RUN/${job}_nms085.log" 2>&1
        code=$?
    fi
    sed -i '$d' "$RUN/status.csv"
    printf '%s,%s,%s,%s,%s,%s\n' "$job" "$variant" "$seed" "$(date --iso-8601=seconds)" "$code" "$work" >> "$RUN/status.csv"
    echo "END $job exit=$code"
    [[ $code -eq 0 ]] || failures=$((failures+1))
done
"$PY" tools/seal_yolo_night.py --run "$RUN" --failures "$failures"
seal_code=$?
if [[ $failures -eq 0 && $seal_code -eq 0 && "${AUTO_SHUTDOWN:-0}" == 1 ]]; then
    # Allow the local monitor to fetch and checksum the bundle; bounded idle time.
    for ((i=0;i<42;i++)); do
        [[ -f "$RUN/download_confirmed" ]] && break
        sleep 10
    done
    date --iso-8601=seconds > "$RUN/shutdown_requested.txt"
    sync
    /usr/bin/shutdown >> "$RUN/shutdown.log" 2>&1
else
    echo "SHUTDOWN_WITHHELD failures=$failures seal_exit=$seal_code" >> "$RUN/launcher.log"
fi
exit "$((failures>0 || seal_code>0))"
