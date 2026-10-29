#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATA_BASE="${DENTAL_DATA_BASE:-/root/autodl-tmp/datasets}"
OUTPUT_BASE="${DENTAL_MATRIX_OUTPUT:-/root/autodl-tmp/experiments/private_frcnn_matrix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_BASE"
STATUS_FILE="$OUTPUT_BASE/status.csv"
printf 'dataset,variant,start_time,end_time,exit_code,log\n' > "$STATUS_FILE"

datasets=(private_only private_single_stomatological)
variants=(baseline ccsw)
failures=0

for dataset in "${datasets[@]}"; do
    data_root="$DATA_BASE/$dataset"
    if [[ ! -f "$data_root/fdi53/train.json" || ! -d "$data_root/images" ]]; then
        printf 'Missing dataset files under %s\n' "$data_root" >&2
        exit 2
    fi
    for variant in "${variants[@]}"; do
        name="${dataset}_${variant}"
        work_dir="/root/autodl-tmp/work_dirs/$name"
        log_file="$OUTPUT_BASE/$name.log"
        start_time="$(date --iso-8601=seconds)"
        printf '%s,%s,%s,,RUNNING,%s\n' "$dataset" "$variant" "$start_time" "$log_file" >> "$STATUS_FILE"
        echo "[$start_time] START $name; data=$data_root; work=$work_dir"
        if DENTAL_DATA_ROOT="$data_root" DENTAL_WORK_DIR="$work_dir" \
            bash tools/run_private_faster_rcnn.sh "$variant" > "$log_file" 2>&1; then
            result=0
            echo "[$(date --iso-8601=seconds)] DONE $name"
        else
            result=$?
            failures=$((failures + 1))
            echo "[$(date --iso-8601=seconds)] FAILED $name exit=$result; see $log_file" >&2
        fi
        end_time="$(date --iso-8601=seconds)"
        sed -i "\$d" "$STATUS_FILE"
        printf '%s,%s,%s,%s,%s,%s\n' "$dataset" "$variant" "$start_time" "$end_time" "$result" "$log_file" >> "$STATUS_FILE"
        printf 'CURRENT_RESULT dataset=%s variant=%s exit=%s log=%s\n' "$dataset" "$variant" "$result" "$log_file" >> "$OUTPUT_BASE/summary.txt"
        grep -E 'coco/bbox_|bbox_mAP|mAP|classwise|extra_tooth|mean' "$log_file" | tail -n 40 >> "$OUTPUT_BASE/summary.txt" || true
    done
done

echo "Finished matrix with $failures failed experiment(s). Status: $STATUS_FILE"
exit "$((failures > 0))"
