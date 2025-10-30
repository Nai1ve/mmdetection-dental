#!/bin/bash
LOG_FILE_BASE="/root/work_dirs/k_fold_teeth_experiment"
LOG_FILE="${LOG_FILE_BASE}/k_fold_run_$(date +'%Y%m%d_%H%M%S').log"
SHUTDOWN_CMD="/usr/bin/shutdown"
mkdir -p ${LOG_FILE_BASE}
echo "--- Script started. Logging to: ${LOG_FILE} ---"
{
    TRAIN_CONFIG_FILE='configs/_teeth_/faster-rcnn_r50_fpn_1x_coco_teeth_k_fold.py'
    TEST_CONFIG_FILE='configs/_teeth_/faster-rcnn_r50_fpn_1x_coco_teeth_k_fold_gnn_train_data.py'

    WORK_DIR_BASE='/root/work_dirs/k_fold_teeth_experiment' 
    K_FOLDS=5
    KFOLD_ANN_DIR="/root/autodl-tmp/dataset/coco/crop_child/annotations"                             # K-Fold标注文件目录
    OUTPUT_PKL_DIR="gnn_training_data_raw"                             # 保存每折预测结果的目录

    # 创建输出目录
    mkdir -p ${OUTPUT_PKL_DIR}

    # 循环 K 折
    for (( FOLD=1; FOLD<=${K_FOLDS}; FOLD++ ))
    do
        echo "========================================================"
        echo "Processing Fold ${FOLD}/${K_FOLDS}"
        echo "========================================================"

        # 1. 定义当前折的文件路径
        TRAIN_ANN_FILE="${KFOLD_ANN_DIR}/train_fold_${FOLD}.json"
        VAL_ANN_FILE="${KFOLD_ANN_DIR}/val_fold_${FOLD}.json" # 验证集即本轮要预测的数据集
        WORK_DIR="${WORK_DIR_BASE}/fold_${FOLD}"
        CHECKPOINT_FILE="${WORK_DIR}/latest.pth" # MMDetection默认保存的最佳或最后模型
        OUTPUT_PKL_FILE="${OUTPUT_PKL_DIR}/preds_fold_${FOLD}.pkl"

        # 2. 训练模型
        echo "--- Training Model for Fold ${FOLD} ---"
        python tools/train.py ${TRAIN_CONFIG_FILE} \
            --work-dir ${WORK_DIR} \
            --cfg-options \
                data.train.ann_file=${TRAIN_ANN_FILE} \
                data.val.ann_file=${VAL_ANN_FILE} \
                # 如果你的配置文件中没有验证器(validator)或需要关闭验证,可以注释掉 data.val.ann_file
                # 或者如果你想在训练时就用这个fold进行验证,则保持
            # 可以添加 --auto-scale-lr 等参数优化训练
        echo "--- Searching for best checkpoint in ${WORK_DIR} ---"
        CHECKPOINT_FILE=$(find "${WORK_DIR}" -maxdepth 1 -name "best_*.pth" | head -n 1)

        # 检查是否找到了最佳模型
        if [ -z "${CHECKPOINT_FILE}" ] || [ ! -f "${CHECKPOINT_FILE}" ]; then
            echo "Warning: Could not find 'best_*.pth' checkpoint in ${WORK_DIR}."
            echo "--- Attempting to fall back to 'latest.pth' ---"
            CHECKPOINT_FILE="${WORK_DIR}/latest.pth"
            
            # 再次检查 'latest.pth' 是否存在
            if [ ! -f "${CHECKPOINT_FILE}" ]; then
                echo "Error: Training for Fold ${FOLD} failed. Neither 'best_*.pth' nor 'latest.pth' found in ${WORK_DIR}"
                echo "Skipping prediction for this fold."
                continue # 跳过此折, 继续下一折
            fi
        fi

        echo "Using checkpoint: ${CHECKPOINT_FILE}"

        # 3. 在对应的验证集 (即本折数据) 上进行预测
        echo "--- Predicting on Fold ${FOLD} using Model trained on other folds ---"
        python tools/test.py ${TEST_CONFIG_FILE} ${CHECKPOINT_FILE} \
            --out ${OUTPUT_PKL_FILE} \
            --cfg-options \
                data.test.ann_file=${VAL_ANN_FILE} \
                # 确保test配置指向的是当前fold的验证文件

        echo "Predictions for Fold ${FOLD} saved to ${OUTPUT_PKL_FILE}"
    done

    echo "========================================================"
    echo "K-Fold Training and Prediction Completed!"
    echo "Raw prediction files are in: ${OUTPUT_PKL_DIR}"
    echo "========================================================"

} 2>&1 | tee -a ${LOG_FILE}

echo "--- K-Fold process finished. ---"
echo "--- Full log file saved to: ${LOG_FILE} ---"

# 等待 10 秒, 给你一个短暂的窗口来按 Ctrl+C (如果你突然反悔)
echo "--- Process complete. Shutting down server in 10 seconds... ---"
echo "--- (Press Ctrl+C NOW to cancel shutdown) ---"
sleep 10

echo "--- Issuing shutdown command: ${SHUTDOWN_CMD} now ---"
echo "--- 拜拜! ---"

# 执行关机
# 再次提醒: 这需要 sudo 权限
sudo ${SHUTDOWN_CMD} now

exit 0