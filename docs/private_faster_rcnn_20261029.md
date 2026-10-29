# 私有数据 Faster R-CNN / CCSW 对照

两组均为 Faster R-CNN R50-FPN、53 类、1600x960 保持比例缩放、batch=2、SGD lr=0.0025、24 轮、16/22 轮衰减、seed=42。继承相同 ImageNet ResNet50 初始化；不加载旧 48 类检测权重。

后续批次资源调整记录：本轮固定 batch=2，不中途改配置。当前服务器观测到的显存余量较大，下一轮可将训练 batch 小幅提高到 3；baseline 与 CCSW 必须同步调整，并记录实际峰值显存、每轮时长和验证集召回/定位指标。 batch 改动应单独登记为新实验，不覆盖本轮结果；若同时按线性规则调整学习率（0.0025 -> 0.00375），需在实验表中明确标注。

唯一训练差异：baseline 使用标准 Shared2FCBBoxHead，ccsw 使用其子类 CCSWBBOXHead，对乳牙正 proposal 的分类 loss 加连续方向权重。回归 loss、RPN、采样和增强保持一致。此版本修订了原 CCSW 的数学定义，不能把它与旧结果视为完全同一方法。

CCSW：53 类乳牙索引为 32–51，上颌牙冠朝下，下颌朝上，正对齐候选在 split=0.5 时权重为 1，向牙冠偏移渐增到 1.5，向牙根偏移渐减到 0.5。额外牙类别不使用固定方向。它是全框提供的弱先验，不是牙冠定位监督。无正样本时沿用标准背景分类和零回归处理。

推理共用 DentalCandidateRoIHead，不改训练损失。RPN 保留 2000 proposals、NMS IoU=0.8；RoI score=0.01、NMS IoU=0.6、最多 500 条类别候选。阈值为首轮固定设置，尚未经临床验证。保留低分候选不能保证检出所有牙，后续需要在验证集报告 recall/FP 和多生牙 FROC。

predictions.pkl 每个候选包含 bboxes/scores/labels、proposal_ids、1024 维 visual_features、含背景的 54 维 all_class_probs、foreground_scores=1-p(background)。同一 proposal 的不同类别候选通过 proposal_ids 关联；不要把每条类别候选当成一颗不同牙。标准 COCO JSON 不保存这些特征，交给 Agent 时使用 PKL（仅加载可信来源）。无 proposal 的空预测沿用标准空结果。

## 数据和启动

1. 解压 private_only.tar.gz（三中心训练+怀柔测试），或 private_single_stomatological.tar.gz（北京口腔训练+三个外部中心）。
2. 服务器使用源码 editable 安装此仓库，确认 PyTorch/MMCV CUDA 环境匹配：`pip install -v -e .`。
3. 设置数据和工作目录；每份数据分别跑两组，共四次训练。

```bash
export DENTAL_DATA_ROOT=/root/autodl-tmp/private_only
export DENTAL_WORK_DIR=/root/autodl-tmp/work_dirs/multi_baseline
mkdir -p "$DENTAL_WORK_DIR"
nohup setsid bash tools/run_private_faster_rcnn.sh baseline > "$DENTAL_WORK_DIR/launcher.log" 2>&1 < /dev/null &
echo $! > "$DENTAL_WORK_DIR/launcher.pid"
```

CCSW 使用 `ccsw` 参数并换工作目录。单中心数据更换 DENTAL_DATA_ROOT 为 private_single_stomatological。两组请串行或放在不同 GPU 服务器运行。训练退出后自动评估最后一轮 checkpoint；保留最新一轮及 best AP 权重。验证集供调参，三个外部中心仅作最终评价。

直接训练配置：

- `configs/_teeth_/faster-rcnn_r50_fpn_24e_private_baseline.py`
- `configs/_teeth_/faster-rcnn_r50_fpn_24e_private_ccsw.py`

打包数据只包含正式划分图像、两套 COCO 视图、划分清单和复核表。review 内含原始病例路径/文件名，仅在受控服务器使用；公开发布需另行去标识。
