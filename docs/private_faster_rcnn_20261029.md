# 私有数据 Faster R-CNN / CCSW 对照

指标更正：下文历史字段 `extra_tooth_precision` 实为 COCO 类别 AP@[0.50:0.95]，不是固定阈值 Precision。详见[候选召回复盘](candidate_recall_and_modern_detectors_20261029.md)。

两组均为 Faster R-CNN R50-FPN、53 类、1600x960 保持比例缩放、batch=2、SGD lr=0.0025、24 轮、16/22 轮衰减、seed=42。继承相同 ImageNet ResNet50 初始化；不加载旧 48 类检测权重。

后续批次资源调整记录：本轮固定 batch=2，不中途改配置。24GB 显存卡当前观察到的峰值占用约 9GB，下一轮优先试 batch=4；若峰值仍留有充足余量且训练稳定，再试 batch=6。baseline 与 CCSW 必须同步调整，并记录实际峰值显存、每轮时长和验证集召回/定位指标。batch 改动应单独登记为新实验，不覆盖本轮结果；若同时按线性规则调整学习率（batch=4 时 0.0025 -> 0.005），需在实验表中明确标注。

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

## 2026-10-29 首轮服务器结果

AutoDL 上四组串行训练和测试均正常结束，均为 exit_code=0。环境为单张 RTX 4090D 24GB；本轮训练 batch=2，观察到显存峰值约 9.5GB，结束后 GPU 空闲。训练、日志和权重均保存在 `/root/autodl-tmp`。四个工作目录总占用约 3.9GB，磁盘仍有约 39GB 可用。14 份 `predictions.pkl` 总计约 2.0GB，包含候选框及 RoI 特征，路径位于各工作目录的 `eval/<split>/predictions.pkl`。

| 数据划分 | 方法 | 测试集 | mAP | mAP50 | mAP75 | extra_tooth_precision |
|---|---|---|---:|---:|---:|---:|
| 三中心训练 | baseline | val | 0.702 | 0.976 | 0.838 | 0.181 |
| 三中心训练 | baseline | internal_test | 0.640 | 0.897 | 0.769 | 0.170 |
| 三中心训练 | baseline | external_test（怀柔） | 0.658 | 0.959 | 0.752 | 0.168 |
| 三中心训练 | CCSW | val | 0.701 | 0.976 | 0.844 | 0.154 |
| 三中心训练 | CCSW | internal_test | 0.644 | 0.908 | 0.767 | 0.188 |
| 三中心训练 | CCSW | external_test（怀柔） | 0.658 | 0.963 | 0.751 | 0.186 |
| 单中心训练（北京口腔） | baseline | val | 0.669 | 0.973 | 0.801 | 0.199 |
| 单中心训练（北京口腔） | baseline | external_fuxingmen | 0.625 | 0.890 | 0.754 | 0.307 |
| 单中心训练（北京口腔） | baseline | external_shijitan | 0.664 | 0.927 | 0.807 | 0.108 |
| 单中心训练（北京口腔） | baseline | external_huairou | 0.643 | 0.956 | 0.745 | 0.131 |
| 单中心训练（北京口腔） | CCSW | val | 0.672 | 0.974 | 0.799 | 0.182 |
| 单中心训练（北京口腔） | CCSW | external_fuxingmen | 0.625 | 0.892 | 0.755 | 0.360 |
| 单中心训练（北京口腔） | CCSW | external_shijitan | 0.664 | 0.928 | 0.806 | 0.138 |
| 单中心训练（北京口腔） | CCSW | external_huairou | 0.640 | 0.955 | 0.738 | 0.134 |

初步解释：CCSW 没有带来整体 mAP 的稳定提升。三中心训练中，怀柔外部测试 mAP 持平（0.658），mAP75 略低（0.752 -> 0.751），多生牙 precision 小幅增加（0.168 -> 0.186）。单中心设定中，三个外部中心的 mAP 基本持平；多生牙 precision 在复兴门提高、世纪坛和怀柔变化很小。类别样本少且各中心差异明显，当前只跑了单次种子，不能据此认定 CCSW 有稳定收益。

本轮只完成 COCO bbox 评估及候选导出，没有运行牙位漏检、定位命中或多生牙专门 recall/FROC 评估。因此 `extra_tooth_precision` 不能代表多生牙检出率，也不能单独证明召回改善。后续应补充 `teeth_error_profile` / 低阈值候选召回、定位误差和中心命中统计，再决定保留 CCSW 与否。下一轮 batch 试验按本记录前述计划先试 batch=4，baseline 与 CCSW 使用相同设置；本轮 checkpoint 与预测文件不覆盖。

四组完整日志、状态表和本地复核结果另存于 [`records/remote_runs/20261029_private_frcnn_matrix/`](../records/remote_runs/20261029_private_frcnn_matrix/结果记录.md)。原始日志仅保存在本地归档，不推送到远端仓库。
