# 候选召回复盘与现代检测器环境

## 指标纠正

MMDetection `coco_metric.py` 的 `extra_tooth_precision` 实际计算的是多生牙类别 AP@[0.50:0.95]，不是固定阈值下的普通 Precision。此前结果表中的 0.168、0.186 等应读作该类 AP；不能推断其召回或固定阈值精确率。

## 已完成分析

14 份可信服务器预测 PKL（约 2GB）已下载到本地 `records/remote_runs/20261029_private_frcnn_matrix/predictions/`，保留原划分与实验名称，不进入 Git。分析工具是 `tools/analysis_tools/analyze_dental_candidates.py`，机器可读输出为同记录目录的 `candidate_recall.json`。

采用类无关、一对一匹配；先最大化匹配数量，再以 IoU 作为次级目标。所有正常牙与多生牙共同参与匹配，再统计多生牙真值匹配数。一个预测框不能同时算作两颗牙。计算 score=0.01/0.05/0.1/0.25/0.5、IoU=0.4/0.5/0.75，并分别分析所有类别候选和每 proposal 只保留最高分候选。匹配忽略类别，不等同于正确识别多生牙身份。

以下固定 score=0.01，使用全部导出候选：

| 训练划分 | 方法 | 外部中心 | 全牙召回 IoU≥0.5 | 多生牙匹配/真值 IoU≥0.5 | 全牙召回 IoU≥0.75 | 多生牙匹配/真值 IoU≥0.75 |
|---|---|---|---:|---:|---:|---:|
| 三中心 | baseline | 怀柔 | 99.35% | 44/53 | 89.50% | 30/53 |
| 三中心 | CCSW | 怀柔 | 99.34% | 44/53 | 89.36% | 28/53 |
| 北京口腔 | baseline | 复兴门 | 99.84% | 29/29 | 94.66% | 21/29 |
| 北京口腔 | CCSW | 复兴门 | 99.81% | 29/29 | 94.56% | 24/29 |
| 北京口腔 | baseline | 世纪坛 | 99.73% | 10/14 | 95.38% | 6/14 |
| 北京口腔 | CCSW | 世纪坛 | 99.77% | 12/14 | 95.15% | 5/14 |
| 北京口腔 | baseline | 怀柔 | 99.42% | 47/53 | 89.56% | 29/53 |
| 北京口腔 | CCSW | 怀柔 | 99.44% | 46/53 | 89.42% | 28/53 |

在三中心怀柔测试中，阈值提高到 0.1 后，baseline 多生牙匹配降为 34/53，CCSW 为 38/53（IoU≥0.5）；每图候选约从 86 个降为 50 个。score=0.01 时约每图 43 个候选未与真值匹配，这包括重复、偏移候选和可能的标注不完整，不能直接解释为病理误诊。

同一测试、同一低阈值下，每 proposal 只留最高分框，baseline 多生牙匹配为 43/53、CCSW 为 44/53，与全部候选的结果接近。需要注意 proposal 去重不等于物理牙齿去重，不同 proposal 仍可能对应同一颗牙。

## 结论边界与下一步

现有检测器并非普遍找不到牙；普通牙低阈值框召回已高。更明确的短板是多生牙、严格定位质量，以及高召回对应的候选冗余。CCSW 在这些指标上没有稳定收益。下一轮应保留 Faster R-CNN，先验证单类 Tooth 任务定义，再以 YOLO11 和 RT-DETR 做架构对照，不预设换架构一定更好。

本次只有后处理后的预测，没有原始 RPN/pre-NMS 预测，不能确定剩余漏检具体发生于哪一级。阈值网格仅为描述性分析，不使用外部测试选择部署阈值；后续调参只用验证集。不同划分重复使用部分图像，不合并计数作为独立样本。多生牙样本少，暂未计算置信区间或完整 FROC；没有用无GPU模式重跑模型。

## 线上环境

- venv：`/root/autodl-tmp/envs/ultralytics`，使用 `--system-site-packages` 共享原 mmdet 的 PyTorch，不是完全独立的 conda 环境。
- Ultralytics=8.4.191，PyTorch=2.3.0+cu121；原 MMDetection 环境的 MMCV=2.1.0、MMEngine=0.10.6 未改动。
- YOLO11（yolo11n YAML）与 Ultralytics RT-DETR-L（rtdetr-l YAML）均已通过 320x320 CPU 模型构建与前向测试。这里只验证可执行性，不代表性能或训练正确性。
- 原 Baidu RT-DETR-R50 和 Ultralytics RT-DETR-L 不是同一个具体配置，本环境采用后者，后续实验必须明确记名。
- 安装日志：`/root/autodl-tmp/ultralytics_install.log`；CPU 冒烟日志与依赖快照：`/root/autodl-tmp/ultralytics_checks/`。
- COCO 预训练权重 `yolo11m.pt`（约39MB）和 `rtdetr-l.pt`（约64MB）均已下载到 `/root/autodl-tmp/models/` 并成功加载。尚未在牙片数据上微调。
- `pip check` 提示共享环境中的 openxlab 与 setuptools 版本约束不一致；未修改原环境，YOLO11/RT-DETR CPU 测试不受此提示影响。
- 无卡模式未验证 CUDA 前向、反向或训练显存；开GPU后需要补做真实图像训练/验证冒烟测试。
