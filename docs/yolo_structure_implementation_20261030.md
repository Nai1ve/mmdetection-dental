# 牙齿结构消融：代码、配置与执行约定

## 实现范围

保持单类Tooth、单一YOLO11检测器，不引入FDI头或第二个检测器。基于Ultralytics 8.4.191固定接口；未修改安装目录里的第三方源码。

主要实现：`tools/dental_yolo_structure.py`；训练入口仍为`tools/run_modern_dental.py`，新增`--structure-config`。不传该参数时保留原有训练路径。

### 方向特征模块

在P3/P4融合输出（16、19层）分别加轻量残差适配器。通道降为1/4，3x3、1x5、5x1深度卷积分支通过逐位置softmax门控融合，再还原通道。残差增益初始0.01，避免突然破坏预训练特征，同时允许首步梯度进入分支。

这是方向偏好，不是真实牙冠、牙根或解剖牙轴预测，也不是可变形卷积。对照改为3x3、3x3、1x1，分支核面积之和同为19，所以新增参数量与理论卷积MAC相同；实际延迟仍需GPU实测。

### 局部关系模块

P3/P4增加3x3邻域Q/K/V注意力，可学习邻域相对位置偏置，边界位置显式屏蔽padding。只有局部特征交互，不使用GT牙位构图，不把真实重叠牙齿强行排斥。

本版不实现候选级图网络、牙弓拟合或框后修正，避免将当前实现夸大为完整解剖关系模型。是否能区分重叠实例需通过消融验证。

### 定位质量分支

P3/P4/P5各加一个1x1质量预测卷积。正样本目标为当前预测框与所分配GT的IoU，目标停止梯度；仅在正样本上计算BCE，均值权重0.2，空正样本安全为零。

原分类、TAL、CIoU和DFL保留。为兼容原生三项loss日志，质量项加在box_loss中；`quality_audit.jsonl`单独记录未乘0.2的质量BCE，避免将box_loss差异误认为纯回归变化。

`DentalQualityPredictor`按原生NMS保留的anchor索引取质量分数，写入`detections.json`的`localization_quality`与PKL同名字段。PKL另保留`anchor_indices`。原始score、NMS和框保留逻辑不变，不用score乘quality进行删除。普通YOLO推理只返回普通检测结果；质量字段须用该自定义predictor获取。

质量只在正样本训练，不能把背景框的q解释为校准概率。它不是框回归头，也没有直接提高坐标准确率的保证。后续实体归并可在保持候选不变的前提下验证代表框排序；目前尚未自动使用q替换代表框。

## 配置矩阵

目录：`configs/dental_yolo_structure/`。

| 配置 | 方向/对照 | 局部关系 | 质量头 | 实例均衡 |
|---|---|---|---|---|
| baseline | 无 | 无 | 无 | 无 |
| control | 等参数卷积对照 | 无 | 无 | 无 |
| direction | 方向分支 | 无 | 无 | 无 |
| direction_balance | 方向分支 | 无 | 无 | 有 |
| relation | 无 | 有 | 无 | 无 |
| quality | 无 | 无 | 有 | 无 |
| direction_relation | 方向分支 | 有 | 无 | 无 |
| direction_quality | 方向分支 | 无 | 有 | 无 |

核心组为前四项，每项seed42/43/44，共12次；次级组为后四项，默认也是三种子，但应在核心组结果复核后另行启动，不自动串接。

单类YOLO11m基础参数量20053779；方向模块与control均新增168456（约0.84%）；独立relation新增163860；独立quality新增1283。方向+relation新增332316，方向+quality新增169739。这里只统计参数，不将其冒充GPU性能或显存测量。

统一private_only、YOLO11m、960、batch8、24轮、AdamW lr0=0.001，与前轮相同增强设置；conf0.01，分别评估NMS0.6/0.85的val/internal/external。先按val比较，外部结果完整披露，不再挑外部最优阈值/种子。

## 初始化和日志

- 先构建原生网络并按原生方式加载COCO预训练权重，再添加模块，避免包装层改变键名而漏载预训练权重。
- `structure_config.json`保存开关、基础/总参数量和初始化说明。新模块随机初始化；新旧网络初始化不可能完全相同，应通过三个种子和容量对照控制。
- 保存原生results.csv、best.pt、last.pt、正样本分配与质量审计。save_period=-1，不保存每轮大权重。
- 记录控制器真实PID、Git版本、源文件SHA256、独立日志与追加式RUNNING/SUCCESS/FAILED状态。
- 单组失败立即退出并保留日志，不更改定义重训。结构checkpoint用于评估；当前明确拒绝resume及从结构权重开启新训练，避免静默丢模块。断连不会停止控制器，但异常中断后的恢复尚未审计。
- 质量分支目前仅支持原生.pt推理；拒绝未审计的导出。只支持当前YOLO11层布局与单卡运行，未验证DDP/compile。

## 运行方式

服务器应已有代码、private_only数据、转换目录和COCO预训练模型。先跑GPU烟雾测试，不直接启动整个矩阵：

```bash
cd /root/mmdetection-dental
/root/autodl-tmp/envs/ultralytics/bin/python tools/run_modern_dental.py \
  --model yolo11m --view tooth1 \
  --data-root /root/autodl-tmp/datasets/private_only \
  --converted /root/autodl-tmp/datasets/private_only_yolo_tooth1_v1 \
  --work-dir /root/autodl-tmp/work_dirs/structure_gpu_smoke \
  --batch 8 --smoke \
  --structure-config configs/dental_yolo_structure/direction_quality.json
```

核心组使用新目录启动，控制器脱离SSH会话；启动前确认无其他GPU任务：

```bash
GROUP=core bash tools/launch_yolo_structure.sh
```

次级组不会自动启动，需要单独运行`GROUP=secondary bash tools/launch_yolo_structure.sh`。可用`SEEDS=42`做先导，但不把单种子当稳定结果。启动脚本不自动关机，本批次未沿用上批次关机授权。

## 验证边界

CPU测试脚本`tools/check_dental_yolo_structure.py`覆盖8个配置的形状、有限梯度、空标注、保存加载、原生预测兼容、质量anchor映射与不改原分数/框的检查；原始baseline与原生loss数值完全一致。等参数对照在32/256/512通道检查；另测YOLO11m布局及前向。

`--trainer-smoke`在临时合成数据上执行真实原生trainer、EMA、验证、checkpoint保存和重新加载，测试同时打开方向、关系、质量、实例均衡。它验证软件流程，不提供任何医学性能证据。合成数据与临时权重自动清理。

本地CPU为Torch2.15.0、Ultralytics8.4.191；服务器此前为Torch2.3.0+cu121。仍需服务器GPU/AMP烟雾测试及960显存测试，不能把CPU通过当GPU已验证。未启动任何正式新实验。

真正的牙冠/根尖/牙轴辅助监督暂未实现，因为现有水平框不支持可靠解剖标注。牙弓先验也未加入，不使用固定牙齿数、单条直线或强拓扑约束筛除多生牙。
