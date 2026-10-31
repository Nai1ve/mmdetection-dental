# 第1—2步本地实现与首轮结果

日期：2026-10-31。按用户分工，仅实现候选上限/归因与可疑区接口；第3步模型选型、视觉复核由用户进行。没有启动训练、远程任务或模型API。

## 服务器结论

目前不需要重新训练YOLO，可以关机。第1—2步已用下载的结果完成本地首轮验证。当前冻结权重完整pre-NMS池仍未重新导出；工具准备好后只需一次短时推理，不需要训练，也不需要现在占用GPU。

## 代码与完成状态

- tools/dental_review_regions.py：预测驱动的可逆分组、双层二次牙弓拟合假说与不确定性、重叠/低分/水平投影空隙/粗图像轴不对称标记。分组是score排序的星形组织，不作传递闭包、不等同牙实例。
- tools/run_dental_stage12.py：按白名单只传图像id/宽高与预测，生成review_interface、独立evaluation、图像映射及原图裁剪请求。GT只在evaluate中使用，case文件不含GT编号或异常标签。
- tools/summarize_dental_raw_audit.py：整理历史同源审计，明确只针对历史权重与多生牙诊断记录，不当成当前池的完整一对一上限。
- tools/export_dental_raw_pool.py：待上机的完整解码NPZ、raw索引与同源NMS重放，conf0.01/0.001/0.0001，NMS0.6/0.85；记录NMS与max_det移除索引，并与原生0.001/0.6逐图核对。
- tools/evaluate_dental_raw_pool.py：重新导出后测pre/post池的一对一框覆盖，并区分几何不足、低分、NMS或预算损失、匹配竞争。几何上限不能称有效检出。
- tools/export_dental_review_crops.py：按中性ID导出原分辨率证据图，无模型调用。已对2张真实图、8张局部裁剪完成烟雾验证。
- tools/check_dental_stage12.py：CPU测试通过，覆盖无GT生成、成员完整性、空候选、面积并集、随机可复现、一对一匹配与低分归因合成测试。raw导出工具通过导入/CLI检查及语法检查，新版真实GPU逐图重放仍待验证。

不新增训练集采样或ROI网络，不改现有检测权重。没有把“未知方向”写成倒置判断；s/d只作投影假说，不能当实际萌出深度。当前空隙触发是水平投影启发式，不宣称已经实现精确沿牙弓序列的间距推理。图像轴对称也不是同名FDI对称。

## 数据来源不能混淆

1. val：完整164张，当前冻结baseline_s42全图候选，取自roi_pilot/single_val_.25/candidates.json的original_boxes/original_scores；**没有使用该文件的ROI新增框**。全牙R@0.5=6896/6915，多生牙7/10；49.18框/图，P95=64，最大82。
2. internal_legacy：完整328张，20261029旧版单类YOLO导出的detections.json，不是当前baseline_s42权重。全牙R@0.5=13646/13709，多生牙14/20；43.47框/图，P95=50，最大61。用于可疑区接口的探索性检查，不能与val合成当前模型的统一性能表。
3. 当前只读取post-NMS候选。无法从这些文件确定每个池外目标究竟因低分、NMS还是无几何响应而漏检，失败原因明确标为unknown_without_pre_nms。

两个子集在IoU0.85分组下均没有折叠候选，成员与代表召回相同，也没有观察到多GT碰撞组。因此**本轮没有实现35–45主簇压缩收益**，也不以牙数先验强制压缩。重复候选的可逆处理能力仅由合成测试验证，真实密集pre-NMS池还待测试。

这与输入本来已经经过硬NMS的性质一致，不能反过来推断pre-NMS池没有重复候选。

## 可疑区协议

- 对照risk、固定grid、可复现random，以及候选中心最远点分散覆盖spread。
- 每区域宽0.22W、高0.30H；K=0/1/2/4/8。相同K具有相同归一化累计查看面积，允许重叠但另报面积并集。
- risk输入只有预测重叠/低分/间距等，无GT或FDI；spread只使用score≥0.25候选的中心，不用异常标签。
- GT覆盖要求：至少90%的牙齿GT框面积位于同一个ROI，不只看中心点。
- 报告全部多生牙覆盖，以及相对于当次全图一对一R@0.5/R@0.75未命中的多生牙覆盖。**ROI覆盖不是检测恢复，也不是临床确诊**。
- 启发式阈值与区域大小属于原型设置，尚未系统调优。添加spread后未反调risk规则；内部结果已查看，全部属于探索性开发记录。

## K=4结果

| 来源 | 策略 | 全部多生牙ROI覆盖 | 原R@0.5漏检多生牙ROI覆盖 | 面积并集/图 | 累计查看面积/图 |
| --- | --- | --- | --- | ---: | ---: |
| val当前模型 | risk | 9/10 | 3/3 | 18.05% | 26.40% |
| val当前模型 | spread | 9/10 | 2/3 | 25.14% | 26.40% |
| val当前模型 | random | 3/10 | 1/3 | 23.12% | 26.40% |
| val当前模型 | grid | 1/10 | 1/3 | 26.40% | 26.40% |
| internal旧模型 | risk | 19/20 | 5/6 | 19.49% | 26.40% |
| internal旧模型 | spread | 19/20 | 6/6 | 25.11% | 26.40% |
| internal旧模型 | random | 12/20 | 5/6 | 23.03% | 26.40% |
| internal旧模型 | grid | 2/20 | 0/6 | 26.40% | 26.40% |

## 预算曲线的关键信息

| 来源 | 策略 | K=2（全部/漏检） | K=4（全部/漏检） | K=8（全部/漏检） |
| --- | --- | --- | --- | --- |
| val | risk | 7/10、2/3 | 9/10、3/3 | 9/10、3/3 |
| val | spread | 9/10、2/3 | 9/10、2/3 | 10/10、3/3 |
| internal旧模型 | risk | 15/20、3/6 | 19/20、5/6 | 19/20、5/6 |
| internal旧模型 | spread | 19/20、6/6 | 19/20、6/6 | 20/20、6/6 |

复杂risk未稳定优于简单spread，因此**不将可疑区规则宣称为已验证算法创新**。建议第3步把spread作为低成本强对照，risk作为另一路实验输入；不能只展示较弱random/grid对照。

这些结果提示“少量区域能让人/视觉模型重新看到漏检牙”有测试空间，但此前同YOLO裁剪重检没有收益，不证明工具已能补检。接下来由用户的第3步判断VLM能否辨认、准确定位并安全修改。

## 给第3步的接口

本地目录records/remote_runs/20261031_stage12：

- val/review_interface.json、internal_legacy/review_interface.json：候选ID、原图与归一化框、分组成员、几何假说、冲突边、排序ROI；不含GT。
- image_index.json：中性image_ref到本地原图相对路径的工具侧映射，不直接作为Prompt。
- crop_requests.json：每种策略最多8个区域，原图像素xyxy坐标与evidence_id。预算K只取rank≤K。
- evaluation.json：含GT分析，仅供离线评价，**禁止喂给模型**。
- handoff_smoke/assets.json及PNG：2图8局部的真实裁剪烟雾产物，不代表全量模型评估。
- e0_historical.json：历史审计证据；有2颗可用粗框分数低于0.001，1颗在NMS0.6/0.75存在跨牙竞争。该文件不能当当前checkpoint统计。

原图根目录：/Users/liupeize/PycharmProjects/Dental/dataset_20261029/private_only。原始坐标、图像与复核产物仅本地，不纳入Git。导出文件使用中性命名不等于影像像素已通过隐私审核，外部模型发送前仍需核查。

## 使用示例

在仓库根目录，使用有numpy/scipy的Python：

```bash
python tools/run_dental_stage12.py --candidates INPUT_CANDIDATES.json --annotations DATA/fdi53/val.json --output NEW_OUTPUT
python tools/export_dental_review_crops.py --interface-root NEW_OUTPUT --image-root DATA --output NEW_CROPS --policy spread --budget 4
```

裁剪工具另需Pillow。候选可用original_boxes/original_scores格式，也可用旧COCO检测JSON；注释文件只作尺寸白名单和独立评价，不参与标记生成。

需要补齐第1步完整当前权重审计时，在现有Ultralytics8.4.191环境执行：

```bash
python tools/export_dental_raw_pool.py --data-root DATA --checkpoint BEST.pt --split val --output NEW_RAW
python tools/evaluate_dental_raw_pool.py --raw-root NEW_RAW --annotations DATA/fdi53/val.json --output NEW_RAW/e0_evaluation.json
```

先用--limit-images 2跑导出烟雾；完整评估拒绝partial_dataset，正式导出需另一个新目录。导出不训练，不修改权重。Soft-NMS尚未加入本次重放矩阵，不把未测设置写成有效结果。

## 下一步边界

本地第1步已完成现存池上限、漏检账本和历史原因整理，完整当前pre-NMS上限仍待短时推理；第2步完成原型、全val/旧internal探索性预算比较与可交接接口。第3步没有调用或选定任何LLM/VLM。
