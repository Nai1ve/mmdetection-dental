# ROI 采样实验部署与真实监督审计

## 训练缓存审计

2026-10-31恢复连接后读取完整train1148缓存，partial_dataset=false，冻结检测器SHA为2f0f34508a1f3874a629118b16f483c21d2b766edd5290c7bd8a16240c69012e。

非overflow有效ROI共51438，overflow另2个。完整GT/ROI：0颗1123个、1颗48007个、2颗2161个、3颗141个、4颗6个。多颗完整GT的ROI共2308（约4.49%）。

互斥分层：困难624个，涉及293张；非困难多实例2188个，涉及887张；普通48626个，涉及1148张。困难定义为至少一颗完整GT未被原始分数≥0.05框以IoU≥0.75覆盖，不等同于多生牙。

完整GT出现次数52776，其中未覆盖出现632次（约1.20%）。出现次数会跨ROI重复，不代表独立牙齿数、一对一召回或临床病例数。

一轮seed42模拟，均抽13776 ROI：

| 策略 | 困难 | 非困难多实例 | 普通 |
| --- | ---: | ---: | ---: |
| uniform | 177 | 575 | 13024 |
| stratified | 571 | 2011 | 11194 |

该分布支持验证困难曝光不足假设，但不能证明重复预测的原因。模拟与实际随机打乱训练不保证逐项相同，实际分布记录于sampling_audit.jsonl。

## 部署

- 训练代码6f5ba79，远端Git bundle快进，原有未跟踪文件保留。
- 远端完整CPU测试通过；两张真实训练图、stratified四槽GPU烟雾2轮loss 2.5839→2.0190，保存权重sampling=stratified且epoch2。烟雾两图没有困难ROI，覆盖困难层的选择由单元测试及全train模拟验证，不作为真实困难学习性能证据。
- 运行目录/root/autodl-tmp/experiments/roi_sampling_matrix_20261031。
- 日志/root/autodl-tmp/roi_sampling_matrix_20261031_launcher.log。
- nohup setsid启动；真实控制器PID1903，PPID1、SID1903；包装1901正常退出。
- 启动时先train/val监督审计，再顺序single_uniform、single_stratified、multi_uniform、multi_stratified。各seed42、6轮，每组两阈值完整val及预算审计，末尾槽位匹配诊断。
- 复用既有7x7缓存，无新增大特征文件；数据盘起始余8.6GB。各组只保留last.pt。
- 已建立10分钟线程监控，正常静默，异常或完成通知。不自动关机、不追加训练。

未修改数据或划分；本轮不运行内部/外部测试，不宣称泛化。完整计划见docs/roi_sampling_plan_20261031.md。
