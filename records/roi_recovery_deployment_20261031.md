# ROI 补漏矩阵部署记录

- 日期：2026-10-31，Asia/Shanghai。
- 远端训练代码提交：43c1326。通过 Git bundle 快进同步，保留远端未跟踪文件；活动脚本启动后冻结。
- GPU：4090D 24GB；启动前数据盘剩余30GB。预计两套新缓存合计约21GB，监控剩余空间。
- 真实 GPU 烟雾：roi_recovery_smoke_20261031，train/val各2张、14x14、conditioned+quality，2轮 loss 6.1431→3.7457。权重540423参数，保存加载和完整烟雾流程通过；不作为性能结论。
- 正式目录：/root/autodl-tmp/experiments/roi_recovery_matrix_20261031_1550。
- 日志：/root/autodl-tmp/roi_recovery_matrix_20261031_1550_launcher.log。
- 真实控制器PID1877，PPID1、SID1877；包装PID1875正常退出。使用nohup setsid，关闭启动SSH会话后另行复核。
- 缓存先后：7/train、7/val、14/train、14/val。五组control7、resolution14、conditioned7、quality7、combined14；各6轮、seed42、四槽，每组两个阈值完整val与预算审计。
- 冻结检测器：/root/autodl-tmp/work_dirs/yolo_structure_core_20261030_0940_baseline_s42/weights/best.pt。
- 数据：/root/autodl-tmp/datasets/private_only。GT、划分保持不变。
- 每组只保留last.pt。完成后下载统计与小头权重，不下载特征缓存。
- 已建立10分钟线程监控；正常进展静默，失败/重大异常/完成通知。不自动关机、不追加实验。

本轮只做val先导比较，未启动内部/外部测试。方法与评价边界见docs/roi_recovery_next_plan_20261031.md。
