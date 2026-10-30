# 牙齿结构实验服务部署记录

- 日期：2026-10-30，北京时间09:39开始核心组。
- 代码：`437f28b`，本地GitHub推送成功；服务器通过离线Git bundle同步，避免外网传输影响部署。
- 远端原有两处保护脚本改动已用限定文件stash保留，未清理权重或其他未跟踪文件。
- 环境：RTX4090D 24GB，Torch2.3.0+cu121，Ultralytics8.4.191；数据盘剩余约36GB。
- GPU烟雾测试：`direction_quality`，YOLO11m、960、batch8，1轮约4张训练样本及164张验证图；成功，loss有限，AMP和权重保存加载通过。该结果仅验证软件可运行，不代表医学性能。
- 测试日志：`/root/autodl-tmp/structure_gpu_smoke_20261030.log`。
- 正式总控：`/root/autodl-tmp/experiments/yolo_structure_core_20261030_0940`。
- 控制器真实PID：2405，已核对PPID=1、SID=2405，不依赖SSH会话。
- 核心组：baseline、control、direction、direction_balance，分别seed42/43/44，共12组，24轮、960、batch8。
- 数据：`/root/autodl-tmp/datasets/private_only`；转换：`/root/autodl-tmp/datasets/private_only_yolo_tooth1_v1`。
- 权重/评估：`/root/autodl-tmp/work_dirs/yolo_structure_core_20261030_0940_<配置>_s<种子>`。
- 每组训练后分别完成NMS0.6/0.85的val/internal_test/external_test；保留best/last，不保存每轮权重。
- 初次核对baseline_s42在第1轮，GPU利用率82%，占用9641MiB；batch保持8，避免改变已锁定的对照配方。
- 独立封存服务：`tools/watch_yolo_structure.py`，不改运行中的总控脚本。全部成功后校验12组24轮及72项子集评估，打包统计并生成SHA256；不包含模型权重或PKL。
- 新批次不默认执行关机，不自动启动次级组。监控发现失败、完成或需人工处理时通知，正常训练保持静默。
