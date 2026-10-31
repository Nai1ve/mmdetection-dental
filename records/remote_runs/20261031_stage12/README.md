# 第1—2步本地结果

完整val164与旧权重internal328的接口和评估已生成。原始候选只取post-NMS全图框，不包含ROI头新增框。两个来源不同权重，不拼成统一模型性能。

目录val、internal_legacy均含review_interface.json、image_index.json、crop_requests.json和evaluation.json。evaluation含真值，仅离线评价，不能作为第3步模型输入。handoff_smoke为2图8局部裁剪验证。

GT覆盖定义为一个ROI内包含至少90%牙齿框面积；不是补检率。risk在K4覆盖val漏检extra3/3、旧internal5/6；简单spread覆盖2/3与6/6，因此没有证明复杂标记稳定更好。

e0_historical.json仅历史权重；当前完整raw池尚待重新导出，不宣称完成当前pre-NMS上限。原始病例坐标与图像均不提交Git。完整报告见docs/stage12_local_results_20261031.md。
