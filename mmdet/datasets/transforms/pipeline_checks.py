import numpy as np
from mmengine import MMLogger

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DebugCheckBBoxValidity:
    """
    一个用于调试的自定义转换器.
    它检查 'gt_bboxes' 中是否存在无效框 (w <= 0 或 h <= 0).
    这个转换器应该放在 pipeline 的末尾, PackDetInputs 之前。
    """

    def __call__(self, results: dict) -> dict:
        logger = MMLogger.get_current_instance()
        logger.info("进入数据筛查.....")
        logger.info(results)
        # results 字典包含了 pipeline 中每一步的结果
        gt_bboxes = results.get('gt_bboxes', None)

        # 检查 'gt_bboxes' 是否存在且不为空
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            # 在 MMDetection 的流水线中, gt_bboxes 此时通常是 numpy 数组
            # 格式为 [x1, y1, x2, y2]
            widths = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            heights = gt_bboxes[:, 3] - gt_bboxes[:, 1]

            # 查找是否存在 w <= 0 或 h <= 0 的情况
            if np.any(widths <= 0) or np.any(heights <= 0):
                logger.info(f"results:{results}存在问题")
                img_path = results.get('img_path', 'Unknown Image')
                bad_boxes = gt_bboxes[(widths <= 0) | (heights <= 0)]

                # 找到问题！立即抛出异常并打印所有信息
                # 这会让训练程序崩溃, 并告诉你是哪张图片和哪个框出了问题
                raise ValueError(
                    f"\n\n[!!!] 调试检查失败：在数据增强后发现无效边界框！\n"
                    f"      图片路径: {img_path}\n"
                    f"      问题框 (w<=0 或 h<=0): \n{bad_boxes}\n"
                    f"      所有框: \n{gt_bboxes}\n"
                    f"      请检查您的数据增强流水线(如Resize, RandomCrop)或原始标注文件。\n"
                )

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'