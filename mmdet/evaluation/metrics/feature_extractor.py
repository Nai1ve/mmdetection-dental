import os
from mmengine import MMLogger
from torch import Tensor
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
import torch
from typing import Sequence, List, Optional
import numpy as np
import pickle


def to_cpu_numpy(tensor_or_array):
    if isinstance(tensor_or_array, Tensor):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)


@METRICS.register_module()
class ExtractRawFeatures(CocoMetric):
    """
    负责提取所有原始特征并保存到一个 .pkl 文件中。
    """

    def __init__(self,
                 data_type: str = 'test',
                 save_dir: str = 'gnn_data/raw_features',
                 **kwargs):
        super().__init__(**kwargs)
        self.data_type = data_type
        self.save_dir = save_dir

        self.raw_feature_list: List[dict] = []

        self.logger = MMLogger.get_current_instance()
        self.logger.info(f"ExtractRawFeatures initialized for data_type: {self.data_type}")

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        处理一个批次, 提取所有原始数据并保存。
        """
        # (我们仍然调用 super().process 来让 CocoMetric 准备 GT，尽管我们不用它)
        super().process(data_batch, data_samples)

        for data_sample in data_samples:
            pred_instances = data_sample.get('pred_instances', {})
            gt_instances = data_sample.get('gt_instances', {})

            # 我们不在这里过滤 (score_threshold), 我们保存所有东西
            # 让下游脚本去灵活地过滤
            if 'scores' not in pred_instances or len(pred_instances['scores']) == 0:
                continue

            # 收集所有原始数据
            raw_data = {
                'img_id': data_sample['img_id'],
                'img_path': data_sample['img_path'],
                'ori_shape': data_sample['ori_shape'],

                # 预测数据 (全部转为Numpy, 方便Pickle)
                'pred_bboxes': to_cpu_numpy(pred_instances.get('bboxes')),
                'pred_scores': to_cpu_numpy(pred_instances.get('scores')),
                'pred_labels': to_cpu_numpy(pred_instances.get('labels')),
                'x_cls': to_cpu_numpy(pred_instances.get('x_cls')),

                # 真实数据
                'gt_bboxes': to_cpu_numpy(gt_instances.get('bboxes')),  # [x, y, w, h]
                'gt_labels': to_cpu_numpy(gt_instances.get('labels'))
            }

            self.raw_feature_list.append(raw_data)

    def compute_metrics(self, results: List) -> dict:
        """
        所有数据处理完后，将“篮子”保存到文件。
        """
        # 我们仍然可以计算mAP，以确认Faster R-CNN的性能
        eval_results = super().compute_metrics(results)
        self.logger.info("MMDetection 评测完成。")
        self.logger.info("开始保存原始特征...")

        os.makedirs(self.save_dir, exist_ok=True)

        save_filename = f'raw_features_{self.data_type}.pkl'
        save_path = os.path.join(self.save_dir, save_filename)

        try:
            with open(save_path, 'wb') as f:
                pickle.dump(self.raw_feature_list, f)
            self.logger.info(f"成功: {len(self.raw_feature_list)} 张图片的原始特征已保存至: {save_path}")
        except Exception as e:
            self.logger.error(f"!!! 保存失败: {e}")

        return eval_results