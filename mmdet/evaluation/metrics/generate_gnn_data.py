from torch import Tensor

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
import torch
from torch_geometric.data import Data

from typing import Dict, List, Optional, Sequence, Union

SCORE_THRESHOLD = 0.3  # 置信度阈值

@METRICS.register_module()
class GenerateGNNData(CocoMetric):



    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        gnn_list = []
        gnn_with_visual_list = []

        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['visual_features'] = pred['visual_features'].cpu().numpy()
            result['all_class_probs'] = pred['all_class_probs'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

            # 生成图对象并保存
            gnn_data,gnn_data_with_visual = self.__generate_gnn_data_and_save(result,gt)
            gnn_list.append(gnn_data)
            gnn_with_visual_list.append(gnn_data_with_visual)

        torch.save(gnn_list,'./gnn_data/gnn_train_data.pt')
        torch.save(gnn_with_visual_list,'./gnn_data/gnn_train_data_with_visual.pt')


    def __generate_gnn_data_and_save(self,result :dict,gt : dict)-> (Data,Data):
        """
        根据数据创建两种
        Args:
            result:
            gt:

        Returns:

        """
        bboxes = result['bboxes']
        # 1.获取坐标

        # 2.计算宽高比
        w = bboxes[2] - bboxes[0]
        h = bboxes[3] - bboxes[1]
        # 3.获取语义特征
        all_class_probs = result['all_class_probs']

        # 4.置信度
        scores = result['scores']


        # 设置中心点

        # 找到邻居节点


        y = self.__ground_truth_y(result,gt)

        # 真实标签，语义特征的标签

        data = Data()
        data_with_visual = Data()
        return data,data_with_visual


    def __ground_truth_y(self,pred_instance:dict,gt:dict)->Tensor:
        """

        Args:
            result:
            gt:

        Returns:

        """
        # 获取预测结果
        score_mask = pred_instance['scores'] >= SCORE_THRESHOLD
        pred_bboxes = pred_instance['bboxes'][score_mask]
        pred_labels_model_idx = pred_instance['labels'][score_mask]
        pred_scores = pred_instance['scores'][score_mask]

        # 获取真实标注



        return None