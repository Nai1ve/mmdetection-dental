import os

from mmengine import MMLogger
from torch import Tensor

from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import to_undirected

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
import torch
from torch_geometric.data import Data

from typing import Sequence, List, Optional
import numpy as np
import pickle

SCORE_THRESHOLD = 0.3  # 置信度阈值
IOU_THRESHOLD = 0.5 # IOU阈值
BACKGROUND = 48
epsilon = 1e-6  # 防止除零
K_NEIGHBORS = 9
info_list = []
TYPE = 'train'
@METRICS.register_module()
class GenerateGNNData(CocoMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gnn_list:List[Optional[Data]] = []
        self.gnn_with_visual_list:List[Optional[Data]] = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        super().process(data_batch,data_samples)

        for data in data_samples:
            gnn_data,gnn_data_with_visual = self.__generate_gnn_data_and_save(data)

            if gnn_data is not None:
                self.gnn_list.append(gnn_data)
            if gnn_data_with_visual is not None:
                self.gnn_with_visual_list.append(gnn_data_with_visual)

    def compute_metrics(self,results:List) -> dict:
        eval_results = super().compute_metrics(results)

        MMLogger.info(MMLogger.get_current_instance(),"所有批次处理完成，开始保存GNN数据")
        MMLogger.info(MMLogger.get_current_instance(),f"总共收集到{len(self.gnn_list)}个GNN图数据")
        MMLogger.info(MMLogger.get_current_instance(),f"总共收集到{len(self.gnn_with_visual_list)}个带视觉特征的GNN图数据")

        save_dir = 'gnn_data'
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.gnn_list, os.path.join(save_dir, f'gnn_{TYPE}_data_{SCORE_THRESHOLD}.pt'))
        torch.save(self.gnn_with_visual_list, os.path.join(save_dir, f'gnn_{TYPE}_data_with_visual_{SCORE_THRESHOLD}.pt'))

        with open(os.path.join(save_dir, f'info_{SCORE_THRESHOLD}.pkl'), 'wb') as f:
            pickle.dump(info_list, f)

        return eval_results

    def __generate_gnn_data_and_save(self,data :dict)-> (Data,Data):
        """
        根据数据创建两种
        Args:
            data:

        Returns:

        """
        pred_instances = data['pred_instances']
        ori_shape = data['ori_shape'] # h,w
        img_h = ori_shape[0]
        img_w = ori_shape[1]
        if 'scores' not in pred_instances or len(pred_instances['scores']) == 0:
            return None, None

        score_mask = pred_instances['scores'] >= SCORE_THRESHOLD
        if not torch.any(score_mask):
            return None, None # 如果过滤后没有框了，也返回空值

        # --- 获取所有需要的数据 ---
        bboxes = pred_instances['bboxes'][score_mask]
        all_class_probs = pred_instances['all_class_probs'][score_mask]
        scores = pred_instances['scores'][score_mask]
        visual_features = pred_instances['visual_features'][score_mask]

        pred_num = bboxes.shape[0]

        # --- 使用Python列表进行高效累积 ---
        feature_list_v1 = []
        feature_list_v2 = []
        pos_list = []

        for i in range(pred_num):
            bbox = bboxes[i]  # 格式应为 [xmin, ymin, xmax, ymax]

            # 1. 提取几何与形状特征
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x_center = bbox[0] + w / 2
            y_center = bbox[1] + h / 2
            aspect_ratio = w / (h + 1e-6)  # 防止除零

            # 2. 准备各类特征
            geom_shape_features = torch.tensor([x_center / img_w, y_center / img_h])
            semantic_features = all_class_probs[i]  # 49维
            confidence_score = scores[i].unsqueeze(0)  # 1维
            visual = visual_features[i]  # 假设是1024或2048维

            # 3. 拼接成最终的特征向量并添加到列表中
            # V1.0 特征
            features_v1 = torch.cat([
                geom_shape_features,
                semantic_features,
                confidence_score
            ])
            feature_list_v1.append(features_v1)

            # V2.0 特征
            features_v2 = torch.cat([
                geom_shape_features,
                semantic_features,
                confidence_score,
                visual.flatten()  # 直接拼接视觉特征
            ])
            feature_list_v2.append(features_v2)

            # 累积位置信息
            pos_list.append(torch.tensor([x_center / img_w, y_center/img_h]))

        # --- [修正点] 在循环外一次性将列表转换为张量 ---
        x_v1 = torch.stack(feature_list_v1)
        x_v2 = torch.stack(feature_list_v2)
        pos = torch.stack(pos_list)

        # 找到邻居节点
        temp_data = Data(pos=pos)
        knn_transform = KNNGraph(k=K_NEIGHBORS)
        graph_data = knn_transform(temp_data)
        edge_index = to_undirected(graph_data.edge_index)

        y = self.__ground_truth_y(data)

        # 安全检查: 确保标签数量和节点数量一致
        if y is None or len(y) != pred_num:
            # 如果y的生成逻辑有问题或数量不匹配，则此样本无效
            # print(f"警告: 标签数量 ({len(y) if y is not None else 'None'}) 与预测框数量 ({pred_num}) 不匹配。跳过此样本。")
            return None, None

        # --- [新增逻辑] 6. 组装并返回最终的Data对象 ---
        data_v1 = Data(x=x_v1, edge_index=edge_index, pos=pos, y=y)
        data_v2 = Data(x=x_v2, edge_index=edge_index, pos=pos, y=y)

        return data_v1,data_v2


    def __ground_truth_y(self,data:dict)->Tensor:
        """

        Args:
            data:

        Returns:

        """
        label_dict = {}
        # 获取预测结果
        pred_instances = data['pred_instances']
        gt_instances = data['gt_instances']

        score_mask = pred_instances['scores'] >= SCORE_THRESHOLD
        pred_bboxes = pred_instances['bboxes'][score_mask].cpu().numpy()
        pred_scores = pred_instances['scores'][score_mask].cpu().numpy()

        gt_bboxes = gt_instances['bboxes'].cpu().numpy()
        gt_labels = gt_instances['labels'].cpu().numpy()

        num_preds = len(pred_bboxes)
        num_gts = len(gt_bboxes)

        y_vector = np.full(num_preds, -1, dtype=int)
        gt_is_matched = np.zeros(gt_bboxes.shape[0],dtype=bool)

        # 按照置信度对预测框进行排序
        if num_preds > 0:
            sorted_pred_indices = np.argsort(pred_scores)[::-1]
        else:
            sorted_pred_indices = []


        # 处理一个特殊情况：如果有预测，但图中没有任何真实物体
        if num_gts == 0 and num_preds > 0:
            y_vector[:] = BACKGROUND # 所有预测都是 "无中生有"
            return torch.from_numpy(y_vector)

        # 循环处理已排序的预测框
        for p_idx in sorted_pred_indices:
            pred_box = pred_bboxes[p_idx]

            # 计算与所有预测框的IOU，找到分数最高的和对应的max_iou
            ious = np.array([self.__calculate_iou(pred_box,gt_box) for gt_box in gt_bboxes])
            #找到最匹配的iou索引
            best_gt_idx = np.argmax(ious)
            max_iou = ious[best_gt_idx]

            if max_iou >= IOU_THRESHOLD:
                label = gt_labels[best_gt_idx]
                if not gt_is_matched[best_gt_idx]:
                    # 未被占用，打标真实标签
                    y_vector[p_idx] = label
                    gt_is_matched[best_gt_idx] = True
                    label_dict[label] = label_dict.get(label, 0) + 1
                else:
                    # 已被占用，标记背景
                    y_vector[p_idx] = BACKGROUND
                    label_dict[BACKGROUND] = label_dict.get(BACKGROUND, 0) + 1
            else:
                y_vector[p_idx] = BACKGROUND
                label_dict[BACKGROUND] = label_dict.get(BACKGROUND, 0) + 1

        # 更新状态
        y_tensor = torch.from_numpy(y_vector).long()
        info_list.append(label_dict)
        return y_tensor

    def __calculate_iou(self,boxA, boxB):
        """计算两个边界框的交并比 (IoU)"""
        # 确保框是 [x1, y1, x2, y2] 格式
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + epsilon)
        return iou