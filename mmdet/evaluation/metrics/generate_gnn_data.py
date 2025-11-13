import os
from mmengine import MMLogger
from torch import Tensor
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import to_undirected
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from typing import Sequence, List, Optional
import numpy as np
import math
from collections import defaultdict

BACKGROUND = 48
epsilon = 1e-6


# [辅助函数] 确保数据在CPU上并转为Numpy
def to_cpu_numpy(tensor_or_array):
    if isinstance(tensor_or_array, Tensor):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)


# [辅助函数] 计算Numpy IoU
def calculate_iou_np(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + epsilon)
    return iou


@METRICS.register_module()
class GenerateGNNData(CocoMetric):

    def __init__(self,
                 score_threshold: float = 0.05,
                 iou_threshold: float = 0.5,
                 k_neighbors: int = 9,
                 data_type: str = 'test',
                 gnn_save_dir: str = 'gnn_data',
                 **kwargs):
        super().__init__(**kwargs)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.k_neighbors = k_neighbors
        self.data_type = data_type
        self.gnn_save_dir = gnn_save_dir

        self.gnn_with_visual_embedding_list: List[Optional[Data]] = []
        self.info_list = []  # 用于__ground_truth_y的兼容性，但现在可以移除
        self.error_stats = defaultdict(int)  # 用于统计错误类型

        logger = MMLogger.get_current_instance()
        logger.info(f"GenerateGNNData (Simplified) initialized with:")
        logger.info(f"  score_threshold: {self.score_threshold}")
        logger.info(f"  iou_threshold: {self.iou_threshold}")
        logger.info(f"  k_neighbors: {self.k_neighbors}")

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        super().process(data_batch, data_samples)
        for data_sample in data_samples:
            cpu_data_sample = data_sample.to_dict()
            gnn_data = self.__generate_gnn_data_and_save(cpu_data_sample)
            if gnn_data is not None:
                self.gnn_with_visual_embedding_list.append(gnn_data)

    def compute_metrics(self, results: List) -> dict:
        eval_results = super().compute_metrics(results)
        logger = MMLogger.get_current_instance()

        logger.info("所有批次处理完成，开始保存GNN数据")
        logger.info(f"总共收集到{len(self.gnn_with_visual_embedding_list)}个GNN图数据")

        # 打印错误统计
        logger.info("--- GNN 训练数据错误类型统计 (来自 classify_errors_refined) ---")
        total_nodes = sum(self.error_stats.values())
        logger.info(f"总节点数 (Nodes): {total_nodes}")
        for key, count in self.error_stats.items():
            logger.info(f"  - {key}: {count} ({(count / total_nodes):.2%})")
        # -------------------------------------------------------------

        os.makedirs(self.gnn_save_dir, exist_ok=True)

        base_filename = (
            f'gnn_{self.data_type}_data_with_visual_embedding'
            f'_score{self.score_threshold:.2f}'
            f'_iou{self.iou_threshold:.2f}'
            f'_k{self.k_neighbors}'
        )
        save_path = os.path.join(self.gnn_save_dir, f'{base_filename}.pt')
        torch.save(self.gnn_with_visual_embedding_list, save_path)
        logger.info(f"GNN 数据已保存至: {save_path}")

        # ... (info_list 的保存逻辑可以移除或保留) ...
        return eval_results

    def __generate_gnn_data_and_save(self, data: dict) -> Optional[Data]:
        pred_instances = data['pred_instances']
        gt_instances = data['gt_instances']
        ori_shape = data['ori_shape']
        img_id = data['img_id']
        img_path = data['img_path']
        img_h, img_w = ori_shape[0], ori_shape[1]

        if 'scores' not in pred_instances or len(pred_instances['scores']) == 0:
            return None

        # --- 1. 准备预测数据 (过滤 + 提取 + 统一移至CPU) ---
        all_scores_np = to_cpu_numpy(pred_instances['scores'])
        score_mask = all_scores_np >= self.score_threshold
        if not np.any(score_mask):
            return None

        final_bboxes_unnormalized_np = to_cpu_numpy(pred_instances['bboxes'])[score_mask]
        final_scores_np = all_scores_np[score_mask]
        final_raw_labels_np = to_cpu_numpy(pred_instances['labels'])[score_mask]
        all_class_probs_np = to_cpu_numpy(pred_instances['all_class_probs'])[score_mask]
        x_cls_np = to_cpu_numpy(pred_instances['x_cls'])[score_mask]

        num_preds = final_bboxes_unnormalized_np.shape[0]

        # --- 2. 准备 GT 数据 (Numpy) ---
        gt_bboxes_coco = to_cpu_numpy(gt_instances['bboxes'])
        gt_labels_coco = to_cpu_numpy(gt_instances['labels'])
        gt_bboxes_xyxy = np.array([[x, y, x + w, y + h] for x, y, w, h in gt_bboxes_coco]) if len(
            gt_bboxes_coco) > 0 else np.empty((0, 4))
        num_gts = gt_bboxes_xyxy.shape[0]
        ground_truths_np = np.hstack((gt_bboxes_xyxy, gt_labels_coco.reshape(-1, 1))) if num_gts > 0 else np.empty(
            (0, 5))

        # --- 3. [新逻辑] 使用 classify_errors_refined 生成 y 标签 ---
        predictions_for_classify = np.hstack((
            final_bboxes_unnormalized_np,
            final_raw_labels_np.reshape(-1, 1),
            final_scores_np.reshape(-1, 1)
        ))

        # (调用类方法)
        pred_assignment, pred_match_info, sort_inds = self.classify_errors_refined(
            predictions_for_classify, ground_truths_np, self.iou_threshold
        )

        # (调用辅助函数)
        y_expected_sorted = self.get_expected_y_vector(pred_assignment, pred_match_info)

        # (恢复原始顺序)
        y_vector_np = np.full_like(y_expected_sorted, BACKGROUND)
        reverse_sort_inds = np.argsort(sort_inds)
        y_vector_np = y_expected_sorted[reverse_sort_inds]

        y_tensor = torch.from_numpy(y_vector_np).long()
        # --- [y 标签生成完毕] ---

        # --- [新逻辑] 统计错误类型 (用于日志) ---
        for category in pred_assignment:
            self.error_stats[category] += 1
        # --- [统计完毕] ---

        # --- 4. 构建图和特征 (在 CPU 上) ---
        final_bboxes_tensor = torch.from_numpy(final_bboxes_unnormalized_np)
        final_scores_tensor = torch.from_numpy(final_scores_np)
        all_class_probs_tensor = torch.from_numpy(all_class_probs_np)
        x_cls_tensor = torch.from_numpy(x_cls_np)
        final_raw_labels_tensor = torch.from_numpy(final_raw_labels_np).long()

        feature_list_v3 = []
        pos_list = []

        for i in range(num_preds):
            bbox = final_bboxes_tensor[i]
            w = bbox[2] - bbox[0];
            h = bbox[3] - bbox[1]
            x_center = bbox[0] + w / 2;
            y_center = bbox[1] + h / 2
            aspect_ratio = w / (h + 1e-6)

            geom_shape_features = torch.tensor(
                [(x_center / img_w) * 2 - 1, (y_center / img_h) * 2 - 1, (w / img_w) * 2 - 1, (h / img_h) * 2 - 1])
            area_feature = torch.tensor([(w * h) / (img_w * img_h)]) * 2 - 1
            aspect_ratio_features = torch.tensor([math.log(aspect_ratio.item())])
            confidence_score = final_scores_tensor[i].unsqueeze(0) * 2 - 1
            normalized_x_cls = F.normalize(x_cls_tensor[i], p=2, dim=-1)

            features_v3 = torch.cat([
                geom_shape_features,  # 4D
                area_feature,  # 1D
                aspect_ratio_features,  # 1D
                normalized_x_cls,  # 1024D
                confidence_score  # 1D
            ])
            feature_list_v3.append(features_v3)
            pos_list.append(torch.tensor([x_center / img_w, y_center / img_h]))

        x_v3 = torch.stack(feature_list_v3)
        pos = torch.stack(pos_list)

        temp_data = Data(pos=pos)
        knn_transform = KNNGraph(k=self.k_neighbors)
        graph_data = knn_transform(temp_data)
        edge_index = to_undirected(graph_data.edge_index)

        # 添加关系特征 (在 Numpy/CPU 上计算)
        max_iou_list, avg_iou_list, avg_dist_list = [], [], []
        for i in range(num_preds):
            neighbors_mask = (edge_index[0] == i)
            neighbors_indices = edge_index[1][neighbors_mask]

            if len(neighbors_indices) == 0:
                max_iou, avg_iou, avg_dist = 0.0, 0.0, 0.0
            else:
                current_box_np = final_bboxes_unnormalized_np[i]
                neighbors_boxes_np = final_bboxes_unnormalized_np[neighbors_indices]
                neighbor_ious = np.array([self.__calculate_iou(current_box_np, nb) for nb in neighbors_boxes_np])
                max_iou = np.max(neighbor_ious) if len(neighbor_ious) > 0 else 0.0
                avg_iou = np.mean(neighbor_ious) if len(neighbor_ious) > 0 else 0.0
                current_pos_np = pos[i].numpy()
                neighbors_pos_np = pos[neighbors_indices].numpy()
                distances = np.linalg.norm(neighbors_pos_np - current_pos_np, axis=1)
                avg_dist = np.mean(distances) if len(distances) > 0 else 0.0

            max_iou_list.append(torch.tensor([max_iou]))
            avg_iou_list.append(torch.tensor([avg_iou]))
            avg_dist_list.append(torch.tensor([(avg_dist * 2) - 1]))

        relational_features = torch.cat([
            torch.stack(max_iou_list),
            torch.stack(avg_iou_list),
            torch.stack(avg_dist_list)
        ], dim=1)

        x_v3_final = torch.cat([x_v3, relational_features], dim=1)

        common_attrs = {
            'edge_index': edge_index, 'pos': pos, 'y': y_tensor, 'img_id': img_id,
            'img_path': img_path, 'ori_shape': torch.tensor(ori_shape),
            'pred_bboxes_raw': final_bboxes_tensor,
            'pred_scores_raw': final_scores_tensor,
            'pred_labels_raw': final_raw_labels_tensor
        }

        data_v3 = Data(x=x_v3_final, **common_attrs)

        # 我们只返回一种数据
        return None, data_v3



    def get_expected_y_vector(self, pred_assignment, pred_match_info):
        """
        根据 classify_errors_refined 的结果生成“期望的”y标签向量 (按分数排序)。
        """
        num_preds = len(pred_assignment)
        y_expected_sorted = np.full(num_preds, -1, dtype=int)

        for i in range(num_preds):
            category = pred_assignment[i]

            if category == 'TP' or category == 'FP_C':
                match_info = pred_match_info[i]
                if match_info and 'gt_label' in match_info:
                    y_expected_sorted[i] = match_info['gt_label']  # 目标是GT的真实标签
                else:
                    y_expected_sorted[i] = BACKGROUND  # 匹配失败的FP-C(罕见)，也抑制
            elif category == 'DUPS' or category == 'FP_H' or category == 'SPAN':
                y_expected_sorted[i] = BACKGROUND  # 所有其他FP都抑制
            else:
                y_expected_sorted[i] = BACKGROUND  # 兜底

        return y_expected_sorted


    def __calculate_iou(self, boxA, boxB):
        # (保持不变, 使用 Numpy)
        xA = max(boxA[0], boxB[0]);
        yA = max(boxA[1], boxB[1]);
        xB = min(boxA[2], boxB[2]);
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + epsilon)

    def __calculate_iou_vectorized(self, box_a: Tensor, boxes_b: Tensor) -> Tensor:
        # (保持不变, PyTorch)
        xA = torch.max(box_a[:, 0], boxes_b[:, 0]);
        yA = torch.max(box_a[:, 1], boxes_b[:, 1])
        xB = torch.min(box_a[:, 2], boxes_b[:, 2]);
        yB = torch.min(box_a[:, 3], boxes_b[:, 3])
        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
        boxAArea = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        boxBArea = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        unionArea = boxAArea + boxBArea - interArea
        return interArea / (unionArea + epsilon)

    def classify_errors_refined(self, predictions: np.ndarray, ground_truths: np.ndarray,
                                iou_threshold: Optional[float] = None) -> (dict, np.ndarray, np.ndarray):
        if iou_threshold is None: iou_threshold = self.iou_threshold
        num_preds, num_gts = predictions.shape[0], ground_truths.shape[0]

        if num_gts == 0:
            return (np.full(num_preds, 'FP_H', dtype=object),
                    np.full(num_preds, None, dtype=object),
                    np.argsort(predictions[:, 5])[::-1])  # 仍然返回 sort_inds
        if num_preds == 0:
            return (np.array([], dtype=object),
                    np.array([], dtype=object),
                    np.array([], dtype=int))

        sort_inds = np.argsort(predictions[:, 5])[::-1]
        predictions_sorted = predictions[sort_inds]

        gt_matched = np.zeros(num_gts, dtype=bool)
        pred_assignment = np.full(num_preds, 'UNMATCHED', dtype=object)
        pred_match_info = np.full(num_preds, None, dtype=object)

        iou_matrix = np.array([[self.__calculate_iou(p[:4], g[:4]) for g in ground_truths] for p in predictions_sorted])

        for i in range(num_preds):
            best_gt_idx, max_iou = -1, iou_threshold
            for j in range(num_gts):
                if not gt_matched[j] and iou_matrix[i, j] >= max_iou:
                    max_iou, best_gt_idx = iou_matrix[i, j], j
            if best_gt_idx != -1:
                gt_matched[best_gt_idx] = True
                gt_label = int(ground_truths[best_gt_idx, 4])
                pred_assignment[i] = 'TP' if int(predictions_sorted[i, 4]) == gt_label else 'FP_C'
                pred_match_info[i] = {'gt_idx': best_gt_idx, 'iou': max_iou, 'gt_label': gt_label}

        for i in range(num_preds):
            if pred_assignment[i] == 'UNMATCHED':
                best_gt_idx = np.argmax(iou_matrix[i, :])
                max_iou = iou_matrix[i, best_gt_idx]
                if max_iou >= iou_threshold:
                    gt_label = int(ground_truths[best_gt_idx, 4])
                    pred_assignment[i] = 'DUPS' if int(predictions_sorted[i, 4]) == gt_label else 'FP_C'
                    pred_match_info[i] = {'gt_idx': best_gt_idx, 'iou': max_iou, 'gt_label': gt_label}
                else:
                    pred_assignment[i] = 'FP_H'
                    pred_match_info[i] = None

        for i in range(num_preds):
            num_overlapping_gts = np.sum(iou_matrix[i, :] >= iou_threshold)
            if num_overlapping_gts > 1:
                pred_assignment[i] = 'SPAN'

        return pred_assignment, pred_match_info, sort_inds
