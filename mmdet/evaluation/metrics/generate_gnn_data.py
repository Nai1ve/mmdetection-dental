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
import pickle
import math
from collections import defaultdict

# --- 常量 ---
BACKGROUND = 48  # 你的背景标签ID
epsilon = 1e-6  # 防止除零


# --- 辅助函数 ---
def to_cpu_numpy(tensor_or_array):
    if isinstance(tensor_or_array, Tensor):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)


def calculate_iou_np(boxA, boxB):
    """(Numpy) 计算两个边界框的IoU"""
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
                 debug_image_filename: str = None,  # [新增] 指定一个文件名来触发详细日志
                 **kwargs):
        super().__init__(**kwargs)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.k_neighbors = k_neighbors
        self.data_type = data_type
        self.gnn_save_dir = gnn_save_dir


        self.debug_image_filename = debug_image_filename

        self.gnn_with_visual_embedding_list: List[Optional[Data]] = []
        self.y_label_stats = defaultdict(int)
        self.logger = MMLogger.get_current_instance()

        self.logger.info(f"GenerateGNNData initialized with:")
        self.logger.info(f"  score_threshold: {self.score_threshold}")
        self.logger.info(f"  iou_threshold: {self.iou_threshold}")
        self.logger.info(f"  k_neighbors: {self.k_neighbors}")
        if self.debug_image_filename:
            self.logger.warning(f"  [DEBUG MODE] Verbose logging is ON for image: {self.debug_image_filename}")

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """处理一个批次的数据"""
        super().process(data_batch, data_samples)
        for data_sample in data_samples:
            gnn_data = self.__generate_gnn_data_and_save(data_sample)
            if gnn_data is not None:
                self.gnn_with_visual_embedding_list.append(gnn_data)

    def compute_metrics(self, results: List) -> dict:
        """在所有数据处理完后，计算指标并保存GNN数据"""
        eval_results = super().compute_metrics(results)
        self.logger.info("所有批次处理完成，开始保存GNN数据")
        self.logger.info(f"总共收集到{len(self.gnn_with_visual_embedding_list)}个GNN图数据")


        self.logger.info("--- GNN 训练数据 y 标签生成统计 ---")
        total_nodes = sum(self.y_label_stats.values())
        self.logger.info(f"总节点数 (Nodes): {total_nodes}")
        sorted_stats = sorted(self.y_label_stats.items(), key=lambda item: item[0])
        for label, count in sorted_stats:
            label_name = "BACKGROUND" if label == BACKGROUND else f"Class_{label}"
            self.logger.info(f"  - {label_name}: {count} ({(count / total_nodes):.2%})")
        # ----------------------------------------

        os.makedirs(self.gnn_save_dir, exist_ok=True)
        base_filename = (
            f'gnn_{self.data_type}_data'
            f'_score{self.score_threshold:.2f}'
            f'_iou{self.iou_threshold:.2f}'
            f'_k{self.k_neighbors}'
        )
        save_path = os.path.join(self.gnn_save_dir, f'{base_filename}.pt')
        torch.save(self.gnn_with_visual_embedding_list, save_path)
        self.logger.info(f"GNN 数据已保存至: {save_path}")

        return eval_results

    def __generate_gnn_y_labels(self,
                                pred_bboxes_np: np.ndarray,
                                pred_scores_np: np.ndarray,
                                gt_bboxes_np: np.ndarray,
                                gt_labels_np: np.ndarray,
                                verbose_logging: bool = False) -> (np.ndarray, dict):


        num_preds = pred_bboxes_np.shape[0]
        num_gts = gt_bboxes_np.shape[0]

        if verbose_logging:
            self.logger.info(f"\n[DEBUG {self.debug_image_filename}] --- Entering __generate_gnn_y_labels ---")
            self.logger.info(f"[DEBUG] Num Preds: {num_preds}, Num GTs: {num_gts}")

            for j in range(num_gts):
                self.logger.info(f"[DEBUG] GT {j}: label={gt_labels_np[j]}, box={gt_bboxes_np[j]}")

        y_vector_np = np.full(num_preds, BACKGROUND, dtype=int)

        if num_preds == 0:
            return y_vector_np, {}
        if num_gts == 0:
            if verbose_logging: self.logger.info("[DEBUG] No GTs found. All nodes assigned BACKGROUND.")
            labels, counts = np.unique(y_vector_np, return_counts=True)
            return y_vector_np, dict(zip(labels, counts))

        sort_inds = np.argsort(pred_scores_np)[::-1]
        gt_matched = np.zeros(num_gts, dtype=bool)

        iou_matrix = np.array([
            [calculate_iou_np(pred_bboxes_np[i], gt_box) for gt_box in gt_bboxes_np]
            for i in sort_inds
        ])

        if verbose_logging:
            self.logger.info(f"[DEBUG] IoU Matrix (SortedPreds x GTs) shape: {iou_matrix.shape}")
            self.logger.info(f"[DEBUG] IoU Matrix Max (All): {np.max(iou_matrix) if iou_matrix.size > 0 else 0}")
            self.logger.info(f"[DEBUG] --- Starting Match Loop (Thresh = {self.iou_threshold}) ---")

        for i_sorted in range(num_preds):
            pred_original_index = sort_inds[i_sorted]
            pred_score = pred_scores_np[pred_original_index]

            best_gt_idx = -1
            max_iou = 0.0

            for j in range(num_gts):
                if not gt_matched[j] and iou_matrix[i_sorted, j] >= max_iou:
                    max_iou = iou_matrix[i_sorted, j]
                    best_gt_idx = j

            if verbose_logging:
                log_msg = f"[DEBUG] Pred_Sorted {i_sorted} (OrigIdx {pred_original_index}, Score {pred_score:.2f}): "
                log_msg += f"Found Best GT {best_gt_idx} with MaxIoU {max_iou:.4f}."

            if best_gt_idx != -1 and max_iou >= self.iou_threshold:
                gt_matched[best_gt_idx] = True
                gt_label = gt_labels_np[best_gt_idx]
                y_vector_np[pred_original_index] = gt_label

                if verbose_logging:
                    self.logger.info(log_msg + f" --> ASSIGNED Label {gt_label}")
            else:
                if verbose_logging:
                    reason = "IoU < Thresh" if best_gt_idx != -1 else "No available GT found"
                    if max_iou > 0.0 and best_gt_idx == -1: reason = "Best GT already matched (DUPS)"
                    self.logger.info(log_msg + f" --> ASSIGNED BACKGROUND (Reason: {reason})")

        labels, counts = np.unique(y_vector_np, return_counts=True)
        stats = dict(zip(labels, counts))

        if verbose_logging:
            self.logger.info("[DEBUG] --- Exiting __generate_gnn_y_labels ---")

        return y_vector_np, stats

    def __generate_gnn_data_and_save(self, data: dict) -> Optional[Data]:


        pred_instances = data['pred_instances']
        gt_instances = data['gt_instances']
        ori_shape = data['ori_shape']
        img_id = data['img_id']
        img_path = data['img_path']
        img_h, img_w = ori_shape[0], ori_shape[1]
        verbose_logging = False
        if self.debug_image_filename and self.debug_image_filename in img_path:
            verbose_logging = True
            self.logger.info(f"\n\n[DEBUG] ******** Processing debug target image: {img_path} *********")
        if 'scores' not in pred_instances or len(pred_instances['scores']) == 0:
            return None

        all_scores_np = to_cpu_numpy(pred_instances['scores'])
        score_mask = all_scores_np >= self.score_threshold
        if not np.any(score_mask):
            return None
        final_bboxes_unnormalized_np = to_cpu_numpy(pred_instances['bboxes'])[score_mask]
        final_scores_np = all_scores_np[score_mask]
        final_raw_labels_np = to_cpu_numpy(pred_instances['labels'])[score_mask]
        x_cls_np = to_cpu_numpy(pred_instances['x_cls'])[score_mask]
        num_preds = final_bboxes_unnormalized_np.shape[0]


        # 1. 准备 GT 数据 (Numpy)

        gt_bboxes_xyxy = to_cpu_numpy(gt_instances.get('bboxes', np.empty((0, 4))))
        gt_labels_coco = to_cpu_numpy(gt_instances.get('labels', np.empty((0,))))

        # 2. 确保gt_labels是1D的
        if gt_labels_coco.ndim > 1: gt_labels_coco = gt_labels_coco.squeeze()
        if gt_labels_coco.ndim == 0 and gt_labels_coco.size == 1:
            gt_labels_coco = np.array([gt_labels_coco.item()])
        elif gt_labels_coco.ndim == 0 and gt_labels_coco.size == 0:
            gt_labels_coco = np.empty((0,))


        # --- 3. 调用y标签生成函数 ---
        y_vector_np, y_stats = self.__generate_gnn_y_labels(
            final_bboxes_unnormalized_np,
            final_scores_np,
            gt_bboxes_xyxy,  # [!!!] 现在这是正确的 [x1, y1, x2, y2] 格式
            gt_labels_coco,
            verbose_logging=verbose_logging
        )
        y_tensor = torch.from_numpy(y_vector_np).long()

        for label, count in y_stats.items():
            self.y_label_stats[int(label)] += count

        # --- 4. 构建图和特征---
        final_bboxes_tensor = torch.from_numpy(final_bboxes_unnormalized_np)
        final_scores_tensor = torch.from_numpy(final_scores_np)
        x_cls_tensor = torch.from_numpy(x_cls_np)
        final_raw_labels_tensor = torch.from_numpy(final_raw_labels_np).long()

        feature_list_v3 = []
        pos_list = []
        diag_length = math.sqrt(img_w ** 2 + img_h ** 2) + epsilon  # 图像对角线长度

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
                geom_shape_features, area_feature, aspect_ratio_features,
                normalized_x_cls, confidence_score
            ])
            feature_list_v3.append(features_v3)
            pos_list.append(torch.tensor([x_center, y_center]))

        x_v3 = torch.stack(feature_list_v3)
        pos = torch.stack(pos_list)

        temp_data = Data(pos=pos)
        knn_transform = KNNGraph(k=self.k_neighbors)
        graph_data = knn_transform(temp_data)
        edge_index = to_undirected(graph_data.edge_index)

        edge_attr_list = []
        row, col = edge_index
        for i in range(len(row)):
            src_node_idx = row[i].item()
            dest_node_idx = col[i].item()
            dist = torch.norm(pos[src_node_idx] - pos[dest_node_idx], p=2)
            normalized_dist = (dist / diag_length) * 2 - 1
            edge_attr_list.append(torch.tensor([normalized_dist]))

        edge_attr = torch.stack(edge_attr_list).float()

        max_iou_list, avg_iou_list = [], []
        for i in range(num_preds):
            neighbors_mask = (edge_index[0] == i)
            neighbors_indices = edge_index[1][neighbors_mask]

            if len(neighbors_indices) == 0:
                max_iou, avg_iou = 0.0, 0.0
            else:
                current_box_np = final_bboxes_unnormalized_np[i]
                neighbors_boxes_np = final_bboxes_unnormalized_np[neighbors_indices.cpu().numpy()]
                if neighbors_boxes_np.ndim == 1: neighbors_boxes_np = neighbors_boxes_np.reshape(1, -1)
                neighbor_ious = np.array([calculate_iou_np(current_box_np, nb) for nb in neighbors_boxes_np])
                max_iou = np.max(neighbor_ious) if len(neighbor_ious) > 0 else 0.0
                avg_iou = np.mean(neighbor_ious) if len(neighbor_ious) > 0 else 0.0

            max_iou_list.append(torch.tensor([max_iou]))
            avg_iou_list.append(torch.tensor([avg_iou]))

        relational_features = torch.cat([
            torch.stack(max_iou_list),
            torch.stack(avg_iou_list),
        ], dim=1).float()  # [N, 2]

        x_v3_final = torch.cat([x_v3, relational_features], dim=1)  # 最终特征

        pos_normalized = torch.stack([
            torch.tensor([(p[0] / img_w) * 2 - 1, (p[1] / img_h) * 2 - 1]) for p in pos
        ])

        # --- 5. 组装并返回最终的Data对象 ---
        common_attrs = {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'pos': pos_normalized,
            'y': y_tensor,
            'img_id': img_id,
            'img_path': img_path,
            'ori_shape': torch.tensor(ori_shape),
            'pred_bboxes_raw': final_bboxes_tensor,
            'pred_scores_raw': final_scores_tensor,
            'pred_labels_raw': final_raw_labels_tensor
        }
        data_v3 = Data(x=x_v3_final, **common_attrs)

        if verbose_logging:
            self.logger.info(f"[DEBUG] Final `y` stats for this image: {y_stats}")
            self.logger.info(f"[DEBUG] ******** Finished processing debug image *********\n")

        return data_v3