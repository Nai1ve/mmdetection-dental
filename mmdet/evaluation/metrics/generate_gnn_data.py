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
from collections import defaultdict # 确保导入


#SCORE_THRESHOLD = 0.05  # 置信度阈值
#IOU_THRESHOLD = 0.5 # IOU阈值
BACKGROUND = 48
epsilon = 1e-6  # 防止除零
#K_NEIGHBORS = 9
#info_list = []
#TYPE = 'test'
@METRICS.register_module()
class GenerateGNNData(CocoMetric):

    def __init__(self,
                 score_threshold: float = 0.05,  # 置信度阈值默认值
                 iou_threshold: float = 0.5,  # IoU阈值默认值
                 k_neighbors: int = 9,  # K近邻默认值
                 data_type: str = 'test',  # 数据类型默认值 ('test', 'val', 'train_fold_x')
                 gnn_save_dir: str = 'gnn_data',  # GNN数据保存目录
                 **kwargs):
        super().__init__(**kwargs)

        # 将参数存储为成员变量
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.k_neighbors = k_neighbors
        self.data_type = data_type  # 替换全局 TYPE
        self.gnn_save_dir = gnn_save_dir  # 替换全局 save_dir


        self.gnn_list:List[Optional[Data]] = [] # 语义信息的
        self.gnn_with_visual_embedding_list:List[Optional[Data]] = [] #嵌入信息的
        self.info_list = []  # 将 info_list 作为成员变量
        self.verification_stats = defaultdict(int)

        # 打印初始化参数 (方便调试)
        logger = MMLogger.get_current_instance()
        logger.info(f"GenerateGNNData initialized with:")
        logger.info(f"  score_threshold: {self.score_threshold}")
        logger.info(f"  iou_threshold: {self.iou_threshold}")
        logger.info(f"  k_neighbors: {self.k_neighbors}")
        logger.info(f"  data_type: {self.data_type}")
        logger.info(f"  gnn_save_dir: {self.gnn_save_dir}")

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        super().process(data_batch,data_samples)

        for data in data_samples:
            gnn_data,gnn_data_with_visual_embedding = self.__generate_gnn_data_and_save(data)

            if gnn_data is not None:
                self.gnn_list.append(gnn_data)
            if gnn_data_with_visual_embedding is not None:
                self.gnn_with_visual_embedding_list.append(gnn_data_with_visual_embedding)

    def compute_metrics(self,results:List) -> dict:
        eval_results = super().compute_metrics(results)

        MMLogger.info(MMLogger.get_current_instance(),"所有批次处理完成，开始保存GNN数据")
        MMLogger.info(MMLogger.get_current_instance(),f"总共收集到{len(self.gnn_list)}个GNN图数据")

        save_dir = 'gnn_data'
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.gnn_list, os.path.join(save_dir, f'gnn_{self.data_type}_data_{self.score_threshold}.pt'))
        torch.save(self.gnn_with_visual_embedding_list,os.path.join(save_dir, f'gnn_{self.data_type}_data_with_visual_embedding_{self.score_threshold}.pt'))

        with open(os.path.join(self.gnn_save_dir, f'info_score{self.score_threshold:.2f}.pkl'), 'wb') as f:
            pickle.dump(self.info_list, f)

        return eval_results

    def __generate_gnn_data_and_save(self,data :dict)-> (Data,Data):
        """
        根据数据创建两种
        Args:
            data:

        Returns:

        """
        pred_instances = data['pred_instances']
        gt_instances = data['gt_instances']  # 获取GT实例
        ori_shape = data['ori_shape']
        img_id = data['img_id']
        img_path = data['img_path']
        img_h, img_w = ori_shape[0], ori_shape[1]

        if 'scores' not in pred_instances or len(pred_instances['scores']) == 0:
            return None, None, None

        # --- 1. 准备预测数据 (过滤 + 提取) ---
        score_mask = pred_instances['scores'] >= self.score_threshold
        if not torch.any(score_mask): return None, None, None

        # 获取所有需要的数据 (应用 mask)
        final_bboxes_unnormalized = pred_instances['bboxes'][score_mask]
        final_scores = pred_instances['scores'][score_mask]
        final_raw_labels = pred_instances['labels'][score_mask]  # Faster R-CNN 原始预测索引
        # ... (提取其他特征) ...
        all_class_probs = pred_instances['all_class_probs'][score_mask]
        visual_features = pred_instances['visual_features'][score_mask]
        x_cls = pred_instances['x_cls'][score_mask]

        num_preds = final_bboxes_unnormalized.shape[0]

        # --- 2. 准备 GT 数据 ---
        gt_bboxes_coco = gt_instances['bboxes'].cpu().numpy()
        gt_labels_coco = gt_instances['labels'].cpu().numpy()  # 这是 GT 的类别索引

        # 将 GT 坐标从 [x,y,w,h] 转换为 [x1,y1,x2,y2]
        gt_bboxes_xyxy = []
        for j in range(len(gt_bboxes_coco)):
            x, y, w, h = gt_bboxes_coco[j]
            gt_bboxes_xyxy.append([x, y, x + w, y + h])
        gt_bboxes_xyxy = np.array(gt_bboxes_xyxy) if len(gt_bboxes_xyxy) > 0 else np.empty((0, 4))
        num_gts = gt_bboxes_xyxy.shape[0]
        # 组合 GT 数据: [xmin, ymin, xmax, ymax, gt_cls_idx]
        ground_truths_np = np.hstack((gt_bboxes_xyxy, gt_labels_coco.reshape(-1, 1))) if num_gts > 0 else np.empty(
            (0, 5))


        # --- 3. 生成 GNN 训练标签 y (使用您的 __ground_truth_y) ---
        # 注意: __ground_truth_y 内部需要确保 gt_bboxes 是 xyxy 格式
        # 传递 xyxy 格式的 gt_bboxes
        y_tensor = self.__ground_truth_y(
            pred_bboxes=final_bboxes_unnormalized.cpu().numpy(),
            pred_scores=final_scores.cpu().numpy(),
            gt_bboxes_xyxy=gt_bboxes_xyxy,  # 传递转换后的GT BBox
            gt_labels=gt_labels_coco,
            iou_threshold = self.iou_threshold
        )

        # 安全检查: 确保标签数量和节点数量一致
        if y_tensor is None or len(y_tensor) != num_preds:
            MMLogger.warning(MMLogger.get_current_instance(),
                             f"图片 {img_id}: 标签数量 ({len(y_tensor) if y_tensor is not None else 'None'}) 与预测框数量 ({num_preds}) 不匹配。跳过。")
            return None, None, None

        # --- 4. [新增] 运行 classify_errors_refined 进行交叉验证 ---
        # 准备 classify_errors_refined 的输入格式: [xmin, ymin, xmax, ymax, pred_cls_idx, score]
        predictions_for_classify = np.hstack((
            final_bboxes_unnormalized.cpu().numpy(),
            final_raw_labels.unsqueeze(1).cpu().numpy(),  # 使用原始预测标签
            final_scores.unsqueeze(1).cpu().numpy()
        ))

        # 使用 classify_errors_refined 进行分析
        errors_dict, pred_match_info = self.classify_errors_refined(predictions_for_classify, ground_truths_np)

        # --- 5. [新增] 逐节点验证 y 的正确性 ---
        y_vector_np = y_tensor.numpy()
        verification_passed_for_img = True

        # classify_errors_refined 返回的 predictions 是按分数排序后的
        # 我们需要一种方法将 y_vector_np 与 errors_dict 中的预测对应起来
        # 最可靠的方法是利用 classify_errors_refined 返回的 original_indices

        # 提取 classify_errors_refined 返回的所有预测及其原始索引
        classified_preds_with_orig_idx = []
        for category, preds_and_indices in errors_dict.items():
            if category == 'FN': continue
            for pred_box_list, orig_idx in preds_and_indices:
                classified_preds_with_orig_idx.append({
                    'box': pred_box_list,
                    'orig_idx': orig_idx,  # 这是在 score_mask 之后的索引
                    'category': category
                })

        # 按原始索引排序，确保与 y_vector_np 对齐
        classified_preds_with_orig_idx.sort(key=lambda item: item['orig_idx'])

        if len(classified_preds_with_orig_idx) != num_preds:
            MMLogger.error(MMLogger.get_current_instance(),
                           f"图片 {img_id}: 验证错误！ classify_errors 返回的预测数 ({len(classified_preds_with_orig_idx)}) 与过滤后的预测数 ({num_preds}) 不匹配！")
            verification_passed_for_img = False
        else:
            for i in range(num_preds):
                item = classified_preds_with_orig_idx[i]
                error_category = item['category']
                generated_y = y_vector_np[i]  # 获取 __ground_truth_y 生成的标签

                # 根据 error_category 判断期望的 y 值
                expected_y = -1  # 初始化为无效值
                match_info = pred_match_info[np.where(errors_dict['original_indices'] == item['orig_idx'])[0][
                    0]] if 'original_indices' in errors_dict else None  # 获取匹配信息

                if error_category == 'TP' or error_category == 'FP_C':
                    # 期望 y 是匹配到的 GT 标签
                    if match_info and 'gt_idx' in match_info:
                        expected_y = int(ground_truths_np[match_info['gt_idx'], 4])
                    else:
                        MMLogger.warning(MMLogger.get_current_instance(),
                                         f"图片 {img_id}, 节点 {i} ({error_category}): 未找到匹配的GT信息，无法验证期望标签。")
                        continue  # 无法验证，跳过
                elif error_category == 'DUPS' or error_category == 'FP_H' or error_category == 'SPAN':
                    expected_y = BACKGROUND

                # 进行比较
                if expected_y != -1 and generated_y != expected_y:
                    verification_passed_for_img = False
                    self.verification_stats['failed'] += 1
                    MMLogger.warning(MMLogger.get_current_instance(),
                                     f"!!! GNN标签验证失败 (图片ID: {img_id}, 节点索引: {i}) !!!")
                    MMLogger.warning(MMLogger.get_current_instance(),
                                     f"  预测框信息(原始): {final_bboxes_unnormalized[i].cpu().numpy()}, 预测标签: {final_raw_labels[i].item()}, 分数: {final_scores[i].item():.3f}")
                    MMLogger.warning(MMLogger.get_current_instance(), f"  classify_errors 判定类别: {error_category}")
                    MMLogger.warning(MMLogger.get_current_instance(),
                                     f"  __ground_truth_y 生成标签 (generated_y): {generated_y}")
                    MMLogger.warning(MMLogger.get_current_instance(),
                                     f"  基于 classify_errors 的期望标签 (expected_y): {expected_y}")
                    if match_info and 'gt_idx' in match_info:
                        gt_idx = match_info['gt_idx']
                        MMLogger.warning(MMLogger.get_current_instance(),
                                         f"  匹配的GT信息: GT索引 {gt_idx}, GT标签 {int(ground_truths_np[gt_idx, 4])}, IoU {match_info['iou']:.3f}")

                else:
                    self.verification_stats['passed'] += 1

        # --- 6. 构建图和特征 (如果验证通过或选择忽略失败) ---
        if not verification_passed_for_img and False:  # 如果需要严格验证，则失败时返回 None
            return None, None

        # ... [您原来的特征构建和图构建代码] ...
        feature_list_v1 = []
        feature_list_v3 = []
        pos_list = []
        # ... (循环构建特征) ...
        for i in range(num_preds):
            bbox = final_bboxes_unnormalized[i]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x_center = bbox[0] + w / 2
            y_center = bbox[1] + h / 2
            aspect_ratio = w / (h + 1e-6)
            geom_shape_features = torch.tensor(
                [(x_center / img_w) * 2 - 1, (y_center / img_h) * 2 - 1, (w / img_w) * 2 - 1,
                 (h / img_h) * 2 - 1])  # 归一化到 [-1, 1]
            area_feature = torch.tensor([(w * h) / (img_w * img_h)]) * 2 - 1  # 归一化面积到 [-1, 1]
            aspect_ratio_features = torch.tensor([math.log(aspect_ratio)])  # 对数可能更稳定
            confidence_score = final_scores[i].unsqueeze(0) * 2 - 1  # 归一化到 [-1, 1]
            normalized_x_cls = F.normalize(x_cls[i], p=2, dim=-1)
            semantic_features = all_class_probs[i]  # 49维
            normalized_semantic_features = 2 * semantic_features - 1

            feature_v1 = torch.cat([
                geom_shape_features, # 4D
                area_feature,# 1D
                aspect_ratio_features, #1D
                normalized_semantic_features, #49D
                confidence_score
            ])

            features_v3 = torch.cat([
                geom_shape_features,  # 4D
                area_feature,  # 1D (新增)
                aspect_ratio_features,  # 1D
                normalized_x_cls,  # 1024D
                confidence_score  # 1D
            ])
            feature_list_v1.append(feature_v1)
            feature_list_v3.append(features_v3)
            pos_list.append(torch.tensor([x_center / img_w, y_center / img_h]))  # 位置仍用 [0,1]

        x_v1 = torch.stack(feature_list_v1)
        x_v3 = torch.stack(feature_list_v3)
        pos = torch.stack(pos_list)

        temp_data = Data(pos=pos)
        knn_transform = KNNGraph(k= self.k_neighbors)
        graph_data = knn_transform(temp_data)
        edge_index = to_undirected(graph_data.edge_index)

        # 添加 IOU 和距离特征 (需要在循环外一次性计算以提高效率)
        max_iou_list = []
        avg_iou_list = []
        avg_dist_list = []

        # 预计算所有节点对的 IoU 和距离 (如果节点数不多)
        # 或者在循环内计算 k 个邻居的
        for i in range(num_preds):
            neighbors_mask = (edge_index[0] == i)
            neighbors_indices = edge_index[1][neighbors_mask]

            if len(neighbors_indices) == 0:
                max_iou, avg_iou, avg_dist = 0.0, 0.0, 0.0
            else:
                current_box_np = final_bboxes_unnormalized[i].cpu().numpy()
                neighbors_boxes_np = final_bboxes_unnormalized[neighbors_indices].cpu().numpy()

                # 计算与邻居的IoU
                neighbor_ious = np.array([self.__calculate_iou(current_box_np, nb) for nb in neighbors_boxes_np])
                max_iou = np.max(neighbor_ious) if len(neighbor_ious) > 0 else 0.0
                avg_iou = np.mean(neighbor_ious) if len(neighbor_ious) > 0 else 0.0

                # 计算与邻居的中心点距离
                current_pos = pos[i].numpy()
                neighbors_pos = pos[neighbors_indices].numpy()
                distances = np.linalg.norm(neighbors_pos - current_pos, axis=1)
                avg_dist = np.mean(distances) if len(distances) > 0 else 0.0

            # 归一化 (简单示例，可能需要更复杂的归一化)
            max_iou_norm = max_iou  # IoU 已经在 [0, 1]
            avg_iou_norm = avg_iou
            # 距离需要根据数据分布进行归一化，这里暂时用简单缩放
            avg_dist_norm = (avg_dist * 2) - 1  # 假设平均距离在 [0, 0.5] 之间

            max_iou_list.append(torch.tensor([max_iou_norm]))
            avg_iou_list.append(torch.tensor([avg_iou_norm]))
            avg_dist_list.append(torch.tensor([avg_dist_norm]))

        # 将关系特征拼接到 x_v3
        relational_features = torch.cat([
            torch.stack(max_iou_list),
            torch.stack(avg_iou_list),
            torch.stack(avg_dist_list)
        ], dim=1)

        x_v1_final = torch.cat([x_v1.to(relational_features.device),relational_features],dim=1)
        # 确保 relational_features 和 x_v3 的设备一致
        x_v3_final = torch.cat([x_v3.to(relational_features.device), relational_features], dim=1)

        common_attrs = {
            'edge_index': edge_index, 'pos': pos, 'y': y_tensor, 'img_id': img_id,
            'img_path': img_path, 'ori_shape': torch.tensor(ori_shape),
            'pred_bboxes_raw': final_bboxes_unnormalized, 'pred_scores_raw': final_scores,
            'pred_labels_raw': final_raw_labels
        }

        # --- 返回最终的Data对象 ---
        data_v1 = Data(x=x_v1_final, **common_attrs) # 如果需要保留 V1

        data_v3 = Data(x=x_v3_final, **common_attrs)  # 使用包含关系特征的 V3

        # return data_v1, data_v2, data_v3
        return data_v1, data_v3


    def __ground_truth_y(self, pred_bboxes, pred_scores, gt_bboxes_xyxy, gt_labels, iou_threshold) -> Optional[Tensor]:
        num_preds = len(pred_bboxes)
        num_gts = len(gt_bboxes_xyxy)
        y_vector = np.full(num_preds, -1, dtype=int)
        gt_is_matched = np.zeros(num_gts, dtype=bool)

        if num_preds == 0: return None
        sorted_pred_indices = np.argsort(pred_scores)[::-1]

        if num_gts == 0:
            y_vector[:] = BACKGROUND
            return torch.from_numpy(y_vector).long()

        # 使用传入的 iou_threshold
        current_img_info = defaultdict(int) # 用于记录当前图片的标签分布

        for p_idx in sorted_pred_indices:
            pred_box = pred_bboxes[p_idx]
            ious = np.array([self.__calculate_iou(pred_box, gt_box) for gt_box in gt_bboxes_xyxy])
            best_gt_idx = np.argmax(ious)
            max_iou = ious[best_gt_idx]

            # 使用传入的 iou_threshold
            if max_iou >= iou_threshold:
                label = gt_labels[best_gt_idx]
                if not gt_is_matched[best_gt_idx]:
                    y_vector[p_idx] = label
                    gt_is_matched[best_gt_idx] = True
                    current_img_info[int(label)] += 1 # 强制转换为int
                else:
                    y_vector[p_idx] = BACKGROUND
                    current_img_info[BACKGROUND] += 1
            else:
                y_vector[p_idx] = BACKGROUND
                current_img_info[BACKGROUND] += 1

        if np.any(y_vector == -1):
             MMLogger.warning(MMLogger.get_current_instance(), f"在 __ground_truth_y 中发现未分配标签的预测！将它们设为背景。")
             y_vector[y_vector == -1] = BACKGROUND
             current_img_info[BACKGROUND] += np.sum(y_vector == -1) # 更新背景计数

        # 将当前图片的标签分布添加到 self.info_list
        self.info_list.append(dict(current_img_info))

        return torch.from_numpy(y_vector).long()

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

    def __calculate_iou_vectorized(self,box_a:Tensor,boxes_b:Tensor)->Tensor:
        """
        计算一个边界框与一组边界框之间的IoU

        Args:
            box_a（Tensor）: 形状为【1，4】的单个边界框【x1，y1，x2，y2】
            boxes_b（Tensor）: 形状为【k，4】的k个边界框【x1，y1，x2，y2】

        Returns:
            Tensor：形状为【k】的IoU
        """

        # 找到左上角和右下角
        xA = torch.max(box_a[:,0],boxes_b[:,0])
        yA = torch.max(box_a[:,1],boxes_b[:,1])
        xB = torch.max(box_a[:,2],boxes_b[:,2])
        yB = torch.max(box_a[:,3],boxes_b[:,3])

        # 计算交集面积
        interArea = torch.clamp(xB - xA,min=0) * torch.clamp(yB - yA,min=0)

        # 计算各自的面积
        boxAArea = (box_a[:,2]- box_a[:,0]) * (box_a[:,3] - box_a[:,1])
        boxBArea = (boxes_b[:,2]-boxes_b[:,0]) * (boxes_b[:,3] - boxes_b[:,1])

        # 计算并集
        unionArea = boxAArea + boxBArea - interArea

        # 防止除零
        iou = interArea / (unionArea + epsilon)
        return  iou

    def classify_errors_refined(self,predictions, ground_truths, iou_threshold=0.5):
        num_preds, num_gts = predictions.shape[0], ground_truths.shape[0]
        if num_gts == 0: return {'TP': [], 'FP_C': [], 'DUPS': [], 'SPAN': [], 'FP_H': predictions.tolist(), 'FN': []}
        if num_preds == 0: return {'TP': [], 'FP_C': [], 'DUPS': [], 'SPAN': [], 'FP_H': [],
                                   'FN': ground_truths.tolist()}
        # 确保按分数排序
        sort_inds = np.argsort(predictions[:, 5])[::-1]
        predictions = predictions[sort_inds]

        gt_matched = np.zeros(num_gts, dtype=bool)
        pred_assignment = np.full(num_preds, 'UNMATCHED', dtype=object)
        # 存储每个预测匹配到的GT索引和IoU，方便后续查找
        pred_match_info = np.full(num_preds, None, dtype=object)

        iou_matrix = np.array([[self.__calculate_iou(p[:4], g[:4]) for g in ground_truths] for p in predictions])

        for i in range(num_preds):
            best_gt_idx, max_iou = -1, iou_threshold
            for j in range(num_gts):
                if not gt_matched[j] and iou_matrix[i, j] >= max_iou:
                    max_iou, best_gt_idx = iou_matrix[i, j], j
            if best_gt_idx != -1:
                gt_matched[best_gt_idx] = True
                pred_assignment[i] = 'TP' if predictions[i, 4] == ground_truths[best_gt_idx, 4] else 'FP_C'
                pred_match_info[i] = {'gt_idx': best_gt_idx, 'iou': max_iou}  # 记录匹配信息

        for i in range(num_preds):
            if pred_assignment[i] == 'UNMATCHED':
                best_gt_idx = np.argmax(iou_matrix[i, :])
                max_iou = iou_matrix[i, best_gt_idx] if num_gts > 0 else 0  # 处理无GT情况
                if max_iou >= iou_threshold:
                    pred_assignment[i] = 'DUPS' if predictions[i, 4] == ground_truths[best_gt_idx, 4] else 'FP_C'
                    pred_match_info[i] = {'gt_idx': best_gt_idx, 'iou': max_iou}  # 记录匹配信息
                else:
                    pred_assignment[i] = 'FP_H'
                    pred_match_info[i] = None  # 没有匹配

        for i in range(num_preds):
            num_overlapping_gts = np.sum(iou_matrix[i, :] >= iou_threshold) if num_gts > 0 else 0
            if num_overlapping_gts > 1:
                pred_assignment[i] = 'SPAN'
                # SPAN也可能匹配了一个主GT，保留匹配信息可能有用
                # 如果 pred_match_info[i] is None:
                #    best_gt_idx = np.argmax(iou_matrix[i, :])
                #    pred_match_info[i] = {'gt_idx': best_gt_idx, 'iou': iou_matrix[i, best_gt_idx]}

        # 编译结果时，带上原始排序的索引，方便后续查找
        results = defaultdict(list)
        original_indices = sort_inds  # 保存排序前的索引
        for i in range(num_preds):
            # 保存 (原始预测框, 原始索引)
            results[pred_assignment[i]].append((predictions[i].tolist(), original_indices[i]))

        results['FN'] = ground_truths[~gt_matched].tolist()
        for key in ['MATCHED', 'UNMATCHED']:
            if key in results: del results[key]

        # 返回错误分类结果和每个预测框匹配的GT信息(按排序后的顺序)
        return dict(results), pred_match_info