# 实现一个 全局条件的空间加权损失函数（Class-Conditional Spatially-Weighted Loss）

import torch
from mmengine import ConfigDict, MMLogger
from torch import Tensor
from typing import Tuple, Optional, List

from mmdet.registry import MODELS
from .convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.losses import accuracy
from ...task_modules import SamplingResult
from mmdet.structures.bbox import get_box_tensor

@MODELS.register_module()
class GCSWBBOXHead(Shared2FCBBoxHead):
    """
    全局条件空间加权 BBox Head (Global-Conditional Spatially-Weighted BBox Head)

    此 Head 继承自 Shared2FCBBoxHead, 但修改了损失计算逻辑,
    使其在计算乳牙 (primary teeth) 损失时,
    动态地为 "牙冠区域的候选框" 和 "牙根区域的候选框" 赋予不同权重。
    """

    def __init__(self,
                 *args,
                 crown_weight:float = 1.5, # 超参数 牙冠权重
                 root_weight:float = 0.5, # 超参数 牙根权重
                 crown_root_split_ratio: float =0.5, # 牙冠牙根分割比例
                 **kwargs
                 ):
        super().__init__(*args,**kwargs)
        self.logger = MMLogger.get_current_instance()
        self.logger.info(f"crown_weight:{crown_weight},"
                         f"root_weight:{root_weight},"
                         f"crown_root_split_ratio:{crown_root_split_ratio}")
        self.crown_weight = crown_weight
        self.root_weight = root_weight
        self.crown_root_split_ratio = crown_root_split_ratio


    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        self.logger.info("-- 开始调用自定义loss---")
        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override,
            sampling_results = sampling_results
        )

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)


    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None,
             sampling_results:list[SamplingResult] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            sampling_results(list):
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """
        self.logger.info("-- 进入自定义loss---")

        pos_inds = (labels >=0) & (labels < self.num_classes)
        num_pos_samples = pos_inds.sum()
        self.logger.info(f"start label_weights:{label_weights}")
        if num_pos_samples > 0:
            # 创建自定义空间权重
            custom_label_weights = label_weights.clone()

            # 获取真实框并转换成Tesnor
            pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
            pos_gt_bboxes = torch.cat(pos_gt_bboxes_list)

            # 获取符合条件的真实标签索引及其提议框
            pos_labels = labels[pos_inds]
            self.logger.info(f"pos_labels:{pos_labels}")
            pos_proposal_boxes = rois[pos_inds][:,1:]# 去掉 batch_index

            for i in range(num_pos_samples):

                gt_box = pos_gt_bboxes[i]
                proposal_box = pos_proposal_boxes[i]

                # 定义牙冠/牙根区域
                gt_h = gt_box[3] - gt_box[1]
                # y坐标中点 (假设 y=0 在顶部)
                gt_crown_root_midline = gt_box[1] + gt_h * self.crown_root_split_ratio
                proposal_y_center = (proposal_box[1] + proposal_box[3]) / 2
                original_index = torch.where(pos_inds)[0][i]

                self.logger.info(f"proposal_y_center:{proposal_y_center},gt_crown_root_midline:{gt_crown_root_midline}")
                if proposal_y_center < gt_crown_root_midline:
                    # 候选框中心在牙冠区, 增加权重
                    self.logger.info(f"center before custom_label_weights :{custom_label_weights[original_index]}")
                    custom_label_weights[original_index] *= self.crown_weight
                    self.logger.info(f"center after custom_label_weights :{custom_label_weights[original_index]}")
                else:
                    # 候选框中心在牙根区, 降低权重
                    self.logger.info(f"no center before custom_label_weights :{custom_label_weights[original_index]}")
                    custom_label_weights[original_index] *= self.root_weight
                    self.logger.info(f"no center after custom_label_weights:{custom_label_weights[original_index]}")

            # --- 7. 使用我们修改后的权重 ---
            label_weights = custom_label_weights
            self.logger.info(f"final weights:{label_weights}")
            losses = dict()

            if cls_score is not None:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                if cls_score.numel() > 0:
                    loss_cls_ = self.loss_cls(
                        cls_score,
                        labels,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                    if isinstance(loss_cls_, dict):
                        losses.update(loss_cls_)
                    else:
                        losses['loss_cls'] = loss_cls_
                    if self.custom_activation:
                        acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                        losses.update(acc_)
                    else:
                        losses['acc'] = accuracy(cls_score, labels)
            if bbox_pred is not None:
                bg_class_ind = self.num_classes
                # 0~self.num_classes-1 are FG, self.num_classes is BG
                pos_inds = (labels >= 0) & (labels < bg_class_ind)
                # do not perform bounding box regression for BG anymore.
                if pos_inds.any():
                    if self.reg_decoded_bbox:
                        # When the regression loss (e.g. `IouLoss`,
                        # `GIouLoss`, `DIouLoss`) is applied directly on
                        # the decoded bounding boxes, it decodes the
                        # already encoded coordinates to absolute format.
                        bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                        bbox_pred = get_box_tensor(bbox_pred)
                    if self.reg_class_agnostic:
                        pos_bbox_pred = bbox_pred.view(
                            bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                    else:
                        pos_bbox_pred = bbox_pred.view(
                            bbox_pred.size(0), self.num_classes,
                            -1)[pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)]]
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                else:
                    losses['loss_bbox'] = bbox_pred[pos_inds].sum()

            return losses