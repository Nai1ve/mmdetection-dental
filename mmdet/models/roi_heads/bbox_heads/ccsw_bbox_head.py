# 实现一个 类别条件的空间加权损失函数（Class-Conditional Spatially-Weighted Loss）

import torch

from mmdet.models.losses import accuracy
from ...task_modules import SamplingResult
from mmengine import MMLogger

from .convfc_bbox_head import Shared2FCBBoxHead
from torch import Tensor
from typing import List, Optional, Tuple, Union
from mmengine.config import ConfigDict
from mmdet.models.layers import multiclass_nms
from mmdet.registry import MODELS
from mmdet.models.utils import empty_instances
from mmdet.utils import  InstanceList
from mmengine.structures import InstanceData
import torch.nn.functional as F
from mmdet.structures.bbox import get_box_tensor, scale_boxes


@MODELS.register_module()
class CCSWBBOXHead(Shared2FCBBoxHead):
    """
    类别条件空间加权 BBox Head (Class-Conditional Spatially-Weighted BBox Head)

    此 Head 继承自 Shared2FCBBoxHead, 但修改了损失计算逻辑,
    使其在计算乳牙 (primary teeth) 损失时,
    动态地为 "牙冠区域的候选框" 和 "牙根区域的候选框" 赋予不同权重。
    """

    def __init__(self,
                 *args,
                 crown_weight:float = 1.5, # 超参数 牙冠权重
                 root_weight:float = 0.5, # 超参数 牙根权重
                 crown_root_split_ratio: float =0.5, # 牙冠牙根分割比例
                 primary_teeth_indices: Tuple[int,int] = (28,47), # 对应 51-85 (假设总共48类 0-47)
                 **kwargs
                 ):
        super().__init__(*args,**kwargs)

        self.crown_weight = crown_weight
        self.root_weight = root_weight
        self.crown_root_split_ratio = crown_root_split_ratio

        self.logger = MMLogger.get_current_instance()
        self.logger.info(f"crown_weight:{crown_weight},"
                         f"root_weight:{root_weight},"
                         f"crown_root_split_ratio:{crown_root_split_ratio}")

        self.primary_min_idx = primary_teeth_indices[0]
        self.primary_max_idx = primary_teeth_indices[1]

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
        logger = MMLogger.get_current_instance()
        #logger.info("-- 开始调用自定义loss---")
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
             sampling_results:List[SamplingResult] = None) -> dict:
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
        logger = MMLogger.get_current_instance()
        #logger.info("-- 进入自定义loss---")

        pos_inds = (labels >=0) & (labels < self.num_classes)
        num_pos_samples = pos_inds.sum()

        #self.logger.info(f"start label_weights:{label_weights}")
        if num_pos_samples > 0:
            # 创建自定义空间权重
            custom_label_weights = label_weights.clone()

            # 获取真实框并转换成Tesnor
            pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
            pos_gt_bboxes = torch.cat(pos_gt_bboxes_list)

            # 获取符合条件的真实标签索引及其提议框
            pos_labels = labels[pos_inds]
            pos_proposal_boxes = rois[pos_inds][:,1:]# 去掉 batch_index

            for i in range(num_pos_samples):
                label = pos_labels[i]

                # 判断提议框是否是乳牙
                if self.primary_min_idx <= label <= self.primary_max_idx:
                    gt_box = pos_gt_bboxes[i]
                    proposal_box = pos_proposal_boxes[i]

                    # 定义牙冠/牙根区域
                    gt_h = gt_box[3] - gt_box[1]
                    # y坐标中点 (假设 y=0 在顶部)
                    gt_crown_root_midline = gt_box[1] + gt_h * self.crown_root_split_ratio
                    proposal_y_center = (proposal_box[1] + proposal_box[3]) / 2
                    original_index = torch.where(pos_inds)[0][i]

                    if proposal_y_center < gt_crown_root_midline:
                        # 候选框中心在牙冠区, 增加权重
                        custom_label_weights[original_index] *= self.crown_weight
                    else:
                        # 候选框中心在牙根区, 降低权重
                        custom_label_weights[original_index] *= self.root_weight

            # --- 7. 使用我们修改后的权重 ---
            label_weights = custom_label_weights
            #self.logger.info(f"final weights:{label_weights}")
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


    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.
        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
                - feature_embed (Tensor): Feature embedding for all \
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        # 特征信息是x_cls
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred,x_cls


    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        x_cls: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        bbox_feats:Tuple[Tensor]=None) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        MMLogger.debug(MMLogger.get_current_instance(),"进入添加视觉特征的ConvFCBBoxHeadAddVisualFeature.predict_by_feat方法")
        assert len(cls_scores) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single_add_visual_feature(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                x_cls=x_cls[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg,
                bbox_feat = bbox_feats[img_id])
            MMLogger.debug(MMLogger.get_current_instance(),"-----------------results_info_begin------------------------")
            MMLogger.debug(MMLogger.get_current_instance(),results)
            MMLogger.debug(MMLogger.get_current_instance(),"-----------------results_info_end--------------------------")
            result_list.append(results)

        return result_list

    def _predict_by_feat_single_add_visual_feature(
            self,
            roi: Tensor,
            cls_score: Tensor,
            x_cls: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None,
            bbox_feat : Tensor = None ) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            # 获取到保留的掩码
            det_bboxes, det_labels,keep_idx = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim,
                return_inds=True
            )
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
            # 转换
            proposal_inds = keep_idx // self.num_classes

            # 使用keep_idx 筛选出匹配的视觉特征和完整概率
            final_all_class_probs = scores[proposal_inds]
            final_visual_features = bbox_feat[proposal_inds]
            final_x_cls = x_cls[proposal_inds]

            results.set_field(
                value=final_visual_features,
                name='visual_features',
                dtype=Tensor,
            )

            results.set_field(
                value=final_all_class_probs,
                name='all_class_probs',
                dtype=Tensor,
            )

            results.set_field(
                value=final_x_cls,
                name='x_cls',
                dtype=Tensor,
            )

        return results