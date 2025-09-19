# 复现论文 10.1038/s41598-019-40414-y 中的后处理方法
# 1. Filtering of excessive overlapped boxes.
# 2. Application of teeth arrangement rules.

import torch
import numpy as np
from typing import List,Tuple

from mmengine import MMLogger
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.roi_heads import StandardRoIHead
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import InstanceList
from os import path


@MODELS.register_module()
class DentalFasterRCNNRoIHead(StandardRoIHead):
    """
    复现论文doi:10.1038/s41598-019-40414-y 中的后处理方法中的两个
    1.Filtering of excessive overlapped boxes.
        在标准NMS后，移除高度重叠的框
    2.Application of teeth arrangement rules.
        使用先验模版修正预测的标签
    """

    def __init__(self,
                 similarity_matrix_dic = None,
                 template=None,
                 class_nms_thresh: float = 0.7,
                 **kwargs
                 ):
        """
        初始化函数
        Args:
            similarity_matrix_path :
            template: 牙齿模版列表
            class_nms_thresh:NMS后处理阈值

            牙齿代号：
            W ：智齿，M：磨牙，P：前臼齿，Ca：犬齿，La：侧切牙，Ce：中切牙，I：正切牙，pI：乳门牙，pCa：乳犬齿，pM：乳磨牙

            牙齿FDI和代号映射字典：
            W ： 18，28，38，48；
            M ： 17，16，26，27，36，37，46，47；
            P ： 15，14，24，25，35，34，44，45；
            Ca： 13，23，33，43；
            La： 12，22；
            Ce： 11，21；
            I ： 42，41，31，32；
            pI： 51，52，61,62，71，72，81，82；
            pCa：53，63，73，83；
            pM：54，55，64，65，74，75，84，85；

        """
        super().__init__(**kwargs)

        # 获取Logger
        self.logger = MMLogger.get_current_instance()

        print("调用初始化")

        # 定义映射关系
        self.FDI_code_name_dic = {
            '18':'W', '28':'W','38':'W','48':'W',
            '17':'M','16':'M','27':'M','26':'M','37':'M','36':'M','47':'M','46':'M',
            '15':'P','14':'P','25':'P','24':'P','35':'P','34':'P','45':'P','44':'P',
            '13':'Ca','23':'Ca','33':'Ca','43':'Ca',
            '12':'La','22':'La',
            '11':'Ce','21':'Ce',
            '42':'I','41':'I','32':'I','31':'I',
            '51':'pI','52':'pI','61':'pI','62':'pI','71':'pI','72':'pI','81':'pI','82':'pI',
            '53':'pCa','63':'pCa','73':'pCa','83':'pCa',
            '54':'pM','55':'pM','64':'pM','65':'pM','74':'pM','75':'pM','84':'pM','85':'pM'

        }


        self.permanent_upper_code_name_idx_dic = {'W':0,'M':1,'P':2,'Ca':3,'La':4,'Ce':5}
        self.permanent_lower_code_name_idx_dic ={'W':0,'M':1,'P':2,'Ca':3,'I':4}
        self.primary_upper_code_name_idx_dic = {'pM':0,'pCa':1,'pI':2}
        self.primary_lower_code_name_idx_dic = {'pM':0,'pCa':1,'pI':2}

        if template is None:
            self.template = [
                [18, 17, 16, 15, 14, 13, 12, 11], [21, 22, 23, 24, 25, 26, 27, 28],
                [48, 47, 46, 45, 44, 43, 42, 41], [31, 32, 33, 34, 35, 36, 37, 38],
                [55, 54, 53, 52, 51], [61, 62, 63, 64, 65],
                [85, 84, 83, 82, 81], [71, 72, 73, 74, 75]
            ]


        if similarity_matrix_dic is None:
            self.similarity_matrix_dic = {
                'permanent_upper' : [[0.9,0.8,0,0,0,0],
                                     [0.8,0.9,0,0,0,0],
                                     [0,0,0.9,0.6,0.4,0.4],
                                     [0,0,0.6,0.9,0.6,0.8],
                                     [0,0,0.4,0.6,0.9,0.8],
                                     [0,0,0.4,0.8,0.8,0.9]
                                     ],
                'permanent_lower' : [[0.9,0.7,0,0,0],
                                     [0.7,0.9,0,0,0],
                                     [0,0,0.9,0.5,0.3],
                                     [0,0,0.5,0.9,0.5],
                                     [0,0,0.3,0.5,0.9]
                                     ],
                'primary_upper' : [[0.9,0.6,0],
                                   [0.6,0.9,0.5],
                                   [0,0.5,0.9]
                                   ],
                'primary_lower' : [[0.9,0.5,0],
                                   [0.5,0.9,0.4],
                                   [0,0.4,0.9]
                                   ]
            }


        self.class_nms_thresh = class_nms_thresh

        self.logger.info(f"Dental RoI Head 初始化成功。")
        self.logger.info(f" - Stage 2 (Inter-Class NMS) 阈值: {self.class_nms_thresh}")
        self.logger.info(f" - Stage 3 Template (len={len(self.template)}): {self.template}...")



    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """
        NMS进行增强
        Args:
            x: 主干网络和特征网络传入的特征图
            rpn_results_list: RPN阶段的输出结果，META INFORMATION labels:tensor , bboxes:tensor,scores:tensor
            batch_data_samples: 包含数据图像的原信息，META INFORMATION
            rescale:

        Returns:

        """
        self.logger.info("---------开始执行后处理算法--------------------")
        # 进行预测
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        for data_sample,results in zip(batch_data_samples,results_list):

            data_sample.pred_instances = results


        result_list = []

        # 遍历处理图像
        for i,data_sample in enumerate(batch_data_samples):
            self.logger.info(f"--------开始处理图片:{data_sample.metainfo.get('img_path','').split(path.sep)}")
            # 跨类别NMS处理
            result = self._cross_category_nms_processing(data_sample)
            # 模版匹配
            result = self._template_matching(data_sample)

            result_list.append(result)


        self.logger.info("---------后处理算法执行完成----------------")

        return result_list


    def _cross_category_nms_processing(self,data_sample:DetDataSample) -> DetDataSample :
        """
        跨类别NMS
        计算任意一对框的IOU，若两框的IOU > 阈值，则删除分数较低的那个
        Args:
            data_sample:

        Returns:
        """
        self.logger.info(f"------ post cross_category_nms 开始进行NMS处理，数据元信息是:{data_sample.metainfo}")
        pred_instances = data_sample.pred_instances
        if len(pred_instances) == 0:
            return data_sample

        bboxes = pred_instances.bboxes
        scores = pred_instances.scores
        labels = pred_instances.labels

        n = len(bboxes)
        self.logger.info(f"------ 跨类别NMS前，共有:{n}个预测框")

        keep_mask = torch.tensor(n,dtype=torch.bool,device=bboxes.device)

        # 计算所有框间的IOU 形状为（n,n）
        ious = bbox_overlaps(bboxes,bboxes)

        for i in range(n):
            if not keep_mask[i]:# 若该框已被移除，则不进行循环
                continue
            for j in range(i+1,n): # 遍历上三角，跳过对角线
                if not keep_mask[j]:# 该框已被移除，不进行循环
                    continue

                # 检查是否超过阈值
                if ious[i,j] > self.class_nms_thresh:
                    # 超过阈值，删除更小的那个
                    if scores[i] < scores[j]:
                        self.logger.info(f"-----post cross_category_nms label:{labels[i]} 为重叠框，与:{labels[j]} 重叠，删除")
                        keep_mask[i] = False
                        break
                    else:
                        keep_mask[j] = False
                        self.logger.info(f"-----post cross_category_nms label:{labels[j]} 为重叠框，与:{labels[i]} 重叠，删除")

        # 只保存需要保存的
        data_sample.pred_instances = pred_instances[keep_mask]
        self.logger.info(f"跨类别NMS后，共有:{len(data_sample.pred_instances)}个检测框")

        return data_sample




    def _template_matching(self,data_sample:DetDataSample) -> DetDataSample:
        """
        模版匹配，乳牙会和恒牙混淆，需要用y的信息进行区分？
        1. 根据空间位置排序分别构建四个牙弓
            - 上恒牙弓
            - 下恒牙弓
            - 上乳牙弓
            - 下乳牙弓
        2. 用模版在上面进行滑动，计算得到一个匹配分数并记录
        3. 选取最高的分的模版，将其数据调整为与模版一致
        4. 返回
        Args:
            data_sample:

        Returns:

        """
        pass
