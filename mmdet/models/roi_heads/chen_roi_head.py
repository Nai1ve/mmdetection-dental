# 复现论文 10.1038/s41598-019-40414-y 中的后处理方法
# 1. Filtering of excessive overlapped boxes.
# 2. Application of teeth arrangement rules.

import torch
import numpy as np
from typing import List,Tuple

from mmengine.registry import MODELS
from mmengine.structures import InstanceData

from mmdet.models.roi_heads import StandardRoIHead
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox_overlaps


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
                 similarity_matrix_path = None,
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

        print("调用初始化")

        # 定义映射关系
        FDI_code_name_dic = {
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

        if template is None:
            template = [
                [18, 17, 16, 15, 14, 13, 12, 11], [21, 22, 23, 24, 25, 26, 27, 28],
                [48, 47, 46, 45, 44, 43, 42, 41], [31, 32, 33, 34, 35, 36, 37, 38],
                [55, 54, 53, 52, 51], [61, 62, 63, 64, 65],
                [85, 84, 83, 82, 81], [71, 72, 73, 74, 75]
            ]
        if similarity_matrix_path is None:
            similarity_matrix_path


        self.class_nms_thresh = class_nms_thresh

        self.templates = [torch.tensor(t,dtype=torch.int64) for t in template]
        print(f"Dental RoI Head 初始化成功。")
        print(f" - Stage 2 (Inter-Class NMS) 阈值: {self.inter_class_nms_thresh}")
        print(f" - Stage 3 Template (len={len(self.templates)}): {self.template[:5]}...")
        print(f" - Stage 3 Sim Matrix loaded from: {similarity_matrix_path} with shape {self.similarity_matrix.shape}")