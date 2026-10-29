"""Global weighting with the standard RoI interfaces."""
from mmdet.registry import MODELS
from .ccsw_bbox_head import CCSWBBOXHead


@MODELS.register_module()
class GCSWBBOXHead(CCSWBBOXHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # An extra tooth has no reliable orientation derived from its label.
        self.primary_teeth_indices = (0, 51 if self.num_classes == 53
                                      else self.num_classes-1)
