"""CPU checks of weighting math/config parity without MMDetection imports.

This isolates the real CCSW class with a minimal parent. It is not a replacement
for an MMDetection/MMCV forward/backward and inference smoke test on the server.
"""
import ast
import runpy
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]


class Parent:
    def __init__(self, num_classes=53, **kwargs):
        self.num_classes = num_classes

    def get_targets(self, *args, **kwargs):
        return self.targets

    def loss(self, cls, bbox, rois, labels, weights, targets, bbox_weights, **kw):
        return {'loss_cls': (cls.square().sum(1)*weights).sum(),
                'loss_bbox': bbox.sum()*0}


def main():
    source = ast.parse((ROOT/'mmdet/models/roi_heads/bbox_heads/ccsw_bbox_head.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
    cls.decorator_list = []
    scope = {'torch':torch, 'Shared2FCBBoxHead':Parent,'get_box_tensor':lambda x:x}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'ccsw-isolated','exec'),scope)
    head = scope['CCSWBBOXHead']()
    labels = torch.tensor([32,32,42,42,0,52,53,32])
    centers = torch.tensor([75.,25.,25.,75.,75.,75.,75.,50.])
    rois = torch.stack([torch.zeros(8),torch.zeros(8),centers-20,
                        torch.ones(8)*100,centers+20],dim=1)
    gt = torch.tensor([[0.,0.,100.,100.]]).repeat(7,1)
    sampling = [SimpleNamespace(pos_gt_bboxes=gt)]
    weights = head.spatial_label_weights(rois,labels,torch.ones(8),sampling)
    torch.testing.assert_close(weights,torch.tensor([1.25,.75,1.25,.75,1.,1.,1.,1.]))
    # Two-image ordering: positives are concatenated in sampling-result order.
    torch.testing.assert_close(weights,head.spatial_label_weights(rois,labels,torch.ones(8),
        [SimpleNamespace(pos_gt_bboxes=gt[:3]),SimpleNamespace(pos_gt_bboxes=gt[3:])]))
    neutral = scope['CCSWBBOXHead'](crown_weight=1,root_weight=1)
    torch.testing.assert_close(neutral.spatial_label_weights(rois,labels,torch.ones(8),sampling),torch.ones(8))
    head.targets = (torch.tensor([53,53]),torch.ones(2),torch.zeros(2,4),torch.zeros(2,4))
    scores = torch.ones(2,54,requires_grad=True)
    loss = head.loss_and_target(scores,torch.zeros(2,4),torch.zeros(2,5),
        [SimpleNamespace(pos_gt_bboxes=torch.empty(0,4))],{})['loss_bbox']['loss_cls']
    loss.backward()
    assert scores.grad is not None
    base = runpy.run_path(str(ROOT/'configs/_teeth_/faster-rcnn_r50_fpn_24e_private_baseline.py'))
    variant = runpy.run_path(str(ROOT/'configs/_teeth_/faster-rcnn_r50_fpn_24e_private_ccsw.py'))
    assert variant['_base_'].endswith('private_baseline.py')
    assert set(variant['model']) == {'roi_head'}
    assert set(variant['model']['roi_head']) == {'bbox_head'}
    assert len(base['classes']) == 53
    assert base['train_cfg']['max_epochs'] == base['param_scheduler'][-1]['end'] == 24
    assert not any(x['type']=='PhotoMetricDistortion' for x in base['test_pipeline'])
    print('PASS: jaw direction, neutral weights, non-primary/extra/background, multi-image ordering, no-positive backward, config parity.')
    print('Pending on server: real MMDetection/MMCV training and candidate-feature alignment smoke test.')


if __name__ == '__main__':
    main()
