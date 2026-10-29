"""Audited YOLO11 feature adapters and an optional non-filtering quality head."""
from copy import deepcopy
import json

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils import nms, ops

from dental_yolo_ablation import DentalPositiveLoss


class DirectionAdapter(nn.Module):
    """Three gated branches; control has exactly the same parameter count."""
    def __init__(self, channels, control=False):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                    nn.GroupNorm(1, hidden), nn.SiLU())
        kernels = [(3, 3), (3, 3), (1, 1)] if control else [(3, 3), (1, 5), (5, 1)]
        self.branches = nn.ModuleList(nn.Conv2d(hidden, hidden, k,
            padding=(k[0] // 2, k[1] // 2), groups=hidden, bias=False) for k in kernels)
        self.gate = nn.Conv2d(hidden, 3, 1)
        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.gain = nn.Parameter(torch.tensor(.01))

    def forward(self, x):
        z = self.reduce(x)
        gates = self.gate(z).softmax(1)
        value = sum(branch(z) * gates[:, i:i+1] for i, branch in enumerate(self.branches))
        return x + self.gain * self.expand(value)


class LocalRelationAdapter(nn.Module):
    """3x3 feature-neighborhood attention, not anatomical/FDI topology."""
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 16)
        self.hidden = hidden
        self.qkv = nn.Conv2d(channels, hidden * 3, 1, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(9))
        self.project = nn.Conv2d(hidden, channels, 1, bias=False)
        self.gain = nn.Parameter(torch.tensor(.01))

    def forward(self, x):
        b, _, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, 1)
        k = F.unfold(k, 3, padding=1).view(b, self.hidden, 9, h * w)
        v = F.unfold(v, 3, padding=1).view(b, self.hidden, 9, h * w)
        logits = (q.flatten(2).unsqueeze(2) * k).sum(1) / self.hidden ** .5
        logits = logits + self.relative_bias[None, :, None]
        valid = F.unfold(x.new_ones(1, 1, h, w), 3, padding=1).bool()
        attention = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(1)
        value = (v * attention[:, None]).sum(2).view(b, self.hidden, h, w)
        return x + self.gain * self.project(value)


class AdaptedLayer(nn.Module):
    def __init__(self, base, adapter):
        super().__init__()
        self.base, self.adapter = base, adapter
        for name in ['i', 'f', 'type', 'np']:
            setattr(self, name, getattr(base, name))

    def forward(self, x):
        return self.adapter(self.base(x))


class QualityDetect(Detect):
    def forward(self, x):
        if self.export:
            raise NotImplementedError('Quality-preserving export is not audited; use native .pt inference')
        return super().forward(x)

    @classmethod
    def from_head(cls, head, channels):
        assert not head.end2end and not head.export
        new = cls(nc=head.nc, ch=channels)
        for name in ['cv2', 'cv3', 'dfl']:
            setattr(new, name, deepcopy(getattr(head, name)))
        new.stride = head.stride.clone()
        for name in ['i', 'f', 'type', 'np', 'legacy', 'max_det']:
            setattr(new, name, getattr(head, name))
        new.quality = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in channels)
        return new

    def forward_head(self, x, box_head=None, cls_head=None):
        result = super().forward_head(x, box_head, cls_head)
        if result:
            result['quality_logits'] = torch.cat(
                [layer(feature).flatten(2) for layer, feature in zip(self.quality, x)], 2)
        return result


class DentalQualityPredictor(DetectionPredictor):
    """Carry quality by native pre-NMS anchor index; never multiply detection score."""
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        assert isinstance(preds, (tuple, list)) and isinstance(preds[1], dict)
        quality = preds[1]['quality_logits'].sigmoid().squeeze(1)
        boxes, indices = nms.non_max_suppression(preds[0], self.args.conf, self.args.iou,
            self.args.classes, self.args.agnostic_nms, max_det=self.args.max_det,
            nc=len(self.model.names), return_idxs=True)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        results = self.construct_results(boxes, img, orig_imgs)
        for i, (result, selected) in enumerate(zip(results, indices)):
            selected = selected.long()  # Native empty-detection indices may be floating point.
            result.dental_quality = quality[i, selected]
            result.dental_anchor_indices = selected
        return results


class DentalStructureLoss(DentalPositiveLoss):
    def __init__(self, model):
        super().__init__(model, model.dental_options['loss_variant'])
        self.quality_weight = model.dental_options['quality_weight']
        self.quality_audit = torch.zeros(2, device=self.device)

    def get_assigned_targets_and_loss(self, preds, batch, **kwargs):
        assignment, loss, items = super().get_assigned_targets_and_loss(preds, batch, **kwargs)
        if 'quality_logits' in preds:
            fg, _, target_boxes, anchors, stride = assignment
            fg = fg.bool()
            logits = preds['quality_logits'].squeeze(1)
            # Supervise quality on positives only; target cannot improve boxes by leakage.
            boxes = self.bbox_decode(anchors, preds['boxes'].permute(0, 2, 1)) * stride
            target = bbox_iou(boxes.detach()[fg], target_boxes[fg], xywh=False).squeeze(-1).clamp(0, 1)
            quality_loss = F.binary_cross_entropy_with_logits(logits[fg], target, reduction='sum')
            quality_loss = quality_loss / fg.sum().clamp(min=1)
            loss = loss.clone()
            loss[0] += self.quality_weight * quality_loss
            items[self.loss_names[0]] = loss[0].detach()
            self.quality_audit += torch.stack([quality_loss.detach(), logits.new_tensor(1.)])
        return assignment, loss, items


class DentalDetectionModel(DetectionModel):
    def init_criterion(self):
        return DentalStructureLoss(self)


def attach_adapters(model, options):
    assert len(model.model) == 24 and list(model.model[-1].f) == [16, 19, 22], 'Only audited YOLO11 layout'
    head = model.model[-1]
    channels = tuple(layer[0].conv.in_channels for layer in head.cv2)
    for index, channel in zip([16, 19], channels[:2]):
        adapters = []
        if options['feature'] != 'none':
            adapters.append(DirectionAdapter(channel, control=options['feature'] == 'control'))
        if options['relation']:
            adapters.append(LocalRelationAdapter(channel))
        if adapters:
            model.model[index] = AdaptedLayer(model.model[index], nn.Sequential(*adapters))
    if options['quality']:
        model.model[-1] = QualityDetect.from_head(head, channels)
    model.dental_options = deepcopy(options)
    return model


class DentalDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, dental_options, **kwargs):
        self.dental_options = dental_options
        super().__init__(*args, **kwargs)
        if self.args.resume:
            raise ValueError('Structure-run resume is not yet audited; do not silently rebuild from vanilla YAML')

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data['nc'] != 1:
            raise ValueError('Dental structure experiments require the single-class Tooth dataset')
        if weights is not None and hasattr(weights, 'dental_options'):
            raise ValueError('Use the original COCO pretrained weights, not a structure checkpoint for fresh training')
        model = self.set_model_names_for_load(DentalDetectionModel(
            cfg, nc=self.data['nc'], ch=self.data['channels'], verbose=verbose))
        if weights is not None:
            model.load(weights)
        base_parameters = sum(p.numel() for p in model.parameters())
        # Attach only after loading to preserve every compatible pretrained tensor.
        attach_adapters(model, self.dental_options)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'structure_config.json').write_text(json.dumps(dict(
            options=self.dental_options, base_parameters=base_parameters,
            total_parameters=sum(p.numel() for p in model.parameters()),
            initialization='native COCO load before adapters; new adapters random residual gain 0.01'), indent=2))
        return model


def log_structure_audit(trainer):
    model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    criterion = model.criterion
    values = criterion.quality_audit.detach().cpu().tolist()
    with (trainer.save_dir / 'quality_audit.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(epoch=trainer.epoch + 1,
            quality_bce_mean=values[0] / max(values[1], 1), batches=values[1])) + '\n')
    criterion.quality_audit.zero_()
