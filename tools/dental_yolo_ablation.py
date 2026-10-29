"""Version-pinned positive-instance reweighting; no FDI/extra classifier."""
import json

import torch
from ultralytics.utils.loss import v8DetectionLoss


class RecordingBCE(torch.nn.BCEWithLogitsLoss):
    def __init__(self):
        super().__init__(reduction='none')
        self.record = None

    def forward(self, logits, targets):
        loss = super().forward(logits, targets)
        self.record = (loss, logits.detach(), targets.detach())
        return loss


def positive_weights(fg, indices, logits, targets, balance, hard):
    """Preserve foreground weight mass per image; leave background unchanged."""
    counts = torch.zeros((fg.shape[0], int(indices.max())+1), device=fg.device)
    counts.scatter_add_(1, indices, fg.float())
    weights = torch.ones_like(fg, dtype=torch.float)
    if balance:
        average = fg.sum(1, keepdim=True) / (counts > 0).sum(1, keepdim=True).clamp(min=1)
        per_anchor = counts.gather(1, indices).clamp(min=1)
        weights *= (average/per_anchor).sqrt().clamp(.5, 2.)
    if hard:
        # Under-prediction relative to TAL soft targets, not forced hard labels.
        weights *= 1. + (targets-logits.sigmoid()).clamp(min=0).sum(-1).clamp(max=1)
    mass = (weights*fg).sum(1, keepdim=True) / fg.sum(1, keepdim=True).clamp(min=1)
    weights = torch.where(fg, weights/mass.clamp(min=1e-6), torch.ones_like(weights))
    return weights.detach(), counts


class DentalPositiveLoss(v8DetectionLoss):
    def __init__(self, model, variant):
        super().__init__(model)
        self.variant = variant
        self.bce = RecordingBCE()
        self.audit = torch.zeros(7, device=self.device)

    def get_assigned_targets_and_loss(self, preds, batch, **kwargs):
        assignment, loss, items = super().get_assigned_targets_and_loss(preds,batch,**kwargs)
        fg, indices = assignment[:2]
        bce, logits, targets = self.bce.record
        self.bce.record = None
        w, counts = positive_weights(fg,indices,logits,targets,
            self.variant in ['balance','combined'], self.variant in ['hard','combined'])
        if self.variant != 'baseline':
            correction = ((w-1)*bce.sum(-1)).sum()/targets.sum().clamp(min=1)*self.hyp.cls
            loss = loss.clone()
            loss[1] = loss[1]+correction
            items[self.loss_names[1]] = loss[1].detach()
        gt_per_image = torch.bincount(batch['batch_idx'].long(), minlength=fg.shape[0])
        width = max(counts.shape[1], batch['cls'].shape[0]+1)
        padded = torch.nn.functional.pad(counts,(0,width-counts.shape[1]))
        valid = torch.arange(width,device=fg.device)[None,:] < gt_per_image[:,None]
        stats = torch.stack([torch.as_tensor(batch['cls'].shape[0],device=fg.device),
            ((padded==0)&valid).sum(), fg.sum(), ((padded>0)&valid).sum(),
            torch.as_tensor(1.,device=fg.device), (w*fg).sum(),
            w.masked_fill(~fg,0).max()]).float().detach()
        self.audit[:6] += stats[:6]
        self.audit[6] = torch.maximum(self.audit[6], stats[6])
        return assignment,loss,items


def install(trainer):
    variant = trainer._dental_variant
    trainer.model.criterion = DentalPositiveLoss(trainer.model,variant)


def log_assignment(trainer):
    criterion = trainer.model.criterion
    values = criterion.audit.detach().cpu().tolist()
    row = dict(epoch=trainer.epoch+1,variant=criterion.variant,
        gt=values[0],zero_positive_gt=values[1],positive_anchors=values[2],
        covered_gt=values[3],batches=values[4],weighted_positive_mass=values[5],
        max_effective_weight=values[6])
    with (trainer.save_dir/'assignment_audit.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    criterion.audit.zero_()
