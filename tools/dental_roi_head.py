"""Frozen-FPN ROI set head. No FDI labels, exclusivity constraints or GT-driven proposals."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou, generalized_box_iou, roi_align
from scipy.optimize import linear_sum_assignment


class ToothROIHead(nn.Module):
    def __init__(self, channels, slots=4, hidden=128, roi_size=7, conditioned=False, quality=False):
        super().__init__()
        self.slots = slots
        self.project = nn.Conv2d(channels, hidden, 1)
        self.position = nn.Parameter(torch.randn(1,roi_size*roi_size,hidden)*.01)
        self.conditioned = conditioned
        self.use_quality = quality
        if conditioned:
            self.context = nn.Sequential(nn.Linear(5,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        if quality:
            self.quality = nn.Linear(hidden,1)
            self.novelty = nn.Linear(hidden,1)
        self.queries = nn.Parameter(torch.randn(slots,hidden)*.01)
        self.geometry = nn.Linear(4,hidden)
        layer = nn.TransformerDecoderLayer(hidden,4,hidden*2,dropout=0.,batch_first=True)
        self.decoder = nn.TransformerDecoder(layer,2)
        self.box = nn.Linear(hidden,4)
        self.objectness = nn.Linear(hidden,1)
        nn.init.constant_(self.objectness.bias, math.log(.05/.95))

    def forward(self, features, geometry, context=None):
        memory = self.project(features).flatten(2).transpose(1,2)+self.position
        query = self.queries[None]+self.geometry(geometry)[:,None]
        if self.conditioned:
            if context is None:raise ValueError('Conditioned head requires predicted-box context')
            query = query+self.context(context).mean(1)[:,None]
        decoded = self.decoder(query,memory)
        raw = self.box(decoded).sigmoid()
        center,size = raw[...,:2],raw[...,2:]
        boxes = torch.cat([(center-size/2).clamp(0,1),(center+size/2).clamp(0,1)],-1)
        result=dict(boxes=boxes,logits=self.objectness(decoded).squeeze(-1))
        if self.use_quality:
            result.update(quality=self.quality(decoded).squeeze(-1),novelty=self.novelty(decoded).squeeze(-1))
        return result


def set_loss(pred, targets, partial, overflow, mode='multi', existing=None):
    """Targets are complete instances; partial instances only mask uncertain negatives."""
    losses=[]
    for i,(gt,cut,skip) in enumerate(zip(targets,partial,overflow)):
        boxes,logits=pred['boxes'][i],pred['logits'][i]
        if skip:
            losses.append((boxes.sum()+logits.sum())*0)
            continue
        if mode=='single':
            # Other complete teeth cannot be represented by one slot; do not label them background.
            cut=torch.cat([cut,gt[1:]],0)
            gt=gt[:1]  # Cache orders the seed-associated instance first.
        if len(gt)>len(boxes):raise ValueError('Overflow must be marked, never truncate multi-instance GT')
        object_target=torch.zeros_like(logits)
        keep=torch.ones_like(logits,dtype=torch.bool)
        if len(cut):
            keep=box_iou(boxes.detach(),cut).max(1).values<.5
        regression=boxes.sum()*0
        quality_target=torch.zeros_like(logits)
        novelty_target=torch.zeros_like(logits)
        if len(gt):
            giou=generalized_box_iou(boxes,gt)
            cost=5*torch.cdist(boxes,gt,p=1)+2*(1-giou)-logits.sigmoid()[:,None]
            rows,cols=linear_sum_assignment(cost.detach().float().cpu().numpy())
            rows=torch.as_tensor(rows,device=boxes.device);cols=torch.as_tensor(cols,device=boxes.device)
            object_target[rows]=1;keep[rows]=True
            quality_target[rows]=box_iou(boxes[rows].detach(),gt[cols]).diag()
            if existing is not None and len(existing[i]):
                novelty_target[rows]=(box_iou(gt[cols],existing[i]).max(1).values<.75).float()
            else:novelty_target[rows]=1
            regression=5*F.l1_loss(boxes[rows],gt[cols],reduction='sum')/len(gt)
            regression+=2*(1-giou[rows,cols]).mean()
        weight=torch.where(object_target>0,1.,.25)
        classification=(F.binary_cross_entropy_with_logits(logits,object_target,reduction='none')*weight*keep).sum()/keep.sum().clamp(min=1)
        auxiliary=logits.sum()*0
        if 'quality' in pred:
            auxiliary=(F.binary_cross_entropy_with_logits(pred['quality'][i],quality_target,reduction='none')*keep).sum()/keep.sum().clamp(min=1)
            auxiliary+=(F.binary_cross_entropy_with_logits(pred['novelty'][i],novelty_target,reduction='none')*keep).sum()/keep.sum().clamp(min=1)
        losses.append(regression+classification+auxiliary)
    return torch.stack(losses).mean()


def make_regions(boxes,scores,width,height,limit=64,expansion=1.6):
    """Only predictions determine the ROI list; an empty detector yields no ROIs."""
    order=scores.argsort(descending=True)
    order=order[scores[order]>=.05][:limit]
    seeds=boxes[order]
    center=(seeds[:,:2]+seeds[:,2:])/2
    size=(seeds[:,2:]-seeds[:,:2])*expansion
    regions=torch.cat([center-size/2,center+size/2],1)
    regions[:,[0,2]]=regions[:,[0,2]].clamp(0,width)
    regions[:,[1,3]]=regions[:,[1,3]].clamp(0,height)
    return regions,seeds


def roi_targets(regions,seeds,gt,capacity=4):
    complete,partial,overflow=[],[],[]
    covered=set()
    for region,seed in zip(regions,seeds):
        xy1=torch.maximum(region[:2],gt[:,:2]);xy2=torch.minimum(region[2:],gt[:,2:])
        inter=(xy2-xy1).clamp(min=0).prod(1)
        area=(gt[:,2:]-gt[:,:2]).prod(1).clamp(min=1e-6)
        visible=inter/area
        # Complete target boxes must lie inside the ROI; do not clip full-tooth GT.
        full=(visible>=.9999)
        ids=full.nonzero().flatten()
        if len(ids):
            ids=ids[box_iou(seed[None],gt[ids])[0].argsort(descending=True)]
        size=(region[2:]-region[:2]).repeat(2)
        origin=region[:2].repeat(2)
        complete.append((gt[ids]-origin)/size)
        cuts=(visible>0)&~full
        clipped=torch.cat([xy1[cuts],xy2[cuts]],1)
        partial.append(((clipped-origin)/size).clamp(0,1))
        overflow.append(len(ids)>capacity)
        if len(ids)<=capacity:covered.update(ids.tolist())
    return complete,partial,overflow,len(covered)


def extract_rois(features,regions,width,height,imgsz=960,roi_size=7):
    gain=min(imgsz/width,imgsz/height)
    padx=round((imgsz-round(width*gain))/2-.1)
    pady=round((imgsz-round(height*gain))/2-.1)
    padded=regions*gain+regions.new_tensor([padx,pady,padx,pady])
    pooled=[]
    for feat in features[:2]:
        assert feat.shape[-1]==feat.shape[-2], 'rect=False square input required'
        pooled.append(roi_align(feat,[padded],output_size=roi_size,
                               spatial_scale=feat.shape[-1]/imgsz,aligned=True))
    return torch.cat(pooled,1)
