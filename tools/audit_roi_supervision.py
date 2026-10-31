"""Aggregate cache supervision and checkpoint matching; no feature cache download needed."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import torch
from torchvision.ops import box_iou, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from dental_roi_head import ToothROIHead
from dental_roi_sampling import strata,sample_rois
from run_dental_roi import load,geometry,box_context


def audit(cache,head=None,device='cpu'):
    manifest=json.loads((cache/'manifest.json').read_text());counts=Counter();gt_hist=Counter();slots=Counter();sample_counts={p:Counter() for p in ['uniform','stratified']}
    model=None
    if head:
        h=load(head);assert h['cache_manifest']['detector_sha256']==manifest['detector_sha256']
        options=h.get('head_options',dict(roi_size=7,conditioned=False,quality=False))
        assert options['roi_size']==manifest['roi_size']
        model=ToothROIHead(h['channels'],1 if h['mode']=='single' else 4,**options).to(device)
        model.load_state_dict(h['state_dict']);model.eval()
    rngs={p:random.Random(42) for p in sample_counts}
    for filename in manifest['entries']:
        item=load(cache/filename);pools=strata(item);counts['images']+=1
        for k,indices in pools.items():counts[k+'_rois']+=len(indices);counts[k+'_images']+=bool(indices)
        valid=[i for pool in pools.values() for i in pool]
        counts['overflow_rois']+=sum(item['overflow'])
        for i in valid:
            gt=item['targets'][i];gt_hist[len(gt)]+=1;counts['complete_gt_occurrences']+=len(gt)
            r=item['regions'][i];absolute=gt*(r[2:]-r[:2]).repeat(2)+r[:2].repeat(2)
            boxes=item['boxes'][item['scores']>=.05]
            best=box_iou(absolute,boxes).max(1).values if len(boxes) else torch.zeros(len(gt))
            counts['uncovered_gt_occurrences']+=int((best<.75).sum())
        for policy in sample_counts:
            picked=sample_rois(item,policy,rng=rngs[policy])
            sample_counts[policy]['selected_rois']+=len(picked)
            for k,indices in pools.items():sample_counts[policy][k+'_rois']+=len(set(picked)&set(indices))
        if model is not None and valid:
            with torch.no_grad():
                context,_=box_context(item,valid,device)
                pred=model(item['features'][valid].float().to(device),geometry(item,valid,device),context)
                for j,i in enumerate(valid):
                    gt=item['targets'][i].to(device)
                    if h['mode']=='single':gt=gt[:1]
                    if not len(gt):continue
                    boxes=pred['boxes'][j];cost=5*torch.cdist(boxes,gt,p=1)+2*(1-generalized_box_iou(boxes,gt))-pred['logits'][j].sigmoid()[:,None]
                    rows,cols=linear_sum_assignment(cost.cpu().numpy())
                    for row,col in zip(rows,cols):
                        slots[f'slot_{row}_matches']+=1
                        slots[f'slot_{row}_iou075']+=int(box_iou(boxes[row:row+1],gt[col:col+1])[0,0]>=.75)
    return dict(images=counts['images'],partial_dataset=manifest['partial_dataset'],split=manifest['split'],
        detector_sha256=manifest['detector_sha256'],counts=dict(counts),complete_gt_per_roi=dict(gt_hist),
        sampling_one_epoch_seed42={k:dict(v) for k,v in sample_counts.items()},matching=dict(slots),
        note='GT occurrences repeat across overlapping ROIs; not unique tooth counts or detection recall.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--head',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    result=audit(a.cache,a.head,a.device)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
