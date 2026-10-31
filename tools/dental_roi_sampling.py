"""Train-only ROI stratification. GT controls sampling, never inference proposals."""
import random
import torch
from torchvision.ops import box_iou


def strata(item):
    result={'hard':[], 'multi':[], 'ordinary':[]}
    boxes=item['boxes'][item['scores']>=.05]
    for i,gt in enumerate(item['targets']):
        if item['overflow'][i]:continue
        r=item['regions'][i]
        absolute=gt*(r[2:]-r[:2]).repeat(2)+r[:2].repeat(2)
        covered=box_iou(absolute,boxes).max(1).values if len(boxes) else torch.zeros(len(gt))
        key='hard' if len(gt) and (covered<.75).any() else 'multi' if len(gt)>1 else 'ordinary'
        result[key].append(i)
    return result


def sample_rois(item,policy='uniform',limit=12,rng=None):
    rng=rng or random
    if policy=='uniform':
        valid=[i for i,skip in enumerate(item['overflow']) if not skip]
        return rng.sample(valid,min(limit,len(valid)))
    if policy!='stratified':raise ValueError(policy)
    pools=strata(item);selected=[]
    for key in ['hard','multi','ordinary']:
        selected.extend(rng.sample(pools[key],min(limit//3,len(pools[key]))))
    rest=[i for pool in pools.values() for i in pool if i not in selected]
    selected.extend(rng.sample(rest,min(limit-len(selected),len(rest))))
    rng.shuffle(selected)
    return selected
