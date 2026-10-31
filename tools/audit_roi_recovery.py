"""Audit frozen candidate files and baseline-protected, reversible recovery budgets."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).parent/'analysis_tools'))
from analyze_dental_candidates import analyze


def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    return inter/np.maximum(1e-9,(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter)


def protect(record,budget,ranking='objectness'):
    """Never remove an original. Every grouped alternative remains in the provenance."""
    originals=record['original_boxes']; additions=record['additions']
    def score(a):
        if ranking=='quality':
            if a.get('quality') is None:raise ValueError('Quality ranking requires quality-head outputs')
            return a['score']*a['quality']*a['novelty']
        return a['score']
    order=sorted(range(len(additions)),key=lambda i:score(additions[i]),reverse=True)
    boxes=list(originals);scores=list(record['original_scores']);groups=[[] for _ in boxes];selected=[]
    for i in order:
        a=additions[i];overlap=iou([a['bbox_xyxy']],boxes)[0]
        if len(overlap) and overlap.max()>=.85:
            groups[int(overlap.argmax())].append(i)
        elif len(selected)<budget:
            selected.append(i);boxes.append(a['bbox_xyxy']);scores.append(a['score']);groups.append([i])
    return dict(img_id=record['image_id'],pred_instances=dict(bboxes=np.asarray(boxes).reshape(-1,4),
        scores=np.asarray(scores),proposal_ids=np.arange(len(boxes)))),dict(image_id=record['image_id'],
        original_count=len(originals),selected_addition_ids=selected,groups=groups,
        deferred_addition_ids=[i for i in order if i not in {j for g in groups for j in g}])


def audit(records,coco):
    ids={r['image_id'] for r in records}
    if ids!={x['id'] for x in coco['images']}:raise ValueError('Require full annotation split, not partial results')
    results={};interfaces={}
    for ranking in ['objectness','quality']:
        if ranking=='quality' and any(a.get('quality') is None for r in records for a in r['additions']):continue
        for budget in [0,8,16,32]:
            pairs=[protect(r,budget,ranking) for r in records]
            key=f'{ranking}_extra_budget_{budget}'
            counts=np.array([len(p[0]['pred_instances']['scores']) for p in pairs])
            results[key]=dict(metrics=analyze([p[0] for p in pairs],coco),
                candidate_mean=float(counts.mean()),candidate_p95=float(np.quantile(counts,.95,method='higher')),
                candidate_max=int(counts.max()))
            interfaces[key]=[p[1] for p in pairs]
    slot_stats={};gt_audit=[];within_roi=[]
    by_image={}
    for a in coco['annotations']:
        if not a.get('iscrowd',0) and not a.get('ignore',False):by_image.setdefault(a['image_id'],[]).append(a)
    for r in records:
        grouped={}
        for a in r['additions']:
            slot=str(a.get('slot_id','legacy_unknown'));slot_stats[slot]=slot_stats.get(slot,0)+1
            grouped.setdefault(str(a.get('roi_id',a['region'])),[]).append(a['bbox_xyxy'])
        for boxes in grouped.values():
            if len(boxes)>1:
                matrix=iou(boxes,boxes);within_roi.extend(matrix[np.triu_indices(len(boxes),1)].tolist())
        for a in by_image.get(r['image_id'],[]):
            x,y,w,h=a['bbox'];gt=[[x,y,x+w,y+h]]
            original=iou(gt,r['original_boxes']);added=iou(gt,[b['bbox_xyxy'] for b in r['additions']])
            gt_audit.append(dict(image_id=r['image_id'],annotation_id=a['id'],category_id=a['category_id'],
                best_original_iou=float(original.max()) if original.size else 0.,
                best_added_iou=float(added.max()) if added.size else 0.))
    return dict(images=len(records),partial_dataset=False,slot_output_counts=slot_stats,
        within_roi_pairs=len(within_roi),within_roi_duplicate_fraction_085=float(np.mean(np.array(within_roi)>=.85)) if within_roi else None,
        gt_box_coverage_diagnostic=gt_audit,comparisons=results),interfaces


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summary,interface=audit(json.loads(a.candidates.read_text()),json.loads(a.annotations.read_text()))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'audit.json').write_text(json.dumps(summary,indent=2))
    (a.output/'interface.json').write_text(json.dumps(interface))
