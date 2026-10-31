"""E0 one-to-one pool ceilings and conservative per-GT failure attribution."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from dental_review_regions import overlaps
from run_dental_stage12 import matched


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.raw_root/'manifest.json').read_text())
    assert not manifest['partial_dataset'],'Do not report partial exports as full evaluation'
    assert hashlib.sha256(a.annotations.read_bytes()).hexdigest()==manifest['annotation_sha256']
    coco=json.loads(a.annotations.read_text());anns=defaultdict(list)
    extra_ids={x['id'] for x in coco['categories'] if x['name']=='extra_tooth'}
    for x in coco['annotations']:
        if not x.get('ignore',False) and not x.get('iscrowd',0):anns[x['image_id']].append(x)
    settings={}
    for path in sorted(a.raw_root.glob('conf*.json')):
        records=json.loads(path.read_text());assert len(records)==len(coco['images'])
        settings[path.stem]={r['image_id']:r for r in records}
    totals=defaultdict(lambda:defaultdict(int));failures=[]
    for im in coco['images']:
        records=anns[im['id']];gt=np.array([[x['bbox'][0],x['bbox'][1],x['bbox'][0]+x['bbox'][2],x['bbox'][1]+x['bbox'][3]] for x in records]).reshape(-1,4)
        extra=np.array([x['category_id'] in extra_ids for x in records],bool)
        with np.load(a.raw_root/'raw'/f'{im["id"]}.npz',allow_pickle=False) as raw:
            boxes=raw['boxes'];scores=raw['scores'];matrix=overlaps(gt,boxes)[0]
            pools={'pre_geometry_all':boxes}
            pools.update({f'pre_score_{floor}':boxes[scores>floor] for floor in [.0001,.001,.01]})
            pools.update({name:np.asarray(images[im['id']]['original_boxes']).reshape(-1,4) for name,images in settings.items()})
            for name,pool in pools.items():
                for threshold in [.5,.75]:
                    hit=matched(gt,pool,threshold);r=totals[f'{name}/iou{threshold}']
                    r['gt']+=len(gt);r['matched']+=len(hit);r['extra_gt']+=int(extra.sum());r['extra_matched']+=int(extra[hit].sum());r['candidates']+=len(pool);r['images']+=1
            for name,images in settings.items():
                floor=float(name.split('_')[0][4:]);post=np.asarray(images[im['id']]['original_boxes']).reshape(-1,4)
                hit=set(matched(gt,post,.5));post_iou=overlaps(gt,post)[0]
                for i,x in enumerate(records):
                    if i in hit:continue
                    max_score=float(scores[matrix[i]>=.5].max()) if (matrix[i]>=.5).any() else None
                    if max_score is None:cause='no_decoded_geometry_iou05'
                    elif max_score<=floor:cause='below_score_floor'
                    elif post_iou.shape[1] and post_iou[i].max()>=.5:cause='one_to_one_assignment_conflict'
                    else:cause='nms_or_max_det_removal'
                    failures.append(dict(image_id=im['id'],annotation_id=x['id'],extra=bool(extra[i]),setting=name,
                        cause=cause,max_score_iou05=max_score,best_raw_iou=float(matrix[i].max()) if matrix.shape[1] else None))
    result=dict(manifest=manifest,one_to_one_counts={k:dict(v) for k,v in totals.items()},failures=failures,
        warning='Pre-NMS geometric ceilings include extremely weak responses, not clinically valid detections. NMS vs max_det can be inspected in trace.json.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print('E0_COMPLETE',len(coco['images']),len(settings))


if __name__=='__main__':main()
