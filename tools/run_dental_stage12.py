"""Build prediction-only review interfaces; evaluate separately against FDI annotations."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from dental_review_regions import build_case,overlaps,select_regions,union_area


def matched(gt,boxes,threshold):
    matrix=overlaps(gt,boxes)[0];valid=matrix>=threshold
    rows,cols=linear_sum_assignment(-(valid*(len(gt)+1)+matrix))
    return rows[valid[rows,cols]]


def evaluate(cases,coco):
    assert len(cases)==len(coco['images']) and {x['image_id'] for x in cases}=={x['id'] for x in coco['images']}
    annotations=defaultdict(list);extra_ids={x['id'] for x in coco['categories'] if x['name']=='extra_tooth'}
    for a in coco['annotations']:
        if not a.get('iscrowd',0) and not a.get('ignore',False):annotations[a['image_id']].append(a)
    totals=defaultdict(lambda:defaultdict(float));failures=[];collision_groups=0
    for case in cases:
        anns=annotations[case['image_id']];gt=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns]).reshape(-1,4)
        extra=np.array([a['category_id'] in extra_ids for a in anns],bool)
        boxes=np.array([c['bbox_xyxy'] for c in case['candidates']]).reshape(-1,4)
        leaders=[g['leader'] for g in case['groups']];members=[i for g in case['groups'] for i in g['members']]
        assert sorted(members)==list(range(len(boxes))), 'Grouping lost/duplicated candidates'
        scale=np.array([case['image_size']['width'],case['image_size']['height']]*2)
        normalized=gt/scale
        baseline={t:set(matched(gt,boxes,t)) for t in [.5,.75]}
        matrix=overlaps(gt,boxes)[0];valid=matrix>=.5
        rows,cols=linear_sum_assignment(-(valid*(len(gt)+1)+matrix))
        assignment={int(c):int(r) for r,c in zip(rows,cols) if valid[r,c]}
        collision_groups+=sum(len({assignment[i] for i in g['members'] if i in assignment})>1 for g in case['groups'])
        for label,indices in [('all_members',list(range(len(boxes)))),('representatives',leaders)]:
            for t in [.5,.75]:
                ids=matched(gt,boxes[indices],t);row=totals[f'candidate/{label}/iou{t}']
                row['gt']+=len(gt);row['matched']+=len(ids);row['extra_gt']+=int(extra.sum());row['extra_matched']+=int(extra[ids].sum());row['candidates']+=len(indices);row['images']+=1
        for i,a in enumerate(anns):
            if i not in baseline[.5]:failures.append(dict(image_id=case['image_id'],annotation_id=a['id'],extra=bool(extra[i]),reason='unmatched_in_saved_post_nms_pool',raw_cause='unknown_without_pre_nms'))
        for policy in ['risk','grid','random','spread']:
            for k in [0,1,2,4,8]:
                regions=select_regions(case,policy,k);roi=np.asarray(regions).reshape(-1,4)
                inter=np.maximum(0,np.minimum(normalized[:,None,2:],roi[None,:,2:])-np.maximum(normalized[:,None,:2],roi[None,:,:2])).prod(2)
                area=np.maximum(1e-9,(normalized[:,2:]-normalized[:,:2]).prod(1))
                covered=(inter/area[:,None]>=.9).any(1)
                row=totals[f'region/{policy}/k{k}'];row['images']+=1;row['gt']+=len(gt);row['covered']+=int(covered.sum())
                row['extra_gt']+=int(extra.sum());row['extra_covered']+=int((extra&covered).sum())
                for t in [.5,.75]:
                    miss=np.array([i not in baseline[t] for i in range(len(gt))],bool)
                    row[f'missed_gt_iou{t}']+=int(miss.sum());row[f'missed_covered_iou{t}']+=int((miss&covered).sum())
                    row[f'missed_extra_iou{t}']+=int((miss&extra).sum());row[f'missed_extra_covered_iou{t}']+=int((miss&extra&covered).sum())
                row['roi_count']+=len(regions);row['union_area_fraction']+=union_area(regions)
                row['summed_area_fraction']+=sum((b[2]-b[0])*(b[3]-b[1]) for b in regions)
    result={}
    for key,v in totals.items():
        row=dict(v)
        if key.startswith('candidate/'):
            row.update(recall=v['matched']/v['gt'] if v['gt'] else None,
                extra_recall=v['extra_matched']/v['extra_gt'] if v['extra_gt'] else None,candidates_mean=v['candidates']/v['images'])
        else:
            row.update(coverage=v['covered']/v['gt'] if v['gt'] else None,
                extra_coverage=v['extra_covered']/v['extra_gt'] if v['extra_gt'] else None,
                mean_roi_count=v['roi_count']/v['images'],mean_union_area_fraction=v['union_area_fraction']/v['images'],
                mean_summed_area_fraction=v['summed_area_fraction']/v['images'])
        result[key]=row
    counts=np.array([len(c['candidates']) for c in cases]);groups=np.array([len(c['groups']) for c in cases])
    burden=lambda a:dict(mean=float(a.mean()),p95=float(np.quantile(a,.95,method='higher')),maximum=int(a.max()))
    return dict(images=len(cases),partial_dataset=False,pool_scope='saved_post_nms_only',
        no_claim_about_raw_pool_ceiling=True,roi_coverage_definition='at_least_90_percent_gt_area_inside_one_roi',
        candidate_burden=burden(counts),group_burden=burden(groups),groups_covering_multiple_matched_gt=collision_groups,
        results=result,failures=failures)


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records=json.loads(a.candidates.read_text());coco=json.loads(a.annotations.read_text())
    images={x['id']:x for x in coco['images']}
    if records and 'original_boxes' not in records[0]:
        # COCO-format detections are a legacy post-NMS artifact, never raw proposals.
        grouped={i:dict(image_id=i,original_boxes=[],original_scores=[]) for i in images}
        for det in records:
            x,y,w,h=det['bbox'];g=grouped[det['image_id']]
            g['original_boxes'].append([x,y,x+w,y+h]);g['original_scores'].append(det['score'])
        records=list(grouped.values())
    if len({r['image_id'] for r in records})!=len(records):raise ValueError('Duplicate image IDs')
    if set(images)!={r['image_id'] for r in records}:raise ValueError('Require complete split')
    # Whitelist metadata. No filenames, patient tags, FDI, ages or GT fields enter generation.
    cases=[build_case({k:images[r['image_id']][k] for k in ['id','width','height']},r) for r in records]
    image_index={};requests=[]
    for index,case in enumerate(cases):
        ref=f'image_{index:06d}';case['image_ref']=ref
        image_index[ref]=dict(image_id=case['image_id'],relative_path=images[case['image_id']]['file_name'])
        w=case['image_size']['width'];h=case['image_size']['height']
        for policy in ['risk','grid','random','spread']:
            for j,b in enumerate(select_regions(case,policy,8)):
                requests.append(dict(evidence_id=f'{ref}_{policy}_{j}',image_ref=ref,policy=policy,rank=j+1,
                    bbox_xyxy=[max(0,int(np.floor(b[0]*w))),max(0,int(np.floor(b[1]*h))),min(w,int(np.ceil(b[2]*w))),min(h,int(np.ceil(b[3]*h)))],
                    use_original_image_pixels=True))
    report=evaluate(cases,coco);report['input_sha256']=hashlib.sha256(a.candidates.read_bytes()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'review_interface.json').write_text(json.dumps(cases))
    (a.output/'evaluation.json').write_text(json.dumps(report,indent=2))
    (a.output/'image_index.json').write_text(json.dumps(image_index))
    (a.output/'crop_requests.json').write_text(json.dumps(requests))
    print('STAGE12_OK',len(cases),json.dumps(report['candidate_burden']),flush=True)


if __name__=='__main__':main()
