"""Read-only dataset and archived prediction audit; no training or clinical inference."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from analyze_dental_candidates import overlaps


def xyxy(rows):
    a = np.asarray([r['bbox'] for r in rows], float).reshape(-1, 4)
    a[:, 2:] += a[:, :2]
    return a


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(data, predictions, out):
    out.mkdir(parents=True, exist_ok=False)
    issues, review, error_rows = [], [], []
    inventory, hashes, identities = {}, defaultdict(list), defaultdict(list)
    datasets = {}
    for split in ['train', 'val', 'internal_test', 'external_test']:
        coco = json.loads((data / 'fdi53' / (split + '.json')).read_text())
        datasets[split] = coco
        images = {im['id']: im for im in coco['images']}
        categories = {c['id']: c['name'] for c in coco['categories']}
        assert len(images) == len(coco['images']) and len(categories) == len(coco['categories'])
        assert len({a['id'] for a in coco['annotations']}) == len(coco['annotations'])
        annotations = defaultdict(list)
        for ann in coco['annotations']:
            assert ann['image_id'] in images and ann['category_id'] in categories
            annotations[ann['image_id']].append(ann)
        sizes, extras = [], []
        for iid, im in images.items():
            file = data / im['file_name']
            tags = dict(split=split, image_id=iid, file_name=im['file_name'])
            if not file.exists():
                issues.append(dict(**tags, kind='missing_image', detail=''))
                continue
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            hashes[digest].append(tags)
            if im.get('image_sha256') != digest:
                issues.append(dict(**tags, kind='sha_metadata_mismatch', detail=digest))
            with Image.open(file) as image:
                if image.size != (im['width'], im['height']):
                    issues.append(dict(**tags, kind='size_mismatch', detail=str(image.size)))
                image.verify()
            for key in ['record_id', 'split_group_id']:
                if im.get(key):
                    identities[key, im[key]].append(tags)
            anns = annotations[iid]
            sizes.append(len(anns))
            extra = [a for a in anns if categories[a['category_id']] == 'extra_tooth']
            extras.extend(extra)
            seen = Counter()
            for ann in anns:
                x, y, w, h = ann['bbox']
                if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
                    issues.append(dict(**tags, kind='invalid_bbox', detail=str(ann['id'])))
                elif x < -.001 or y < -.001 or x+w > im['width']+.001 or y+h > im['height']+.001:
                    issues.append(dict(**tags, kind='out_of_bounds_bbox', detail=str(ann['id'])))
                if abs(ann.get('area', w*h)-w*h) > max(.001, abs(w*h)*1e-5):
                    issues.append(dict(**tags, kind='area_mismatch', detail=str(ann['id'])))
                key = (ann['category_id'], tuple(ann['bbox']))
                seen[key] += 1
            if any(v > 1 for v in seen.values()):
                issues.append(dict(**tags, kind='exact_duplicate_annotation', detail=''))
            fdi = Counter(categories[a['category_id']] for a in anns if categories[a['category_id']] != 'extra_tooth')
            for name, count in fdi.items():
                if count > 1:
                    review.append(dict(**tags, kind='repeated_fdi', annotation_ids=str([a['id'] for a in anns if categories[a['category_id']] == name]), detail=name))
            b = xyxy(anns)
            pair_iou = overlaps(b, b)
            for i, j in zip(*np.where(np.triu(pair_iou >= .9, 1))):
                review.append(dict(**tags, kind='high_iou_gt_pair', annotation_ids=f"{anns[i]['id']},{anns[j]['id']}", detail=str(float(pair_iou[i,j]))))
        tooth = json.loads((data / 'tooth1' / (split + '.json')).read_text())
        assert len(tooth['annotations']) == len(coco['annotations'])
        assert tooth['images'] == coco['images'], 'FDI/tooth1 image metadata drift'
        assert {c['name'] for c in tooth['categories']} == {'tooth'}
        assert all(a['id'] == b['id'] and a['image_id'] == b['image_id'] and a['bbox'] == b['bbox']
                   for a,b in zip(coco['annotations'], tooth['annotations']))
        inventory[split] = dict(images=len(images), teeth=len(coco['annotations']),
            extra_teeth=len(extras), extra_images=len({a['image_id'] for a in extras}),
            centers=dict(Counter(im.get('domain_id', 'unknown') for im in images.values())),
            teeth_per_image=dict(mean=float(np.mean(sizes)), min=min(sizes), max=max(sizes)),
            tooth1_geometry_identical=True)
        print('DATA', split, inventory[split], flush=True)
    duplicates = [v for v in hashes.values() if len(v) > 1]
    cross_ids = [dict(field=k[0], value=k[1], occurrences=v) for k,v in identities.items()
                 if len({t['split'] for t in v}) > 1]
    summaries = []
    for split in ['val', 'internal_test']:
        coco = datasets[split]
        gt = defaultdict(list)
        for a in coco['annotations']:
            if not a.get('iscrowd', 0) and not a.get('ignore', False):
                gt[a['image_id']].append(a)
        extra_ids = {c['id'] for c in coco['categories'] if c['name'] == 'extra_tooth'}
        image_ids = {im['id'] for im in coco['images']}
        methods = ['baseline', 'control', 'direction', 'direction_balance'] if split == 'val' else ['baseline']
        for method in methods:
            for seed in [42,43,44]:
                path = predictions / f'yolo_structure_core_20261030_0940_{method}_s{seed}' / 'eval_nms085' / split / 'detections.json'
                pred = defaultdict(list)
                for p in json.loads(path.read_text()):
                    assert p['image_id'] in image_ids
                    assert np.isfinite(p['bbox'] + [p['score']]).all()
                    if p['score'] >= .01:
                        pred[p['image_id']].append(p)
                summary = Counter()
                burdens = []
                for im in coco['images']:
                    anns, ps = gt[im['id']], pred[im['id']]
                    b, pb = xyxy(anns), xyxy(ps)
                    iou = overlaps(b, pb)
                    match = {}
                    for threshold in [.5,.75]:
                        valid = iou >= threshold
                        ii,jj = linear_sum_assignment(-(valid*(len(b)+1)+iou))
                        match[threshold] = {int(i):int(j) for i,j in zip(ii,jj) if valid[i,j]}
                    burdens.append(len(ps))
                    normal = [i for i,a in enumerate(anns) if a['category_id'] not in extra_ids]
                    for i,a in enumerate(anns):
                        is_extra = a['category_id'] in extra_ids
                        candidates = [j for j in normal if j != i]
                        inter = np.maximum(np.minimum(b[i,2:],b[candidates,2:])-np.maximum(b[i,:2],b[candidates,:2]),0).prod(1)
                        ioa = inter / np.maximum((b[i,2:]-b[i,:2]).prod(),1e-10)
                        neighbor = candidates[int(ioa.argmax())] if candidates else None
                        max_ioa = float(ioa.max()) if candidates else 0.
                        state = 'strict_match' if i in match[.75] else 'coarse_only' if i in match[.5] else 'no_match05'
                        cohort = 'extra' if is_extra else 'normal'
                        summary[cohort+'_'+state] += 1
                        summary[cohort+('_ioa_gt05' if max_ioa>.5 else '_ioa_le05')+'_'+state] += 1
                        if not is_extra:
                            continue
                        j = match[.5].get(i)
                        row = dict(split=split, method=method, seed=seed, image_id=im['id'], annotation_id=a['id'],
                            file_name=im['file_name'], state=state, gt_bbox=json.dumps(a['bbox']),
                            neighbor_annotation_id=anns[neighbor]['id'] if neighbor is not None else '',
                            max_neighbor_ioa=max_ioa, best_retained_iou=float(iou[i].max()) if len(ps) else 0.,
                            match_conflict05=bool(j is None and len(ps) and iou[i].max()>=.5),
                            selected_bbox='', selected_iou='', score='', area_ratio='', center_drift_to_neighbor='',
                            dx1='',dy1='',dx2='',dy2='', review_label='', review_notes='')
                        if j is not None:
                            w,h = b[i,2:]-b[i,:2]
                            delta = (pb[j]-b[i]) / np.array([w,h,w,h])
                            center_delta = (pb[j,:2]+pb[j,2:]-b[i,:2]-b[i,2:])/2
                            neighbor_delta = (b[neighbor,:2]+b[neighbor,2:]-b[i,:2]-b[i,2:])/2 if neighbor is not None else np.zeros(2)
                            drift = float(center_delta@neighbor_delta / max(np.linalg.norm(neighbor_delta)*np.sqrt(w*h),1e-10))
                            row.update(selected_bbox=json.dumps(ps[j]['bbox']), selected_iou=float(iou[i,j]),
                                score=ps[j]['score'], area_ratio=float((pb[j,2:]-pb[j,:2]).prod()/(w*h)),
                                center_drift_to_neighbor=drift, **dict(zip(['dx1','dy1','dx2','dy2'],delta.tolist())))
                        error_rows.append(row)
                summaries.append(dict(split=split, method=method, seed=seed, counts=dict(summary),
                    candidate_burden=dict(mean=float(np.mean(burdens)),p95=float(np.percentile(burdens,95)),max=max(burdens))))
    write_csv(out/'annotation_issues.csv',issues)
    write_csv(out/'annotation_review.csv',review)
    write_csv(out/'extra_tooth_review.csv',error_rows)
    result = dict(inventory=inventory, exact_byte_duplicate_groups=duplicates,
        cross_split_record_or_group_ids=cross_ids, annotation_issue_counts=dict(Counter(r['kind'] for r in issues)),
        annotation_review_counts=dict(Counter(r['kind'] for r in review)), prediction_summaries=summaries,
        boundaries=['No patient identity verification', 'No perceptual near-duplicate detection',
                    'Post-NMS predictions only; missing low-score/NMS causes not inferable',
                    'Internal-test error inspection is exploratory, not an untouched confirmation',
                    'Geometry review flags are not clinical annotation errors'])
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('ISSUES',result['annotation_issue_counts'],'REVIEW',result['annotation_review_counts'],
          'DUPLICATE_GROUPS',len(duplicates),'CROSS_IDS',len(cross_ids),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.data,a.predictions,a.output)
