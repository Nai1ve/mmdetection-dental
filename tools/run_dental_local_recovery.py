"""Fit a train-only region prior, or evaluate frozen YOLO local recovery."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from dental_local_recovery import fit_spatial_prior, recover, validate_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--fit-prior',type=Path)
    p.add_argument('--prior',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--split',choices=['val','internal_test','external_test'],default='val')
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--device',default='0')
    p.add_argument('--limit-images',type=int,default=0,help='Smoke subset only; zero evaluates the complete split')
    a=p.parse_args()
    assert a.limit_images>=0
    if a.fit_prior:
        prior=fit_spatial_prior(a.data_root/'fdi53/train.json')
        with a.fit_prior.open('x') as stream:
            json.dump(prior,stream,indent=2)
        return
    if not all([a.config,a.checkpoint,a.output]):
        p.error('Evaluation needs --config, --checkpoint and --output')
    import torch
    import ultralytics
    from ultralytics import YOLO
    assert ultralytics.__version__=='8.4.191'
    assert torch.cuda.is_available() or a.device=='cpu'
    config=json.loads(a.config.read_text());validate_config(config)
    prior=json.loads(a.prior.read_text()) if a.prior else None
    if prior is not None:
        import hashlib
        assert prior['source_sha256']==hashlib.sha256((a.data_root/'fdi53/train.json').read_bytes()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=False)
    model=YOLO(str(a.checkpoint))
    if len(model.names)!=1:
        raise ValueError('Only single-class Tooth checkpoints are supported')
    def predict(image):
        result=model.predict(image,imgsz=a.imgsz,device=a.device,conf=config['conf'],
            iou=config['nms_iou'],rect=False,max_det=500,verbose=False)[0]
        return result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()
    sys.path.insert(0,str(Path(__file__).parent/'analysis_tools'))
    from analyze_dental_candidates import analyze
    from evaluate_dental_candidate_file import evaluate
    from validate_dental_interface import group_candidates
    coco=json.loads((a.data_root/'fdi53'/f'{a.split}.json').read_text())
    if a.limit_images:
        coco['images']=coco['images'][:a.limit_images]
        ids={im['id'] for im in coco['images']}
        coco['annotations']=[ann for ann in coco['annotations'] if ann['image_id'] in ids]
    all_predictions,full_predictions,provenance=[],[],[]
    timings=[]
    for im in coco['images']:
        with Image.open(a.data_root/im['file_name']) as source:
            image=np.asarray(source.convert('RGB'))[...,::-1].copy()
        start=time.perf_counter();records=recover(image,predict,config,prior)
        if a.device!='cpu':torch.cuda.synchronize()
        timings.append(time.perf_counter()-start)
        def pack(rows):
            return dict(img_id=im['id'],pred_instances=dict(
                bboxes=np.array([r['bbox_xyxy'] for r in rows],float).reshape(-1,4),
                scores=np.array([r['score'] for r in rows]),proposal_ids=np.arange(len(rows))))
        combined=pack(records);all_predictions.append(combined)
        full_predictions.append(pack([r for r in records if r['source']=='full']))
        groups=group_candidates(combined['pred_instances'],.85)
        provenance.append(dict(image_id=im['id'],candidates=records,
            provisional_groups=[dict(representative=i,members=members) for i,members in groups]))
        print(im['id'],len(records),flush=True)
    reports=dict(config=config,checkpoint=str(a.checkpoint.resolve()),split=a.split,
        partial_evaluation=bool(a.limit_images),images=len(coco['images']),
        prior=prior,full=analyze(full_predictions,coco),recovery=evaluate(all_predictions,coco),
        calls_per_image=1+config['crop_count']*len(config['rotations']),
        seconds=dict(mean=float(np.mean(timings)),p95=float(np.percentile(timings,95)),
                     includes_first_call_warmup=True),
        note='Raw recovery is a superset; grouped representatives may lose recall. No global cross-view NMS.')
    reports['candidate_burden']={}
    for label,values in dict(full=[len(p['pred_instances']['scores']) for p in full_predictions],
        raw=[len(p['candidates']) for p in provenance],grouped=[len(p['provisional_groups']) for p in provenance]).items():
        reports['candidate_burden'][label]=dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=max(values))
    (a.output/'metrics.json').write_text(json.dumps(reports,indent=2))
    (a.output/'candidates.json').write_text(json.dumps(provenance))


if __name__=='__main__':main()
