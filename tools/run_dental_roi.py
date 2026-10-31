"""Cache frozen YOLO FPN ROIs, train a small set head, and evaluate reversible additions."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from dental_roi_head import ToothROIHead,extract_rois,make_regions,roi_targets,set_loss
from dental_roi_sampling import sample_rois,strata


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def cache(a):
    from ultralytics import YOLO
    import ultralytics
    assert ultralytics.__version__=='8.4.191'
    assert torch.cuda.is_available() or a.device=='cpu'
    source=a.data_root/'fdi53'/f'{a.split}.json'
    coco=json.loads(source.read_text());anns={im['id']:[] for im in coco['images']}
    for ann in coco['annotations']:
        if not ann.get('iscrowd',0) and not ann.get('ignore',False):anns[ann['image_id']].append(ann)
    model=YOLO(str(a.checkpoint));assert len(model.names)==1
    captured={}
    def hook(module,args):captured['features']=[f.detach() for f in args[0]]
    handle=model.model.model[-1].register_forward_pre_hook(hook)
    a.output.mkdir(parents=True,exist_ok=False)
    entries=[];total_gt=covered_gt=overflows=0
    images=coco['images'][:a.limit_images] if a.limit_images else coco['images']
    try:
        for im in images:
            with torch.inference_mode():
                result=model.predict(str(a.data_root/im['file_name']),imgsz=960,rect=False,
                    conf=.01,iou=.85,max_det=500,device=a.device,verbose=False)[0]
                boxes=result.boxes.xyxy;scores=result.boxes.conf
                regions,seeds=make_regions(boxes,scores,im['width'],im['height'])
                gt=boxes.new_tensor([x['bbox'] for x in anns[im['id']]]).reshape(-1,4)
                gt[:,2:]+=gt[:,:2]
                target,partial,overflow,covered=roi_targets(regions,seeds,gt)
                features=extract_rois(captured['features'],regions,im['width'],im['height'],roi_size=a.roi_size)
                channels=features.shape[1]
                item=dict(image_id=im['id'],file_name=im['file_name'],width=im['width'],height=im['height'],
                    features=features.cpu().half(),regions=regions.cpu(),seeds=seeds.cpu(),
                    targets=[x.cpu() for x in target],partial=[x.cpu() for x in partial],overflow=overflow,
                    boxes=boxes.cpu(),scores=scores.cpu())
                file=f'{im["id"]}.pt';torch.save(item,a.output/file);entries.append(file)
                total_gt+=len(gt);covered_gt+=covered;overflows+=sum(overflow)
            print('CACHE',im['id'],'ROIs',len(regions),'GT covered',covered,'/',len(gt),flush=True)
    finally:handle.remove()
    manifest=dict(schema=1,split=a.split,annotation_sha256=digest(source),detector_sha256=digest(a.checkpoint),
        channels=channels,entries=entries,images=len(entries),partial_dataset=bool(a.limit_images),
        region=dict(limit=64,expansion=1.6,score=.05),capacity=4,roi_size=a.roi_size,imgsz=960,
        total_gt=total_gt,gt_covered_by_nonoverflow_rois=covered_gt,overflow_rois=overflows)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


def geometry(item,indices,device):
    scale=torch.tensor([item['width'],item['height']]*2,device=device)
    return item['regions'][indices].to(device)/scale


def box_context(item,indices,device):
    """Eight nearby predicted boxes per ROI, never GT-conditioned at inference."""
    contexts=[];existing=[]
    boxes=item['boxes'][item['scores']>=.05]
    scores=item['scores'][item['scores']>=.05]
    for i in indices:
        r=item['regions'][i];size=(r[2:]-r[:2]).clamp(min=1)
        local=(boxes-r[:2].repeat(2))/size.repeat(2)
        distance=((local[:,:2]+local[:,2:])/2-.5).square().sum(1)
        order=distance.argsort()[:8]
        values=torch.cat([local[order].clamp(-2,3),scores[order,None]],1)
        values=torch.cat([values,values.new_zeros((8-len(values),5))])
        contexts.append(values);existing.append(local.to(device))
    return torch.stack(contexts).to(device),existing


def train(a):
    manifest=json.loads((a.cache/'manifest.json').read_text())
    assert manifest['split']=='train', 'Never train on validation or test cache'
    random.seed(a.seed);torch.manual_seed(a.seed)
    options=dict(roi_size=manifest['roi_size'],conditioned=a.conditioned,quality=a.quality)
    model=ToothROIHead(manifest['channels'],slots=1 if a.mode=='single' else 4,**options).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    a.output.mkdir(parents=True,exist_ok=False)
    entries=manifest['entries'][:a.limit_images] if a.limit_images else manifest['entries']
    for epoch in range(1,a.epochs+1):
        random.shuffle(entries);model.train();losses=[];skipped=0
        sampled=dict(hard=0,multi=0,ordinary=0)
        for filename in entries:
            item=load(a.cache/filename)
            valid=[i for i,x in enumerate(item['overflow']) if not x]
            if not valid:skipped+=1;continue
            indices=sample_rois(item,getattr(a,'sampling','uniform'))
            pools=strata(item)
            for key,values in pools.items():sampled[key]+=len(set(indices)&set(values))
            features=item['features'][indices].float().to(a.device)
            targets=[item['targets'][i].to(a.device) for i in indices]
            partial=[item['partial'][i].to(a.device) for i in indices]
            context,existing=box_context(item,indices,a.device)
            pred=model(features,geometry(item,indices,a.device),context)
            loss=set_loss(pred,targets,partial,[False]*len(indices),a.mode,existing)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite ROI loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        assert losses,'No eligible ROIs'
        record=dict(epoch=epoch,loss=float(np.mean(losses)),images=len(losses),skipped=skipped)
        with (a.output/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        with (a.output/'sampling_audit.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(epoch=epoch,policy=getattr(a,'sampling','uniform'),selected_rois=sampled))+'\n')
        print(record,flush=True)
        # Fixed final epoch checkpoint, no test-driven or manual best-epoch selection.
        torch.save(dict(state_dict=model.state_dict(),mode=a.mode,channels=manifest['channels'],
            epoch=epoch,seed=a.seed,cache_manifest=manifest,learning_rate=1e-4,
            sampling=getattr(a,'sampling','uniform'),head_options=options,parameters=sum(p.numel() for p in model.parameters())),a.output/'last.pt')


def evaluate(a):
    sys.path.insert(0,str(Path(__file__).parent/'analysis_tools'))
    from analyze_dental_candidates import analyze
    from evaluate_dental_candidate_file import evaluate as candidate_evaluate
    manifest=json.loads((a.cache/'manifest.json').read_text());checkpoint=load(a.head)
    assert manifest['split']!='train'
    assert manifest['detector_sha256']==checkpoint['cache_manifest']['detector_sha256']
    source=a.data_root/'fdi53'/f'{manifest["split"]}.json'
    assert digest(source)==manifest['annotation_sha256']
    options=checkpoint.get('head_options',dict(roi_size=7,conditioned=False,quality=False))
    assert options['roi_size']==manifest['roi_size'],'ROI cache resolution differs from trained head'
    model=ToothROIHead(checkpoint['channels'],1 if checkpoint['mode']=='single' else 4,**options).to(a.device)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    original=[];combined=[];records=[]
    for filename in manifest['entries']:
        item=load(a.cache/filename);n=len(item['regions'])
        boxes=item['boxes'].numpy();scores=item['scores'].numpy()
        def pack(b,s):return dict(img_id=item['image_id'],pred_instances=dict(bboxes=b,scores=s,proposal_ids=np.arange(len(s))))
        original.append(pack(boxes,scores));added=[]
        if n:
            with torch.no_grad():
                context,_=box_context(item,list(range(n)),a.device)
                pred=model(item['features'].float().to(a.device),geometry(item,list(range(n)),a.device),context)
            for roi_id,(region,local,confidence) in enumerate(zip(item['regions'],pred['boxes'].cpu(),pred['logits'].sigmoid().cpu())):
                for slot_id,(box,score) in enumerate(zip(local,confidence)):
                    if score>=a.threshold:
                        restored=box*(region[2:]-region[:2]).repeat(2)+region[:2].repeat(2)
                        q=float(pred['quality'][roi_id,slot_id].sigmoid()) if 'quality' in pred else None
                        novelty=float(pred['novelty'][roi_id,slot_id].sigmoid()) if 'novelty' in pred else None
                        added.append(dict(bbox_xyxy=restored.tolist(),score=float(score),region=region.tolist(),
                            roi_id=roi_id,slot_id=slot_id,quality=q,novelty=novelty,source='roi_head'))
        b=np.array([x['bbox_xyxy'] for x in added],float).reshape(-1,4);s=np.array([x['score'] for x in added])
        combined.append(pack(np.concatenate([boxes,b]),np.concatenate([scores,s])))
        records.append(dict(image_id=item['image_id'],original_boxes=boxes.tolist(),
                            original_scores=scores.tolist(),additions=added))
    coco=json.loads(source.read_text());ids={x['img_id'] for x in original}
    coco['images']=[x for x in coco['images'] if x['id'] in ids]
    coco['annotations']=[x for x in coco['annotations'] if x['image_id'] in ids]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'metrics.json').write_text(json.dumps(dict(mode=checkpoint['mode'],threshold=a.threshold,
        images=len(ids),partial_dataset=manifest['partial_dataset'],epoch=checkpoint['epoch'],
        parameters=checkpoint['parameters'],baseline=analyze(original,coco),
        recovery=candidate_evaluate(combined,coco)),indent=2))
    (a.output/'candidates.json').write_text(json.dumps(records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cache','train','eval'])
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets/private_only'))
    p.add_argument('--checkpoint',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--head',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',default='train',choices=['train','val','internal_test','external_test'])
    p.add_argument('--mode',choices=['single','multi'],default='multi');p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threshold',type=float,default=.25)
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit-images',type=int,default=0)
    p.add_argument('--roi-size',type=int,choices=[7,14],default=7)
    p.add_argument('--conditioned',action='store_true');p.add_argument('--quality',action='store_true')
    p.add_argument('--sampling',choices=['uniform','stratified'],default='uniform')
    a=p.parse_args();assert a.epochs>0 and a.limit_images>=0 and 0<a.threshold<1
    # Native YOLO device argument accepts numeric CUDA IDs, head training accepts torch devices.
    if a.action=='cache' and a.device=='cuda:0':a.device='0'
    dict(cache=cache,train=train,eval=evaluate)[a.action](a)
