"""GPU export for E0: retain full decoded pools; replay NMS from identical forward passes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import nms
import ultralytics
from ultralytics import YOLO
from audit_yolo_suppression import AuditPredictor


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['val','internal_test'],default='val');p.add_argument('--device',default='0')
    p.add_argument('--limit-images',type=int,default=0)
    a=p.parse_args();assert ultralytics.__version__=='8.4.191' and a.limit_images>=0
    source=a.data_root/'fdi53'/f'{a.split}.json';coco=json.loads(source.read_text())
    images=coco['images'][:a.limit_images] if a.limit_images else coco['images']
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'raw').mkdir()
    model=YOLO(str(a.checkpoint));assert len(model.names)==1
    outputs={f'conf{conf}_nms{threshold}':[] for conf in [.01,.001,.0001] for threshold in [.6,.85]}
    trace=[]
    for im in images:
        native=model.predict(str(a.data_root/im['file_name']),predictor=AuditPredictor,
            device=a.device,imgsz=960,rect=False,conf=.001,iou=.6,max_det=500,verbose=False)[0]
        raw=model.predictor.audit;boxes=raw['boxes'];scores=raw['scores'];network=raw['network_boxes']
        np.savez_compressed(a.output/'raw'/f'{im["id"]}.npz',boxes=boxes.cpu().numpy(),
            network_boxes=network.cpu().numpy(),scores=scores.cpu().numpy(),raw_indices=np.arange(len(scores)))
        for conf in [.01,.001,.0001]:
            selected=torch.where(scores>conf)[0]
            for threshold in [.6,.85]:
                uncapped=selected[nms(network[selected],scores[selected],threshold)];kept=uncapped[:500]
                if conf==.001 and threshold==.6:
                    np.testing.assert_allclose(boxes[kept].cpu().numpy(),native.boxes.xyxy.cpu().numpy(),atol=1e-3)
                    np.testing.assert_allclose(scores[kept].cpu().numpy(),native.boxes.conf.cpu().numpy(),atol=1e-6)
                key=f'conf{conf}_nms{threshold}'
                outputs[key].append(dict(image_id=im['id'],original_boxes=boxes[kept].cpu().tolist(),
                    original_scores=scores[kept].cpu().tolist(),raw_candidate_indices=kept.cpu().tolist()))
                trace.append(dict(image_id=im['id'],setting=key,above_threshold=len(selected),
                    kept_raw_indices=kept.cpu().tolist(),nms_removed_raw_indices=sorted(set(selected.cpu().tolist())-set(uncapped.cpu().tolist())),
                    cap_removed_raw_indices=uncapped[500:].cpu().tolist()))
        print('RAW_POOL',im['id'],len(scores),flush=True)
    for key,records in outputs.items():(a.output/f'{key}.json').write_text(json.dumps(records))
    (a.output/'trace.json').write_text(json.dumps(trace))
    (a.output/'manifest.json').write_text(json.dumps(dict(images=len(images),partial_dataset=bool(a.limit_images),
        split=a.split,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        annotation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),ultralytics=ultralytics.__version__,
        decoded_pool_is_not_detection=True,native_replay_verified=True,imgsz=960,rect=False,max_det=500),indent=2))


if __name__=='__main__':main()
