"""Validation-only, existing MMCV Soft-NMS control on a trained checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mmcv.ops import soft_nms
from ultralytics import YOLO

from audit_yolo_suppression import AuditPredictor
from validate_yolo_operating_points import measure


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets/private_only'))
    args=p.parse_args()
    coco=json.loads((args.data_root/'fdi53/val.json').read_text())
    model=YOLO(str(args.checkpoint)); predictions=[]
    for image in coco['images']:
        model.predict(str(args.data_root/image['file_name']),predictor=AuditPredictor,
            device=0,imgsz=960,rect=False,conf=.001,iou=.6,max_det=500,verbose=False)
        raw=model.predictor.audit
        ids=torch.where(raw['scores']>.001)[0]
        dets,keep=soft_nms(raw['network_boxes'][ids].float().cpu(),raw['scores'][ids].float().cpu(),
            iou_threshold=.6,sigma=.5,min_score=.001,method='linear')
        original=ids[keep.to(ids.device)[:500]]
        predictions.append((image,dict(bboxes=raw['boxes'][original].cpu().numpy(),scores=dets[:500,4].numpy())))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(method='MMCV linear Soft-NMS',iou=.6,
        results=measure(predictions,coco,.6)),indent=2))
    print('SOFT_NMS_OK',len(predictions),flush=True)


if __name__=='__main__':main()
