"""Train one detector and evaluate original private COCO splits, on GPU only."""
import argparse
import csv
import gc
import json
import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO, RTDETR

sys.path.insert(0, str(Path(__file__).parent/'analysis_tools'))
from evaluate_dental_candidate_file import evaluate


def predict_in_batches(model, paths, batch, **kwargs):
    # A list of paths may become one in-memory batch inside Ultralytics.
    # Bound the source list itself, not only the predictor's batch argument.
    for start in range(0, len(paths), batch):
        yield from model.predict(source=paths[start:start+batch], stream=True, batch=batch, **kwargs)


def train_model(cls, args):
    weights = Path('/root/autodl-tmp/models')/(args.model+'.pt')
    model = cls(str(weights))
    train_options = {}
    if args.structure_config:
        from dental_yolo_structure import DentalDetectionTrainer, log_structure_audit
        from dental_yolo_ablation import log_assignment
        train_options['trainer'] = partial(DentalDetectionTrainer, dental_options=args.structure_options)
        model.add_callback('on_train_epoch_end', log_assignment)
        model.add_callback('on_train_epoch_end', log_structure_audit)
    elif args.loss_variant != 'standard':
        import ultralytics
        assert ultralytics.__version__ == '8.4.191', 'Custom loss requires audited Ultralytics 8.4.191'
        assert args.view == 'tooth1' and args.model == 'yolo11m'
        from dental_yolo_ablation import install, log_assignment
        def before_start(trainer):
            trainer._dental_variant = args.loss_variant
            install(trainer)
        model.add_callback('on_train_start', before_start)
        model.add_callback('on_train_epoch_end', log_assignment)
    model.train(data=str(args.converted/'val.yaml'), project=str(args.work_dir.parent),
                name=args.work_dir.name, exist_ok=False, epochs=1 if args.smoke else args.epochs,
                fraction=.004 if args.smoke else 1., imgsz=args.imgsz,
                batch=args.batch, device=0, workers=4, seed=args.seed, deterministic=False,
                optimizer='AdamW', lr0=.001 if args.model == 'yolo11m' else .0001,
                amp=True, rect=False, mosaic=0., mixup=0., fliplr=0., flipud=0.,
                degrees=0., translate=0., scale=0., shear=0., perspective=0.,
                hsv_h=0., hsv_s=0., hsv_v=0., patience=0, save_period=-1,
                max_det=500, plots=False, **train_options)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['yolo11m', 'rtdetr-l'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--converted', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--view', choices=['fdi53', 'tooth1'], default='fdi53')
    p.add_argument('--smoke', action='store_true', help='One epoch on four training images; no final test')
    p.add_argument('--evaluate-only', action='store_true', help='Use existing best.pt without retraining')
    p.add_argument('--loss-variant', choices=['standard','baseline','balance','hard','combined'], default='standard')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--nms-iou', type=float, default=.6)
    p.add_argument('--eval-suffix', default='')
    p.add_argument('--structure-config', type=Path, help='Audited dental structure JSON, single-class YOLO11m only')
    args = p.parse_args()
    if args.structure_config:
        import ultralytics
        assert ultralytics.__version__ == '8.4.191'
        assert args.model == 'yolo11m' and args.view == 'tooth1'
        assert args.loss_variant == 'standard', 'Specify loss variant in structure config only'
        options = json.loads(args.structure_config.read_text())
        assert set(options) == {'feature', 'relation', 'quality', 'quality_weight', 'loss_variant'}
        assert options['feature'] in ['none', 'control', 'direction']
        assert type(options['relation']) is bool and type(options['quality']) is bool
        assert options['loss_variant'] in ['baseline', 'balance']
        assert 0 <= options['quality_weight'] <= 1
        args.structure_options = options
    assert torch.cuda.is_available(), 'GPU required; refusing CPU training'
    assert args.batch > 0
    assert not (args.smoke and args.evaluate_only)
    cls = YOLO if args.model == 'yolo11m' else RTDETR
    if not args.evaluate_only:
        train_model(cls, args)
        gc.collect()
        torch.cuda.empty_cache()
    if args.smoke:
        with (args.work_dir/'results.csv').open() as stream:
            row = list(csv.DictReader(stream))[-1]
        losses = [float(value) for key, value in row.items() if key.strip().startswith('train/')]
        assert losses and np.isfinite(losses).all(), 'Non-finite smoke-training loss'
        print('GPU_SMOKE_OK', args.model, losses, flush=True)
        return
    # Best is selected by native validation metrics only, never by test performance.
    checkpoint = args.work_dir/'weights'/'best.pt'
    model = cls(str(checkpoint))
    if args.structure_config:
        assert getattr(model.model, 'dental_options', None) == args.structure_options, 'Checkpoint/config mismatch'
    predictor_options = {}
    if getattr(model.model, 'dental_options', {}).get('quality'):
        from dental_yolo_structure import DentalQualityPredictor
        predictor_options['predictor'] = DentalQualityPredictor
    reports = []
    for split in ['val', 'internal_test', 'external_test']:
        source = args.data_root/args.view/f'{split}.json'
        coco = json.loads(source.read_text())
        categories = sorted(coco['categories'], key=lambda x: x['id'])
        paths = [str((args.data_root/image['file_name']).resolve()) for image in coco['images']]
        results = predict_in_batches(model, paths, args.batch, device=0,
                                     imgsz=args.imgsz, rect=False, conf=.01, iou=args.nms_iou,
                                     max_det=500, verbose=False, **predictor_options)
        predictions, detections = [], []
        for image, result in zip(coco['images'], results, strict=True):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)
            predictions.append(dict(img_id=image['id'], pred_instances=dict(
                bboxes=boxes, scores=scores, labels=labels, proposal_ids=np.arange(len(boxes)))))
            qualities = None
            if hasattr(result, 'dental_quality'):
                qualities = result.dental_quality.cpu().numpy()
                predictions[-1]['pred_instances']['localization_quality'] = qualities
                predictions[-1]['pred_instances']['anchor_indices'] = result.dental_anchor_indices.cpu().numpy()
            for index, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = map(float, box)
                detections.append(dict(image_id=image['id'], category_id=categories[int(label)]['id'],
                                       bbox=[x1,y1,x2-x1,y2-y1], score=float(score)))
                if qualities is not None:
                    detections[-1]['localization_quality'] = float(qualities[index])
        out = args.work_dir/('eval'+args.eval_suffix)/split
        out.mkdir(parents=True, exist_ok=True)
        with (out/'predictions.pkl').open('wb') as stream:
            pickle.dump(predictions, stream)
        gt = COCO(str(source))
        gt.dataset.setdefault('info', {})
        if detections:
            evaluator = COCOeval(gt, gt.loadRes(detections), 'bbox')
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            stats = evaluator.stats.tolist()
        else:
            stats = None
        reference = json.loads((args.data_root/'fdi53'/f'{split}.json').read_text())
        reports.append(dict(split=split, annotation_view=args.view, nms_iou=args.nms_iou, coco_stats=stats,
                            candidate_metrics=evaluate(predictions, reference)))
        (out/'detections.json').write_text(json.dumps(detections))
        (out/'metrics.json').write_text(json.dumps(reports[-1], indent=2))
    (args.work_dir/('evaluation_summary'+args.eval_suffix+'.json')).write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
