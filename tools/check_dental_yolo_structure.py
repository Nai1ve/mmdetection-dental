"""CPU smoke/invariant checks; not a substitute for GPU training validation."""
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import argparse
from functools import partial

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils.loss import v8DetectionLoss

from dental_yolo_structure import (DentalDetectionModel, DirectionAdapter,
    DentalQualityPredictor, LocalRelationAdapter, attach_adapters,
    DentalDetectionTrainer, log_structure_audit)
from dental_yolo_ablation import log_assignment


def check():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for channels in [32, 256, 512]:
        control, direction = DirectionAdapter(channels, True), DirectionAdapter(channels)
        assert sum(p.numel() for p in control.parameters()) == sum(p.numel() for p in direction.parameters())
        for block in [control, direction, LocalRelationAdapter(channels)]:
            x = torch.randn(2, channels, 7, 9, requires_grad=True)
            y = block(x)
            assert y.shape == x.shape and y.isfinite().all()
            y.square().mean().backward()
            assert all(p.grad is not None and p.grad.isfinite().all() for p in block.parameters())
    configs = Path(__file__).resolve().parents[1] / 'configs/dental_yolo_structure'
    for path in sorted(configs.glob('*.json')):
        options = json.loads(path.read_text())
        # Nano uses the same audited layer layout; medium is separately instantiated below.
        model = DentalDetectionModel('yolo11n.yaml', nc=1, verbose=False)
        native = deepcopy(model).eval()
        model = attach_adapters(model, options)
        model.args = get_cfg()
        model.train()
        batch = dict(img=torch.rand(2, 3, 64, 64), batch_idx=torch.tensor([0., 1.]),
                     cls=torch.zeros(2, 1), bboxes=torch.tensor([[.4, .4, .3, .3], [.6, .5, .2, .4]]))
        losses, items = model(batch)
        if path.stem == 'baseline':
            native.args = get_cfg()
            with torch.no_grad():
                predictions = model(batch['img'])
                native_loss = v8DetectionLoss(native)(predictions, batch)[0]
                custom_loss = model.criterion(predictions, batch)[0]
            torch.testing.assert_close(native_loss, custom_loss, rtol=0, atol=0)
        assert losses.isfinite().all()
        losses.sum().backward()
        new_parameters = [p for name, p in model.named_parameters() if '.adapter.' in name or '.quality.' in name]
        assert all(p.grad is not None and p.grad.isfinite().all() for p in new_parameters)
        model.zero_grad(set_to_none=True)
        empty = dict(batch, batch_idx=torch.empty(0), cls=torch.empty(0, 1), bboxes=torch.empty(0, 4))
        loss_empty, _ = model(empty)
        assert loss_empty.isfinite().all()
        loss_empty.sum().backward()
        model.eval()
        with torch.no_grad():
            before = model(batch['img'])[0]
        if options['feature'] == 'none' and not options['relation']:
            # Quality additions must not change decoded boxes/scores before training.
            # Training forward updated BN stats, so compare all native shared tensors.
            native_state = native.state_dict()
            native.load_state_dict({k: v for k, v in model.state_dict().items() if k in native_state})
            with torch.no_grad():
                assert torch.equal(before, native(batch['img'])[0])
        model.names = {0: 'tooth'}
        # Match native checkpoint semantics: criterion isn't retained by EMA checkpoints.
        del model.criterion
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'model.pt'
            torch.save(dict(model=model, train_args={'task': 'detect'}), checkpoint)
            loaded = YOLO(str(checkpoint))
            with torch.no_grad():
                after = loaded.model(batch['img'])[0]
            assert torch.equal(before, after), path.name
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            predictor = DentalQualityPredictor if options['quality'] else DetectionPredictor
            results = loaded.predict(image, predictor=predictor, imgsz=64, conf=.000001, max_det=50,
                                     device='cpu', verbose=False)
            if options['quality']:
                assert len(results[0].dental_quality) == len(results[0].boxes)
                assert len(results[0].boxes) > 0, 'Quality index mapping must be tested on nonempty detections'
                assert results[0].dental_quality.isfinite().all()
                with torch.no_grad():
                    raw_quality = loaded.model(torch.zeros(1, 3, 64, 64))[1]['quality_logits'].sigmoid()[0, 0]
                torch.testing.assert_close(results[0].dental_quality,
                    raw_quality[results[0].dental_anchor_indices])
                loaded.predictor = None
                normal = loaded.predict(image, predictor=DetectionPredictor, imgsz=64, conf=.000001,
                                        max_det=50, device='cpu', verbose=False)
                assert torch.equal(results[0].boxes.data, normal[0].boxes.data)
                loaded.predictor = None
                empty_result = loaded.predict(image, predictor=DentalQualityPredictor,
                    imgsz=64, conf=.99, device='cpu', verbose=False)[0]
                assert len(empty_result.boxes) == len(empty_result.dental_quality) == 0
        print(path.stem, 'forward/backward/empty-GT/save/reload/predict PASS', flush=True)
    medium = DentalDetectionModel('yolo11m.yaml', nc=1, verbose=False)
    attach_adapters(medium, json.loads((configs / 'direction_relation.json').read_text()))
    medium.eval()
    with torch.no_grad():
        assert medium(torch.rand(1, 3, 64, 64))[0].shape[1] == 5
    print('YOLO11m layout and forward PASS', flush=True)


def trainer_smoke():
    from PIL import Image
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for split in ['train', 'val']:
            (root / 'images' / split).mkdir(parents=True)
            (root / 'labels' / split).mkdir(parents=True)
            for i in range(2):
                image = np.full((64, 64, 3), 100 + i * 30, dtype=np.uint8)
                Image.fromarray(image).save(root / 'images' / split / f'{i}.png')
                (root / 'labels' / split / f'{i}.txt').write_text('0 0.5 0.5 0.4 0.4\n')
        data = root / 'data.yaml'
        data.write_text(f'path: {root}\ntrain: images/train\nval: images/val\nnames: [tooth]\n')
        options = dict(feature='direction', relation=True, quality=True,
                       quality_weight=.2, loss_variant='balance')
        model = YOLO('yolo11n.yaml')
        model.add_callback('on_train_epoch_end', log_assignment)
        model.add_callback('on_train_epoch_end', log_structure_audit)
        model.train(trainer=partial(DentalDetectionTrainer, dental_options=options),
            data=str(data), project=str(root), name='smoke', epochs=1, batch=2,
            imgsz=64, device='cpu', workers=0, amp=False, pretrained=False,
            plots=False, save=True, val=True, mosaic=0., optimizer='AdamW')
        work = root / 'smoke'
        for filename in ['weights/best.pt', 'structure_config.json',
                         'assignment_audit.jsonl', 'quality_audit.jsonl', 'results.csv']:
            assert (work / filename).is_file(), filename
        loaded = YOLO(str(work / 'weights/best.pt'))
        assert loaded.model.dental_options == options
        result = loaded.predict(np.zeros((64, 64, 3), dtype=np.uint8),
            predictor=DentalQualityPredictor, imgsz=64, device='cpu', verbose=False)[0]
        assert len(result.dental_quality) == len(result.boxes)
    print('Native trainer/EMA/validation/best-checkpoint/quality-interface PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer-smoke', action='store_true')
    args = parser.parse_args()
    trainer_smoke() if args.trainer_smoke else check()
