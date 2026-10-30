"""Single-detector local recovery with training-only spatial hints and reversible candidates."""
import hashlib
import json

import numpy as np


def fit_spatial_prior(train_path):
    content = train_path.read_bytes()
    coco = json.loads(content)
    images = {i['id']: i for i in coco['images']}
    extra = {c['id'] for c in coco['categories'] if c['name'] == 'extra_tooth'}
    per_image = {}
    for a in coco['annotations']:
        if a['category_id'] in extra and not a.get('iscrowd', 0) and not a.get('ignore', False):
            per_image.setdefault(a['image_id'], []).append(a)
    points = []
    for iid, annotations in per_image.items():
        im = images[iid]
        for a in annotations:
            x, y, w, h = a['bbox']
            points.append([(x+w/2)/im['width'], (y+h/2)/im['height'], 1/len(annotations)])
    if not points:
        raise ValueError('Training annotations have no extra_tooth instances')
    return dict(schema=1, source_sha256=hashlib.sha256(content).hexdigest(),
                training_file=str(train_path), images=len(per_image), instances=len(points), points=points)


def validate_config(config):
    assert set(config) == {'region_policy', 'crop_count', 'crop_fraction', 'rotations', 'conf', 'nms_iou'}
    assert config['region_policy'] in ['uniform', 'spatial']
    assert type(config['crop_count']) is int and 0 <= config['crop_count'] <= 4
    assert len(config['crop_fraction']) == 2 and all(0 < x <= 1 for x in config['crop_fraction'])
    assert config['rotations'] and len(set(config['rotations'])) == len(config['rotations'])
    assert all(x in [0,90,180,270] for x in config['rotations'])
    assert 0 < config['conf'] < 1 and 0 < config['nms_iou'] <= 1


def select_regions(width, height, config, prior=None):
    validate_config(config)
    count = config['crop_count']
    if not count:
        return []
    fw, fh = config['crop_fraction']
    cw, ch = max(1, round(width*fw)), max(1, round(height*fh))
    centers = [(x,y) for y in [fh/2,.5,1-fh/2] for x in [fw/2,.5,1-fw/2]]
    if config['region_policy'] == 'spatial':
        if prior is None:
            raise ValueError('Spatial policy requires a training-only prior')
        points = np.asarray(prior['points'])
        values = [float((np.exp(-.5*((points[:,0]-x)/.16)**2-.5*((points[:,1]-y)/.20)**2)*points[:,2]).sum()) for x,y in centers]
        order = sorted(range(9), key=lambda i: (-values[i],i))
    else:
        order = [0,2,6,8,4,1,3,5,7]
    result = []
    for i in order:
        x,y = centers[i]
        x1 = min(max(round(x*width-cw/2),0),width-cw)
        y1 = min(max(round(y*height-ch/2),0),height-ch)
        rect = (x1,y1,x1+cw,y1+ch)
        if rect not in result:
            result.append(rect)
        if len(result) == count:
            break
    return result


def invert_boxes(boxes, width, height, rotation, origin=(0,0)):
    boxes = np.asarray(boxes, dtype=float).reshape(-1,4)
    corners = np.stack([boxes[:,[0,1]],boxes[:,[0,3]],boxes[:,[2,1]],boxes[:,[2,3]]],1)
    x,y = corners[...,0].copy(),corners[...,1].copy()
    if rotation == 90:
        x,y = width-y,x
    elif rotation == 180:
        x,y = width-x,height-y
    elif rotation == 270:
        x,y = y,height-x
    elif rotation != 0:
        raise ValueError(rotation)
    x = x.clip(0,width)+origin[0]
    y = y.clip(0,height)+origin[1]
    return np.stack([x.min(1),y.min(1),x.max(1),y.max(1)],1)


def recover(image, predict, config, prior=None):
    """predict accepts BGR ndarray and returns xyxy/scores; no GT is passed here."""
    validate_config(config)
    height,width = image.shape[:2]
    records = []
    full_boxes,full_scores = predict(image)
    # Keep the full-frame prefix unchanged, including scores and original coordinates.
    for box,score in zip(full_boxes,full_scores,strict=True):
        records.append(dict(bbox_xyxy=np.asarray(box).tolist(), score=float(score),
                            source='full', region=None, rotation=0, crop_boundary=False))
    for region in select_regions(width,height,config,prior):
        x1,y1,x2,y2 = region
        crop = image[y1:y2,x1:x2]
        for rotation in config['rotations']:
            view = np.ascontiguousarray(np.rot90(crop,rotation//90))
            boxes,scores = predict(view)
            restored = invert_boxes(boxes,x2-x1,y2-y1,rotation,(x1,y1))
            for box,score in zip(restored,scores,strict=True):
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                boundary = min(box[0]-x1,box[1]-y1,x2-box[2],y2-box[3]) < 2
                records.append(dict(bbox_xyxy=box.tolist(),score=float(score),source='local',
                    region=list(region),rotation=rotation,crop_boundary=bool(boundary)))
    return records
