"""Geometry and retention invariants without model weights or a GPU."""
import json
from pathlib import Path
import tempfile
import numpy as np
from dental_local_recovery import fit_spatial_prior, invert_boxes, recover, select_regions


def forward_boxes(boxes,w,h,rotation):
    # Independent closed-form xyxy transforms for np.rot90 (counterclockwise).
    x1,y1,x2,y2=np.asarray(boxes).T
    if rotation==0:return np.asarray(boxes)
    if rotation==90:return np.stack([y1,w-x2,y2,w-x1],1)
    if rotation==180:return np.stack([w-x2,h-y2,w-x1,h-y1],1)
    return np.stack([h-y2,x1,h-y1,x2],1)


def main():
    boxes=np.array([[2.,1.,7.,4.],[0,0,10,6]])
    for rotation in [0,90,180,270]:
        transformed=forward_boxes(boxes,10,6,rotation)
        np.testing.assert_allclose(invert_boxes(transformed,10,6,rotation),boxes)
        np.testing.assert_allclose(invert_boxes(transformed,10,6,rotation,(5,9)),boxes+[5,9,5,9])
        assert invert_boxes([],10,6,rotation).shape==(0,4)
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'train.json'
        p.write_text(json.dumps(dict(images=[dict(id=1,width=100,height=60)],
            categories=[dict(id=1,name='extra_tooth')],annotations=[
                dict(image_id=1,category_id=1,bbox=[20,10,10,20]),
                dict(image_id=1,category_id=1,bbox=[40,10,10,20]) ])))
        prior=fit_spatial_prior(p)
        assert prior['images']==1 and prior['instances']==2
        assert sum(x[2] for x in prior['points'])==1
    image=np.zeros((60,100,3),dtype=np.uint8)
    image[20:35,35:50]=255
    def predict(view):
        y,x=np.where(view[...,0]>0)
        if len(x)==0:return np.empty((0,4)),np.empty(0)
        return np.array([[x.min(),y.min(),x.max()+1,y.max()+1]],float),np.array([.8])
    configs=Path(__file__).resolve().parents[1]/'configs/dental_local_recovery'
    for path in sorted(configs.glob('*.json')):
        config=json.loads(path.read_text());records=recover(image,predict,config,prior)
        assert records[0]['source']=='full' and records[0]['bbox_xyxy']==[35,20,50,35]
        assert records[0]['score']==.8
        regions=select_regions(100,60,config,prior)
        assert len(regions)==config['crop_count']
        for r in records:
            x1,y1,x2,y2=r['bbox_xyxy']
            assert 0<=x1<x2<=100 and 0<=y1<y2<=60
            if not r['crop_boundary']:
                np.testing.assert_allclose(r['bbox_xyxy'],[35,20,50,35])
        empty=recover(np.zeros_like(image),predict,config,prior)
        assert not empty
        print(path.stem,'PASS',flush=True)
    config=json.loads((configs/'spatial.json').read_text())
    try:select_regions(100,60,config)
    except ValueError:pass
    else:raise AssertionError('Missing prior must fail')
    print('PASS: rotation roundtrip, offsets, empty boxes/images, case weighting, full-prefix invariance')


if __name__=='__main__':main()
