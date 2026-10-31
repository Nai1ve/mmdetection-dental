"""CPU tests for set supervision, partial instances, ROI coverage and serialization."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
from dental_roi_head import ToothROIHead,extract_rois,make_regions,roi_targets,set_loss
from run_dental_roi import train,load


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    regions,seeds=make_regions(torch.tensor([[10.,10,30,30]]),torch.tensor([.9]),100,60)
    targets,partial,overflow,covered=roi_targets(regions,seeds,
        torch.tensor([[10.,10,30,30],[33.,10,50,30]]))
    assert covered==1 and len(targets[0])==1 and len(partial[0])==1
    assert not overflow[0]
    empty=make_regions(torch.empty(0,4),torch.empty(0),100,60)
    assert empty[0].shape==(0,4)
    fpn=[torch.randn(1,8,120,120),torch.randn(1,16,60,60)]
    pooled=extract_rois(fpn,regions,100,60)
    assert pooled.shape==(1,24,7,7) and pooled.isfinite().all()
    assert extract_rois(fpn,torch.empty(0,4),100,60).shape==(0,24,7,7)
    gt=[torch.tensor([[.1,.1,.4,.5],[.5,.4,.9,.9]]),torch.empty(0,4),torch.tensor([[.2,.2,.8,.8]])]
    for mode,slots in [('single',1),('multi',4)]:
        model=ToothROIHead(24,slots);features=torch.randn(3,24,7,7);geom=torch.rand(3,4)
        pred=model(features,geom)
        loss=set_loss(pred,gt,[torch.empty(0,4)]*3,[False]*3,mode)
        assert loss.isfinite();loss.backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
        if mode=='multi':
            flipped=deepcopy(gt);flipped[0]=flipped[0].flip(0)
            torch.testing.assert_close(loss,set_loss(pred,flipped,[torch.empty(0,4)]*3,[False]*3,mode))
        ignored=set_loss(pred,gt,[torch.empty(0,4)]*3,[True]*3,mode)
        assert float(ignored.detach())==0
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'head.pt';torch.save(model.state_dict(),path)
            clone=ToothROIHead(24,slots);clone.load_state_dict(load(path))
            torch.testing.assert_close(model(features,geom)['boxes'],clone(features,geom)['boxes'])
        print(mode,'loss, gradient, overflow, serialization PASS',flush=True)
    # Real optimizer and checkpoint loop on a synthetic trusted cache, not performance data.
    import json
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);cache=root/'cache';cache.mkdir()
        item=dict(features=torch.randn(2,24,7,7).half(),regions=torch.tensor([[0.,0,80,60],[10,0,90,60]]),
                  width=100,height=60,targets=gt[:2],partial=[torch.empty(0,4)]*2,overflow=[False]*2)
        torch.save(item,cache/'1.pt')
        (cache/'manifest.json').write_text(json.dumps(dict(split='train',channels=24,entries=['1.pt'])))
        train(SimpleNamespace(cache=cache,output=root/'run',seed=42,mode='multi',device='cpu',epochs=2,limit_images=0))
        assert load(root/'run/last.pt')['epoch']==2
    print('ROIAlign, partial-GT handling, empty ROI and optimizer loop PASS',flush=True)


if __name__=='__main__':main()
