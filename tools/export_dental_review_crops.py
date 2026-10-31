"""Materialize neutral evidence crops for model evaluation; no model or GT access."""
import argparse
import json
from pathlib import Path
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--interface-root',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',choices=['risk','spread','grid','random'],default='spread')
    p.add_argument('--budget',type=int,choices=[1,2,4,8],default=4);p.add_argument('--limit-images',type=int,default=0)
    a=p.parse_args();assert a.limit_images>=0
    index=json.loads((a.interface_root/'image_index.json').read_text());cases=json.loads((a.interface_root/'review_interface.json').read_text())
    cases={x['image_ref']:x for x in cases};requests=json.loads((a.interface_root/'crop_requests.json').read_text())
    refs=list(index)[:a.limit_images] if a.limit_images else list(index)
    a.output.mkdir(parents=True,exist_ok=False);assets=[];root=a.image_root.resolve()
    for ref in refs:
        path=(root/index[ref]['relative_path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Image path escapes dataset root')
        with Image.open(path) as image:
            size=cases[ref]['image_size'];assert image.size==(size['width'],size['height'])
            image=image.convert('RGB');image.save(a.output/f'{ref}_full.png')
            selected=[x for x in requests if x['image_ref']==ref and x['policy']==a.policy and x['rank']<=a.budget]
            for request in selected:
                name=request['evidence_id']+'.png';image.crop(request['bbox_xyxy']).save(a.output/name)
                assets.append(dict(**request,file=name,full_image=f'{ref}_full.png'))
    (a.output/'assets.json').write_text(json.dumps(dict(images=len(refs),partial_dataset=bool(a.limit_images),
        policy=a.policy,budget=a.budget,assets=assets),indent=2))
    print('CROPS_OK',len(refs),len(assets))


if __name__=='__main__':main()
