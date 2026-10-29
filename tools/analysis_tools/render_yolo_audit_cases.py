"""Render private, local-only diagnostic crops; not clinical interpretation."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cases = [x for x in json.loads(args.cases.read_text()) if x['image_id'] in [18,23,76,129,150]]
    sheet = Image.new('RGB', (1000, 480*((len(cases)+1)//2)), 'white')
    for k, case in enumerate(cases):
        image = Image.open(args.data_root/case['file_name']).convert('RGB')
        x1,y1,x2,y2 = case['gt_xyxy']
        cx,cy = (x1+x2)/2,(y1+y2)/2
        size = max(x2-x1,y2-y1)*2.5
        size = max(size,320)
        left,top = max(0,int(cx-size/2)),max(0,int(cy-size/2))
        crop = image.crop((left,top,min(image.width,int(cx+size/2)),min(image.height,int(cy+size/2))))
        draw = ImageDraw.Draw(crop)
        def box(coords,color):
            a,b,c,d=coords
            draw.rectangle((a-left,b-top,c-left,d-top),outline=color,width=3)
        box(case['gt_xyxy'],'red')
        box(case['best_raw_box'],'deepskyblue')
        if 'candidate_box' in case: box(case['candidate_box'],'orange')
        for blocker in case['nms']['0.6'].get('higher_score_overlapping_survivors',[]): box(blocker['box'],'lime')
        crop.thumbnail((480,410))
        ox,oy=(k%2)*500,(k//2)*480
        sheet.paste(crop,(ox+(500-crop.width)//2,oy+55))
        ImageDraw.Draw(sheet).text((ox+10,oy+5),f"image {case['image_id']}; best IoU={case['best_raw_iou']:.3f}",fill='black')
        ImageDraw.Draw(sheet).text((ox+10,oy+22),'GT:red  raw geometry:blue  scored:orange  blocker:green',fill='black')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(args.output)


if __name__ == '__main__':
    main()
