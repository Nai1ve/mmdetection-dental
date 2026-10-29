"""Convert existing private COCO splits to YOLO labels without changing splits."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def prepare(root, output, view='fdi53'):
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for source in sorted((root/view).glob('*.json')):
        data = json.loads(source.read_text())
        categories = sorted(data['categories'], key=lambda c: c['id'])
        mapping = {c['id']: i for i, c in enumerate(categories)}
        annotations = defaultdict(list)
        for ann in data['annotations']:
            assert not ann.get('iscrowd', 0) and not ann.get('ignore', False), 'Ignored annotations need explicit handling'
            annotations[ann['image_id']].append(ann)
        image_dir = output/'images'/source.stem
        label_dir = output/'labels'/source.stem
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        stems = set()
        for image in data['images']:
            src = (root/image['file_name']).resolve()
            assert src.is_file()
            name = Path(image['file_name']).name
            assert Path(name).stem not in stems, 'Image stem collision'
            stems.add(Path(name).stem)
            (image_dir/name).symlink_to(src)
            lines = []
            w, h = image['width'], image['height']
            for ann in annotations[image['id']]:
                x, y, bw, bh = ann['bbox']
                assert bw > 0 and bh > 0 and x >= 0 and y >= 0 and x+bw <= w+1e-3 and y+bh <= h+1e-3
                lines.append(f'{mapping[ann["category_id"]]} {(x+bw/2)/w:.10f} {(y+bh/2)/h:.10f} {bw/w:.10f} {bh/h:.10f}')
            (label_dir/(Path(name).stem+'.txt')).write_text('\n'.join(lines)+'\n')
        summary[source.stem] = dict(images=len(data['images']), boxes=len(data['annotations']))
        config = dict(path=str(output.resolve()), train='images/train', val=f'images/{source.stem}',
                      names={i: c['name'] for i, c in enumerate(categories)})
        (output/f'{source.stem}.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))
    (output/'conversion.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--view', choices=['fdi53', 'tooth1'], default='fdi53')
    args = p.parse_args()
    print(prepare(args.root, args.output, args.view))
