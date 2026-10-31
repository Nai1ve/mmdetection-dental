"""Historical E0 evidence only: per-GT geometry diagnostics are not one-to-one recall."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(cases):
    counts=Counter();rows=[]
    for case in cases:
        score=case['max_score_iou05'];reason=[]
        if case['best_raw_iou']<.5:reason.append('no_raw_box_iou05')
        elif score is not None:
            for threshold in [.01,.001,.0001]:
                if score<threshold:counts[f'geometry_exists_but_score_below_{threshold}']+=1
        for threshold,info in case['nms'].items():
            if info.get('candidate_survived') is False and info.get('higher_score_overlapping_survivors'):
                reason.append(f'nms_conflict_{threshold}');counts[f'nms_conflict_{threshold}']+=1
        if not reason:reason=['score_or_localization_requires_inspection']
        rows.append(dict(image_id=case['image_id'],annotation_id=case['annotation_id'],
            best_raw_iou=case['best_raw_iou'],max_score_iou05=score,diagnostic_flags=reason))
    return dict(instances=len(cases),counts=dict(counts),diagnostics=rows,
        historical_checkpoint_only=True,not_one_to_one_recall=True,
        not_current_frozen_checkpoint_reexport=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(summarize(json.loads(a.cases.read_text())),f,indent=2)
