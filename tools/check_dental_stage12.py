"""CPU checks for reversibility, GT isolation, exact area and matching cardinality."""
from copy import deepcopy
import numpy as np
from dental_review_regions import build_case,group_boxes,union_area,select_regions
from run_dental_stage12 import matched,evaluate


def main():
    image=dict(id=1,width=100,height=100)
    record=dict(original_boxes=[[10,10,30,30],[10.5,10.5,29.5,29.5],[60,60,80,80]],original_scores=[.9,.4,.8])
    case=build_case(image,record);assert len(case['groups'])==2
    assert sorted(i for g in case['groups'] for i in g['members'])==[0,1,2]
    polluted=deepcopy(image);polluted.update(valid_slot_mask=[1],filename='qy_dsy',annotations=[dict(category_id=9)])
    assert build_case(polluted,record)==case
    assert union_area([[0,0,.5,1],[.25,0,.75,1]])==.75
    assert union_area([])==0
    assert len(select_regions(case,'risk',0))==0
    assert select_regions(case,'random',4)==select_regions(case,'random',4)
    gt=np.array([[10,10,30,30],[10,10,30,30]])
    assert len(matched(gt,np.array([[10,10,30,30]]),.5))==1
    empty=build_case(image,dict(original_boxes=[],original_scores=[]));assert empty['arch_hypotheses']==[]
    assert len(select_regions(empty,'risk',8))==8
    assert len(select_regions(empty,'spread',4))==4
    from summarize_dental_raw_audit import summarize
    raw=summarize([dict(image_id=1,annotation_id=1,best_raw_iou=.8,max_score_iou05=.0002,nms={})])
    assert raw['counts']['geometry_exists_but_score_below_0.001']==1
    assert raw['not_one_to_one_recall']
    coco=dict(images=[image],categories=[dict(id=1,name='extra_tooth')],annotations=[dict(id=1,image_id=1,category_id=1,bbox=[10,10,20,20])])
    report=evaluate([case],coco);assert report['results']['candidate/all_members/iou0.5']['extra_matched']==1
    import tempfile,json,hashlib,sys
    from pathlib import Path
    from evaluate_dental_raw_pool import main as raw_main
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/'raw').mkdir()
        annotation=root/'annotations.json';annotation.write_text(json.dumps(coco))
        (root/'manifest.json').write_text(json.dumps(dict(partial_dataset=False,annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest())))
        np.savez_compressed(root/'raw/1.npz',boxes=np.array([[10.,10,30,30]]),scores=np.array([.0002]))
        (root/'conf0.001_nms0.6.json').write_text(json.dumps([dict(image_id=1,original_boxes=[],original_scores=[])]))
        previous=sys.argv
        try:
            sys.argv=['audit','--raw-root',str(root),'--annotations',str(annotation),'--output',str(root/'result.json')];raw_main()
        finally:sys.argv=previous
        result=json.loads((root/'result.json').read_text())
        assert result['failures'][0]['cause']=='below_score_floor'
        assert result['one_to_one_counts']['pre_score_0.0001/iou0.5']['extra_matched']==1
    print('STAGE12_TESTS_PASS: grouping, GT isolation, empty cases, area, cardinality, reproducibility')


if __name__=='__main__':main()
