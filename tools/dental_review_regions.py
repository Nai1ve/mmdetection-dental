"""Prediction-only reversible candidates and provisional review regions, not diagnoses."""
import numpy as np


def overlaps(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    aa=np.maximum(0,a[:,2:]-a[:,:2]).prod(1);bb=np.maximum(0,b[:,2:]-b[:,:2]).prod(1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9),inter/np.maximum(aa[:,None],1e-9)


def group_boxes(boxes,scores,threshold=.85):
    pending=np.argsort(-np.asarray(scores),kind='stable');result=[]
    while len(pending):
        leader=int(pending[0]);mask=overlaps(boxes[leader:leader+1],boxes[pending])[0][0]>=threshold
        mask[0]=True
        result.append(dict(leader=leader,members=pending[mask].tolist()))
        pending=pending[~mask]
    return result


def arch_features(boxes,scores):
    """Two robust quadratic hypotheses. Offset is image geometry, not eruption depth."""
    centers=(boxes[:,:2]+boxes[:,2:])/2;strong=np.flatnonzero(scores>=.25)
    if len(strong)<12:return [],[None]*len(boxes)
    mid=float(np.median(centers[strong,1]));fits=[];coords=[None]*len(boxes)
    for side in [0,1]:
        ids=strong[(centers[strong,1]>=mid)==bool(side)]
        if len(ids)<6 or np.ptp(centers[ids,0])<.3:continue
        xy=centers[ids];keep=np.ones(len(ids),bool)
        for _ in range(3):
            coef=np.polyfit(xy[keep,0],xy[keep,1],2)
            residual=np.abs(np.polyval(coef,xy[:,0])-xy[:,1])
            keep=residual<=max(.015,float(np.quantile(residual,.8)))
            if keep.sum()<4:break
        rmse=float(np.sqrt(np.mean(residual[keep]**2)))
        grid=np.linspace(0,1,241);curve=np.stack([grid,np.polyval(coef,grid)],1)
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
        fits.append(dict(layer_hypothesis=side,coefficients=coef.tolist(),rmse_normalized=rmse,
                         fit_quality='uncertain' if rmse>.04 else 'provisional',clinical_layer_verified=False))
        for i,c in enumerate(centers):
            if (c[1]>=mid)!=bool(side):continue
            j=int(np.argmin(((curve-c)**2).sum(1)));slope=2*coef[0]*grid[j]+coef[1]
            normal=np.array([-slope,1.])/np.sqrt(1+slope*slope)
            coords[i]=dict(layer_hypothesis=side,s=float(2*length[j]/length[-1]-1),
                           d_projected=float((c-curve[j])@normal),fit_rmse=rmse)
    return fits,coords


def crop_at(center,size=(.22,.30)):
    center=np.clip(center,np.array(size)/2,1-np.array(size)/2)
    return np.r_[center-np.array(size)/2,center+np.array(size)/2].tolist()


def build_case(image,record):
    width,height=image['width'],image['height'];scale=np.array([width,height,width,height],float)
    if width<=0 or height<=0:raise ValueError('Image dimensions must be positive')
    boxes=np.asarray(record['original_boxes'],float).reshape(-1,4);scores=np.asarray(record['original_scores'],float)
    if len(boxes)!=len(scores) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():raise ValueError('Invalid candidates')
    if ((scores<0)|(scores>1)).any():raise ValueError('Expected raw probabilities in [0,1]')
    if len(boxes) and ((boxes[:,2:]<=boxes[:,:2]).any() or (boxes<0).any() or (boxes>scale+1).any()):raise ValueError('Invalid image coordinates')
    norm=boxes/scale;groups=group_boxes(norm,scores);fits,arch=arch_features(norm,scores)
    centers=(norm[:,:2]+norm[:,2:])/2;iou,ioa=overlaps(norm,norm);edges=[];proposals=[]
    def propose(center,risk,reason,ids):
        proposals.append(dict(bbox_normalized=crop_at(center),risk=float(risk),reason=reason,candidate_ids=ids))
    for g in groups:
        if len(g['members'])>1:propose(centers[g['leader']],1.2,'ambiguous_overlap_group',g['members'])
    for i in range(len(boxes)):
        if .01<=scores[i]<.25:propose(centers[i],1-float(scores[i]),'low_score_response',[i])
        neighbors=np.argsort(np.linalg.norm(centers-centers[i],axis=1),kind='stable')
        for j in neighbors[neighbors!=i][:4]:
            if i<j and (iou[i,j]>.2 or max(ioa[i,j],ioa[j,i])>.5):
                edges.append(dict(a=i,b=int(j),iou=float(iou[i,j]),ioa_a=float(ioa[i,j]),ioa_b=float(ioa[j,i])))
                propose((centers[i]+centers[j])/2,1+max(ioa[i,j],ioa[j,i]),'overlap_conflict',[i,int(j)])
    leaders=[g['leader'] for g in groups if scores[g['leader']]>=.25]
    mid=float(np.median(centers[leaders,1])) if leaders else .5
    for side in [0,1]:
        ids=sorted([i for i in leaders if (centers[i,1]>=mid)==bool(side)],key=lambda i:centers[i,0])
        for i,j in zip(ids,ids[1:]):
            gap=centers[j,0]-centers[i,0];typical=max(.01,float(np.mean(norm[[i,j],2]-norm[[i,j],0])))
            if gap/typical>1.5:propose((centers[i]+centers[j])/2,min(2.,gap/typical/2),'projected_gap',[i,j])
        # Coarse image-axis asymmetry only, not contralateral same-FDI matching.
        for i in ids:
            mirror=np.array([1-centers[i,0],centers[i,1]])
            distance=min([np.linalg.norm(centers[j]-mirror) for j in ids],default=1.)
            if distance>.08:propose(mirror,min(1.5,distance*5),'coarse_asymmetry',[i])
    grid=[crop_at((x,y)) for y in [.35,.65] for x in [.15,.38,.62,.85]]
    for box in grid:proposals.append(dict(bbox_normalized=box,risk=0.,reason='exploration_fallback',candidate_ids=[]))
    ranked=[]
    for p in sorted(proposals,key=lambda p:-p['risk']):
        if ranked and overlaps([p['bbox_normalized']],[x['bbox_normalized'] for x in ranked])[0].max()>.7:continue
        ranked.append(p)
    candidates=[dict(candidate_id=i,bbox_xyxy=b.tolist(),bbox_normalized=norm[i].tolist(),score_raw=float(scores[i]),
                     source='full_frame_post_nms',raw_candidate_index=record.get('raw_candidate_indices',[None]*len(boxes))[i],
                     nearest_candidate_ids=[int(j) for j in np.argsort(np.linalg.norm(centers-centers[i],axis=1),kind='stable') if j!=i][:4],
                     arch_coords=arch[i],is_inverted=None) for i,b in enumerate(boxes)]
    return dict(schema_version=1,image_id=image['id'],image_size=dict(width=width,height=height),
        orientation_status='unverified',candidate_stage='post_nms_existing_export',raw_pre_nms_available=False,
        candidates=candidates,groups=groups,grouping_is_provisional=True,arch_hypotheses=fits,
        overlap_edges=edges,ranked_regions=ranked,grid_regions=grid,
        clinical_claims=False,no_gt_used_in_generation=True)


def select_regions(case,policy,k):
    if policy=='risk':return [x['bbox_normalized'] for x in case['ranked_regions'][:k]]
    grid=case['grid_regions']
    if policy=='grid':return grid[:k]
    if policy=='spread':
        centers=np.array([[(c['bbox_normalized'][0]+c['bbox_normalized'][2])/2,
                           (c['bbox_normalized'][1]+c['bbox_normalized'][3])/2] for c in case['candidates'] if c['score_raw']>=.25]).reshape(-1,2)
        if not len(centers):return grid[:k]
        selected=[];pending=list(range(len(centers)))
        while pending and len(selected)<k:
            if not selected:idx=min(pending,key=lambda i:np.linalg.norm(centers[i]-.5))
            else:idx=max(pending,key=lambda i:min(np.linalg.norm(centers[i]-centers[j]) for j in selected))
            selected.append(idx);pending.remove(idx)
        return [crop_at(centers[i]) for i in selected]+grid[:max(0,k-len(selected))]
    if policy=='random':
        import hashlib
        seed=int(hashlib.sha256(str(case['image_id']).encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed)
        return [crop_at(c) for c in rng.uniform([.11,.15],[.89,.85],size=(k,2))]
    raise ValueError(policy)


def union_area(rectangles):
    if not rectangles:return 0.
    xs=sorted({x for b in rectangles for x in (b[0],b[2])});area=0.
    for left,right in zip(xs,xs[1:]):
        intervals=sorted((b[1],b[3]) for b in rectangles if b[0]<right and b[2]>left)
        end=-1.;length=0.
        for low,high in intervals:length+=max(0,high-max(low,end));end=max(end,high)
        area+=(right-left)*length
    return area
