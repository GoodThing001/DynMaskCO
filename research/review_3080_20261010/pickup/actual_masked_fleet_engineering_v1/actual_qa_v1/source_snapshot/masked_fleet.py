"""Real CVRP encoder+decoder proposal on observed multi-vehicle tails.

The original3D model has no temperature, current-position or TW input. Its logits
are proposals only; actual public-state C0 certifies and scores every complete plan.
No private terminal day utility is read.
"""
import copy
import hashlib
import math
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import public_fleet as pf
MODES=('pretrained','random','discard')

def bootstrap(scripts):
    scripts=pf.bootstrap(scripts)
    sys.path.insert(0,str(scripts/'delivery'))
    return scripts

def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def topology(packet,max_orders=5):
    pf.validate_public(packet)
    vids,pool,_=pf.family(packet,max_orders)
    if len(pool)<2:raise ValueError('two mutable public orders required')
    ids=sorted({o for v in vids for o in packet['plan'][str(v)]})
    loc={o:i+1 for i,o in enumerate(ids)}
    ds=packet['dataset']
    inst={'coords':np.asarray([ds['coords'][0,0]]+[ds['coords'][0,o] for o in ids]),
          'demands':np.asarray([0.]+[ds['demands'][0,o] for o in ids])}
    routes=[[loc[o] for o in packet['plan'][str(v)]] for v in vids]
    committed={int(packet['snapshot']['committed_next'][v]) for v in vids
               if packet['snapshot']['vehicle_status'][v]=='committed'}
    return vids,pool,ids,loc,inst,routes,committed

def propose(provider,packet,scripts,mode='pretrained',seed=202610101903,mask_fraction=.3,guide_weight=1.,deadline=None):
    scripts=bootstrap(scripts)
    from delivery_maskco import partial_adjacency
    if mode not in MODES:raise ValueError('unregistered proposal mode')
    if not math.isfinite(mask_fraction) or not 0<mask_fraction<=1:raise ValueError('mask fraction')
    if not math.isfinite(guide_weight) or guide_weight<0:raise ValueError('guide weight')
    deadline=time.perf_counter()+10 if deadline is None else deadline
    vids,pool,ids,loc,inst,routes,committed=topology(packet)
    rng=np.random.default_rng(seed)
    removed=set(map(int,rng.choice(pool,size=max(1,math.ceil(mask_fraction*len(pool))),replace=False)))
    removed_local={loc[o] for o in removed};n=len(ids)+1
    adj=partial_adjacency(routes,removed_local,n)
    feats=provider.features(inst,packet['capacity'])
    logits=provider.logits(feats,adj)
    if time.perf_counter()>deadline:raise TimeoutError('encoder/decoder exceeded submission deadline')
    # Pay the same complete model call in discard; random uses same architecture.
    valid=~np.eye(n,dtype=bool)
    scale=max(1e-8,float(np.std(logits[valid])))
    dist=np.linalg.norm(inst['coords'][:,None,:]-inst['coords'][None,:,:],axis=-1)
    distance_scale=max(1e-8,float(dist[valid].mean()))
    candidate=copy.deepcopy(packet['plan'])
    for v in vids:candidate[str(v)]=[o for o in candidate[str(v)] if o not in removed]
    remaining=sorted(removed);audit=[]
    while remaining:
        choices=[]
        for o in remaining:
            for v in vids:
                route=candidate[str(v)]
                for at in range(len(route)+1):
                    if time.perf_counter()>deadline:raise TimeoutError('proposal/certificate exceeded deadline')
                    cn=int(packet['snapshot']['committed_next'][v])
                    if packet['snapshot']['vehicle_status'][v]=='committed' and cn>0 and at==0:continue
                    a=loc[route[at-1]] if at else 0;b=loc[route[at]] if at<len(route) else 0
                    if adj[a,b]>0:continue
                    # Certify only complete accepted sets: unplaced orders remain on
                    # their original owner, at the end, during this tentative check.
                    trial=copy.deepcopy(candidate);trial[str(v)].insert(at,o)
                    for pending in remaining:
                        if pending!=o:
                            owner=packet['ownership'][str(pending)];trial[str(owner)].append(pending)
                    try:cert,_=pf.certificate(packet,trial)
                    except ValueError as exc:
                        audit.append(dict(order=o,vehicle=v,position=at,failure=str(exc)));continue
                    if cert['qualified'] is not True:continue
                    k=loc[o];old=0. if a==b else float(logits[a,b])
                    gain=(float(logits[a,k])+float(logits[k,b])-old)/scale
                    delta=dist[a,k]+dist[k,b]-dist[a,b]
                    score=float(delta/distance_scale-(guide_weight*gain if mode!='discard' else 0.))
                    choices.append((score,o,v,at));audit.append(dict(order=o,vehicle=v,position=at,proposal_score=score,certificate=cert))
        if not choices:
            return copy.deepcopy(packet['plan']),dict(complete=False,fallback=True,reason='no certified insertable gap; mask never silently extended',masked_orders=sorted(removed),decoder_calls=1,audit=audit)
        _,o,v,at=min(choices);candidate[str(v)].insert(at,o);remaining.remove(o)
    rebuilt=partial_adjacency([[loc[o] for o in candidate[str(v)]] for v in vids],set(),n)
    if not np.all(rebuilt[adj>0]>0):raise ValueError('retained original edges changed')
    cert,tr=pf.certificate(packet,candidate)
    baseline,_=pf.certificate(packet,packet['plan'])
    if time.perf_counter()>deadline:raise TimeoutError('final certification exceeded deadline')
    improves=cert['qualified'] is True and cert['evaluation']['utility']>baseline['evaluation']['utility']+1e-10
    selected=candidate if improves else copy.deepcopy(packet['plan'])
    return selected,dict(mode=mode,complete=True,masked_orders=sorted(removed),committed_orders=sorted(committed),
        retained_edges=adj.tolist(),retained_edges_preserved=True,decoder_calls=1,encoder_calls=1,
        features_shape=[1,n,3],local_order_map=loc,logits_sha256=array_hash(logits),
        raw_candidate=candidate,selected=selected,candidate_certificate=cert,baseline_certificate=baseline,
        accepted_on_known_score_only=improves,audit=audit,provider_identity=provider.identity,
        formal_online_gain_established=False,model_physical_state_features=False)
