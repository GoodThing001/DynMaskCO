"""Finite registered same-incumbent delivery pilot; creates no final-test data."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_baselines import improve_ortools,improve_regret_lns
from delivery_maskco import MaskCORouteProvider,improve_masked

def dump(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False))
def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def generate(reg,i):
    ds=reg['dataset'];rng=np.random.default_rng(np.random.SeedSequence([ds['seed'],i]));n=ds['customers']
    if n!=50:raise ValueError('Undeclared common route construction')
    data={'coords':np.r_[np.array([[5.,5.]]),rng.uniform(0,10,(n,2))].tolist(),
        'demands':np.r_[0,rng.integers(1,6,n)].tolist(),
        'temp_class':np.r_[0,rng.integers(0,3,n)].tolist(),
        'reveal_time':[0.]*(n+1),'ready_time':[0.]*(n+1),'tw_start':[0.]*(n+1),
        'tw_end':[8.]*(n+1),'service_time':[0.]+[.03]*n}
    routes=[list(range(1+10*k,11+10*k)) for k in range(5)]
    return data,routes


def main(protocol,upstream,checkpoint,out,manifest_path):
    out=Path(out);out.mkdir(exist_ok=False)
    protocol=Path(protocol);reg=json.loads(protocol.read_text())
    if hashlib.sha256(protocol.read_bytes()).hexdigest()!='8e40c705e57a6b6f2ed2244fc9c73108b43db7757596ff8ed03283f110cc3642':
        raise ValueError('Delivery preregistration changed')
    contract=DeliveryContract(**reg['contract_overrides'])
    contract.validate()
    # Preserve a source/weight/timing identity before creating any development data.
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text())
    source_root=manifest_path.parent/'src'
    for p,h in manifest.items():
        if hashlib.sha256((source_root/p).read_bytes()).hexdigest()!=h:raise ValueError('Source not sealed')
    cp_sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if cp_sha!=reg['weights']['sha256']:raise ValueError('Pretrained weight mismatch')
    identity={'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'contract_hash':contract.identity(),'source_sha256':digest(manifest),'checkpoint_sha256':cp_sha,
        'timing_hash':digest(reg['timing']),'hardware':'CPU54_BLAS1_OMP1',
        'claim_scope':'route_transfer_pilot_only_no_online_gain'}
    dump(out/'identity_start.json',identity)
    # Both graphs use only a fixed synthetic input for compilation; no development scores.
    from ortools.constraint_solver import pywrapcp
    pretrained=MaskCORouteProvider(checkpoint,upstream)
    random=MaskCORouteProvider(checkpoint,upstream,random_seed=42)
    for model in (pretrained,random):model.prewarm(51,contract.capacity)
    probe={'coords':np.stack([np.arange(51),np.zeros(51)],axis=1),'demands':np.r_[0.,np.ones(50)]}
    a=np.zeros((51,51),np.float32)
    pa=pretrained.logits(pretrained.features(probe,contract.capacity),a)
    ra=random.logits(random.features(probe,contract.capacity),a)
    off=~np.eye(51,dtype=bool)
    if np.max(np.abs(pa[off]-ra[off]))<=1e-6:raise ValueError('Random architecture control not distinct')
    dump(out/'MODEL_ENGINEERING_PASS.json',{'pretrained':pretrained.identity,'random':random.identity,
        'fixed_shape_precompile_only':True,'actual_random_weights_distinct':True,
        'online_gain_established':False,'development_data_not_yet_generated':True})
    instances={};initial_reports={}
    for i in reg['dataset']['instance_ids']:
        data,routes=generate(reg,i);checked=evaluate(data,routes,contract)
        if not checked['hard']:raise ValueError('Do not screen or resample invalid initial route: '+str(i))
        instances[str(i)]={'instance':data,'initial_routes':routes,'data_initial_hash':digest([data,routes])}
        initial_reports[i]=checked
    dataset_sha=digest(instances);identity['data_sha256']=dataset_sha
    dump(out/'dataset.json',instances);dump(out/'identity_start.json',identity)
    dump(out/'INITIAL_ROUTES_PASS.json',{'unique_instances':len(instances),'hard':True,'dataset_sha256':dataset_sha})
    limit=reg['timing']['total_per_arm_per_instance_s'];rows=[]
    with (out/'progress.jsonl').open('x') as progress:
        for i in reg['dataset']['instance_ids']:
            item=instances[str(i)];data=item['instance'];routes=item['initial_routes'];initial=initial_reports[i]
            seed=int(np.random.SeedSequence([reg['dataset']['seed'],i,41]).generate_state(1)[0])
            order=list(np.random.default_rng(np.random.SeedSequence([reg['dataset']['seed'],i,999])).permutation(reg['arms'][1:]))
            records={'incumbent':{'routes':routes,'report':initial,'wall_s':0.,'timeout':False}}
            for arm in order:
                started=time.perf_counter()
                if arm=='ortools_gls':r=improve_ortools(data,routes,contract,time_limit_s=limit)
                elif arm=='destroy_regret_lns':r=improve_regret_lns(data,routes,contract,time_limit_s=limit,seed=seed)
                else:r=improve_masked(data,routes,contract,random if arm=='maskco_random_single' else pretrained,
                    time_limit_s=limit,seed=seed,iterative=(arm=='maskco_iterative'),
                    mask_fraction=reg['reconstruction']['mask_fraction'],guide_weight=reg['reconstruction']['guide_weight'])
                submit_wall=time.perf_counter()-started
                if submit_wall>=limit:
                    r.update(routes=routes,report=initial,timeout=True)
                r['wall_s']=submit_wall
                # Independent re-evaluation is after submission, not credited as search.
                checked=evaluate(data,r['routes'],contract)
                if not checked['hard'] or abs(checked['cost']-r['report']['cost'])>1e-9:
                    raise RuntimeError('Post-submit independent delivery certification failed')
                records[arm]=r
            row={'instance_id':i,'data_initial_hash':item['data_initial_hash'],'identity':identity,
                'arm_order':order,'arms':records}
            rows.append(row);progress.write(json.dumps(row,allow_nan=False)+'\n');progress.flush()
    for p,h in manifest.items():
        if hashlib.sha256((source_root/p).read_bytes()).hexdigest()!=h:raise ValueError('End source mismatch')
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()!=cp_sha:raise ValueError('End weight mismatch')
    dump(out/'identity_end.json',identity)
    dump(out/'result.json',{'identity':identity,'source_stable':True,'identity_verified':True,'rows':rows,
        'scope':'registered_same_initial_delivery_route_pilot','online_gain_established':False})
    print(json.dumps({'pilot_complete':True,'instances':len(rows),'online_gain_established':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ['protocol','upstream','checkpoint','out','manifest-path']:p.add_argument('--'+a,required=True)
    a=p.parse_args();main(a.protocol,a.upstream,a.checkpoint,a.out,a.manifest_path)
