"""Actual CVRP encoder/decoder QA on fixed toy routes; no research gains."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'delivery'))
from delivery_maskco import MaskCORouteProvider,partial_adjacency,improve_masked
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from test_delivery_contract import fixture
from test_delivery_maskco import test


def main(checkpoint,upstream,out):
    out=Path(out);out.mkdir(exist_ok=False)
    test()
    data=fixture();contract=DeliveryContract(capacity=6.,vehicles=2)
    p=MaskCORouteProvider(checkpoint,upstream)
    p.prewarm(4,contract.capacity)
    h=p.features(data,contract.capacity)
    a=partial_adjacency([[1,2,3]],{2},4)
    zero=p.logits(h,np.zeros((4,4),np.float32));partial=p.logits(h,a)
    valid=~np.eye(4,dtype=bool)
    difference=float(np.max(np.abs(zero[valid]-partial[valid])))
    assert difference>1e-6, 'Actual decoder must use the partial adjacency'
    arms={}
    for name,iterative in [('single',False),('iterative',True)]:
        r=improve_masked(data,[[1,2,3]],contract,p,time_limit_s=2.,iterative=iterative,mask_fraction=.5)
        checked=evaluate(data,r['routes'],contract)
        assert checked['hard'] and r['same_initial_assignment'] and r['retained_edges_preserved']
        assert checked['cost']<=evaluate(data,[[1,2,3]],contract)['cost']+1e-9
        assert r['wall_s']<2.1
        arms[name]={k:r[k] for k in ['wall_s','timeout','cycles','retained_edges_preserved']}
    result={'actual_route_provider_verified':True,'checkpoint_identity':p.identity,
        'encoder_and_decoder_active':True,'partial_adjacency_changes_logits':difference,
        'same_initial_route_certified':True,'arms':arms,
        'qualification':'fixed_toy_engineering_only','formal_pilot_started':False,
        'online_gain_established':False}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--upstream',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();main(a.checkpoint,a.upstream,a.out)
