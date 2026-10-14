"""Route adapter invariants with a stub; real pretrained decoder is separate QA."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'delivery'))
from delivery_maskco import partial_adjacency, improve_masked
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from test_delivery_contract import fixture


class ProbeProvider:
    def __init__(self):self.calls=[]
    def features(self,data,capacity):return len(data['coords'])
    def logits(self,h,adj):
        self.calls.append(adj.copy())
        out=np.arange(h*h,dtype=float).reshape(h,h)
        np.fill_diagonal(out,-np.inf)
        return out


def test():
    adj=partial_adjacency([[1,2,3]],{2},4)
    assert adj[0,1]==adj[0,3]==1 and adj[1,3]==0
    data=fixture();contract=DeliveryContract(capacity=6.,vehicles=2)
    for iterative in [False,True]:
        p=ProbeProvider()
        r=improve_masked(data,[[1,2,3]],contract,p,time_limit_s=.1,
            iterative=iterative,mask_fraction=1.)
        assert p.calls and all(not a.any() for a in p.calls)
        assert r['retained_edges_preserved'] and r['same_initial_assignment']
        assert evaluate(data,r['routes'],contract)['hard'] and r['wall_s']<.15
        assert not r['online_gain_established']
    r=improve_masked(data,[[1,2,3]],contract,ProbeProvider(),time_limit_s=.1,mask_fraction=.34)
    assert r['retained_edges_preserved'] and r['report']['hard']
    print(json.dumps({'masked_adapter_stub_pass':True,'no_shortcut_leak':True,
        'same_incumbent_fallback_certified':True,'retained_edges_preserved':True,
        'real_pretrained_encoder_decoder_verified':False,'online_gain_established':False}))


if __name__=='__main__':test()
