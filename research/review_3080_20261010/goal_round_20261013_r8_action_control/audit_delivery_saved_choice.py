"""New-process saved-array audit; no head, chain, or solver import."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'delivery'
SPLIT=ROOT/'score_split_timed_execution_root_v1'/'downloaded'
COMP=ROOT/'compiled_head_identity_root_v1'/'downloaded'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def independent_pick(packet, vals):
    cursor=0; winners=[]
    for g in packet['customers']:
        choices=[]
        for row in g['rows']:
            if row[5] is True and row[6] is True:
                choices.append((cursor, float(vals[cursor]), row))
            cursor+=1
        if choices:
            choices=sorted(choices,key=lambda x:(-x[1],float(x[2][8]),int(x[2][1]),int(x[2][2])))
            margin=None if len(choices)==1 else choices[0][1]-choices[1][1]
            winners.append((int(g['customer']),margin,choices[0]))
    assert cursor==len(vals) and winners
    winners.sort(key=lambda x:(x[1] is None,0. if x[1] is None else x[1],-x[0]),reverse=True)
    return winners[0][2][0],winners[0][1]

def main():
    result=json.loads((HERE/'DELIVERY_SAVED_CHOICE_RESULT.json').read_text(encoding='utf-8'))
    original=json.loads((COMP/'RESULT.json').read_text(encoding='utf-8'))['records']
    manifest=json.loads((COMP/'SAVED_HEAD_INPUTS.json').read_text(encoding='utf-8'))
    errors=[];checks=0
    def ck(name,test):
        nonlocal checks
        checks+=1
        if not test: errors.append(name)
    ck('all_fixed_inputs',len(result['records'])==len(original)==len(manifest)==34)
    for k,(out,raw,m) in enumerate(zip(result['records'],original,manifest)):
        folder='pretrained_initial_fault0_worker' if raw['arm']=='pretrained_initial' else 'random888_initial_fault0_worker'
        inp=SPLIT/folder/'request'/('%02d.npz'%raw['step'])
        meta=SPLIT/folder/'request'/('%02d.json'%raw['step'])
        comp=COMP/raw['output_file']
        ck('%d_input_hash'%k,digest(inp)==m['sha256']==raw['input_sha256']==out['input_sha256'])
        ck('%d_meta_hash'%k,digest(meta)==out['saved_request_sha256'])
        ck('%d_output_hash'%k,digest(comp)==raw['output_sha256']==out['compiled_output_sha256'])
        packet=json.loads(meta.read_text(encoding='utf-8'))
        with np.load(inp,allow_pickle=False) as z: eager=z['scores'].copy();acts=z['actions'].copy()
        with np.load(comp,allow_pickle=False) as z: candidate=z['scores'].copy()
        a,ar=independent_pick(packet,eager)
        b,br=independent_pick(packet,candidate)
        ck('%d_eager_index'%k,a==out['eager_index'])
        ck('%d_compiled_index'%k,b==out['compiled_index'])
        ck('%d_original_action'%k,acts[a].tolist()==packet['chosen'][:5]==out['eager_action'])
        ck('%d_candidate_action'%k,acts[b].tolist()==out['compiled_action'])
        ck('%d_regrets'%k,ar==out['eager_regret'] and br==out['compiled_regret'])
        planned=[list(route) for route in packet['before']]
        c=acts[b].tolist();planned[c[1]].insert(c[2],c[0])
        ck('%d_route'%k,planned==out['compiled_after'] and out['saved_after_same']==(planned==packet['after']))
        ck('%d_choice_flag'%k,out['same_choice']==(a==b))
    ck('aggregate',result['same_choice_count']==sum(x['same_choice'] for x in result['records'])
       and result['same_after_count']==sum(x['saved_after_same'] for x in result['records']))
    audit=dict(checks=checks,failures=errors,raw_replay_qualified=not errors,
        original_compiled_strict_failure_preserved=True,scientific_gate_pass=False)
    with (HERE/'INDEPENDENT_DELIVERY_CHOICE_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
