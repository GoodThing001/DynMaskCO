"""Separate complete stable-order comparison from archived eager/compiled bytes."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
COMP=HERE.parent/'delivery/compiled_head_identity_root_v1/downloaded'
SPLIT=HERE.parent/'delivery/score_split_timed_execution_root_v1/downloaded'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    reg=json.loads((HERE/'DELIVERY_AUDIT_REGISTRATION.json').read_text(encoding='utf-8'))
    report=json.loads((HERE/'DELIVERY_SAVED_RANK_RESULT.json').read_text(encoding='utf-8'))
    assert sha(__file__)==reg['audit_sha256'] and sha(HERE/'DELIVERY_SAVED_RANK_RESULT.json')==reg['result_sha256']
    records=json.loads((COMP/'RESULT.json').read_text(encoding='utf-8'))['records']
    inputs=json.loads((COMP/'SAVED_HEAD_INPUTS.json').read_text(encoding='utf-8'))
    checks=[]
    def ck(n,x):checks.append(dict(name=n,pass_=bool(x)))
    ck('all_original_records',len(records)==len(inputs)==len(report['rows'])==34)
    total_unequal=0
    for r,m,row in zip(records,inputs,report['rows']):
        folder=('pretrained_initial_fault0_worker' if r['arm']=='pretrained_initial' else 'random888_initial_fault0_worker')
        p=SPLIT/folder/'request'/('%02d.npz'%r['step']);q=COMP/r['output_file']
        ck('input_sha:'+str(r['index']),sha(p)==m['sha256']==r['input_sha256'])
        ck('output_sha:'+str(r['index']),sha(q)==r['output_sha256'])
        with np.load(p,allow_pickle=False) as z:original=z['scores'].copy();actions=z['actions'].copy()
        with np.load(q,allow_pickle=False) as z:candidate=z['scores'].copy()
        a=sorted(range(len(original)),key=lambda i:(-float(original[i]),i))
        b=sorted(range(len(candidate)),key=lambda i:(-float(candidate[i]),i))
        unequal=sum(int(x!=y) for x,y in zip(original.view(np.uint32),candidate.view(np.uint32)))
        groups=set(int(x) for x in actions[:,0]);changes=[]
        for customer in groups:
            indices=[i for i in range(len(original)) if int(actions[i,0])==customer]
            left=max(indices,key=lambda i:(float(original[i]),-i));right=max(indices,key=lambda i:(float(candidate[i]),-i))
            if left!=right:changes.append(customer)
        ck('full_order:'+str(r['index']),a==b and row['full_stable_rank_same'] and row['top_action_same'])
        ck('bit_count:'+str(r['index']),unequal==row['score_bitwise_unequal_count'])
        ck('customer_best:'+str(r['index']),sorted(changes)==sorted(row['customer_top_action_changes']))
        total_unequal+=unequal
    ck('totals',report['records']==34 and report['top_action_same_count']==34 and report['full_stable_rank_same_count']==34 and report['adjacent_swaps_total']==0 and report['customer_top_action_changes_total']==0 and report['score_bitwise_unequal_total']==total_unequal)
    ck('barriers_unchanged',not report['strict_output_identity_qualified'] and not report['complete_chain_qualified'] and not report['full_request_budget_qualified'] and not report['scientific_family3_started'] and not report['stable_online_gain'])
    out=dict(checks=checks,checks_count=len(checks),failed=sum(not x['pass_'] for x in checks),
        old_strict_bitwise_gate_passed=False,scientific_family3_started=False,stable_online_gain=False)
    with (HERE/'INDEPENDENT_DELIVERY_RANK_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    if out['failed']:raise SystemExit(1)
if __name__=='__main__':main()
