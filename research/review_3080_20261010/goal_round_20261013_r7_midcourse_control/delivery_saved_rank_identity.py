"""Compare archived eager/compiled action rankings; no model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
COMP=HERE.parent/'delivery/compiled_head_identity_root_v1/downloaded'
SPLIT=HERE.parent/'delivery/score_split_timed_execution_root_v1/downloaded'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    reg=json.loads((HERE/'DELIVERY_SAVED_RANK_REGISTRATION.json').read_text(encoding='utf-8'))
    assert reg['stage']=='saved_original_vs_compiled_action_rank_identity_v1'
    manifests=json.loads((COMP/'SAVED_HEAD_INPUTS.json').read_text(encoding='utf-8'))
    original=json.loads((COMP/'RESULT.json').read_text(encoding='utf-8'))
    assert len(manifests)==len(original['records'])==34
    rows=[]
    for m,r in zip(manifests,original['records']):
        assert m['arm']==r['arm'] and m['step']==r['step']
        folder='pretrained_initial_fault0_worker' if r['arm']=='pretrained_initial' else 'random888_initial_fault0_worker'
        p=SPLIT/folder/'request'/('%02d.npz'%r['step'])
        q=COMP/r['output_file']
        assert sha(p)==m['sha256']==r['input_sha256'] and sha(q)==r['output_sha256']
        with np.load(p,allow_pickle=False) as z:
            eager=z['scores'].copy();actions=z['actions'].copy()
        with np.load(q,allow_pickle=False) as z:compiled=z['scores'].copy()
        assert eager.dtype==compiled.dtype==np.float32 and eager.shape==compiled.shape and len(eager)==r['actions']
        assert actions.dtype==np.int32 and actions.shape==(len(eager),5)
        assert np.isfinite(eager).all() and np.isfinite(compiled).all()
        eager_order=np.argsort(-eager.astype(np.float64),kind='stable')
        compiled_order=np.argsort(-compiled.astype(np.float64),kind='stable')
        groups=sorted(set(int(x) for x in actions[:,0]))
        changed_customer=[]
        for customer in groups:
            indices=np.flatnonzero(actions[:,0]==customer)
            e=indices[np.argmax(eager[indices])]
            c=indices[np.argmax(compiled[indices])]
            if e!=c:changed_customer.append(customer)
        position=np.empty(len(eager),np.int32);position[compiled_order]=np.arange(len(eager),dtype=np.int32)
        adjacent_swaps=sum(int(position[a]>position[b]) for a,b in zip(eager_order,eager_order[1:]))
        rows.append(dict(index=r['index'],arm=r['arm'],step=r['step'],actions=len(eager),customers=len(groups),
            original_input_sha256=m['sha256'],compiled_output_sha256=r['output_sha256'],
            score_bitwise_unequal_count=int(np.count_nonzero(eager.view(np.uint32)!=compiled.view(np.uint32))),
            max_abs_difference=float(np.max(np.abs(eager.astype(np.float64)-compiled.astype(np.float64)))),
            top_action_same=bool(eager_order[0]==compiled_order[0]),full_stable_rank_same=bool(np.array_equal(eager_order,compiled_order)),
            original_top_index=int(eager_order[0]),compiled_top_index=int(compiled_order[0]),
            original_adjacent_swaps_in_compiled=int(adjacent_swaps),customer_top_action_changes=changed_customer))
    out=dict(stage=reg['stage'],rows=rows,records=len(rows),
        top_action_same_count=sum(x['top_action_same'] for x in rows),
        full_stable_rank_same_count=sum(x['full_stable_rank_same'] for x in rows),
        adjacent_swaps_total=sum(x['original_adjacent_swaps_in_compiled'] for x in rows),
        customer_top_action_changes_total=sum(len(x['customer_top_action_changes']) for x in rows),
        score_bitwise_unequal_total=sum(x['score_bitwise_unequal_count'] for x in rows),
        strict_output_identity_qualified=False,complete_chain_qualified=False,full_request_budget_qualified=False,
        scientific_family3_started=False,stable_online_gain=False)
    with (HERE/'DELIVERY_SAVED_RANK_RESULT.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
if __name__=='__main__':main()
