"""Independent fresh-process replay of one new padded-head component result."""
import hashlib,json,os,sys
from pathlib import Path
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research/review_3080_20261010/delivery/score_split_timed_execution_root_v1/downloaded'
BASE=ROOT/'research/review_3080_20261010/goal_round_20261013_r10_continuous_strong_delivery'
NPZ=OLD/'pretrained_initial_fault0_worker/request/13.npz'
sys.path.insert(0,str(OLD))
from delivery_actual_action_head_v1 import construct_head
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    psutil.Process().cpu_affinity([7]);reg=json.loads((BASE/'DELIVERY_PADDED_HEAD_AUDIT_REGISTRATION.json').read_text(encoding='utf-8'));prod=json.loads((BASE/'DELIVERY_PADDED_HEAD_LOCAL_RESULT.json').read_text(encoding='utf-8'));checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    ck('registered_input',sha(NPZ)==reg['input_sha256']);ck('producer_result_identity',sha(BASE/'DELIVERY_PADDED_HEAD_LOCAL_RESULT.json')==reg['production_sha256']);ck('head_source_identity',sha(OLD/'delivery_actual_action_head_v1.py')==reg['head_source_sha256'])
    with np.load(NPZ,allow_pickle=False) as z:
        h=z['latent'].copy();d={k:z[k].copy() for k in ('actions','route_weights','prefix_weights','suffix_weights','slot_position_length','context','timestep')};saved=z['scores'].copy()
    n=len(d['actions']);p={}
    for k,v in d.items():p[k]=np.concatenate([v,np.tile(v[:1],(750-n,)+(1,)*(v.ndim-1))],axis=0) if k in ('actions','route_weights','prefix_weights','suffix_weights','slot_position_length') else v.copy()
    ck('padded_first_rows_exact',all(np.array_equal(p[k][:n],d[k]) for k in ('actions','route_weights','prefix_weights','suffix_weights','slot_position_length')))
    model=construct_head(seed=20261012);direct=np.asarray(model(h,d));padded=np.asarray(model(h,p))[:n];a=direct.view(np.uint32);b=padded.view(np.uint32)
    ck('fresh_direct_digest',hashlib.sha256(direct.tobytes()).hexdigest()==prod['direct_sha256']);ck('fresh_padded_digest',hashlib.sha256(padded.tobytes()).hexdigest()==prod['fixed_sha256']);ck('strict_mismatch_count',int(np.count_nonzero(a!=b))==n-prod['strict_equal_count']==3);ck('max_abs_difference',float(np.max(np.abs(direct.astype(np.float64)-padded.astype(np.float64))))==prod['max_abs_diff']);ck('remote_saved_diagnostic_only',float(np.max(np.abs(direct.astype(np.float64)-saved.astype(np.float64))))==prod['local_direct_vs_remote_saved_max_abs'])
    result=dict(checks=checks,failed=[c['name'] for c in checks if not c['pass_']],strict_head_candidate_qualified=False,full_request_budget_qualified=False,stable_online_gain=False)
    with (BASE/'INDEPENDENT_DELIVERY_PADDED_HEAD_LOCAL.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(checks=len(checks),failed=result['failed'])));return int(bool(result['failed']))
if __name__=='__main__':raise SystemExit(main())
