"""One new fixed-shape head transform on a sealed intermediate step.

Engineering-only: no backbone, decoder, C0, optimizer, route selection or old QA.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
import psutil
import jax

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'research/review_3080_20261010/delivery/score_split_timed_execution_root_v1/downloaded'
OUT = ROOT/'research/review_3080_20261010/goal_round_20261013_r10_continuous_strong_delivery/DELIVERY_PADDED_HEAD_LOCAL_RESULT.json'
NPZ = OLD/'pretrained_initial_fault0_worker/request/13.npz'
sys.path.insert(0,str(OLD))
from delivery_actual_action_head_v1 import construct_head

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    psutil.Process().cpu_affinity([7])
    reg=json.loads((OUT.parent/'DELIVERY_PADDED_HEAD_LOCAL_REGISTRATION.json').read_text(encoding='utf-8'))
    if sha(NPZ)!=reg['input_sha256'] or sha(__file__)!=reg['program_sha256'] or sha(OLD/'delivery_actual_action_head_v1.py')!=reg['original_head_source_sha256']:
        raise ValueError('pre-execution identity gate')
    with np.load(NPZ,allow_pickle=False) as z:
        latent=z['latent'].copy()
        binding={k:z[k].copy() for k in ('actions','route_weights','prefix_weights','suffix_weights','slot_position_length','context','timestep')}
        saved_scores=z['scores'].copy()
    original_count=len(binding['actions']);target=reg['fixed_action_count']
    if original_count!=516 or target!=750:
        raise ValueError('registered fixed dimensions')
    padded={k:v.copy() for k,v in binding.items()}
    for key in ('actions','route_weights','prefix_weights','suffix_weights','slot_position_length'):
        v=padded[key]
        padded[key]=np.concatenate((v,np.repeat(v[:1],target-original_count,axis=0)),axis=0)
    head=construct_head(seed=20261012)
    direct=np.asarray(head(latent,binding));jax.block_until_ready(direct)
    fixed=np.asarray(head(latent,padded))[:original_count];jax.block_until_ready(fixed)
    delta=np.abs(direct.astype(np.float64)-fixed.astype(np.float64))
    saved_delta=np.abs(direct.astype(np.float64)-saved_scores.astype(np.float64))
    result=dict(input_sha256=sha(NPZ),actual_jax_version=jax.__version__,actual_numpy_version=np.__version__,local_cpu=7,
                original_count=original_count,fixed_count=target,strict_equal_count=int(np.count_nonzero(direct.view(np.uint32)==fixed.view(np.uint32))),
                strict_equal_total=original_count,max_abs_diff=float(delta.max()),nonfinite=int(np.count_nonzero(~np.isfinite(fixed))),
                local_direct_vs_remote_saved_max_abs=float(saved_delta.max()),
                direct_sha256=hashlib.sha256(direct.tobytes()).hexdigest(),fixed_sha256=hashlib.sha256(fixed.tobytes()).hexdigest(),
                scientific_model_qualified=False,full_request_budget_qualified=False,learned_output_qualified=False,stable_online_gain=False)
    with OUT.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('strict_equal_count','strict_equal_total','max_abs_diff','local_direct_vs_remote_saved_max_abs')}))
if __name__=='__main__':main()
