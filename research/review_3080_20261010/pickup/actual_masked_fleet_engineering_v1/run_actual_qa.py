"""Register and execute real checkpoint engineering QA; no fresh days."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
import numpy as np
import masked_fleet as mf
import public_fleet as pf

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    scripts=mf.bootstrap(a.scripts);upstream=Path(a.upstream).resolve()
    ckpt=Path(a.checkpoint).resolve();packet_path=Path(a.packet).resolve()
    reg=dict(id='P-ACTUAL-CVRP-PUBLIC-FLEET-ENGINEERING-v1',registered_actual=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        known_archived_packet=True,new_days=0,checkpoint_sha256=sha(ckpt),public_packet_pickle_sha256=sha(packet_path),
        fixed_random_architecture_seed=42,fixed_mask_seed=202610101903,mask_fraction=.3,guide_weight=1.,
        runtime_cap=10,setup_excluded='read-only weights/imports and fixed synthetic prewarm of registered node shape only',
        modes=list(mf.MODES),protected_final=[20260927,20260928],no_private_terminal_utility=True,
        source_registration={str(p.resolve()):sha(p) for p in (Path(mf.__file__),Path(__file__),Path(pf.__file__))})
    (out/'REGISTRATION.json').write_text(json.dumps(reg,indent=2),encoding='utf-8')
    from delivery_maskco import MaskCORouteProvider
    from flax import nnx
    from audit_io import encode
    packet=pickle.loads(packet_path.read_bytes());pf.validate_public(packet)
    vids,pool,ids,loc,inst,routes,comm=mf.topology(packet)
    # Fixed shape derives solely from archived observed graph; no result-based warmup.
    providers={}
    for mode,random in [('pretrained',None),('random',42)]:
        t=time.perf_counter();p=MaskCORouteProvider(ckpt,upstream,random_seed=random);p.prewarm(len(ids)+1,packet['capacity'])
        state=nnx.state(p.model,nnx.Param);flat=state.flat_state()
        h=hashlib.sha256();shapes={}
        for key,val in flat:
            arr=np.asarray(val.value);h.update(str(key).encode());h.update(arr.tobytes());shapes[str(key)]=list(arr.shape)
        p.identity.update(actual_param_hash=h.hexdigest(),param_count=len(shapes),actual_param_shapes=shapes,setup_wall=time.perf_counter()-t)
        providers[mode]=p
    rows=[]
    for mode in mf.MODES:
        p=providers['random' if mode=='random' else 'pretrained'];t=time.perf_counter()
        try:
            chosen,info=mf.propose(p,packet,scripts,mode=mode,deadline=t+10)
            row=dict(mode=mode,wall_seconds=time.perf_counter()-t,selected=chosen,proposal=info,ran=True)
        except Exception as exc:row=dict(mode=mode,wall_seconds=time.perf_counter()-t,ran=False,failure=str(exc),selected=packet['plan'])
        rows.append(row)
    paths={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).endswith('.py')}
    paths={p for p in paths if p.is_relative_to(scripts) or p.is_relative_to(upstream) or p.is_relative_to(Path(__file__).resolve().parents[1])}
    result=dict(registration=reg,rows=rows,actual_sources={str(p):sha(p) for p in sorted(paths)},
        full_encoder_decoder_used=all(r.get('proposal',{}).get('decoder_calls')==1 for r in rows),
        random_params_differ=providers['pretrained'].identity['actual_param_hash']!=providers['random'].identity['actual_param_hash'],
        known_archived_only=True,formal_online_gain_established=False,qualification_gaps=['no full online fresh deployment','three-dimensional CVRP has no resource/thermal input','small observed graph no attribution power','production persistent warm model subprocess budget not yet qualified'])
    (out/'result.json').write_text(json.dumps(encode(result),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['full_encoder_decoder_used','random_params_differ','formal_online_gain_established']}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scripts',required=True);ap.add_argument('--upstream',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--packet',required=True);ap.add_argument('--out',required=True);run(ap.parse_args())
