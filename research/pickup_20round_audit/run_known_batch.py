"""Execute only the preregistered16-state engineered known-packet screen."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from known_exact import solve


def run(protocol,scripts,out):
    raw=Path(protocol).read_bytes();p=json.loads(raw);cfg=p['next_bounded_batch']
    if p['version']!='P-KNOWN-EXACT-v1' or cfg['n_states']!=16 or cfg['revealed_customers_per_state']!=4:
        raise ValueError('unregistered batch')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    # Registration is actually written before generating any next-batch packet.
    (out/'BATCH_REGISTRATION.json').write_text(json.dumps(dict(protocol=p,
        protocol_sha256=hashlib.sha256(raw).hexdigest(),registered_unix=time.time(),
        parent_program_sha256='bc8cfcb51dba940af539e23194fad7c7e7caf681a1e98e68a5b296c7286632d5',
        qualification='engineered_boundary_screen_not_online_confirmation'),indent=2),encoding='utf-8')
    rng=np.random.default_rng(cfg['seed']);rows=[]
    for i in range(cfg['n_states']):
        xy=rng.uniform(.1,.9,size=(4,2))
        orders=[dict(id=j+1,reveal=0.,x=float(xy[j,0]),y=float(xy[j,1]),demand=2.,
            temp_class=cfg['classes'][j],tw_start=0.,tw_end=20.,service_time=.05) for j in range(4)]
        packet=dict(clock=0.,depot=[.5,.5],depot_deadline=22.,orders=orders,
                    accepted=[1,2,3,4],baseline=[1,2,3,4],budget=p['budget_B'])
        (out/('public_%02d.json'%i)).write_text(json.dumps(packet,indent=2),encoding='utf-8')
        try:
            r=solve(packet,scripts,out/('state_%02d'%i),p['time_cap_seconds_per_state'])
            rows.append(dict(state=i,complete=r['exact_within_declared_subproblem'],
                             gain=r['utility_gain'],hard=r['best_evaluation']['hard_feasible']))
        except Exception as exc:
            rows.append(dict(state=i,complete=False,gain=None,hard=False,failure=repr(exc)))
    summary=dict(states=rows,formal_online_gain_established=False,MaskCO_attribution=False,
                 claims_impossibility_if_negative=False)
    # No drop-and-recompute: CI requires all16 fixed states fully qualified.
    if all(r['complete'] and r['hard'] and r['gain'] is not None for r in rows):
        delta=np.array([r['gain'] for r in rows]);boot=np.random.default_rng(cfg['bootstrap_seed'])
        means=delta[boot.integers(0,16,size=(cfg['bootstrap_repetitions'],16))].mean(axis=1)
        summary.update(mean_J_change=float(delta.mean()),descriptive_ci95=np.percentile(means,[2.5,97.5]).tolist())
    else:summary['descriptive_ci95']=None
    (out/'BATCH_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--protocol',type=Path,required=True)
    a.add_argument('--scripts',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    p=a.parse_args();print(json.dumps(run(p.protocol,p.scripts,p.out)))
