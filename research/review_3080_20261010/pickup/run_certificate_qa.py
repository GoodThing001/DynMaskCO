"""Re-use one archived engineering state; no new day/model/server is generated."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from public_fleet import bootstrap,export_public,digest,bounded_select,certificate,source_hashes


def run(scripts,archive,out):
    scripts=bootstrap(scripts);archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    from scenario_saa import SaaReplanner,make_c0_contract_v2
    from recourse_snapshot import restore_recourse_snapshot
    from strict_online_env import StrictOnlineEnv
    from coldchain_contract import ColdChainContractV2
    from audit_io import decode,encode
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [archive/'private_capture.pkl',archive/'baseline_trace.jsonl',archive/'QA_RESULT.json']}
    # Genuine registration before reading/processing the already-observed state.
    reg=dict(protocol_id='P-PUBLIC-FLEET-CERTIFICATE-ENGINEERING-v1',registered=datetime.now(timezone.utc).isoformat(),
        archive_source_sha256=inputs,source_start=source_hashes(scripts),
        fixed_archive_seed=202610101801,scope='one already observed engineer state; no new online gain inference',
        actor_scope='known-only <=5 order two-vehicle ordered partitions plus own incumbent; ten-second submission cap',
        forbidden_final=[20260927,20260928])
    (out/'QA_REGISTRATION.json').write_text(json.dumps(reg,indent=2),encoding='utf-8')
    saved=pickle.loads((archive/'private_capture.pkl').read_bytes());snap=saved['private']
    start=json.loads((archive/'baseline_trace.jsonl').open(encoding='utf-8').readline())
    # Dataset can be independently decoded without decoding dataclass contract.
    mapping=start['payload']['__mapping__']
    ds=decode(next(v for k,v in mapping if k=='dataset'))
    contract=make_c0_contract_v2()
    class Sampler:
        def sample(self,*args):raise RuntimeError('QA never calls private-day admission/sampling')
    def environment(data):
        rp=SaaReplanner(710.6548152249378,50.,16.,contract,0.,Sampler(),{0:5.,1:10.,2:15.},K=10,time_limit=10.)
        rp.restore_state(snap['replanner_state'])
        e=StrictOnlineEnv(data,50.,1.5,3,rp,contract,16.)
        fleet,traces,served=restore_recourse_snapshot(snap)
        return e,fleet,traces,served
    env,fleet,traces,served=environment(ds)
    universe=list(snap['customer_universe']);clock=float(snap['clock'])
    packet=export_public(env,0,clock,snap['event_id'],snap['reveal_idx'],fleet,traces,served,universe)
    hidden=[o for o in universe if ds['reveal_time'][0,o]>clock+1e-6]
    alt=copy.deepcopy(ds)
    for key in ('coords','demands','temp_class','tw_start','tw_end','service_time'):
        alt[key][0,hidden]=0
    e2,f2,t2,s2=environment(alt)
    alt_packet=export_public(e2,0,clock,snap['event_id'],snap['reveal_idx'],f2,t2,s2,universe)
    invariance=digest(packet)==digest(alt_packet)
    selected=bounded_select(packet,scripts)
    repeated=bounded_select(alt_packet,scripts)
    basecert,basetr=certificate(packet,packet['plan'])
    bestcert,besttr=certificate(packet,selected['selected_plan'])
    bare=copy.deepcopy(packet);bare['contract']=ColdChainContractV2();bare['contract_hash']=bare['contract'].contract_hash
    bare_rejected=False
    try:certificate(bare,bare['plan'])
    except ValueError as exc:bare_rejected='unit speed' in str(exc)
    result=dict(registration=reg,public_packet_sha=digest(packet),future_mutation_packet_invariant=invariance,
        future_mutation_choice_invariant=selected['selected_plan']==repeated['selected_plan'],
        baseline_certificate=basecert,selected_certificate=bestcert,selection=selected,
        bare_contract_unit_mismatch_rejected=bare_rejected,formal_online_gain_established=False,
        maskco_attribution=False,full_future_day_used_for_selection=False,
        qualification_gaps=['single archived engineering state; no fresh day/full online deployment',
            'at most5 orders/2vehicles append family; not global assignment optimum',
            'known immediate-close score lacks future opportunity costs',
            'no MaskCO model or guide in actor; nonlearning control only'])
    for name,value in [('public_packet',packet),('baseline_known_raw',dict(certificate=basecert,traces=basetr)),
                       ('selected_known_raw',dict(certificate=bestcert,traces=besttr))]:
        (out/(name+'.json')).write_text(json.dumps(encode(value),indent=2),encoding='utf-8')
    (out/'public_packet.pkl').write_bytes(pickle.dumps(packet,protocol=4))
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scripts',required=True);p.add_argument('--archive',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=run(a.scripts,a.archive,a.out)
    print(json.dumps({k:r[k] for k in ('future_mutation_packet_invariant','future_mutation_choice_invariant','bare_contract_unit_mismatch_rejected','formal_online_gain_established')}))
