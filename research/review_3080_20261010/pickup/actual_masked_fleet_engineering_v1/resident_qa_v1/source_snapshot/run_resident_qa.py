"""New worker/deadline integration regression; does not repeat old three-arm QA."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import time
import resident_actor as ra
import public_fleet as pf

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);pf.bootstrap(a.scripts)
    from audit_io import encode
    source_files=[Path(__file__),Path(ra.__file__),Path(ra.mf.__file__),Path(pf.__file__)]
    reg=dict(id='P-ACTUAL-WARMED-WORKER-INTEGRATION-ENGINEERING-v1',registered=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.resolve()):sha(p) for p in source_files},packet_pickle_sha256=sha(a.packet),checkpoint_sha256=sha(a.checkpoint),
        already_observed_packet=True,new_days=0,mode='pretrained',registered_shapes=[4],calls=['one certified public proposal','one .001s timeout kill'],
        fixed_setup_excluded='weight/import and declared synthetic4node prewarm before day',max_admission_cap=10,
        forbidden_final=[20260927,20260928],formal_online_gain_established=False)
    (out/'REGISTRATION.json').write_text(json.dumps(reg,indent=2),encoding='utf-8')
    packet=pickle.loads(Path(a.packet).read_bytes())
    actor=ra.ResidentActor(a.scripts,a.upstream,a.checkpoint,'pretrained',[4])
    try:
        started=time.perf_counter();selection=actor.select(packet,started,10.)
        cert,raw=pf.certificate(packet,selection['selected_plan'])
        before=time.perf_counter();timed=actor.select(packet,before,.001)
        result=dict(registration=reg,actual_ready=actor.ready,selection=selection,selected_certificate=cert,selected_known_raw=raw,
                    timeout_case=timed,timeout_worker_reaped=actor.process.poll() is not None,
                    timeout_keeps_own=timed['selected_plan']==packet['plan'],new_day_or_online_gain=False)
    finally:actor.close()
    if selection.get('qualified') is not True:raise ValueError('real worker proposal did not qualify')
    if timed.get('timeout') is not True or not result['timeout_worker_reaped'] or not result['timeout_keeps_own']:raise ValueError('actual worker timeout regression failed')
    (out/'result.json').write_text(json.dumps(encode(result),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(actual_public_proposal_qualified=True,timeout_kill_and_own_fallback=True,new_day_or_online_gain=False)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scripts',required=True);p.add_argument('--upstream',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--packet',required=True);p.add_argument('--out',required=True)
    run(p.parse_args())
