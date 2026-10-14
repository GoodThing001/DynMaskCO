"""Independent original terminal physics/evaluator and actual raw-byte identities."""
import os,sys,json,copy,pickle,struct,hashlib,ctypes
from pathlib import Path
import numpy as np,psutil
import qualify_pickup_joint_policy_transaction_root_v3 as old
import qualify_pickup_observed_day_extension_v1 as observed
from strict_online_env import StrictOnlineEnv
from run_exp_energy_c0 import C0ReserveReplanner
from coldchain_evaluator_a1 import evaluate_trace_a1
from pickup_loaded_closure_v1 import capture,sha
ROOT=old.ROOT;O=old.P/'terminal_restore_root_v2';SOURCE=old.P/'resident_day_tail_root_v1/ACTUAL_SHARED_PAYLOAD_63.bin'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();template,data=observed.setup();initial=pickle.loads(SOURCE.read_bytes())['state'];t=decode(read(O/'TERMINAL.json'));actual=decode(read(O/'TERMINAL_EVENTS.json'));r=read(O/'RESULT.json');restore=decode(read(O/'RESTORE_IN_REQUEST.json'));reg=read(O/'REGISTRATION.json');events=[];checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            value=super()._advance_fleet(i,clock,fleet,traces,served);events.append(dict(sequence=64,clock=clock,event_id=self.event_id,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()));return value
    class RP(C0ReserveReplanner):
        def on_reveal(self,*args):raise AssertionError('no new orders allowed at original terminal source')
    rp=RP(template['budget'],template['capacity'],template['booking_horizon'],template['contract'],0.);rp.restore_state(copy.deepcopy(initial['state']));env=Env(copy.deepcopy(data),template['capacity'],template['tw_speed'],template['snapshot']['num_vehicles'],rp,template['contract'],template['booking_horizon']);env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];env.force_suffix={};fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy();customers=[int(j) for j in range(1,env.num_nodes) if env.demands[0,j]>0];reveals=sorted(set(float(env.reveal_time[0,j]) for j in customers if env.reveal_time[0,j]>0));clock=initial['clock']
    ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name','cmdline']));capture();identity={str(p):sha(p) for p in list(O.iterdir())+[SOURCE,Path(__file__)] if p.is_file()};save('AUDIT_REGISTRATION.json',dict(inputs=identity,scope='New terminal-only original loop/evaluator exact replay from actual selected63,0admission/0SAA. All10 physical event records/rawbytes/complete80 evaluation/time/load/process qualification only. Original parent runFAIL preserved, no50 rerun.'));start=capture(identity);save('AUDIT_SOURCE_START.json',start)
    env._run_loop(0,clock,sum(v<=clock+1e-6 for v in reveals),fleet,traces,served,customers,reveals,float(env.tw_end[0].max()));final=dict(state=rp.export_state(),fleet=fleet,traces=traces,served=served,clock=events[-1]['clock'],event_id=env.event_id,replan_count=env.replan_count)
    ck('ten_actual_physical_events',len(events)==len(actual)==10)
    for k,v in final.items():ck('original_terminal_field:'+k,old.pf.digest(v)==old.pf.digest(t['state'][k]))
    for i,(x,y) in enumerate(zip(events,actual)):
        for k,v in x.items():ck('actual_physical:'+str(i)+':'+k,old.pf.digest(v)==old.pf.digest(y[k]))
    ev=evaluate_trace_a1(traces,{k:v[0] for k,v in data.items()},template['contract'],set(rp._accepted),set(rp._rejected),template['budget'],template['penalties'])
    for k,v in ev.items():ck('original_complete80_C0:'+k,old.pf.digest(v)==old.pf.digest(t['evaluation'][k]))
    ck('complete_hard_return',ev['hard_feasible'] and not ev['failures'] and all(v.status=='closed' for v in fleet) and set(np.flatnonzero(served))-{0}==set(rp._accepted) and set(customers)==set(rp._accepted)|set(rp._rejected))
    for k,v in initial.items():ck('actual_selected63_restored:'+k,old.pf.digest(v)==old.pf.digest(restore['state'][k]))
    ck('source_restore_no_calls',restore['source_epoch']==63 and restore['source_payload_sha256']==sha(SOURCE) and restore['admission_calls']==restore['physical_calls']==t['admission_calls']==t['total_new_requests']==0 and t['resident_before']==old.pf.digest(initial))
    payload=(O/'ACTUAL_WRITER_PAYLOAD_64.bin').read_bytes();cp=pickle.loads(payload);envelope=(O/'ACTUAL_SHARED_ENVELOPE_64.bin').read_bytes();head=struct.Struct('<QQ32s32s32s');epoch,size,bh,ph,sh=head.unpack_from(envelope)
    ck('terminal_actual_raw_bytes',epoch==64 and size==len(payload) and envelope==(O/'RETAINED_ENVELOPE.bin').read_bytes() and payload==envelope[head.size:]==(O/'ACTUAL_SHARED_PAYLOAD_64.bin').read_bytes()==(O/'RETAINED_PAYLOAD.bin').read_bytes() and ph==hashlib.sha256(payload).digest() and sh.hex()==old.pf.digest(final) and bh.hex()==cp['base_digest']==old.pf.digest(initial) and old.pf.digest(cp['state'])==old.pf.digest(final))
    h=r['host'];v=h['validation'];ck('restore_terminal_choice_reap_time',h['started']<=t['resident_started']<=restore['started']<=restore['finished']<=t['resident_finished']<=v['started']<=v['finished']<=h['chosen_at']<h['cutoff'] and h['finished']<h['started']+10. and h['wall_s']==h['finished']-h['started'] and h['ack_present'] and h['event_present'] and h['reaped'] and r['actual_worker_exit']==0)
    begin=read(O/'SOURCE_START.json');end=read(O/'SOURCE_END_64.json');ck('actual_workload_closure',begin['files']==end['files'] and begin['module_files']==end['module_files'] and t['loaded_closure_qualified'])
    for p,h in begin['files'].items():ck('production_actual_bytes:'+p,sha(p)==h)
    for p,h in identity.items():ck('audit_input:'+p,sha(p)==h)
    for p,h in reg['sources'].items():ck('registration:'+p,sha(p)==h)
    finish=capture(identity);save('AUDIT_SOURCE_END.json',finish)
    for p,h in start['files'].items():ck('audit_actual_bytes:'+p,sha(p)==h==finish['files'].get(p))
    ck('audit_loaded_closure',start['files']==finish['files'] and start['module_files']==finish['module_files']);ck('actual_resident_exited',not [p.pid for p in psutil.process_iter(['cmdline']) if p.pid!=os.getpid() and any(Path(x).name=='qualify_pickup_terminal_restore_v2.py' for x in p.info['cmdline'] or [])]);ck('original_whole_run_not_PASS',read(old.P/'resident_day_tail_root_v1/RUN_EXIT.json')['exit_code']==1 and not r['whole_previous_run_PASS'] and not r['full_live_request_qualified'])
    failures=[x['name'] for x in checks if not x['passed']];save('INDEPENDENT_TERMINAL_RESTORE.json',dict(checks=checks,checks_count=len(checks),failed=len(failures),failure_names=failures,scoped_terminal_only_qualified=not failures,original_whole_run_PASS=False,full_live_request_qualified=False,stable_online_gain=False));print(json.dumps(dict(checks=len(checks),failed=len(failures),failures=failures)));return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
