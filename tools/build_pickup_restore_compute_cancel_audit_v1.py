"""Generate static independent original-library reference, not producer replay."""
import ast
from pathlib import Path
ROOT=Path('D:/PyCharm_/MASKCO-Main')
old=(ROOT/'tools/audit_pickup_timed_restore_live_v1.py').read_text(encoding='utf-8')
imports=old[:old.index('ROOT=old.ROOT;')]
body=old[old.index("        name='new:'"):old.index("        private_rp.restore_state(copy.deepcopy(replay));private_env._plan_and_commit")]
body=body.replace("row['callback']==21+index","row['callback']==23")
body=body.replace("state_dict(response['compiled_state'])==replay==row['after']","state_dict(response['compiled_state'])==replay")
code=imports+'''ROOT=old.ROOT;P=old.P;OUT=P/'restore_compute_cancel_root_v1';SOURCE=P/'timed_restore_live_root_v1/ACTUAL_SHARED_PAYLOAD_5.bin'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();reg=read(OUT/'REGISTRATION.json');raw=read(OUT/'RESULT.json');initial=pickle.loads(SOURCE.read_bytes())['state'];checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    witnesses={i:decode(read(OUT/('CASE_'+str(i))/'FAULT_WITNESS.json')) for i in (1,2)}
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    with fixture.TRACE.open(encoding='utf-8') as f:dataset=decode(json.loads(next(f))['payload'])['dataset']
    details=witnesses[2]['details'];first=details['packet'];boundary={};reference_events=[];index=0
    class Stop(Exception):pass
    class Policy(C0ReserveReplanner):
        def on_reveal(self,env,i,clock,fleet,served,visible):boundary.update(clock=clock,visible=visible);raise Stop()
    class Env(StrictOnlineEnv):
        def _advance_fleet(self,i,clock,fleet,traces,served):
            value=super()._advance_fleet(i,clock,fleet,traces,served);reference_events.append(dict(sequence=6,clock=clock,event_id=self.event_id,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy()));return value
    private_rp=Policy(first['budget'],first['capacity'],first['booking_horizon'],first['contract'],0.);private_rp.restore_state(copy.deepcopy(initial['state']));private_env=Env(copy.deepcopy(dataset),first['capacity'],first['tw_speed'],first['snapshot']['num_vehicles'],private_rp,first['contract'],first['booking_horizon']);private_env.event_id=initial['event_id'];private_env.replan_count=initial['replan_count'];private_env.force_suffix={};private_fleet=copy.deepcopy(initial['fleet']);private_traces=copy.deepcopy(initial['traces']);private_served=initial['served'].copy();private_clock=initial['clock'];customers=[int(j) for j in range(1,private_env.num_nodes) if private_env.demands[0,j]>0];reveals=sorted(set(float(private_env.reveal_time[0,j]) for j in customers if private_env.reveal_time[0,j]>0))
    def snapshot():return dict(state=private_rp.export_state(),fleet=private_fleet,traces=private_traces,served=private_served,clock=private_clock,event_id=private_env.event_id,replan_count=private_env.replan_count)
    dll=ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32/SHCore.dll'));list(psutil.process_iter(['name','cmdline']));capture()
    identity={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()};identity.update({str(SOURCE):sha(SOURCE),str(Path(__file__)):sha(Path(__file__)),str(ROOT/'tools/build_pickup_restore_compute_cancel_audit_v1.py'):sha(ROOT/'tools/build_pickup_restore_compute_cancel_audit_v1.py')})
    with (OUT/'AUDIT_REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(dict(inputs=identity,scope='Stored two new stage faults; actual original raw sealed state, cancellation times/processes/loaded identities. Only second new next-reveal original admission/all20K10/C0/private precommit physical reference; no producer/engine/projector/checkpoint execution.'),f,indent=2)
    begin=capture(identity)
    with (OUT/'AUDIT_SOURCE_START.json').open('x',encoding='utf-8') as f:json.dump(begin,f,indent=2)
    partial=witnesses[1]['details']
    for k,v in initial.items():ck('partial_selected_source:'+k,old.pf.digest(v)==old.pf.digest(partial['selected_state'][k]))
    ck('partial_replanner_exact_not_complete_env',old.pf.digest(partial['restored_replanner'])==old.pf.digest(initial['state']) and not partial['environment_restore_complete'] and partial['admission_calls']==partial['physical_calls']==0)
    restore=decode(read(OUT/'CASE_2/RESTORE_IN_REQUEST.json'))
    for k,v in initial.items():ck('compute_selected_restore:'+k,old.pf.digest(v)==old.pf.digest(restore['state'][k]))
    # This view only maps stored fields to the original static reference below.
    row=dict(packet=first,response=details['response'],order=first['admission_input']['current_order'],clock=first['snapshot']['clock'],event_id=first['snapshot']['event_id'],before=details['before'],resident_before=restore['selected_digest'],callback=23)
    for _ in (0,):
''' +body+'''    for k,v in snapshot().items():ck('uninstalled_private_at_reveal:'+k,old.pf.digest(v)==old.pf.digest(details['private_at_reveal'][k]))
    ck('all_private_precommit_physical_events',len(reference_events)==len(details['events']))
    for i,(x,y) in enumerate(zip(reference_events,details['events'])):
        for k,v in x.items():ck('physical:'+str(i)+':'+k,old.pf.digest(v)==old.pf.digest(y[k]))
    head=struct.Struct('<QQ32s32s32s')
    for i,h in zip((1,2),raw['host']):
        d=OUT/('CASE_'+str(i));w=witnesses[i];b=read(d/'SOURCE_START.json');e=read(d/'SOURCE_AT_FAULT.json');envelope=(d/'RETAINED_SHARED_ENVELOPE.bin').read_bytes();epoch,size,bh,ph,sh=head.unpack_from(envelope);payload=envelope[head.size:];cp=pickle.loads(payload)
        ck(str(i)+':actual_original_raw_retained',envelope==(d/'INITIAL_SHARED_ENVELOPE.bin').read_bytes() and payload==SOURCE.read_bytes() and len(payload)==size and hashlib.sha256(payload).digest()==ph and epoch==cp['epoch']==5 and bh.hex()==cp['base_digest'] and sh.hex()==old.pf.digest(initial))
        ck(str(i)+':reached_and_cancelled_in10',h['stage']==w['stage']==i and h['sequence']==w['sequence']==6 and h['started']<=w['reached']<h['cutoff']<h['finished']<h['started']+10. and h['wall_s']==h['finished']-h['started'] and w['deadline']==h['cutoff'])
        ck(str(i)+':no_publication_new_owner_reply',not h['publication_signaled'] and h['reply_sequence']==0 and h['reaped'] and h['authority_digest']==old.pf.digest(initial) and not (d/'OWNER_STATE_6.json').exists() and not (d/'ACTUAL_WRITER_PAYLOAD_6.bin').exists())
        ck(str(i)+':fault_compute_segment_closure',b['files']==e['files'] and b['module_files']==e['module_files'] and w['loaded_closure_qualified'])
        for p,v in e['files'].items():ck(str(i)+':loaded:'+p,sha(p)==v)
        ck(str(i)+':actual_worker_gone',not psutil.pid_exists(h['pid']) or Path(psutil.Process(h['pid']).exe()).name.lower()!='python.exe')
    ck('runtime_all_checks_pass',raw['failed']==0 and raw['checks_count']==13)
    for p,v in reg['sources'].items():ck('registered:'+p,sha(p)==v)
    ck('all_actual_inputs_preserved',all(sha(p)==v for p,v in identity.items()))
    end=capture(identity)
    ck('actual_independent_closure',begin['files']==end['files'] and begin['module_files']==end['module_files'])
    for p,v in begin['files'].items():ck('audit_loaded:'+p,end['files'].get(p)==v==sha(p))
    with (OUT/'AUDIT_SOURCE_END.json').open('x',encoding='utf-8') as f:json.dump(end,f,indent=2)
    failed=[x['name'] for x in checks if not x['pass_']];result=dict(checks=checks,checks_count=len(checks),failed=len(failed),failure_names=failed,shared_original_physical_and_canonical_implementation=True,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,physical_ack_loss_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_RESTORE_COMPUTE_CANCEL.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=len(checks),failed=len(failed),failure_names=failed,events=len(reference_events),order=row['order'])));return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
'''
ast.parse(code)
with (ROOT/'tools/audit_pickup_restore_compute_cancel_v1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(code)
print('Prospective independent audit created; not executed.')
