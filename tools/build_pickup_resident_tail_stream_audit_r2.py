"""Stream independent first-time replay of fifty saved submitted requests only."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
def main():
    src=ROOT/'research/review_3080_20261010/pickup/resident_day_tail_root_v1/FROZEN_audit_pickup_resident_day_tail_v1.py';a=src.read_text(encoding='utf-8')
    old="psutil.Process().cpu_affinity([7]);_,decode=old.codecs();raw=read(OUT/'RESULT.json');reg=read(OUT/'REGISTRATION.json');actual=decode(read(OUT/'OWNER_STATE_64.json'));initial=pickle.loads(SOURCE.read_bytes())['state'];checks=[]"
    new="""psutil.Process().cpu_affinity([7]);_,decode=old.codecs();reg=read(OUT/'REGISTRATION.json');initial=pickle.loads(SOURCE.read_bytes())['state'];checks=[]
    class StreamRecords:
        def __len__(self):return 50
        def __getitem__(self,i):
            if i<0:i+=50
            if not 0<=i<50:raise IndexError(i)
            return decode(read(OUT/('REQUEST_'+str(i+14)+'.json')))
        def __iter__(self):
            for i in range(50):yield self[i]
    actual=dict(initial_digest=old.pf.digest(initial),records=StreamRecords(),final=pickle.loads((OUT/'ACTUAL_SHARED_PAYLOAD_63.bin').read_bytes())['state'])
    raw=dict(host=[read(OUT/('HOST_'+str(i)+'.json')) for i in range(14,64)],physical_ack_loss_qualified=False,full_live_request_qualified=False,full_loaded_dependency_closure_qualified=False,stable_online_gain=False)
"""
    assert a.count(old)==1;a=a.replace(old,new)
    a=a.replace("'AUDIT_REGISTRATION.json'","'STREAM_AUDIT_REGISTRATION_R2.json'").replace("'AUDIT_SOURCE_START.json'","'STREAM_AUDIT_SOURCE_START_R2.json'").replace("'AUDIT_SOURCE_END.json'","'STREAM_AUDIT_SOURCE_END_R2.json'").replace("'INDEPENDENT_RESIDENT_DAY_TAIL.json'","'INDEPENDENT_RESIDENT_TAIL_STREAM_R2.json'")
    a=a.replace("scope='two new actual reveals, direct original admission/all40K10/C0/actual private events and actual raw checkpoint lineage; no new worker/engine/projector reference'","scope='First independent stream replay of all50 original submitted callbacks31..80. Original producer RUN_EXIT1/terminal64 unpublished and absent cumulative physical archive preserved. All1000K10, original C0, per-request whole private post-commit states/physical suffix and actual raw checkpoints; original per-advance event list unavailable, cannot certify raw intermediate event ledger. Do not rerun production, whole frozen auditor, terminal or old qualified QA.'")
    start=a.index('    reference_sequence=64;');end=a.index('    for k,v in snapshot().items():',start)
    a=a[:start]+"    ck('reference_physics_matches_stored_event_increment',len(reference_events)==actual['final']['event_id']-initial['event_id'])\n"+a[end:]
    start=a.index("    ck('same_process_and_terminal_reap'");end=a.index("    ck('no_formal_upgrade'",start)
    a=a[:start]+"    ck('same_resident50_original_terminal_failed',len({h['pid'] for h in raw['host']})==1 and all(not h['reaped'] for h in raw['host']) and read(OUT/'RUN_EXIT.json')['exit_code']==1 and not (OUT/'ACTUAL_SHARED_PAYLOAD_64.bin').exists() and not (OUT/'OWNER_STATE_64.json').exists())\n"+a[end:]
    start=a.index("    ck('natural_worker0_exit'");end=a.index("    ck('all_actual_audit_inputs_unchanged'",start)
    a=a[:start]+"    ck('all50_continuous_callbacks_no_selection',len(actual['records'])==50 and all(h['sequence']==i+14 for i,h in enumerate(raw['host'])))\n"+a[end:]
    a=a.replace("scoped_continuous_last50_and_terminal_from_actual13=True","scoped_fifty_submitted_request_replay_qualified=not failed,original_whole_run_PASS=False,registered_barrier_failures=['original_terminal64_not_selected','original_cumulative_physical_event_archive_absent'],raw_replay_failure_names=failed,reference_physical_events=len(reference_events),original_terminal_not_reexecuted=True")
    ast.parse(a)
    with (ROOT/'tools/audit_pickup_resident_tail_stream_r2.py').open('x',encoding='utf-8',newline='\n') as f:f.write(a)
    print('new stream-only scoped auditor saved, old frozen auditor unrun and unchanged')
if __name__=='__main__':main()
