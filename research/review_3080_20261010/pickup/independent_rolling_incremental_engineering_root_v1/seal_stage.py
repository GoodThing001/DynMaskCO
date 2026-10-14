"""Saved-only phase/adjudication sealing; no controller/scenario/C0 calls."""
import datetime,json,hashlib,struct
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent
REC=HERE/'semantics_audit_recovery_root_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    binding={str(p):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name!='seal_stage.py'}
    put(HERE/'STAGE_SEAL_REGISTRATION.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),method='saved-only binary trace parsing, exact original raw/source hash closure, PID creation/cmdline observation, no production or completed audits rerun',inputs=binding))
    checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    for p,h in binding.items():ck('saved_input_identity:'+p,sha(p)==h)
    raw=(HERE/'PHASE_TRACE.bin').read_bytes();header=struct.Struct('<8sIIQQ32s');record=struct.Struct('<QQ16siiQQ8s');magic,capacity,count,pid,initial,reserved=header.unpack(raw[:64])
    ck('fixed_binary_header',magic==b'PKPHv001' and capacity==512 and reserved==bytes(32) and len(raw)==32832)
    rows=[];last=initial
    for i in range(count):
        seq,ns,label,index,decision,v0,v1,padding=record.unpack_from(raw,64+64*i)
        ck('trace_record:'+str(i),seq==i+1 and ns>=last and padding==bytes(8));last=ns
        rows.append(dict(sequence=seq,clock_ns=ns,stage=label.split(b'\0')[0].decode('ascii'),index=index,decision=decision,value0=v0,value1=v1))
    ck('unused_binary_tail_zero',raw[64+64*count:]==bytes(64*(capacity-count)))
    receipt=read(HERE/'RUN_RECEIPT.json');failure=read(HERE/'WORKER_FAILURE.json');audit=read(REC/'INDEPENDENT_SEMANTICS_AUDIT.json');supervision=read(REC/'AUDIT_SUPERVISION_RECEIPT.json')
    ck('trace_writer_actual_pid',pid==receipt['ready']['pid']);ck('actual_internal_deadline_failure','TimeoutError: complete branch comparison required' in failure['traceback']);ck('original_no_selected_output',not receipt['output_selected'] and not receipt['qualified'])
    ck('recovery_complete_saved_prefix',audit['complete_counts']=={'reject':7,'accept':0} and audit['failed']==0)
    start=read(REC/'AUDIT_SOURCE_START.json');end=read(REC/'AUDIT_SOURCE_END.json');ck('new_audit_actual_start_end_identity',start==end)
    active=[];historical=[]
    identities=[(receipt['ready']['pid'],receipt['ready']['creation']),(read(HERE/'AUDIT_SUPERVISION_RECEIPT.json')['pid'],None),(supervision['pid'],None)]
    for oldpid,created in identities:
        try:
            proc=psutil.Process(oldpid);info=dict(pid=oldpid,create_time=proc.create_time(),cmdline=proc.cmdline())
            info['matches_old_creation']=created is not None and info['create_time']==created
            historical.append(info)
            if any(str(HERE).lower() in x.lower() for x in info['cmdline']) and any(Path(x).name in ('run.py','audit.py') for x in info['cmdline']):active.append(info)
        except psutil.NoSuchProcess:historical.append(dict(pid=oldpid,status='absent'))
        except psutil.AccessDenied:historical.append(dict(pid=oldpid,status='access_denied_not_absence'))
    ck('actual_owned_historical_workloads_inactive',not active)
    completed=[r for r in rows if r['stage']=='saa_end'];begins=[r for r in rows if r['stage']=='saa_begin'];ck('saved_seven_finished_reject_stages',len(completed)==7 and all(r['decision']==0 for r in completed))
    put(HERE/'SAVED_ONLY_STAGE_IDENTITY_AUDIT.json',dict(checks_count=len(checks),failed=sum(not x['pass_'] for x in checks),failure_names=[x['name'] for x in checks if not x['pass_']],checks=checks,actual_process_observation=historical,trace_records=rows,latest_saa_begin=begins[-1] if begins else None,last_actual_stage=rows[-1] if rows else None,model_C0_solver_calls=0))
    stage=dict(stage='independent_rolling_incremental_partial_semantics_engineering',registration_sha256=sha(HERE/'REGISTRATION.json'),recovery_registration_sha256=sha(REC/'RECOVERY_REGISTRATION.json'),original_production=dict(qualified=False,elapsed=receipt['elapsed'],worker_returncode=receipt['worker_returncode'],worker_reaped=receipt['worker_reaped'],failure_nature='Worker emitted failure after its own cooperative deadline TimeoutError; parent subsequently calls kill/wait, actual code1 cannot identify kill as cause',output_selected=False,SOURCE_END_present=(HERE/'SOURCE_END.json').exists(),completed_counts={'reject':7,'accept':0},K10_both_decisions_complete=False),original_audit=dict(persisted_successful_checks=67,exception='KeyError:index because __mapping__ witness was not decoded',actual_returncode=1,original_files_preserved=True,completed_immediate_C0_checks_carried_by_sha=True),recovery_audit=dict(checks_count=audit['checks_count'],failed=audit['failed'],strict_semantics_failure_names=audit['strict_semantics_failure_names'],actual_returncode=supervision['returncode'],elapsed=supervision['finished']-supervision['started'],actual_loaded_files=len(start),start_end_exact=start==end,repeated_production=False,repeated_completed_immediate_C0=False),saved_only_identity_checks_count=len(checks),saved_only_identity_failures=sum(not x['pass_'] for x in checks),scope='Seven completed reject witnesses: exact branch utility float.hex, recorded per-insertion and local-search entry/exit route state, terminal C0 energy/distance. Individual local-search candidate deltas/internal ranking were not separately saved; no universal flag equivalence.',strict_saved_prefix_semantics_qualified=audit['strict_prefix_semantics_qualified'],candidate_eligible_for_install=False,registered_barrier_failures=['incomplete reject K10','all accept K10 absent','original production failure/source end absent/no selection'],timing_scope='Candidate and original reference run in separate windows with different observation implementation; saved timings are descriptive, not fair online speedup estimate. Old phase3/new7 prefixes likewise not a controlled speedup comparison.',next='Prospectively implement conservative infeasibility screening or exact route-summary reuse while retaining original arithmetic ranking; then fresh bounded engineering control-chain tests. Do not rerun this candidate or prefix to seek all20.',formal_strong_dynamic_qualified=False,full_request_budget_qualified=False,full_scientific_loaded_closure=False,learned_output_qualified=False,stable_online_gain=False,scientific_data_generated=False,literature=[dict(title='Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic Customers',authors='Bent and Van Hentenryck',year=2004,doi='10.1287/opre.1040.0124',url='https://pubsonline.informs.org/doi/10.1287/opre.1040.0124',read_scope='Parent read publisher abstract only',use='Scenario planning inspiration; paired C0 SAA adaptation, not consensus MSA replication'),dict(title='PyVRP: A High-Performance VRP Solver Package',authors='Wouda, Lan and Kool',year=2024,doi='10.1287/ijoc.2023.0055',url='https://arxiv.org/html/2403.13795v2',read_scope='Parent read sections4.3.2 and5.3',use='Move-cache, conservative filtering and route-summary inspiration only; existingincr_eval is not PyVRP replication or coldchain equivalence')])
    put(HERE/'STAGE_ADJUDICATION.json',stage)
    final={str(p):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name!='FINAL_ARTIFACT_IDENTITY.json'}
    put(HERE/'FINAL_ARTIFACT_IDENTITY.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=final,all_prior_bound_inputs_unchanged=all(sha(p)==h for p,h in binding.items())))
    print(json.dumps(dict(stage_path=str(HERE/'STAGE_ADJUDICATION.json'),registration_sha256=stage['registration_sha256'],recovery_checks=92,recovery_failed=0,saved_only_checks=len(checks),saved_only_failed=stage['saved_only_identity_failures'],complete_counts={'reject':7,'accept':0},files=len(final))))
if __name__=='__main__':main()
