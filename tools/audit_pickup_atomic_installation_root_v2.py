"""Independent old-record identity and actual live installation component replay."""
import hashlib,json,sys,math
from pathlib import Path
from dataclasses import is_dataclass
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'atomic_plan_installation_root_v2'
sys.path.insert(0,str(P))
for sub in ('evaluation','simulation','coldchain','correction_engineering'):sys.path.insert(0,str(E/'scripts'/sub))
import public_fleet as pf
import coldchain_contract,coldchain_state,strict_online_env,scenario_saa
from audit_io import decode
CLASSES={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env,scenario_saa) for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    reg=json.loads((OUT/'REGISTRATION.json').read_text());raw=decode(json.loads((OUT/'RESULT.json').read_text()),CLASSES)
    source=P/'actual_common_candidate_root_v2/RESULT.json';original=decode(json.loads(source.read_text()),CLASSES);packet=original['public_packet'];checks=[];summaries=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('actual_numpy_not_assumed',np.__version__==reg['required_numpy']=='1.26.4'==raw['ready']['runtime']['numpy'])
    ck('interpreter_start_end_identity',sys.executable==reg['actual_executable']==raw['ready']['runtime']['executable'] and sha(sys.executable)==reg['executable_sha256'])
    ck('registered_input_identity',sha(source)==reg['input_sha256']==raw['raw_input_sha256'])
    ck('registration_bound',raw['registration_sha256']==sha(OUT/'REGISTRATION.json'))
    ck('registered_actual_code',sha(ROOT/'tools/qualify_pickup_atomic_installation_root_v2.py')==reg['source_sha256'] and sha(OUT/'atomic_install.py')==reg['installer_sha256'])
    normals=[r for r in raw['records'] if 'mode' in r];ck('five_expected_old_plans',len(normals)==5 and [r['mode'] for r in normals]==reg['normal_modes'])
    start=raw['ready']['source_start']
    for p,h in start.items():ck('actual_source_or_runtime:'+p,sha(p)==h)
    for r,old in zip(normals,original['records']):
        mode=r['mode'];answer=r['prepared'];expected={int(k):list(v) for k,v in old['actual']['selected_plan'].items()}
        current,_=pf.certificate(packet,{str(k):v for k,v in answer['compiled_plan'].items()})
        ck(mode+':actual_worker_in_same_time_window',r['started']<=answer['worker_finished']<=r['received']<=r['finished']<r['deadline'] and r['deadline']-r['started']<=10.00000001)
        ck(mode+':actual_legacy_candidate_only',answer['compiled_plan']==expected=={int(k):v for k,v in r['after']['plan'].items()})
        ck(mode+':actual_current_full_physical_replay',pf.digest(current)==pf.digest(answer['certificate']) and current['qualified'])
        ck(mode+':original_certificate_exact_not_relaxed',pf.digest(answer['certificate'])==pf.digest(old['actual']['selected_certificate']))
        ck(mode+':accepted_rejected_mode_budget_same',all(r['before'][k]==r['after'][k] for k in ('accepted','rejected','status_inst','budget','mode')))
        ck(mode+':source_start_end_actual_same',answer['source_end']==start and answer['source_qualified'])
        ck(mode+':real_install_with_no_parent_C0',r['installed'] and r['reason']=='published' and r['parent_certification_called'] is False)
        ck(mode+':candidate_bytes_hash',answer['compiled_plan_sha256']==hashlib.sha256(json.dumps(answer['compiled_plan'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
        ck(mode+':complete_reply_identity',answer['request_id']==mode and answer['packet_digest']==pf.digest(packet) and answer['deadline']==r['deadline'] and answer['cpu']==[7])
        summaries.append({'mode':mode,'component_wall_s':r['finished']-r['started'],'receive_to_commit_s':r['finished']-r['received'],'real_state_installed':True,'utility_gain_claim':False})
    faults=[r for r in raw['records'] if 'fault' in r]
    expected=['changed_plan_pointer','mutated_plan_in_place','changed_admissions','changed_fleet','changed_event','expired','wrong_request_identity','unqualified_worker','tampered_prepared_plan','replayed_token','certification_delay','compile_delay']
    ck('all_registered_fault_cases_retained',[r['fault'] for r in faults]==expected)
    for r in faults:
        if r['fault']=='replayed_token':v=r['first_installed'] and not r['second_installed']
        elif r['fault'].endswith('_delay'):v=r['reply_kind']=='timeout' and r['worker_reaped'] and r['plan_pointer_unchanged']
        else:v=not r['installed'] and r['plan_pointer_unchanged']
        ck(r['fault']+':no_invalid_publication',v)
    own=[]
    for p in psutil.process_iter(['name','cmdline']):
        try:
            cmd=p.info['cmdline'] or []
            if '--worker' in cmd and any('qualify_pickup_atomic_installation_root_v' in a for a in cmd):own.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    ck('no_owned_resident_worker_remaining',not own)
    ck('old_installer_original_preserved',sha(P/'atomic_plan_installation_root_v1/atomic_install.py')==reg['installer_sha256'])
    ck('limits_still_false',not raw['public_projection_included'] and not raw['actual_admission_rejection_transaction_included'] and not raw['all_five_formal_deployment_qualified'] and not raw['stable_online_gain'])
    failed=sum(not c['pass'] for c in checks)
    out={'checks':checks,'checks_count':len(checks),'failed':failed,'summaries':summaries,'raw_sha256':sha(OUT/'RESULT.json'),
         'candidate_installation_component_qualified':failed==0,'old_input_records_unmodified':True,'full_transaction_qualified':False,'stable_online_gain':False,
         'independence_scope':'Independent stored data/source/actual state/certificate/time replay; certificate shares physical evaluator; no new policy/SAA run'}
    with (OUT/'INDEPENDENT_INSTALLATION_COMPONENT_REPLAY.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':failed,'normal_components':summaries,'full_transaction_qualified':False}));return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
