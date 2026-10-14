"""Narrow offline replay of failed Windows path membership only; no model/C0."""
import hashlib,json
from pathlib import Path,PureWindowsPath
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'research/review_3080_20261010/delivery/timed_actual_request_execution_root_v1'
A=O/'downloaded'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    reg=read(O/'STAGE_PATH_MEMBERSHIP_RECOVERY_R2_REGISTRATION.json')
    assert sha(__file__)==reg['checker_sha256']
    assert all(sha(O/n)==h for n,h in reg['inputs'].items())
    checks=[]
    def ck(name,value):checks.append(dict(name=name,passed=bool(value)))
    original=read(A/'INDEPENDENT_TIMED_REQUEST_AUDIT.json')
    lines=[json.loads(x) for x in (A/'TIMED_AUDIT_CHECKS.jsonl').read_text(encoding='utf-8').splitlines()]
    ck('original_count_and_failures_exact',len(lines)==original['checks_count'] and [x['name'] for x in lines if not x['passed']]==original['failure_names'] and original['failed']==24)
    required={2:['request/FEATURES.npz'],3:['request/FEATURES.npz','request/INCUMBENT_C0.json','request/ENCODED.npz'],4:['request/00.npz']}
    receipts={x['arm']+'_fault'+str(x['fault_stage']):x for x in read(A/'RESULT.json')['receipts']}
    resolved=[]
    for item in reg['failed_memberships']:
        name=item['name'];tag,_,marker=name.partition(':required_stage_evidence:')
        root=A/(tag+'_worker');p=root/(marker+'.json');st=read(p);raw=st['evidence_files'];normalized={}
        ck(name+':original_marker_identity',sha(p)==item['marker_sha256'])
        ck(name+':literal_forward_slash_test_reproduces_failure',not all(f in raw for f in required[st['stage']]))
        for k,h in raw.items():
            win=PureWindowsPath(k);relative=win.as_posix()
            ck(name+':bounded_windows_path:'+k,not win.is_absolute() and not win.drive and '..' not in win.parts and relative not in normalized)
            normalized[relative]=h
            artifact=root/Path(*win.parts)
            ck(name+':actual_original_bytes:'+k,artifact.is_file() and artifact.resolve().is_relative_to(root.resolve()) and sha(artifact)==h)
        ck(name+':required_actual_artifacts',all(f in normalized for f in required[st['stage']]))
        r=receipts[tag]
        ck(name+':original_marker_time',st['deadline']==r['child_deadline'] and r['start_clock']<=st['clock']<r['child_deadline'])
        resolved.append(dict(name=name,raw_keys=list(raw),interpreted_relative_windows_keys=list(normalized),marker_sha256=sha(p)))
    ck('all_and_only24_original_failures',len(resolved)==24 and [x['name'] for x in resolved]==original['failure_names'])
    failed=sum(not x['passed'] for x in checks)
    out=dict(checks=checks,checks_count=len(checks),failed=failed,resolved_memberships=resolved,
        original_checks_count=original['checks_count'],original_failed=original['failed'],original_successful_items=original['checks_count']-original['failed'],
        original_audit_sha256=sha(A/'INDEPENDENT_TIMED_REQUEST_AUDIT.json'),original_check_stream_sha256=sha(A/'TIMED_AUDIT_CHECKS.jsonl'),
        original_audit_exit=read(A/'AUDIT_RUN_EXIT.json'),original_chain_exit=read(A/'CHAIN_EXIT.json'),
        combined_persisted_raw_replay_qualified=failed==0,original_result_rewritten=False,
        recovered_scope='Only relative Windows path spelling in24 required-stage membership checks. Existing files and recorded SHA/time preserved; no numerical replay or production retry.',
        all_registered_faults_exercised=False,complete_all_request_dependency_closure=False,complete_chain_within_budget=False,
        full_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,formal_controls_qualified=False,learned_output_qualified=False,
        model_or_C0_executed=False,scientific_family3_started=False,stable_online_gain=False)
    create(O/'INDEPENDENT_STAGE_PATH_MEMBERSHIP_RECOVERY_R2.json',out)
    print(json.dumps(dict(checks=len(checks),failed=failed,original_failed=24,scope='Windows path membership only',model_or_C0_executed=False)))
    return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
