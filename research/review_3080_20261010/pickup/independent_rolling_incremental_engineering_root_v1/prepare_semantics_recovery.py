"""Seal the unfinished witness-reference audit without rerunning production/C0."""
import ast,json,hashlib,datetime
from pathlib import Path
import psutil
BASE=Path(__file__).resolve().parent
OUT=BASE/'semantics_audit_recovery_root_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    observed=[];active=[]
    for p in psutil.process_iter(['pid','create_time','cmdline']):
        try:
            c=p.info.get('cmdline') or []
            if any(str(BASE).lower() in s.lower() for s in c) and not any('prepare_semantics_recovery.py' in s for s in c):
                observed.append(p.info)
                if any(Path(s).name in ('run.py','audit.py','supervise_audit.py') for s in c):active.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    if active:raise RuntimeError('Actual owned workload still running: '+repr(active))
    OUT.mkdir(exist_ok=False)
    old=(BASE/'audit.py').read_text(encoding='utf-8')
    new=old.replace("HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]", "OUT=Path(__file__).resolve().parent;HERE=OUT.parent;ROOT=HERE.parents[3]")
    new=new.replace("with (HERE/n).open('x'", "with (OUT/n).open('x'").replace("log=(HERE/'AUDIT_CHECKS.jsonl')", "log=(OUT/'AUDIT_CHECKS.jsonl')")
    start=new.index('    # Actual chosen immediate plans:')
    end=new.index("    with old.HISTORY.open('rb')",start)
    new=new[:start]+"    # Original 67 streamed successful checks include immediate canonical C0.\n    # Preserve and carry them by hash; no rerun of completed physical checks.\n    prior_checks=[json.loads(line) for line in (HERE/'AUDIT_CHECKS.jsonl').read_text(encoding='utf-8').splitlines()]\n    ck('prior_streamed_67_successes',len(prior_checks)==67 and all(c['pass_'] for c in prior_checks))\n    ck('prior_journal_hash',sha(HERE/'AUDIT_CHECKS.jsonl')==read(OUT/'RECOVERY_REGISTRATION.json')['prior_journal_sha256'])\n"+new[end:]
    new=new.replace("saved=read(path);j=saved['index']", "saved=decode(read(path));j=saved['index']")
    new=new.replace("ck(decision+':branch_identity:'", "ck(decision+':decoded_witness_types:'+str(j),isinstance(saved,dict) and isinstance(saved['events'],list) and isinstance(saved['terminal'],list) and isinstance(j,int))\n            ck(decision+':branch_identity:'")
    new=new.replace("    for path,h in reg['sources'].items():", "    for path,h in read(OUT/'RECOVERY_REGISTRATION.json')['frozen_inputs'].items():ck('recovery_frozen_input:'+path,sha(path)==h)\n    for path,h in reg['sources'].items():")
    new=new.replace("complete_counts=complete_counts,complete_K10", "prior_successful_checks_carried=67,complete_counts=complete_counts,complete_K10")
    ast.parse(new);(OUT/'audit.py').write_text(new,encoding='utf-8',newline='\n')
    sup=(BASE/'supervise_audit.py').read_text(encoding='utf-8');ast.parse(sup);(OUT/'supervise_audit.py').write_text(sup,encoding='utf-8',newline='\n')
    frozen={str(p):sha(p) for p in BASE.iterdir() if p.is_file() and p.name!='prepare_semantics_recovery.py'}
    put(OUT/'RECOVERY_REGISTRATION.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal='Finish only seven previously unexecuted original-false reference rollouts with exact decoded witness trajectory/utility hex/C0 gates; no production or completed canonical C0 rerun',old_audit_sha256=sha(BASE/'audit.py'),new_audit_sha256=sha(OUT/'audit.py'),supervisor_sha256=sha(OUT/'supervise_audit.py'),prior_journal_sha256=sha(BASE/'AUDIT_CHECKS.jsonl'),prior_checks_count=67,prior_check_failures=0,frozen_inputs=frozen,process_observation=observed,actual_owned_workloads_active=active,offline_cap_s=120,production_repeated=False,prior_FAIL_preserved=True,qualification_rule='All saved seven branches exact; entire K10 accept+reject and original production eligibility still required for installation',scientific_data_generated=False,stable_online_gain=False))
    print(json.dumps({'root':str(OUT),'registration_sha256':sha(OUT/'RECOVERY_REGISTRATION.json'),'old_checks':67}))
if __name__=='__main__':main()
