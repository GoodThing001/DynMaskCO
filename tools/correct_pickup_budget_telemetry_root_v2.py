"""Correct an unqualified telemetry assertion in an exclusive successor copy."""
import ast,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'research/review_3080_20261010/pickup'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=P/'actual_masked_fleet_followup_v1/hard_admission.py'
    text=old.read_text();before=sha(old)
    assert text.count('all_in_hard_worker=True')==1
    target=P/'actual_masked_fleet_followup_root_v2';target.mkdir(exist_ok=False)
    text=text.replace('all_in_hard_worker=True', "all_in_hard_worker=False,hard_request_transaction_qualified=False,\n                    worker_coverage='admission_and_actor_candidate_only',\n                    parent_outside_preemptive_worker='public_projection_serialization_parent_certificate_restore_and_install',\n                    end_deadline_check_is_preemptive=False")
    ast.parse(text)
    new=target/'hard_admission.py'
    with new.open('x',encoding='utf-8') as f:f.write(text)
    assert sha(old)==before
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'old_source_sha256':before,'new_source_sha256':sha(new),
        'old_source_preserved':True,'current_successor_source':str(new),'change':'Only correct telemetry; transaction behavior is identical and still unqualified.',
        'old_claim':'all_in_hard_worker=True',
        'actual_control_flow':['parent deep-copy and public projection precede bounded worker select','parent full physical certificate and live install follow worker select','post-deadline rollback does not preempt parent computation'],
        'qualification':'AST syntax and unchanged non-telemetry source; no runtime qualification or new experiment',
        'old_logs_interpretation':'The flag is withdrawn as whole-transaction evidence wherever emitted by this pinned v1 source. Preserve all original log bytes.',
        'full_10_second_transaction_qualified':False,'all_five_formal_deployment_qualified':False,'stable_online_gain':False,
        'required_next':'Separate full transaction engineering must move all costly projection/certification work behind interruptible coverage and independently audit atomic in-budget commit.'}
    with (target/'TELEMETRY_CORRECTION.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'old_source_preserved':True,'syntax_pass':True,'whole_transaction_qualified':False}))
if __name__=='__main__':main()
