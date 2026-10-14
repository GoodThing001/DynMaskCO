"""Identity-only final verification of new source preparation, no model imports."""
import ast,json,tomllib
from pathlib import Path
from record_pickup_inactive_slot_cancel_20261012 import sha,read,create
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';O=R/'delivery/action_head_independent_replay_preparation_root_v1'

def main():
    plan=read(O/'EXECUTION_PLAN_PREPARATION.json');receipt=read(O/'PREPARATION_RECEIPT.json')
    entries=read(R/'current_entries_before_action_head_full_replay_preparation_20261012/UPDATE_IDENTITY.json')
    checks={}
    for n,h in plan['package_files'].items():
        checks['package:'+n]=sha(O/n)==h
        if n.endswith('.py'):ast.parse((O/n).read_text(encoding='utf-8'))
    backup=R/'current_entries_before_action_head_full_replay_preparation_20261012'
    for x in entries['files']:
        checks['entry:'+x['path']]=sha(E/x['path'])==x['after_sha256']
        checks['prior_bytes:'+x['path']]=sha(backup/x['path'])==x['before_sha256']
    for kind in ['report','boundary']:checks[kind+'_identity']=sha(receipt[kind+'_path'])==receipt[kind+'_sha256']
    checks['source_preparation_only']=not plan['execution_authorized'] and not plan['model_executed'] and not plan['scientific_family3_started']
    checks['no_actual_execution_artifact']=not any((O/n).exists() for n in ['EXECUTION_REGISTRATION.json','RESULT.json','RUN_EXIT.json','INDEPENDENT_ACTION_HEAD_AUDIT.json'])
    previous=R/'delivery/action_head_execution_chain_preparation_root_v1'
    draft=read(previous/'EXECUTION_CHAIN_DRAFT.json')
    checks['original_draft_preserved']=all(sha(previous/n)==h for n,h in draft['package_files'].items())
    config=tomllib.loads(Path('C:/Users/mmc/.codex/automations/maskco/automation.toml').read_text(encoding='utf-8-sig'))
    checks['automation_active30_full_replay_pending']=config['status']=='ACTIVE' and 'INTERVAL=30' in config['rrule'] and config['prompt'].endswith('maskco ACTIVE30分钟。') and '配送独立完整重载核查源码准备（仅源准备' in config['prompt']
    status=read(E/'results/research_20261010_continuation_v2/WORK_STATUS.json')
    checks['authoritative_source_pointer']=status['delivery']['actual_action_head_full_replay_preparation_directory']==str(O.relative_to(ROOT)) and status['delivery']['actual_action_head_execution_closed']
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    create(O/'FINAL_SOURCE_PREPARATION_VERIFICATION.json',dict(checks=checks,identity_checks_only=True,AST_only=True,model_executed=False,qualification=False,execution_closed=True,shared_MD_uploaded=False))
    print(json.dumps(dict(identity_checks=len(checks),failed=0,model_executed=False,execution_closed=True)))

if __name__=='__main__':main()
