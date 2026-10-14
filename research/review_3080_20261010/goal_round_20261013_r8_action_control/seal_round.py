"""Append-only round evidence and backed-up local entry updates."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
REPORT=EXT/'docs/当前规划/决策与审计/双线第十轮_同状态接拒控制与保存动作选择复核_2026-10-13.md'
EVIDENCE=HERE.parent/'reproduction/GOAL_ROUND_R8_ACTION_CONTROL_EVIDENCE_20261013.json'
BACK=HERE.parent/'current_entries_before_goal_round_20261013_r8_action_control'
ENTRIES=['项目当前状态.md','docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
         '01_取货线/README.md','02_配送线/README.md','results/research_20261010_continuation_v2/WORK_STATUS.json']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    d=json.loads((HERE/'DELIVERY_SAVED_CHOICE_RESULT.json').read_text(encoding='utf-8'))
    da=json.loads((HERE/'INDEPENDENT_DELIVERY_CHOICE_AUDIT.json').read_text(encoding='utf-8'))
    p=json.loads((HERE/'PICKUP_UNCOND_MIDCOURSE_RESULT.json').read_text(encoding='utf-8'))
    pa=json.loads((HERE/'INDEPENDENT_PICKUP_UNCOND_AUDIT.json').read_text(encoding='utf-8'))
    assert d['count']==d['same_choice_count']==d['same_after_count']==34
    assert da['checks']==342 and not da['failures']
    assert p['decision_same'] and p['plan_same'] and pa['checks_count']==20 and pa['failed']==0
    assert REPORT.exists() and not EVIDENCE.exists() and not BACK.exists()
    evidence=dict(stage='goal_round_20261013_r8_action_control',
        registration_sha256=sha(HERE/'REGISTRATION.json'),
        pickup_execution_registration_sha256=sha(HERE/'PICKUP_EXECUTION_REGISTRATION.json'),
        pickup_audit_registration_sha256=sha(HERE/'PICKUP_AUDIT_REGISTRATION.json'),
        files={name:sha(HERE/name) for name in [
          'delivery_saved_choice.py','DELIVERY_SAVED_CHOICE_RESULT.json','audit_delivery_saved_choice.py',
          'INDEPENDENT_DELIVERY_CHOICE_AUDIT.json','pickup_uncond_midcourse.py',
          'PICKUP_UNCOND_MIDCOURSE_RESULT.json','audit_pickup_uncond_midcourse.py',
          'INDEPENDENT_PICKUP_UNCOND_AUDIT.json']},
        report_sha256=sha(REPORT),pickup_independent_checks=20,pickup_failures=0,
        pickup_same_decision=p['decision_same'],pickup_same_plan=p['plan_same'],
        delivery_original_saved_states=34,delivery_same_chosen_action=34,delivery_same_after_skeleton=34,
        delivery_independent_checks=342,delivery_failures=0,
        delivery_original_strict_bitwise_failures_preserved=34,
        pickup_original_five_day_audit_failures_preserved=2,
        formal_independent_strong_dynamic=False,learned_controls_qualified=False,
        equal_time_maskco_iterations_qualified=False,scientific_family3_started=False,
        stable_online_gain=False)
    write_json(EVIDENCE,evidence)
    BACK.mkdir(exist_ok=False)
    before={}
    for rel in ENTRIES:
        src=EXT/rel;target=BACK/rel;target.parent.mkdir(parents=True,exist_ok=True)
        raw=src.read_bytes();target.write_bytes(raw)
        assert target.read_bytes()==raw
        before[rel]=hashlib.sha256(raw).hexdigest()
    report_rel='docs/当前规划/决策与审计/'+REPORT.name
    for rel in ENTRIES[:-1]:
        src=EXT/rel
        line='\n2026-10-13 双线第十轮：取货固定旧中途状态原无条件与独立滚动均接受61、路线一致，独立20/0；配送保存34状态经完整regret竞争选择和骨架34/34一致，独立342/0。只属限定工程诊断，原严格FAIL及正式强动态/学习/等耗时门不变。见 '+report_rel+'。\n'
        with src.open('a',encoding='utf-8',newline='\n') as f:f.write(line)
    status=EXT/ENTRIES[-1]
    raw=status.read_text(encoding='utf-8')
    marker='\n}'
    assert raw.rstrip().endswith('}') and marker in raw
    cut=raw.rfind(marker)
    prefix=raw[:cut].rstrip()
    comma='' if prefix.endswith('{') else ','
    new='\n  "latest_goal_round_20261013_r8_action_control": '+json.dumps(dict(
        report=report_rel,
        evidence='research/review_3080_20261010/reproduction/'+EVIDENCE.name,
        pickup_independent_checks=20,pickup_formal_strong_dynamic_qualified=False,
        delivery_saved_choices_same=34,delivery_independent_checks=342,
        delivery_original_strict_fail_preserved=True,scientific_family3_started=False,
        stable_online_gain=False),ensure_ascii=False,indent=2).replace('\n','\n  ')
    status.write_text(prefix+comma+new+'\n}\n',encoding='utf-8',newline='\n')
    json.loads(status.read_text(encoding='utf-8'))
    after={rel:sha(EXT/rel) for rel in ENTRIES}
    write_json(BACK/'UPDATE_IDENTITY.json',dict(stage=evidence['stage'],before_sha256=before,
        after_sha256=after,backup_bytes_verified=True,evidence_sha256=sha(EVIDENCE),
        report_sha256=sha(REPORT),prior_bytes_preserved=True))
    print(json.dumps(dict(evidence=str(EVIDENCE),report=str(REPORT),backup=str(BACK),
        pickup_checks=20,delivery_checks=342),ensure_ascii=False))

if __name__=='__main__':main()
