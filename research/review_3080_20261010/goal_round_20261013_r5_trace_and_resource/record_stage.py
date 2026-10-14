"""Preserve current entry bytes before indexing the two new bounded stages."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
TREE=ROOT/'research/review_3080_20261010'
HERE=Path(__file__).resolve().parent
PICK=TREE/'pickup/reservation_trace_engineering_root_v2'
DELV=TREE/'delivery/head_reduction_diagnostic_root_v1'
BACK=TREE/'current_entries_before_goal_round_20261013_r5_trace_and_resource'
REPORT='docs/当前规划/决策与审计/双线第七轮_逐单接拒证据与编译归约定向诊断_2026-10-13.md'
RECEIPT=TREE/'reproduction/GOAL_ROUND_R5_TRACE_RESOURCE_EVIDENCE_20261013.json'

def sha(b):return hashlib.sha256(b).hexdigest()
def put_new(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)

def main():
    a=json.loads((PICK/'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
    b=json.loads((DELV/'SAVED_AUDIT.json').read_text(encoding='utf-8'))
    c=json.loads((HERE/'SERVER_RESOURCE_OBSERVATION.json').read_text(encoding='utf-8'))
    assert a['checks_count']==58 and a['failed']==0 and b['checks_count']==10 and b['failed']==0
    assert c['remote_exit_code']==0 and c['observation']['cpu15_busy_percent']>20
    assert not RECEIPT.exists() and not (BACK/'UPDATE_IDENTITY.json').exists()
    receipt={'stage':'goal_round_20261013_r5_trace_and_resource',
       'pickup_trace_cases':54,'pickup_independent_checks':58,'pickup_independent_failures':0,
       'pickup_v1_failure_preserved':True,'pickup_old_five_day_audit_failures_preserved':2,
       'delivery_synthetic_reduction_stages':7,'delivery_synthetic_first_difference':None,
       'delivery_saved_audit_checks':10,'delivery_original_34_exact_fail_preserved':True,
       'delivery_original_independent_9099_1fail_preserved':True,
       'server_available_ram_gib':c['observation']['available_ram_gib'],
       'server_cpu15_busy_percent':c['observation']['cpu15_busy_percent'],
       'server_research_dispatched':False,'science_family3_started':False,'stable_online_gain':False,
       'report':REPORT,
       'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p.read_bytes()) for p in
          [PICK/'REGISTRATION.json',PICK/'trace_reservation.py',PICK/'run_fixture.py',PICK/'FIXTURE_RESULT.json',PICK/'audit_fixture.py',PICK/'INDEPENDENT_AUDIT.json',
           DELV/'REGISTRATION.json',DELV/'run.py',DELV/'RESULT.json',DELV/'audit_saved.py',DELV/'SAVED_AUDIT.json',HERE/'REGISTRATION.json',HERE/'SERVER_RESOURCE_OBSERVATION.json']}}
    put_new(RECEIPT,(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    notes={
      '项目当前状态.md':'> 2026-10-13 双线第七轮：取货在固定54个合成公开状态上接入原预算预过滤器的逐单证据，28准许/26拒绝，独立58/0；v1运行前 objective 键错误原件保留，v2仅改从原登记读目标，不补旧五日缺日志或升级现场资格。配送本地小合成 float32 归约图七阶段即时/编译逐位一致、保存审计10/0，但原真实34/34严格FAIL仍在；3080 RAM19.423GiB、CPU15忙89.1%且比较任务运行，不派发主干。详见[第七轮记录]('+REPORT+')。稳定核心在线增益未建立。\n\n',
      'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md':'> 2026-10-13 双线第七轮：取货新逐单证据组件54例、独立58/0，不补旧五日缺失记录；配送小合成图七阶段逐位同但原34严格FAIL不变，3080 CPU15忙89.1%/RAM19.423GiB未派发。见[记录](双线第七轮_逐单接拒证据与编译归约定向诊断_2026-10-13.md)。\n\n',
      '01_取货线/README.md':'> 2026-10-13 新接拒证据组件固定54公开合成状态，原规则28准许/26拒绝，独立58/0；未接完整现场或补旧五日逐单日志。见[双线记录](../'+REPORT+').\n\n',
      '02_配送线/README.md':'> 2026-10-13 小合成编译归约七阶段逐位一致，但原34严格身份FAIL保持；3080 CPU15忙89.1%且比较任务在跑，未派发主干。见[双线记录](../'+REPORT+').\n\n'}
    paths=list(notes)+['results/research_20261010_continuation_v2/WORK_STATUS.json']
    old={p:(EXT/p).read_bytes() for p in paths}
    for p,blob in old.items():put_new(BACK/p,blob)
    new={p:n.encode('utf-8')+old[p] for p,n in notes.items()}
    status_path=paths[-1]
    status=json.loads(old[status_path])
    status['latest_goal_round_20261013_r5_trace_and_resource']={
        'report':REPORT,'evidence':str(RECEIPT.relative_to(ROOT)).replace('\\','/'),
        'pickup_trace_cases':54,'pickup_independent_checks':58,'pickup_original_five_day_audit_failures_preserved':2,
        'delivery_synthetic_stages_bitwise_equal':7,'delivery_original_compiled_head_strict_fail_preserved':True,
        'delivery_blocked_by_observed_cpu15_busy':True,'server_permission_missing':False,
        'scientific_family3_started':False,'stable_online_gain':False}
    status['latest_literature_record']=REPORT
    new[status_path]=(json.dumps(status,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    for p in paths:
        if (EXT/p).read_bytes()!=old[p]:raise RuntimeError('concurrent entry edit: '+p)
    for p in paths:
        target=EXT/p;temp=target.with_name(target.name+'.goal_r5_new')
        put_new(temp,new[p]);os.replace(temp,target)
    identity={'files':[{'path':p,'before_sha256':sha(old[p]),'after_sha256':sha(new[p])} for p in paths],
              'report_sha256':sha((EXT/REPORT).read_bytes()),'receipt_sha256':sha(RECEIPT.read_bytes())}
    put_new(BACK/'UPDATE_IDENTITY.json',(json.dumps(identity,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    assert all(sha((EXT/p).read_bytes())==sha(new[p]) for p in paths)
    print(json.dumps({'updated_entries':len(paths),'pickup_checks':a['checks_count'],'delivery_checks':b['checks_count']}))

if __name__=='__main__':main()
