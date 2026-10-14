"""Close a bounded dual-track stage and preserve prior navigation bytes."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
HERE=Path(__file__).resolve().parent
TREE=ROOT/'research/review_3080_20261010'
NAME='双线第九轮_中途独立接拒与原编译动作全排序核查_2026-10-13.md'
REPORT=EXT/'docs/当前规划/决策与审计'/NAME
EVIDENCE=TREE/'reproduction/GOAL_ROUND_R7_MIDCOURSE_RANK_EVIDENCE_20261013.json'
BACKUP=TREE/'current_entries_before_goal_round_20261013_r7_midcourse_rank'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def create(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    pickup=read(HERE/'INDEPENDENT_PICKUP_AUDIT.json');delivery=read(HERE/'INDEPENDENT_DELIVERY_RANK_AUDIT.json')
    ranks=read(HERE/'DELIVERY_SAVED_RANK_RESULT.json');resource=read(HERE/'SERVER_RESOURCE_OBSERVATION.json')
    assert pickup['checks_count']==23 and pickup['failed']==0
    assert delivery['checks_count']==173 and delivery['failed']==0
    assert ranks['records']==34 and ranks['full_stable_rank_same_count']==34 and ranks['customer_top_action_changes_total']==0
    assert ranks['score_bitwise_unequal_total']==16989 and ranks['adjacent_swaps_total']==0
    assert resource['observation']['cpu15_busy_percent']>20 and resource['research_launched'] is False
    names=('REGISTRATION.json','PICKUP_EXECUTION_REGISTRATION.json','PICKUP_MIDCOURSE_RESULT.json',
        'PICKUP_AUDIT_REGISTRATION.json','INDEPENDENT_PICKUP_AUDIT.json',
        'DELIVERY_SAVED_RANK_REGISTRATION.json','DELIVERY_SAVED_RANK_RESULT.json',
        'DELIVERY_AUDIT_REGISTRATION.json','INDEPENDENT_DELIVERY_RANK_AUDIT.json','SERVER_RESOURCE_OBSERVATION.json')
    create(EVIDENCE,dict(stage='goal_round_20261013_r7_midcourse_control',
        artifacts={str((HERE/n).relative_to(ROOT)).replace('\\','/'):sha(HERE/n) for n in names},
        report='docs/当前规划/决策与审计/'+NAME,report_sha256=sha(REPORT),
        pickup_order=61,pickup_independent_checks=23,pickup_formal_strong_dynamic_qualified=False,
        delivery_original_actions=21148,delivery_bitwise_score_differences=16989,
        delivery_full_stable_rank_equal_records=34,delivery_customer_best_changes=0,
        delivery_independent_checks=173,delivery_old_strict_fail_preserved=True,
        server_cpu15_busy_percent=resource['observation']['cpu15_busy_percent'],research_workload_launched=False,
        scientific_family3_started=False,stable_online_gain=False))
    files=('项目当前状态.md','docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
        '01_取货线/README.md','02_配送线/README.md','results/research_20261010_continuation_v2/WORK_STATUS.json')
    BACKUP.mkdir(exist_ok=False)
    prior={}
    for rel in files:
        p=EXT/rel;prior[rel]=sha(p);(BACKUP/rel.replace('/','__')).write_bytes(p.read_bytes())
    leads={
        files[0]:f'> 2026-10-13 双线第九轮：取货固定中途公开状态订单61独立零罚金接拒，保留车辆0已承诺59，原部分C0 hard、独立23/0；非连续强动态正式臂。配送34原/编译评分虽21148项中16989逐位不等，但保存动作完整稳定排序34/34一致、客户首选0变化，另进程173/0；原严门FAIL保持。3080 CPU15忙60%不派发。详见[第九轮记录](docs/当前规划/决策与审计/{NAME})。稳定核心在线增益未建立。\n\n',
        files[1]:f'> 2026-10-13 双线第九轮：取货中途独立接拒单组件23/0保留承诺；配送旧保存原/编译动作全排序34/34同但原逐位FAIL不改，173/0。3080 CPU15忙60%未派发。见[记录]({NAME})。\n\n',
        files[2]:f'> 2026-10-13 中途公开订单61独立零罚金滚动接拒实际接受，承诺59不动、部分C0 hard，独立23/0；真实连续强动态/全10秒仍待。见[第九轮](../docs/当前规划/决策与审计/{NAME})。\n\n',
        files[3]:f'> 2026-10-13 旧34保存输入上原/编译21148分数有16989逐位不等，但完整动作排序34/34一致、客户首选0变化，独立173/0；旧严格门仍FAIL。3080 CPU15忙60%未派发。见[第九轮](../docs/当前规划/决策与审计/{NAME})。\n\n'}
    for rel in files[:4]:
        p=EXT/rel;p.write_bytes(leads[rel].encode('utf-8')+p.read_bytes())
    status=read(EXT/files[4]);status['latest_goal_round_20261013_r7_midcourse_rank']={
        'report':'docs/当前规划/决策与审计/'+NAME,
        'evidence':'research/review_3080_20261010/reproduction/'+EVIDENCE.name,
        'pickup_independent_checks':23,'pickup_formal_strong_dynamic_qualified':False,
        'delivery_full_stable_rank_same_records':34,'delivery_bitwise_score_differences':16989,
        'delivery_independent_checks':173,'delivery_old_strict_fail_preserved':True,
        'delivery_blocked_by_observed_cpu15_busy':True,'server_permission_missing':False,
        'scientific_family3_started':False,'stable_online_gain':False}
    status['latest_literature_record']='docs/当前规划/决策与审计/'+NAME
    (EXT/files[4]).write_text(json.dumps(status,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    create(BACKUP/'UPDATE_IDENTITY.json',dict(files=[dict(path=x,before_sha256=prior[x],after_sha256=sha(EXT/x)) for x in files],
        report_sha256=sha(REPORT),evidence_sha256=sha(EVIDENCE)))
if __name__=='__main__':main()
