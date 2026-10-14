"""Append this bounded stage to local navigation; preserve all previous bytes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
HERE=Path(__file__).resolve().parent
TREE=ROOT/'research/review_3080_20261010'
REPORT=EXT/'docs/当前规划/决策与审计/双线第八轮_独立接拒首事件与配送评分排序误差界_2026-10-13.md'
EVIDENCE=TREE/'reproduction/GOAL_ROUND_R6_CONTROL_RANK_EVIDENCE_20261013.json'
BACKUP=TREE/'current_entries_before_goal_round_20261013_r6_control_rank'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write_new(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    pickup=read(HERE/'INDEPENDENT_AUDIT.json');delivery=read(HERE/'INDEPENDENT_DELIVERY_RANK_AUDIT.json')
    rank=read(HERE/'DELIVERY_RANK_BOUND_RESULT.json');resources=read(HERE/'SERVER_RESOURCE_OBSERVATION.json')
    assert pickup['checks_count']==22 and pickup['failed']==0
    assert delivery['checks_count']==173 and delivery['failed']==0
    assert rank['top_index_robust_count']==34 and rank['vulnerable_adjacent_pairs_total']==9
    assert resources['observation']['cpu15_busy_percent']>20 and not resources['research_launched']
    assert read(HERE/'RESULT.json')['current_decision']=='accepted'
    artifact_paths=[HERE/n for n in ('REGISTRATION.json','EXECUTION_REGISTRATION.json','RESULT.json','AUDIT_REGISTRATION.json',
        'INDEPENDENT_AUDIT.json','DELIVERY_RANK_BOUND_REGISTRATION.json','DELIVERY_RANK_BOUND_RESULT.json',
        'INDEPENDENT_DELIVERY_RANK_AUDIT.json','SERVER_RESOURCE_OBSERVATION.json')]
    write_new(EVIDENCE,dict(stage='goal_round_20261013_r6_control_resource',
        artifacts={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in artifact_paths},
        report=str(REPORT.relative_to(EXT)).replace('\\','/'),report_sha256=sha(REPORT),
        pickup_first_actual_reveal_independent_zero_penalty_admission=True,
        pickup_audit_checks=22,pickup_audit_failures=0,pickup_formal_strong_dynamic_qualified=False,
        delivery_compiled_top_rank_bound_robust=34,delivery_close_adjacent_pairs=9,
        delivery_audit_checks=173,delivery_audit_failures=0,
        delivery_original_34_strict_fail_preserved=True,delivery_original_independent_9099_1_fail_preserved=True,
        cpu15_busy_percent=resources['observation']['cpu15_busy_percent'],
        delivery_workload_launched=False,scientific_family3_started=False,stable_online_gain=False))
    paths=['项目当前状态.md','docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
        '01_取货线/README.md','02_配送线/README.md','results/research_20261010_continuation_v2/WORK_STATUS.json']
    BACKUP.mkdir(exist_ok=False)
    befores={}
    for rel in paths:
        p=EXT/rel;data=p.read_bytes();befores[rel]=sha(p)
        (BACKUP/rel.replace('/','__')).write_bytes(data)
    report_rel='双线第八轮_独立接拒首事件与配送评分排序误差界_2026-10-13.md'
    lead={
        paths[0]:f'> 2026-10-13 双线第八轮：取货原首个现场公开揭示订单46的独立零拒单罚金接拒组件实际接受，部分C0 hard通过、另进程22/0；不等完整强动态对照。配送原34编译评分的最高分索引扰动界34/34稳健，但9对邻接排名不保证、原逐位34FAIL/独立9099/1FAIL不变；另进程173/0。3080 CPU15忙82.8%，未派发。详见[第八轮记录](docs/当前规划/决策与审计/{report_rel})。稳定MaskCO核心在线增益未建立。\n\n',
        paths[1]:f'> 2026-10-13 双线第八轮：独立接拒首实际公开事件部分C0/22项独立核查通过；配送旧34评分最高位排序有界稳健但原严门FAIL、9邻接排序未证。CPU15忙82.8%未派发。详见[记录]({report_rel})。\n\n',
        paths[2]:f'> 2026-10-13 独立零罚金滚动接拒在首实际公开订单46完成单事件职责，另进程22/0；连续现场10秒/正式强动态仍待。见[第八轮](../docs/当前规划/决策与审计/{report_rel})。\n\n',
        paths[3]:f'> 2026-10-13 原34编译评分的最高索引误差界34/34稳健，9对邻接排序未证、原逐位身份FAIL不变；173/0独立算术复核。3080 CPU15忙82.8%未派发。见[第八轮](../docs/当前规划/决策与审计/{report_rel})。\n\n'
    }
    for rel in paths[:4]:
        p=EXT/rel;p.write_bytes(lead[rel].encode('utf-8')+p.read_bytes())
    status=read(EXT/paths[4])
    status['latest_goal_round_20261013_r6_control_rank']={
        'report':'docs/当前规划/决策与审计/'+report_rel,
        'evidence':'research/review_3080_20261010/reproduction/'+EVIDENCE.name,
        'pickup_independent_checks':22,'pickup_formal_strong_dynamic_qualified':False,
        'delivery_top_rank_bound_robust':34,'delivery_close_adjacent_pairs':9,
        'delivery_independent_checks':173,'delivery_original_strict_fail_preserved':True,
        'delivery_blocked_by_observed_cpu15_busy':True,'server_permission_missing':False,
        'scientific_family3_started':False,'stable_online_gain':False}
    status['latest_literature_record']='docs/当前规划/决策与审计/'+report_rel
    (EXT/paths[4]).write_text(json.dumps(status,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    update={'files':[dict(path=rel,before_sha256=befores[rel],after_sha256=sha(EXT/rel)) for rel in paths],
        'report_sha256':sha(REPORT),'evidence_sha256':sha(EVIDENCE)}
    write_new(BACKUP/'UPDATE_IDENTITY.json',update)
if __name__=='__main__':main()
