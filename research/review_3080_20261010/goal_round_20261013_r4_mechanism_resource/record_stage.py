"""Archive previous entry bytes, then append this bounded stage's navigation."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / 'research/review_3080_20261010'
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
HERE = Path(__file__).resolve().parent
BACK = TREE / 'current_entries_before_goal_round_20261013_r4_mechanism_resource'
REPORT = 'docs/当前规划/决策与审计/取货五日效用机制边界与配送3080资源复核_2026-10-13.md'
RECEIPT = TREE / 'reproduction/GOAL_ROUND_R4_MECHANISM_RESOURCE_EVIDENCE_20261013.json'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write_new(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(blob)


def main():
    mech = json.loads((HERE / 'SAVED_MECHANISM_AUDIT.json').read_text(encoding='utf-8'))
    resource = json.loads((HERE / 'SERVER_RESOURCE_OBSERVATION.json').read_text(encoding='utf-8'))
    assert mech['checks_count'] == 22 and mech['failed'] == 0
    assert resource['remote_exit_code'] == 0 and resource['observation']['cpu15_busy_percent'] > 20
    assert not RECEIPT.exists() and not (BACK / 'UPDATE_IDENTITY.json').exists()
    receipt = {'stage': 'goal_round_20261013_r4_mechanism_resource',
               'saved_mechanism_checks': 22, 'saved_mechanism_failures': 0,
               'original_audit_failures_preserved': 2,
               'five_day_total_delta_utility': mech['sum_delta']['delta_utility'],
               'five_day_total_delta_revenue': mech['sum_delta']['delta_revenue'],
               'five_day_total_delta_fuel_cost': mech['sum_delta']['delta_fuel_cost'],
               'direct_reservation_log_available': False,
               'server_available_ram_gib': resource['observation']['available_ram_gib'],
               'server_cpu15_busy_percent': resource['observation']['cpu15_busy_percent'],
               'server_research_dispatched': False, 'forty_day_launched': False,
               'scientific_family3_started': False, 'stable_online_gain': False,
               'report': REPORT,
               'sha256': {name: sha((HERE/name).read_bytes()) for name in (
                   'REGISTRATION.json','analyze_saved.py','SAVED_MECHANISM_AUDIT.json',
                   'DEFAULT_SANDBOX_TRANSPORT_FAILURE.json','inspect_3080_resources.py',
                   'SERVER_RESOURCE_OBSERVATION.json')}}
    write_new(RECEIPT, (json.dumps(receipt, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    notes = {
      '项目当前状态.md': '> 2026-10-13 五日线索机制复核：新增已存结果分解22/0，效用合计+127.678537=收入+140−燃油+12.321463；第2天收入增加但少服务1单、类0服务率下降6个百分点，不能把全部收益归为低能耗类补回。原逐单过滤日志未保存，61/2FAIL保持。3080只读RAM8.3085GiB、CPU15忙95.3%且比较任务在跑，本轮不派发；既有授权有效。详见[机制和资源复核]('+REPORT+').稳定核心在线增益未建立。\n\n',
      'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md': '> 2026-10-13 五日效用分解22/0：合计+127.678537，收入+140/燃油成本+12.321463，第2天类0服务率−6个百分点；原逐单过滤次数不可复原，61/2FAIL不变。3080 CPU15忙95.3%、RAM8.3085GiB，比较进程在跑，不派发主干。详见[机制与资源复核](取货五日效用机制边界与配送3080资源复核_2026-10-13.md)。\n\n',
      '01_取货线/README.md': '> 2026-10-13 五日机制边界：已存分解22/0，效用合计+127.678537；第2天类0率下降且少服务1单，原逐单过滤日志未保存，原61/2FAIL保持。详见[复核](../'+REPORT+').\n\n',
      '02_配送线/README.md': '> 2026-10-13 3080只读复核RAM8.3085GiB、CPU15忙95.3%，比较任务仍在跑；本轮不派发完整主干请求，既有授权有效。详见[复核](../'+REPORT+').\n\n',
    }
    paths = list(notes) + ['results/research_20261010_continuation_v2/WORK_STATUS.json']
    old = {p: (EXT/p).read_bytes() for p in paths}
    for p,b in old.items(): write_new(BACK/p,b)
    new = {p: note.encode('utf-8') + old[p] for p,note in notes.items()}
    status_path = paths[-1]
    status = json.loads(old[status_path])
    status['latest_goal_round_20261013_r4_mechanism_resource'] = {
        'report': REPORT, 'evidence': str(RECEIPT.relative_to(ROOT)).replace('\\','/'),
        'saved_mechanism_checks': 22, 'saved_mechanism_failures': 0,
        'original_audit_failures_preserved': 2, 'per_order_filter_log_available': False,
        'delivery_blocked_by_observed_cpu15_busy': True,
        'server_permission_missing': False, 'stable_online_gain': False}
    new[status_path] = (json.dumps(status, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    for p in paths:
        if (EXT/p).read_bytes() != old[p]: raise RuntimeError('concurrent entry change: '+p)
    for p in paths:
        target = EXT/p
        temp = target.with_name(target.name+'.goal_r4_new')
        write_new(temp,new[p]); os.replace(temp,target)
    identity = {'files':[{'path':p,'before_sha256':sha(old[p]),'after_sha256':sha(new[p])} for p in paths],
                'report_sha256':sha((EXT/REPORT).read_bytes()),'receipt_sha256':sha(RECEIPT.read_bytes())}
    write_new(BACK/'UPDATE_IDENTITY.json',(json.dumps(identity,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    assert all(sha((EXT/p).read_bytes()) == sha(new[p]) for p in paths)
    print(json.dumps({'updated_entries':len(paths),'mechanism_checks':22,'resource_cpu15':resource['observation']['cpu15_busy_percent']}))


if __name__ == '__main__': main()
