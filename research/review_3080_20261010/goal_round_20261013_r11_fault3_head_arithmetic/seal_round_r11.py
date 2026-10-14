from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
EXT = WORKSPACE / 'C-VRP_Cold-chainVehicleRoutingProblem'
ROUND = ROOT / 'goal_round_20261013_r11_fault3_head_arithmetic'
PICKUP = ROOT / 'pickup/strong_independent_owner_fault3_root_v1'
REPORT = EXT / 'docs/当前规划/决策与审计/双线第十三轮_强接拒提交后取消与配送末层算术定位_2026-10-13.md'
EVIDENCE = ROOT / 'reproduction/GOAL_ROUND_R11_STRONG_FAULT_FINAL_LAYER_EVIDENCE_20261013.json'
STATUS = EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_json(path, value):
    assert not path.exists()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    inputs = [
        ROUND / 'PICKUP_FAULT3_REGISTRATION.json',
        ROUND / 'PICKUP_FAULT3_AUDIT_REGISTRATION.json',
        PICKUP / 'REGISTRATION.json',
        PICKUP / 'RESULT.json',
        PICKUP / 'RUN_EXIT.json',
        PICKUP / 'INDEPENDENT_FAULT3_AUTHORITY.json',
        ROUND / 'DELIVERY_LAYER_REGISTRATION.json',
        ROUND / 'DELIVERY_LAYER_DIAGNOSIS.json',
        ROUND / 'DELIVERY_LAYER_AUDIT_REGISTRATION.json',
        ROUND / 'INDEPENDENT_DELIVERY_LAYER_AUDIT.json',
        WORKSPACE / 'tools/qualify_pickup_strong_owner_fault3_v1.py',
        WORKSPACE / 'tools/audit_pickup_strong_owner_fault3_v1.py',
        WORKSPACE / 'tools/diagnose_delivery_padded_head_layers_v1.py',
        WORKSPACE / 'tools/audit_delivery_padded_head_layers_v1.py',
    ]
    assert all(p.is_file() for p in inputs)
    p = read(PICKUP / 'RESULT.json')
    pa = read(PICKUP / 'INDEPENDENT_FAULT3_AUTHORITY.json')
    d = read(ROUND / 'DELIVERY_LAYER_DIAGNOSIS.json')
    da = read(ROUND / 'INDEPENDENT_DELIVERY_LAYER_AUDIT.json')
    assert p['checks_count'] == 5 and p['failed'] == 0
    assert len(pa['checks']) == 12 and not pa['failed']
    assert len(da['checks']) == 16 and not da['failed']
    assert [d['stages'][s]['different_elements'] for s in ('current_features', 'input_linear', 'activation', 'output_score')] == [0, 0, 0, 3]
    assert d['strict_candidate_qualified'] is False and REPORT.is_file()
    new_json(EVIDENCE, {
        'scope': 'one fixed post-selection pickup fault and one saved-input delivery arithmetic diagnosis',
        'report': str(REPORT.relative_to(EXT)).replace('\\', '/'),
        'report_sha256': sha(REPORT),
        'input_sha256': {str(x.relative_to(WORKSPACE)).replace('\\', '/'): sha(x) for x in inputs},
        'pickup_production_checks': 5,
        'pickup_independent_checks': 12,
        'pickup_fault3_actual_total_s': p['records'][0]['wall_s'],
        'pickup_physical_advance_authoritative': False,
        'pickup_full_live_request_qualified': False,
        'delivery_layer_different_elements': [0, 0, 0, 3],
        'delivery_independent_checks': 16,
        'delivery_padded_candidate_strict_qualified': False,
        'delivery_full_request_budget_qualified': False,
        'scientific_family3_started': False,
        'stable_online_gain': False,
    })

    entries = [EXT / '项目当前状态.md',
               EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
               EXT / '01_取货线/README.md', EXT / '02_配送线/README.md', STATUS]
    assert all(x.is_file() for x in entries)
    backup = ROUND / 'current_entries_before_goal_round_r11'
    assert not backup.exists()
    backup.mkdir()
    old = {}
    for n, path in enumerate(entries):
        copied = backup / f'{n}_{path.name}'
        shutil.copyfile(path, copied)
        old[str(path.relative_to(WORKSPACE)).replace('\\', '/')] = {'sha256': sha(path), 'backup': str(copied.relative_to(ROOT)).replace('\\', '/')}
    name = '双线第十三轮_强接拒提交后取消与配送末层算术定位_2026-10-13.md'
    general = (f'> 2026-10-13 双线第十三轮：取货独立强接拒故障入口3实际取消回收8.016秒、生产5/0及独立12/0，原权威所有者未变；配送定形头特征/首层/激活逐位同，末层3/516不同，另进程16/0，严格FAIL未安装。两线稳定核心在线增益未建立。见[记录](docs/当前规划/决策与审计/{name})。\n\n')
    for path in entries[:-1]:
        if path == entries[1]:
            note = general.replace('docs/当前规划/决策与审计/', '')
        elif path == entries[2]:
            note = f'> 2026-10-13 独立接拒故障3在发布后物理推进前取消回收8.016秒，生产5/0、独立12/0；旧所有者不变，正式强动态/学习资格仍缺。见[记录](../docs/当前规划/决策与审计/{name})。\n\n'
        elif path == entries[3]:
            note = f'> 2026-10-13 保存输入层级诊断特征/首层/激活逐位同，末层3/516异，独立16/0；定形候选严格FAIL未安装，8＋2秒全请求未合格。见[记录](../docs/当前规划/决策与审计/{name})。\n\n'
        else:
            note = general
        path.write_text(note + path.read_text(encoding='utf-8'), encoding='utf-8')
    status = read(STATUS)
    status['latest_goal_round_20261013_r11_strong_fault_final_layer'] = {
        'report': str(REPORT.relative_to(EXT)).replace('\\', '/'),
        'evidence': str(EVIDENCE.relative_to(WORKSPACE)).replace('\\', '/'),
        'pickup_production_checks': 5, 'pickup_independent_checks': 12,
        'pickup_authoritative_physical_advance': False,
        'pickup_full_live_request_qualified': False,
        'delivery_different_elements_by_stage': [0, 0, 0, 3],
        'delivery_independent_checks': 16,
        'delivery_strict_candidate_qualified': False,
        'delivery_full_request_budget_qualified': False,
        'scientific_family3_started': False,
        'stable_online_gain': False,
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    new_json(ROUND / 'UPDATE_IDENTITY.json', {
        'prior_entries': old,
        'updated_entries_sha256': {str(x.relative_to(WORKSPACE)).replace('\\', '/'): sha(x) for x in entries},
        'report_sha256': sha(REPORT), 'evidence_sha256': sha(EVIDENCE),
    })


if __name__ == '__main__':
    main()
