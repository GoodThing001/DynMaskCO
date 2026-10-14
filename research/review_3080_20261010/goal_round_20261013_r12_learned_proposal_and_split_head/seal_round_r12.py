from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent
EXT = WORKSPACE / 'C-VRP_Cold-chainVehicleRoutingProblem'
ROUND = ROOT / 'goal_round_20261013_r12_learned_proposal_and_split_head'
PICKUP = ROOT / 'pickup'
REPORT = EXT / 'docs/当前规划/决策与审计/双线第十四轮_独立接拒后学习提议与分段评分头_2026-10-13.md'
EVIDENCE = ROOT / 'reproduction/GOAL_ROUND_R12_LEARNED_PROPOSAL_SPLIT_HEAD_EVIDENCE_20261013.json'
STATUS = EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def new_json(path, value):
    assert not path.exists()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    inputs = [
        ROUND / 'PICKUP_MODEL_REGISTRATION.json', ROUND / 'PICKUP_RECOVERY_REGISTRATION.json',
        PICKUP / 'learned_after_strong_root_v1/REGISTRATION.json',
        PICKUP / 'learned_after_strong_root_v1/RESULT.json',
        PICKUP / 'learned_after_strong_recovery_root_v2/common_score.py',
        PICKUP / 'learned_after_strong_recovery_execution_root_v2/REGISTRATION.json',
        PICKUP / 'learned_after_strong_recovery_execution_root_v2/RESULT.json',
        ROUND / 'DELIVERY_SPLIT_HEAD_REGISTRATION.json', ROUND / 'DELIVERY_SPLIT_HEAD_RESULT.json',
        ROUND / 'INDEPENDENT_R12_AUDIT_REGISTRATION.json', ROUND / 'INDEPENDENT_R12_SAVED_AUDIT.json',
        WORKSPACE / 'tools/qualify_pickup_learned_after_strong_v1.py',
        WORKSPACE / 'tools/qualify_pickup_learned_after_strong_v2.py',
        WORKSPACE / 'tools/qualify_delivery_split_final_head_v1.py',
        WORKSPACE / 'tools/audit_goal_round_r12_saved_v1.py',
    ]
    assert all(p.is_file() for p in inputs) and REPORT.is_file()
    a = json.loads((ROUND / 'INDEPENDENT_R12_SAVED_AUDIT.json').read_text(encoding='utf-8'))
    d = json.loads((ROUND / 'DELIVERY_SPLIT_HEAD_RESULT.json').read_text(encoding='utf-8'))
    assert a['pickup']['checks'] == 24 and a['pickup']['failures'] == 0
    assert a['delivery']['checks'] == 141 and a['delivery']['failures'] == 0
    assert a['pickup']['same_raw_candidate'] and not a['pickup']['model_learning_attribution']
    assert d['strict_equal'] and len(d['rows']) == 34 and sum(x['actions'] for x in d['rows']) == 21148
    new_json(EVIDENCE, {
        'scope': 'one old post-independent-admission pickup proposal state and 34 old saved delivery score inputs',
        'report': str(REPORT.relative_to(EXT)).replace('\\', '/'), 'report_sha256': sha(REPORT),
        'inputs_sha256': {str(p.relative_to(WORKSPACE)).replace('\\', '/'): sha(p) for p in inputs},
        'original_pickup_worker_failure_preserved': True,
        'pickup_recovery_request_seconds': a['pickup']['both_request_seconds'],
        'pickup_independent_checks': 24, 'pickup_distinct_parameters_same_candidate': True,
        'pickup_formal_learning_attribution': False, 'pickup_realized_online_utility_gain': False,
        'delivery_saved_inputs': 34, 'delivery_saved_actions': 21148,
        'delivery_new_split_head_strict_equal': True, 'delivery_independent_checks': 141,
        'delivery_original_padded_failure_preserved': True,
        'delivery_full_request_budget_qualified': False,
        'scientific_family3_started': False, 'stable_online_gain': False,
    })
    entries = [EXT / '项目当前状态.md',
               EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
               EXT / '01_取货线/README.md', EXT / '02_配送线/README.md', STATUS]
    assert all(p.is_file() for p in entries)
    backup = ROUND / 'current_entries_before_goal_round_r12'
    assert not backup.exists()
    backup.mkdir()
    old = {}
    for n, path in enumerate(entries):
        copy = backup / f'{n}_{path.name}'
        shutil.copyfile(path, copy)
        old[str(path.relative_to(WORKSPACE)).replace('\\', '/')] = {'sha256': sha(path), 'backup': str(copy.relative_to(ROOT)).replace('\\', '/')}
    name = REPORT.name
    general = (f'> 2026-10-13 双线第十四轮：取货订单77独立接拒后两模型提议工程请求2.621/2.450秒合格，但参数与logits虽异、最终候选相同，学习归因未建立；原评分接口异常失败保留。配送新分段末层头在34份保存输入/21148动作逐位同、独立141/0，完整8+2请求仍待。稳定核心在线增益未建立。见[记录](docs/当前规划/决策与审计/{name})。\n\n')
    for path in entries[:-1]:
        if path == entries[1]:
            note = general.replace('docs/当前规划/决策与审计/', '')
        elif path == entries[2]:
            note = f'> 2026-10-13 订单77后两模型实际提议请求2.621/2.450秒并认证，但不同参数与logits同候选，影子改善非学习/在线收益；原接口失败保留，正式控制仍缺。见[记录](../docs/当前规划/决策与审计/{name})。\n\n'
        elif path == entries[3]:
            note = f'> 2026-10-13 新分段评分头保存输入34/34、21148动作逐位同且独立141/0，原整头定形失败保留，真实主干8+2请求未通过。见[记录](../docs/当前规划/决策与审计/{name})。\n\n'
        else:
            note = general
        path.write_text(note + path.read_text(encoding='utf-8'), encoding='utf-8')
    status = json.loads(STATUS.read_text(encoding='utf-8'))
    status['latest_goal_round_20261013_r12_learned_proposal_split_head'] = {
        'report': str(REPORT.relative_to(EXT)).replace('\\', '/'),
        'evidence': str(EVIDENCE.relative_to(WORKSPACE)).replace('\\', '/'),
        'pickup_original_failure_preserved': True, 'pickup_recovery_independent_checks': 24,
        'pickup_models_same_candidate': True, 'pickup_learning_attribution_qualified': False,
        'delivery_split_head_saved_inputs_strict_equal': '34/34',
        'delivery_split_head_saved_actions': 21148, 'delivery_independent_checks': 141,
        'delivery_full_request_budget_qualified': False,
        'scientific_family3_started': False, 'stable_online_gain': False,
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    new_json(ROUND / 'UPDATE_IDENTITY.json', {
        'prior_entries': old,
        'updated_entries_sha256': {str(p.relative_to(WORKSPACE)).replace('\\', '/'): sha(p) for p in entries},
        'report_sha256': sha(REPORT), 'evidence_sha256': sha(EVIDENCE),
    })


if __name__ == '__main__':
    main()
