"""OR7 最终核验器（严格只读：绝不写回任何文件）。

从原始证据重新计算预期结果，并与现有 or7_budget_selection.json 逐字段比较：
  - 复验三组 OR7-A 双批（协议 Gate / run_id 不同 / decision parity）；
  - 复验三组 OR7-B 单批（288/288 complete / 0 fallback / 真实 solver call）；
  - 从 OR7-B 原始实例重新读取距离；
  - 调用已冻结的 select_budget()，验证 SELECTED_30；
  - 用冻结的 _mean_relative_diff / _bootstrap_ci_upper 重算 mean-relative 与
    bootstrap CI；
  - 对照 selection_source / 三档距离 / selected_budget / 三层身份 / 证据路径。

所有关键判断用显式错误检查 + 非零退出码（不用可能被 `python -O` 禁用的 assert）。
"""
import hashlib
import json
import os
import sys

_DCC = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC, '..', '..', 'common'))
for p in (_DCC, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

from aggregate_devproto import verify_aggregate, CELLS
from protocol_identity import layer_hash, CONTROL_FILES, ANALYSIS_FILES
from strict_online_runner import code_hash
from ortools_adapter import ORToolsRHDAdapter
import or7_selector as sel

RESULTS_DIR = os.path.join(_DCC, 'results')
SELECTION_PATH = os.path.join(RESULTS_DIR, 'or7_budget_selection.json')
LIMITS = (10, 30, 100)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _verify_or7a(problems):
    for lim in LIMITS:
        a = os.path.join(RESULTS_DIR, f'or7a_9x2_l{lim}_runA')
        b = os.path.join(RESULTS_DIR, f'or7a_9x2_l{lim}_runB')
        try:
            res = sel.validate_batch(lim, a, b, 2)
        except Exception as exc:  # noqa: BLE001
            problems.append(f'OR7-A limit={lim} 复验异常: {exc}')
            continue
        if not res.get('eligible'):
            problems.append(f'OR7-A limit={lim} 未通过协议 Gate')


def _verify_or7b(problems):
    results = []
    for lim in LIMITS:
        run_dir = os.path.join(RESULTS_DIR, f'or7b_9x32_l{lim}')
        try:
            verify_aggregate(run_dir)
        except Exception as exc:  # noqa: BLE001
            problems.append(f'OR7-B limit={lim} 复验异常: {exc}')
            continue
        eligible = True
        per_cell = {}
        for c in CELLS:
            dists = []
            for i in range(32):
                p = os.path.join(run_dir, c, 'instances', f'inst_{i}.json')
                if not os.path.exists(p):
                    problems.append(f'OR7-B limit={lim} 缺实例 {c} inst_{i}')
                    eligible = False
                    dists = [0.0] * 32
                    continue
                r = json.load(open(p, encoding='utf-8'))
                if not r['outcome']['complete']:
                    eligible = False
                if r['stats']['fallback_triggered_events'] != 0:
                    eligible = False
                if r['stats']['n_solver_calls'] <= 0:
                    eligible = False
                dists.append(float(r['outcome']['distance_cost']))
            per_cell[c] = dists
        results.append({'solution_limit': lim, 'eligible': eligible,
                        'per_cell': per_cell})
    return results


def _eq(actual, expected, label, problems, tol=1e-12):
    if isinstance(expected, float):
        if abs(float(actual) - expected) > tol:
            problems.append(f'{label} 不一致: {actual} != {expected}')
    elif actual != expected:
        problems.append(f'{label} 不一致: {actual!r} != {expected!r}')


def main():
    problems = []
    _verify_or7a(problems)
    results = _verify_or7b(problems)
    if not all(r['eligible'] for r in results):
        problems.append('存在未通过协议 Gate 的 OR7-B 预算')

    saved = json.load(open(SELECTION_PATH, encoding='utf-8'))

    # 选择结果
    if all(r['eligible'] for r in results):
        out = sel.select_budget(results)
        _eq(out['verdict'], 'SELECTED_30', 'verdict', problems)
        _eq(out['selected_budget'], 30, 'selected_budget', problems)
        _eq(out['reference'], saved.get('reference'), 'reference', problems)
        for lim in LIMITS:
            expected = next(r for r in results if r['solution_limit'] == lim)
            cell_mean = sum(sum(expected['per_cell'][c]) / 32 for c in CELLS) / 9
            _eq(cell_mean, saved['distances'].get(str(lim)),
                f'distances[{lim}]', problems)

        # mean-relative + bootstrap CI（用冻结实现重算）
        by_lim = {r['solution_limit']: r['per_cell'] for r in results}
        ref = by_lim[30]
        for lim in (10, 100):
            md = sel._mean_relative_diff(by_lim[lim], ref)
            ci = sel._bootstrap_ci_upper(by_lim[lim], ref)
            entry = saved.get('equivalence', {}).get(str(lim), {})
            _eq(md, entry.get('mean_relative_diff'), f'equivalence[{lim}].mean_relative_diff', problems)
            _eq(ci, entry.get('bootstrap_ci_upper'), f'equivalence[{lim}].bootstrap_ci_upper', problems)

    # 选择来源
    _eq(saved.get('selection_source'), 'OR7-B-9x32', 'selection_source', problems)

    # 三层身份
    compute = code_hash(os.path.join(_DCC, 'ortools_adapter.py'),
                        ORToolsRHDAdapter.compute_files())
    control = layer_hash('control', CONTROL_FILES, _DCC)
    analysis = layer_hash('analysis', ANALYSIS_FILES, _DCC)
    ident = saved.get('identity', {})
    _eq(compute, ident.get('compute_hash'), 'identity.compute_hash', problems)
    _eq(control, ident.get('control_hash'), 'identity.control_hash', problems)
    _eq(analysis, ident.get('analysis_hash'), 'identity.analysis_hash', problems)

    # 证据路径存在性
    for lim in LIMITS:
        for run in (f'or7a_9x2_l{lim}_runA', f'or7a_9x2_l{lim}_runB',
                    f'or7b_9x32_l{lim}'):
            if not os.path.isdir(os.path.join(RESULTS_DIR, run)):
                problems.append(f'证据目录缺失: {run}')

    if problems:
        print('OR7 FINAL VERIFY FAILED:')
        for p in problems:
            print(f'  - {p}')
        sys.exit(1)
    print('OR7 FINAL VERIFY PASS: SELECTED_30（严格只读，未写回任何文件）')


if __name__ == '__main__':
    main()
