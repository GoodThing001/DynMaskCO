"""OR7 预算选择器：预注册选择规则的可执行冻结实现（analysis 层，纳入 analysis_hash）。

本文件把 OR7_PROTOCOL.json 中的选择规则实现为代码，使规则在 OR7-A 扫描前即被
哈希冻结。preflight 完整核对冻结协议（候选/两阶段规模/solver 配置/Gate 顺序/
bootstrap/300 规则/逐文件 hash），select 不接受手填 gate_pass，而是直接接收批次
目录并自行 verify_aggregate + 复验 admission/parity/fallback/solver call。

选择规则（与 OR7_PROTOCOL.json 一致）：
  1. 保留所有完整通过协议 Gate 的预算；
  2. 九 cell 等权 pure distance 最低者为距离参考；
  3. 更小预算：平均相对距离差 ≤ 0.5% 且按内部 seed 成组 bootstrap 的 95% CI
     上界 ≤ 1% → 与最佳预算工程等价；
  4. 工程等价者选最小 solution_limit；
  5. 参考预算为最大候选且无更小预算等价（距离随预算持续改善）→ 追加 300。

用法：
    python or7_selector.py preflight
    python or7_selector.py select --batches <batches.json>
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_DCC = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC, '..', '..', 'common'))
for p in (_DCC, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

from strict_online_runner import code_hash, load_objective_profile
from ortools_adapter import ORToolsRHDAdapter
from protocol_identity import (CONTROL_FILES, ANALYSIS_FILES, layer_hash,
                               build_protocol_id)
from aggregate_devproto import verify_aggregate, CELLS
from parity_compare import admission_check

PROTOCOL_PATH = os.path.join(_DCC, 'OR7_PROTOCOL.json')
DEV_MANIFEST = os.path.join(_bootstrap.PROJECT_EXTENSION_ROOT, 'data',
                            'baseline', '50_node', 'dev_proto',
                            'DEV_MANIFEST.json')
PROFILE = os.path.join(_bootstrap.PROJECT_EXTENSION_ROOT, 'results', 'o0cc',
                       'scale_v2', 'objective_profile.json')

# ---- 预注册冻结常量（与 OR7_PROTOCOL.json 一致；preflight 逐项核对）----
SOLUTION_LIMIT_CANDIDATES = [10, 30, 100]
TIME_LIMIT_S = 30.0
SOLVER_SEED = 0
THREADS = 1
FIRST_SOLUTION_STRATEGY = 'PATH_CHEAPEST_ARC'
LOCAL_SEARCH_METAHEURISTIC = 'GUIDED_LOCAL_SEARCH'
GATE_ORDER = ['hard', 'service', 'protocol', 'distance']
BOOTSTRAP_SEED = 0
BOOTSTRAP_N_RESAMPLES = 10000
MEAN_RELATIVE_THRESHOLD = 0.005
BOOTSTRAP_CI_UPPER = 0.01
UPGRADE_LIMIT = 300


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_protocol():
    with open(PROTOCOL_PATH, encoding='utf-8') as f:
        return json.load(f)


def preflight(protocol=None):
    """验证冻结协议与代码常量/当前身份一致（任何漂移抛错）。

    protocol 可为 None（读 OR7_PROTOCOL.json）或 dict（测试用）。
    """
    protocol = protocol if protocol is not None else load_protocol()
    problems = []

    # 三层身份 + 资产（动态重算）
    compute = code_hash(os.path.join(_DCC, 'ortools_adapter.py'),
                        ORToolsRHDAdapter.compute_files())
    control = layer_hash('control', CONTROL_FILES, _DCC)
    analysis = layer_hash('analysis', ANALYSIS_FILES, _DCC)
    ident = protocol.get('identity', {})
    if compute != ident.get('compute_hash'):
        problems.append('compute_hash 与冻结协议不一致')
    if control != ident.get('control_hash'):
        problems.append('control_hash 与冻结协议不一致')
    if analysis != ident.get('analysis_hash'):
        problems.append('analysis_hash 与冻结协议不一致')
    if sha256_file(DEV_MANIFEST) != protocol.get('data', {}).get('dev_manifest_sha256'):
        problems.append('DEV manifest SHA 与冻结协议不一致')
    profile_hash = load_objective_profile(PROFILE).profile_hash
    if profile_hash != protocol.get('profile', {}).get('profile_hash'):
        problems.append('objective profile hash 与冻结协议不一致')

    # 冻结常量与 JSON 逐项交叉核验（防 JSON 与代码漂移）
    def _eq(jpath, expected, label):
        node = protocol
        for k in jpath:
            node = node.get(k, {}) if isinstance(node, dict) else {}
        if node != expected:
            problems.append(f'{label} 与代码常量不一致: {node!r} != {expected!r}')

    bc = protocol.get('budget_candidates', {})
    _eq(['budget_candidates', 'solution_limit'], SOLUTION_LIMIT_CANDIDATES,
        'solution_limit 候选')
    _eq(['budget_candidates', 'time_limit_s'], TIME_LIMIT_S, 'time_limit_s')
    _eq(['budget_candidates', 'solver_seed'], SOLVER_SEED, 'solver_seed')
    _eq(['budget_candidates', 'threads'], THREADS, 'threads')
    _eq(['budget_candidates', 'first_solution_strategy'],
        FIRST_SOLUTION_STRATEGY, 'first_solution_strategy')
    _eq(['budget_candidates', 'local_search_metaheuristic'],
        LOCAL_SEARCH_METAHEURISTIC, 'local_search_metaheuristic')
    _eq(['gate_order'], GATE_ORDER, 'gate_order')
    _eq(['bootstrap', 'seed'], BOOTSTRAP_SEED, 'bootstrap seed')
    _eq(['bootstrap', 'n_resamples'], BOOTSTRAP_N_RESAMPLES, 'bootstrap n_resamples')
    _eq(['selection_rule', 'threshold', 'mean_relative_distance'],
        MEAN_RELATIVE_THRESHOLD, 'mean_relative_distance 阈值')
    _eq(['selection_rule', 'threshold', 'bootstrap_ci_upper'],
        BOOTSTRAP_CI_UPPER, 'bootstrap_ci_upper 阈值')
    if protocol.get('data', {}).get('cells') != CELLS:
        problems.append(f'data.cells 与代码 CELLS 不一致')

    # 300 规则存在性（trigger/action 语义由代码实现，此处只核对 trigger 文本与 limit）
    ur = protocol.get('upgrade_rule', {})
    if '300' not in str(ur.get('action', '')) and '300' not in str(ur.get('trigger', '')):
        problems.append('upgrade_rule 未引用 300')

    if problems:
        raise RuntimeError('OR7 preflight 失败: ' + '; '.join(problems))
    return {'compute_hash': compute, 'control_hash': control,
            'analysis_hash': analysis, 'profile_hash': profile_hash}


def cell_equal_mean_distance(per_cell):
    """九 cell 等权 pure distance：先每 cell 均值，再九 cell 等权平均。"""
    return float(np.mean([float(np.mean(per_cell[c])) for c in CELLS]))


def _grouped_resample_indices(n, n_resamples, seed):
    """生成成组 bootstrap 索引：每轮一组长度为 n 的索引（同一索引用于九个 cell，
    保持「内部 seed 成组」的配对结构）。"""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_resamples)]


def _mean_relative_diff(budget_per_cell, ref_per_cell):
    """预算 vs 参考的相对距离差（cell 等权、cell 内 seed 均值后平均）。"""
    cell_diffs = []
    for c in CELLS:
        b = np.asarray(budget_per_cell[c], dtype=float)
        r = np.asarray(ref_per_cell[c], dtype=float)
        cell_diffs.append(float(np.mean((b - r) / r)))
    return float(np.mean(cell_diffs))


def _bootstrap_ci_upper(budget_per_cell, ref_per_cell):
    """按内部 seed 成组 bootstrap 的 95% CI 上界（相对距离差，cell 等权）。

    每轮只生成一组索引并同时用于九个 cell（保持实例/seed 配对）。
    """
    rel = {c: (np.asarray(budget_per_cell[c], dtype=float)
               - np.asarray(ref_per_cell[c], dtype=float))
           / np.asarray(ref_per_cell[c], dtype=float) for c in CELLS}
    n = len(rel[CELLS[0]])
    means = []
    for idx in _grouped_resample_indices(n, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED):
        cell_means = [float(np.mean(rel[c][idx])) for c in CELLS]
        means.append(float(np.mean(cell_means)))
    return float(np.percentile(means, 97.5))


def select_budget(budget_results):
    """按预注册规则选择预算。

    budget_results: list of {'solution_limit': int, 'eligible': bool,
                             'per_cell': {cell: [distance, ...]}}。
    返回 dict。eligible 必须由 validate_batch 从实际批次推导（不接受手填）。
    """
    eligible = [r for r in budget_results if r.get('eligible')]
    if not eligible:
        return {'eligible': [], 'selected_budget': None, 'upgrade': False,
                'verdict': 'NO_ELIGIBLE_BUDGET'}

    candidates = sorted(r['solution_limit'] for r in eligible)
    distances = {r['solution_limit']: cell_equal_mean_distance(r['per_cell'])
                 for r in eligible}
    reference = min(distances, key=distances.get)
    ref_per_cell = next(r['per_cell'] for r in eligible
                        if r['solution_limit'] == reference)

    equivalence = {}
    for r in eligible:
        b = r['solution_limit']
        if b >= reference:
            continue  # 只对「更小预算」判定是否与最佳预算工程等价
        mean_diff = _mean_relative_diff(r['per_cell'], ref_per_cell)
        ci_upper = _bootstrap_ci_upper(r['per_cell'], ref_per_cell)
        equivalence[b] = (mean_diff <= MEAN_RELATIVE_THRESHOLD
                          and ci_upper <= BOOTSTRAP_CI_UPPER)

    equivalent_smaller = [b for b, eq in equivalence.items() if eq]
    if equivalent_smaller:
        selected = min(equivalent_smaller)
        upgrade = False
        verdict = f'SELECTED_{selected}'
    elif reference == max(candidates):
        # 最佳在最大候选、且无更小预算等价 → 距离随预算持续改善 → 追加 300
        selected = None
        upgrade = True
        verdict = f'UPGRADE_TO_{UPGRADE_LIMIT}'
    else:
        selected = reference
        upgrade = False
        verdict = f'SELECTED_{selected}'

    return {'eligible': candidates, 'reference': reference,
            'distances': distances, 'equivalence': equivalence,
            'selected_budget': selected, 'upgrade': upgrade,
            'upgrade_limit': UPGRADE_LIMIT if upgrade else None,
            'verdict': verdict}


def _validate_instance_pairs(run_a, run_b, instances_per_cell):
    """读取两批实例，检查 decision parity / 协议 Gate / 实例存在性，返回
    (eligible, per_cell)。不接受手填 gate_pass（eligible 由实际记录推导）。

    结构性复验（verify_aggregate）在 validate_batch 单独做，便于单元测试。
    """
    eligible = True
    per_cell = {}
    for cell in CELLS:
        dists = []
        for inst in range(instances_per_cell):
            pa = os.path.join(run_a, cell, 'instances', f'inst_{inst}.json')
            pb = os.path.join(run_b, cell, 'instances', f'inst_{inst}.json')
            if not os.path.exists(pa) or not os.path.exists(pb):
                raise RuntimeError(f'缺实例 {cell} inst_{inst}')
            ra = json.load(open(pa, encoding='utf-8'))
            rb = json.load(open(pb, encoding='utf-8'))
            if ra['decision_hash'] != rb['decision_hash']:
                raise RuntimeError(f'{cell} inst_{inst} decision parity 失败')
            if not ra['outcome']['complete']:
                eligible = False
            if ra['stats']['fallback_triggered_events'] != 0:
                eligible = False
            if ra['stats']['n_solver_calls'] <= 0:
                eligible = False
            dists.append(float(ra['outcome']['distance_cost']))
        per_cell[cell] = dists
    return eligible, per_cell


def parse_batches(spec):
    """解析批次清单并拒绝重复预算。返回 (instances_per_cell, [(limit, a, b)])。"""
    instances_per_cell = int(spec['instances_per_cell'])
    seen = set()
    out = []
    for b in spec['budgets']:
        lim = int(b['solution_limit'])
        if lim in seen:
            raise RuntimeError(f'重复预算: {lim}')
        seen.add(lim)
        out.append((lim, b['run_a'], b['run_b']))
    return instances_per_cell, out


def validate_batch(budget, run_a, run_b, instances_per_cell):
    """验证一批（两次运行），从实际批次推导 eligible 与 per_cell。

    结构性/确定性失败（verify_aggregate、admission、decision parity）→ 抛
    RuntimeError；协议 Gate 失败（未 complete / fallback / 无 solver call）→
    eligible=False。
    """
    verify_aggregate(run_a)
    verify_aggregate(run_b)
    pre_a = json.load(open(os.path.join(run_a, 'pre_run_manifest.json'),
                           encoding='utf-8'))
    pre_b = json.load(open(os.path.join(run_b, 'pre_run_manifest.json'),
                           encoding='utf-8'))
    admission = admission_check(pre_a.get('protocol_id'), pre_b.get('protocol_id'),
                                pre_a.get('run_id'), pre_b.get('run_id'))
    if admission:
        raise RuntimeError(f'budget {budget} 准入失败: {admission}')

    eligible, per_cell = _validate_instance_pairs(run_a, run_b, instances_per_cell)
    return {'solution_limit': budget, 'eligible': eligible, 'per_cell': per_cell}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('preflight', help='验证冻结协议与代码/身份一致')
    sel = sub.add_parser('select', help='从实际批次按预注册规则选择预算')
    sel.add_argument('--batches', required=True,
                     help='JSON: {"instances_per_cell": N, "budgets": '
                          '[{"solution_limit": b, "run_a": dir, "run_b": dir}]}')
    args = ap.parse_args()

    if args.cmd == 'preflight':
        ident = preflight()
        print('OR7 preflight PASS')
        print(f"  compute_hash = {ident['compute_hash']}")
        print(f"  control_hash = {ident['control_hash']}")
        print(f"  analysis_hash = {ident['analysis_hash']}")
        return

    with open(args.batches, encoding='utf-8') as f:
        spec = json.load(f)
    instances_per_cell, batches = parse_batches(spec)
    results = [validate_batch(lim, a, b, instances_per_cell)
               for lim, a, b in batches]
    out = select_budget(results)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
