"""A-v1 对比方法总表采集器（2026-10-13，对比方法线）。

目标：把 `CC_Compare/` 全部已适配对比方法在统一 A-v1 接单协议下的落盘结果，
连同内部臂与主模型臂，采集为**一份机器可读数据集**，供 Markdown/Excel 总表渲染。

关键口径（与项目现有实现一致）：
- 统计 = 按天聚类 bootstrap（`run_exp_reserve._stat`：seed+777、2000 次重采样、
  2.5/97.5 百分位），此处逐位复刻，不另立口径。
- 配对 = 按 day_id 显式配对（`run_a1_external_accept.paired_all_days` 语义）：
  以预期 40 个唯一 day id 验收，缺日 → 不可裁决，不静默过滤。
- **基线重配对**：正式臂落盘时把基线臂（cond_hist/uncond_hist/explicit_feat）
  从旧 `a1_step2_gate_c1_20260926` 导入，该文件无 identity 块 →
  `_pairing_identity.identity_verified=false`。本采集器改以**身份合格的**
  `a1_c1rr2_reveal_40`（含 Q-04 identity 块；与各臂 contract/data/meta/budget/seed
  逐项相同）的 per_day 作为基线，重新计算配对差，并同时保留臂内原始报告值。

用法（在 C-VRP_Cold-chainVehicleRoutingProblem 下）：
    python ../CC_Compare/tools/collect_compare_table.py --json out.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..',
                                        'C-VRP_Cold-chainVehicleRoutingProblem'))
RESULTS = os.path.join(EXT_ROOT, 'results')

BASELINE_ARMS = ('cond_hist', 'uncond_hist', 'explicit_feat')
DEFAULT_BASELINE = os.path.join(RESULTS, 'a1_c1rr2_reveal_40', 'gate.json')

# 主模型 / 内部臂（不同 future_policy / standby 或不同基线的 gate，单独成节，不与外部臂同表比）
MAIN_MODEL_PATTERNS = [
    'i3_online_recovery_1009_v1/online_*/gate.json',
    'a1_a3_online_density_*_20260926_*/gate.json',
    'a1_a3_online_density_*_20260930_*/gate.json',
    'a1_a3_online_density_*_20261001_*/gate.json',
    'a1_a1_formal_40_*/gate.json',
    'a1_rr3_density_40_*/gate.json',
    'a1_rr3_reveal_40_*/gate.json',
    'a1_s3_4_d3_*_40_*/gate.json',
    'a3_baseline_full-cond_*/gate.json',
    'a1_step2_gate_c1_20260926/gate.json',
    'a1_c1rr2_reveal_40/gate.json',
    'a1_c1r_reveal_40/gate.json',
]

# 方法元数据（类别 / 来源 / 权重 / 原生约束能力），用于总表而非计算。
# ckpt_mb = checkpoint 文件字节数（2026-10-13 在 1003 上实测，`_preflight_v2.sh` 输出），
# 作为参考表「参数量(M)/FLOPs(G)」的可核验替代（文件大小而非参数量，脚注已写明）。
# --------------------------------------------------------------------------- #
# CCF 目录版本口径（2026-10-14 用户要求新增「期刊/会议等级」列时确立）
#
# 三个版本均已核对（官方 CCF PDF）：
#   `'2019'` = 第 5 版 · `'2022'` = 第 6 版（更名版） · `'2026'` = 第 7 版（2026-03-31 发布，现行）
# ⚠️ 本批 13 个学习型方法里**只有 ICLR 在版本之间变化**：2019 / 2022 两版均**未列入**，
#    2026 第 7 版**新增为 A**（ccfddl 提交 c4569983「Update CCF Rank (2026)」同证）。
#    期刊（TNNLS=B / INFORMS JOC=B / Transportation Science=未列入 / TMLR=未列入）
#    在 2019 / 2022 / 2026 **三版均无变化**。
# ⇒ 两边都记：`rank`（2022 值）+ `rank_2026`（2026 值）；显示哪一版由**这一个常量**决定。
CCF_EDITION = '2026'


def ccf_rank(meta):
    """把 `rank` / `rank_2026` 解析成**显示串**（含跨版本变化注释）。

    * 两版都没有 ⇒ `未确证`（**绝不**凭印象填）；
    * 两版相同 / 只有一版 ⇒ 该值；
    * 两版不同 ⇒ `主值（另一版：值）`，当前口径由 `CCF_EDITION` 决定。
    """
    r22, r26 = (meta or {}).get('rank'), (meta or {}).get('rank_2026')
    if r22 is None and r26 is None:
        return '未确证'
    if CCF_EDITION == '2026':
        base = r26 if r26 is not None else r22
        if r22 is not None and r22 != base:
            return '%s（2022版：%s）' % (base, r22)
        return base
    if r26 is not None and r26 != r22:
        return '%s（2026版起 %s）' % (r22, r26)
    return r22


METHOD_META = {
    # ⚠️ `venue` 列已于 2026-10-14 逐条**对外核实**（见 `RANK_AUDIT_2026-10-14` 脚注）：
    # 13 个学习型方法的出处均与官方来源一致；`routefinder` 的出处**补全**为
    # 「TMLR 2025 / ICML 2026 (J2C)」（仓库 `README.md` 早已如此记录，此前表里被简写）。
    # `rank` = **CCF 推荐目录**等级（A/B/C）；`—` = 非论文（软件/内部基线/本工作），
    # `未列入` = 在 CCF 目录里查无此项，`未确证` = 未能核实（**绝不凭印象填**）。
    'ortools': dict(label='OR-Tools-RH accept', cls='经典求解器',
                    venue='Google OR-Tools（软件，非论文）', rank='—',
                    backbone='CP-SAT/GLS',
                    native='CVRPTW（完整覆盖）', note='myopic 接单 + 强路由',
                    ckpt_mb=None),
    'pyvrp': dict(label='PyVRP-RH accept', cls='经典求解器',
                  venue='INFORMS JOC 2024', rank='B', backbone='HGS (pyvrp 0.14.0)',
                  native='CVRPTW（完整覆盖）', note='第二经典锚点', ckpt_mb=None),
    'rrnco': dict(label='RRNCO-Ordering accept', cls='学习型（排序）',
                  venue='ICLR 2026', rank='未列入', rank_2026='A', backbone='RRNCO rcvrptw epoch_199',
                  native='RCVRPTW（有 TW）', note='冻结 R1 管线逐车排序',
                  ckpt_mb=51.65),
    'routefinder': dict(label='RouteFinder accept', cls='学习型（基础模型）',
                        venue='TMLR 2025 / ICML 2026 (J2C)', rank='A（ICML 2026 J2C；TMLR 未列入）', backbone='rf-transformer 100',
                        native='CVRP（零样本，TW 由外层认证）', note='',
                        ckpt_mb=20.15),
    'cada': dict(label='CaDA accept', cls='学习型（约束感知）',
                 venue='ICML 2025', rank='A', backbone='CaDA 100 checkpoint-300',
                 native='CVRP（零样本）', note='', ckpt_mb=41.29),
    'mvmoe': dict(label='MVMoE accept', cls='学习型（多任务 MoE）',
                  venue='ICML 2024', rank='A', backbone='pomo_vrptw_n100 epoch-5000',
                  native='VRPTW（TW 感知）', note='POMO 架构 + VRPTW 权重',
                  ckpt_mb=15.28),
    'pomo': dict(label='POMO accept', cls='学习型（构造）',
                 venue='NeurIPS 2020', rank='A', backbone='CVRP100 ckpt-30500',
                 native='CVRP（零样本）',
                 note='与 SGBS/Omni-VRP **同一权重文件**（sha256 d7a08f61…）',
                 ckpt_mb=16.36),
    'symnco': dict(label='Sym-NCO accept', cls='学习型（构造）',
                   venue='NeurIPS 2022', rank='A', backbone='CVRP ckpt-8000',
                   native='CVRP（零样本）', note='对称变换', ckpt_mb=16.05),
    'sgbs': dict(label='SGBS accept', cls='学习型（搜索）',
                 venue='NeurIPS 2022', rank='A', backbone='CVRP100 ckpt-30500',
                 native='CVRP（零样本）', note='beam=10 计入时限；与 POMO 同权重',
                 ckpt_mb=16.36),
    'deepaco': dict(label='DeepACO accept', cls='学习型（ACO 增强）',
                    venue='NeurIPS 2023', rank='A', backbone='cvrp100',
                    native='CVRP（零样本）', note='官方最轻档 T=30', ckpt_mb=0.36),
    'attention': dict(label='AttentionModel accept', cls='学习型（构造）',
                      venue='ICLR 2019', rank='未列入', rank_2026='A', backbone='cvrp_100',
                      native='CVRP（零样本）', note='官方 greedy 档', ckpt_mb=2.78),
    'omnivrp': dict(label='Omni-VRP accept', cls='学习型（构造）',
                    venue='ICML 2023', rank='A', backbone='POMO-CVRP 档',
                    native='CVRP（零样本）', note='官方 aug8 档；与 POMO 同权重',
                    ckpt_mb=16.36),
    'lih': dict(label='Learn-Improvement-Heuristics accept', cls='学习型（改进）',
                venue='IEEE TNNLS 2022', rank='B', backbone='CVRP50 epoch-199',
                native='CVRP（零样本）', note='steps=100 局部改进；跨次不可复现',
                ckpt_mb=19.59),
    # --- 本线新适配（2026-10-13）：CO-enriched-ML 两模式 + MAPT ---
    'coenrich_nopchgs': dict(
        label='CO-enriched-ML accept（纯学习排序）', cls='学习型（预测+优化）',
        venue='Transportation Science 2024', rank='未列入', backbone='官方 MLP 20→10×4→1（NumPy 前向）',
        native='原生动态 VRPTW（波次语义 → 逐揭示改造）',
        note='不用 PC-HGS；0.02 s/决策', ckpt_mb=0.004),
    'coenrich_pchgs': dict(
        label='CO-enriched-ML accept（+官方 PC-HGS）', cls='学习型（预测+优化）',
        venue='Transportation Science 2024', rank='未列入', backbone='官方 MLP + PC-HGS（C++20 原二进制）',
        native='原生动态 VRPTW（波次语义 → 逐揭示改造）',
        note='PC-HGS 作优先级生成器，按池缓存 ⇒ 1 次/决策 ≈2.1 s', ckpt_mb=0.004),
    'mapt': dict(label='MAPT accept', cls='学习型（多智能体指针 Transformer）',
                 venue='AAAI 2026', rank='A', backbone='MAPT 自训 144 updates（1.44 M 参数）',
                 native='MVDPDPSR（无 TW；TW/冷链由外层认证）',
                 note='**弱介入档**：模型只决定前 ~3 位、尾部 EDD；'
                      'dev 效用未超弱贪心（871 vs 951），服务率超（36.4% vs 26.9%）',
                 ckpt_mb=5.82),
    'myopic': dict(label='rule: myopic', cls='规则基线', venue='—', rank='—',
                   backbone='—', native='完整覆盖可行性', note='无前瞻即收',
                   ckpt_mb=None),
    'valuee20': dict(label='rule: value/energy λ=2', cls='规则基线', venue='—', rank='—',
                     backbone='—', native='预算感知门槛', note='固定价值/能耗门槛',
                     ckpt_mb=None),
    'valuee30': dict(label='rule: value/energy λ=3', cls='规则基线', venue='—', rank='—',
                     backbone='—', native='预算感知门槛',
                     note='退化（0 接单）', ckpt_mb=None),
}


def _stat(x, seed):
    """逐位复刻 run_exp_reserve._stat（按天聚类 bootstrap）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    n = len(x)
    r = np.random.default_rng(seed + 777)
    mm = np.array([x[r.integers(0, n, n)].mean() for _ in range(2000)])
    return {"mean": float(np.mean(x)), "ci_lo": float(np.percentile(mm, 2.5)),
            "ci_hi": float(np.percentile(mm, 97.5)), "n": int(n)}


def _summ_rows(rows, seed, expected=40):
    ok = np.array([bool(r['hard_feasible']) for r in rows])
    miss = [r['i'] for r in rows
            if r.get('utility') is None or not np.isfinite(r.get('utility', np.nan))]
    ids = sorted(int(r['i']) for r in rows)
    complete = (len(rows) == expected and ids == list(range(expected)))
    adj = bool(complete and not miss)
    return {
        'n_days': len(rows), 'expected_n': expected,
        'adjudicable': adj, 'missing_day_ids': miss,
        'hard_feasible_rate': float(ok.mean()) if adj else None,
        'utility': _stat([r['utility'] for r in rows if r.get('utility') is not None],
                         seed) if adj else None,
        'mean_timeouts': float(np.mean([r.get('timeouts', 0) for r in rows])),
        'timeout_days': int(sum(1 for r in rows if r.get('timeouts', 0) > 0)),
        'fail_counts': dict(Counter(f for r in rows for f in r.get('failures', []))),
        'elapsed_total_s': float(np.sum([r.get('elapsed_s', 0.0) for r in rows])),
        'mean_served': (float(np.mean([r['n_served'] for r in rows]))
                        if all('n_served' in r for r in rows) else None),
        'mean_rejected': (float(np.mean([r['n_rejected'] for r in rows]))
                          if all('n_rejected' in r for r in rows) else None),
        'mean_energy_kwh': (float(np.mean([r['energy_kwh'] for r in rows]))
                            if all('energy_kwh' in r for r in rows) else None),
        'budget_violation_days': int(sum(1 for r in rows if r.get('budget_violated'))),
        'revenue': _stat([r['revenue'] for r in rows if 'revenue' in r], seed)
        if all('revenue' in r for r in rows) else None,
        'fuel_cost': _stat([r['fuel_cost'] for r in rows if 'fuel_cost' in r], seed)
        if all('fuel_cost' in r for r in rows) else None,
        'reject_loss': _stat([r['reject_loss'] for r in rows if 'reject_loss' in r], seed)
        if all('reject_loss' in r for r in rows) else None,
    }


def _pair(a_rows, b_rows, seed, expected=40):
    """paired_all_days 语义：按 day id 显式配对 + 预期天数验收。"""
    b_by_i = {int(r['i']): r.get('utility') for r in b_rows}
    a_by_i = {int(r['i']): r.get('utility') for r in a_rows}
    pa, pb, missing = [], [], []
    for i in sorted(set(a_by_i) | set(b_by_i)):
        if i not in a_by_i:
            missing.append(('a_missing_day', i))
            continue
        if i not in b_by_i:
            missing.append(('b_missing_day', i))
            continue
        ua, ub = a_by_i[i], b_by_i[i]
        if ua is None or ub is None or not np.isfinite(ua) or not np.isfinite(ub):
            missing.append(('missing_utility', i))
            continue
        pa.append(ua)
        pb.append(ub)
    st = _stat(np.array(pa, float) - np.array(pb, float), seed)
    ids_ok = (sorted(a_by_i) == sorted(b_by_i) == list(range(expected)))
    st['adjudicable'] = bool(not missing and ids_ok
                             and len(a_rows) == expected and len(b_rows) == expected)
    st['n_paired'] = len(pa)
    st['missing'] = missing
    return st


def _load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def load_baseline(path):
    """返回 {p_c: {arm: per_day_rows}} 与元信息。"""
    d = _load(path)
    cont = d.get('gate') or d.get('results')
    per = {}
    for pname, blk in cont.items():
        if pname.startswith('_') or not isinstance(blk, dict):
            continue
        if 'per_day' not in blk:
            continue
        pd = blk['per_day']
        if isinstance(pd, dict):
            per[pname] = {a: rows for a, rows in pd.items() if a in BASELINE_ARMS}
    return d, per


def scan_arm_gates():
    out = []
    # 正式批次 v2（2026-10-13，同版源码 + 身份合格基线）
    for p in sorted(glob.glob(os.path.join(RESULTS, 'a1_ext_formal2', '*', 'gate.json'))):
        out.append(('a1_ext_formal2', os.path.basename(os.path.dirname(p)), p))
    for p in sorted(glob.glob(os.path.join(RESULTS, 'a1_ext_formal2', '*.json'))):
        continue
    # 旧正式/诊断批次（2026-09-29/30）
    for p in sorted(glob.glob(os.path.join(RESULTS, 'a1_ext_extra', '*', 'gate.json'))):
        name = os.path.basename(os.path.dirname(p))
        if 'smoke' in name or 'dbg' in name or 'providerbug' in name:
            continue
        out.append(('a1_ext_extra', name, p))
    for p in sorted(glob.glob(os.path.join(RESULTS, 'a1_ext_a05e_*', 'gate.json'))):
        out.append(('a1_ext_a05e', os.path.basename(os.path.dirname(p)), p))
    return out


# 参数量（百万）——对齐参考表（AEGIS-CD 风格）的「参数量(M)」列。
# 数据来源（**均为实测，不是估计**）：
#   - 11 个 torch provider：`participation/<m>.json` → `C2_random_weights.replaced_params`
#     （C2 权重替换时实际替换的参数量 = 真网络参数量）
#   - rrnco：`tools/param_count.py <ckpt> --bridge` 直接数 state_dict（4,290,252）
#   - coenrich：NumPy MLP 权重元素数（`artifacts_a1/a1_weights.npz` = 550）
#   - 经典求解器与规则臂：无学习参数 → None（表中显示 —）
PARAMS_M = {
    'routefinder': 1.677824, 'cada': 3.421488, 'sgbs': 1.254016, 'pomo': 1.254016,
    'symnco': 1.303424, 'attention': 0.692608, 'omnivrp': 1.254016, 'mvmoe': 1.2544,
    'deepaco': 0.067169, 'lih': 0.70976, 'mapt': 1.437187, 'rrnco': 4.290252,
    'coenrich_nopchgs': 0.00055, 'coenrich_pchgs': 0.00055,
    'ortools': None, 'pyvrp': None,
    'myopic': None, 'valuee20': None, 'valuee30': None,
}


def method_key(name, arm=None):
    n = name
    for pre in ('rule_',):
        if n.startswith(pre):
            n = n[len(pre):]
    if n.endswith('_f'):
        n = n[:-2]
    for k in ('valuee20', 'valuee30', 'valuee2', 'valuee3'):
        if n.startswith(k):
            return k
    for k in METHOD_META:
        if n == k or n.startswith(k + '_'):
            return k
    if arm:
        a = arm
        for suf in ('_accept',):
            if a.endswith(suf):
                a = a[:-len(suf)]
        for k in METHOD_META:
            if a == k or a.startswith(k):
                return k
        return a
    return n


def arm_identity_block(gate):
    if 'identity' in gate:
        return gate['identity']
    for yr, yb in (gate.get('results') or {}).items():
        if isinstance(yb, dict) and '_identity' in yb:
            return yb['_identity']
    return None


def pairing_identity_block(gate):
    for yr, yb in (gate.get('results') or {}).items():
        if isinstance(yb, dict) and '_pairing_identity' in yb:
            return yb['_pairing_identity']
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default=DEFAULT_BASELINE)
    ap.add_argument('--json', default=None)
    ap.add_argument('--include-diagnostic', action='store_true')
    args = ap.parse_args()

    base, base_per = load_baseline(args.baseline)
    base_id = base.get('identity', {})
    seed = int(base['seeds']['gate'])

    dataset = {
        'generated_from': {
            'baseline': args.baseline,
            'baseline_identity_present': bool(base_id),
            'baseline_source_stable': base_id.get('source_stable'),
            'baseline_contract_sha256': base_id.get('contract_sha256'),
            'baseline_data_sha256': base_id.get('data_sha256'),
            'baseline_meta_sha256': base_id.get('data_meta_sha256'),
            'seed': seed,
            'stat': 'day-clustered bootstrap seed+777, 2000 resamples, 2.5/97.5 pct',
        },
        'baseline_arms': {},
        'arms': [],
        'incomparable_or_unadapted': [],
    }

    for pname, arms in sorted(base_per.items()):
        for an, rows in sorted(arms.items()):
            dataset['baseline_arms']['%s|%s' % (pname, an)] = _summ_rows(rows, seed)

    for group, name, path in scan_arm_gates():
        # 正式口径：v2 批次全部、`*_f`（旧修复版正式批次）、`rule_*`、a05e 求解器 40 天臂
        final = (group == 'a1_ext_formal2' or name.endswith('_f')
                 or name.startswith('rule_') or group == 'a1_ext_a05e')
        g = _load(path)
        ident = arm_identity_block(g)
        pair_id = pairing_identity_block(g)
        # 两种 schema：results[year][p_c]（外部驱动）与 gate[p_c]（步骤2驱动）
        cont = g.get('results') or g.get('gate') or {}
        flat = []
        for k, v in cont.items():
            if k.startswith('_') or not isinstance(v, dict):
                continue
            if 'arms' in v or 'per_day' in v:
                flat.append((k, v))
            else:
                for k2, v2 in v.items():
                    if k2.startswith('_') or not isinstance(v2, dict):
                        continue
                    flat.append((k2, v2))
        for pname, blk in sorted(flat):
            if pname.startswith('_') or not isinstance(blk, dict):
                continue
            arms = blk.get('arms')
            if not isinstance(arms, dict):
                continue
            pd = blk.get('per_day')
            if isinstance(pd, dict):
                pd = pd.get(next(iter(arms))) if len(arms) == 1 else None
            mine = [a for a in arms if a not in BASELINE_ARMS]
            if not mine:
                continue
            mine = mine[0]
            rows = pd if isinstance(pd, list) else None
            # gate 内该臂的配对裁决块：形如 `<arm>_minus_cond_hist`（与 arms 同级）。
            _gk = '%s_minus_cond_hist' % mine
            _gb = blk.get(_gk) if isinstance(blk.get(_gk), dict) else {}
            if not _gb and isinstance(blk.get('main_comparison'), dict):
                _gb = blk['main_comparison'].get(_gk) or {}
            gate_adjudication = {
                'key': _gk,
                'formal_adjudication': _gb.get('formal_adjudication'),
                'identity_verified': _gb.get('identity_verified'),
                'adjudicable': _gb.get('adjudicable'),
                'n_paired': _gb.get('n_paired'),
            }
            rec = {
                'group': group, 'arm_dir': name, 'formal': bool(final),
                'method': method_key(name, mine), 'arm': mine, 'p_c': pname,
                'gate_path': os.path.relpath(path, EXT_ROOT),
                'reported': _summ_rows(rows, seed) if rows else None,
                'reported_utility': (arms[mine].get('utility_all_days')
                                     if isinstance(arms.get(mine), dict) else None),
                'solver_meta': blk.get('solver_meta') or blk.get('arm_meta'),
                'pairing_identity_recorded': pair_id,
                # 该臂**落盘时实际配对的基线**（gate config.baseline_from）。
                # 必须逐臂记录：v2 各臂是拿同源新基线 a1_c1fresh_reveal_40 完成裁决的，
                # 而本表的 Δ 列统一另以 a1_c1rr2_reveal_40 重配对 —— 两者基线不同，
                # 渲染器要如实说明，不能笼统宣称「双侧 source_stable=True」。
                'baseline_from': (g.get('config') or {}).get('baseline_from'),
                'identity_present': bool(ident),
                'identity_source_stable': (ident or {}).get('source_stable'),
                'identity_contract_sha256': (ident or {}).get('contract_sha256'),
                'identity_data_sha256': (ident or {}).get('data_sha256'),
                'identity_meta_sha256': (ident or {}).get('data_meta_sha256'),
                'repaired_vs_baseline': {},
                # 该臂**在自身 gate 内**的配对裁决标志（`<arm>_minus_cond_hist` 块里的
                # `formal_adjudication` / `identity_verified`），与 `repaired_vs_baseline`
                # （本表另以 `--baseline` 重算的差）**是两件事**。
                # 为什么要采：`repaired_vs_baseline` 由 `_pair()` 产出，**从不包含**
                # `formal_adjudication` —— 收口脚本原先打印 `repaired_vs_baseline.cond_hist.
                # formal_adjudication`，于是那一列**恒为 None**，读起来像「没有正式裁决」，
                # 而恰恰相反：v2 各臂落盘的就是 `formal_adjudication=true`。这是纯粹的口径混淆，
                # 必须把 gate 内的真值单独采出来。
                'gate_adjudication': gate_adjudication,
            }
            if rows:
                for ban in BASELINE_ARMS:
                    brows = base_per.get(pname, {}).get(ban)
                    if brows:
                        rec['repaired_vs_baseline'][ban] = _pair(rows, brows, seed)
            # 该 gate 自带的基线摘要（旧口径：从旧 beeline gate 导入）留存对照
            for ban in BASELINE_ARMS:
                if isinstance(arms.get(ban), dict):
                    rec.setdefault('reported_baseline_summary', {})[ban] = \
                        arms[ban].get('utility_all_days')
            meta = METHOD_META.get(rec['method'], {})
            # 附上实测参数量（百万），供渲染器的「参数量(M)」列使用；
            # `rank` 覆盖为**按 `CCF_EDITION` 解析后的显示串**（并保留 `rank_2026` 原值）。
            rec['meta'] = dict(meta, params_m=PARAMS_M.get(rec['method']),
                               rank=ccf_rank(meta))
            dataset['arms'].append(rec)

    # 未适配清单：**只放确实无法适配的方法**，且必须写清具体机制原因。
    # 2026-10-13 修正：MAPT 与 CO-enriched-ML 已适配、并作为 v2 正式臂产出结果
    # （见 README §13.2 / §13.4、NEXT_STEPS §3 第 6 项），故从本清单**移除**——
    # 否则它们会同时出现在结果行与「未适配」表里，自相矛盾。
    # 保留的两项都是**机制层面**接不上（不是权重或环境问题：资产其实都能加载）。
    dataset['incomparable_or_unadapted'] = [
        {'method': 'Learning-to-Delegate',
         'path': 'CC_Compare/Learning-to-Delegate/A1_ASSESSMENT.md',
         'reason': '机制不匹配：其回归目标是 HGS 子问题距离，规模 n≥500，而 A-v1 单决策可见客户 ≤90；'
                   '资产本身可加载（3/3 strict，MAE 0.071–0.138），缺的是可迁移到「接单」决策的语义。'},
        {'method': 'PIP-constraint',
         'path': 'CC_Compare/PIP-constraint/A1_ASSESSMENT.md',
         'reason': '机制不匹配：模型是单车辆 TSPTW/TSPDL，其 env 定义里 num_vehicles / vehicle_id / '
                   'capacity 出现 0 次，没有车队维度可对接 A-v1 的 15 车接单+路由决策；'
                   '43/48 个 .pt 权重可加载，但「单车辆 → 多车辆」这一步接不上。'},
    ]

    # ---- 主模型 / 内部臂（单独成节：不同 future_policy / standby / 基线，不与外部臂同表） ----
    dataset['main_model'] = []
    for pat in MAIN_MODEL_PATTERNS:
        for path in sorted(glob.glob(os.path.join(RESULTS, pat))):
            try:
                g = _load(path)
            except Exception:
                continue
            ident = arm_identity_block(g)
            cont = g.get('results') or g.get('gate') or {}
            flat = []
            for k, v in cont.items():
                if k.startswith('_') or not isinstance(v, dict):
                    continue
                if 'arms' in v:
                    flat.append((k, v))
                else:
                    for k2, v2 in v.items():
                        if not k2.startswith('_') and isinstance(v2, dict):
                            flat.append((k2, v2))
            cfg = g.get('config') or {}
            for pname, blk in sorted(flat):
                arms = blk.get('arms') or {}
                pd = blk.get('per_day')
                rows_by_arm = pd if isinstance(pd, dict) else None
                # 配对差既可能在 block 顶层（<arm>_minus_cond_hist），也可能在
                # 嵌套的 main_comparison 里（I3 gate 就是这样）
                mc = {}
                for src in (blk, blk.get('main_comparison') or {}):
                    if not isinstance(src, dict):
                        continue
                    for k, v in src.items():
                        if isinstance(v, dict) and '_minus_' in k:
                            mc[k] = v
                for an, ab in sorted(arms.items()):
                    rows = (rows_by_arm or {}).get(an)
                    dataset['main_model'].append({
                        'gate': os.path.relpath(path, EXT_ROOT),
                        'p_c': pname, 'arm': an,
                        # 该 gate **记录在文件里**的开发集 seed（`seeds.dev` 或 `seeds.gate`）。
                        # 为什么必须采这个：渲染器原先靠「目录名里是否含 `20260926`」筛开发集，
                        # 而 `results/a1_a1_formal_40_1004_171907/gate.json` 的 seed 明明就是
                        # 20260926（实测 seeds={'train':20260925,'gate':20260926,...}，
                        # 且 source_stable=True / contract=e25a12104b20c3ea），
                        # 却因目录名不含该串而被**整个 gate 静默丢掉** —— 连带丢掉
                        # `a1_rollout`（p_c=0 util=1312.6、0 超时）这类合法主模型臂。
                        # 采真实 seed 后，判定不再依赖目录命名习惯。
                        'dev_seed': ((g.get('seeds') or {}).get('dev')
                                     or (g.get('seeds') or {}).get('gate')),
                        'config': {k: cfg.get(k) for k in
                                   ('future_policy', 'standby_orders', 'workers',
                                    'time_limit', 'K', 'rho', 'baseline_from',
                                    'model_bin', 'arm')},
                        'reported': _summ_rows(rows, seed) if rows else None,
                        'reported_utility': ab.get('utility_all_days'),
                        'identity_present': bool(ident),
                        'identity_source_stable': (ident or {}).get('source_stable'),
                        'pairing_identity': pairing_identity_block(g),
                        'main_comparison': mc,
                    })
    txt = json.dumps(dataset, ensure_ascii=False, indent=1)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            f.write(txt)
        print('wrote %s (%d arms)' % (args.json, len(dataset['arms'])))
    else:
        print(txt)


if __name__ == '__main__':
    main()
