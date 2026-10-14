#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配对身份检查器（2026-10-14 用户要求）。

起因是一次真实错误：我把 **I3 gate 内的配对差 −144.680** 与**另一套协议**的 C1 基线值
`a1_c1fresh_reveal_40|cond_hist` = **631.410** 并列写进正文，而正确的同批对照是
**600.997**：

    456.317 − 600.997 = −144.680   ✅ 同批成立
    456.317 − 631.410 = −175.093   ✗ 与 −144.680 不符

`table_consistency` 当时是**全绿**的 —— 它只保证「表与表之间一致」，
**不能**保证正文里每一组「效用 A / 效用 B / 配对差」具有相同的运行身份。本检查器补这个缺口。

三段检查：
  **A. 数据层恒等式（硬判据）**：对 `dataset.json` 里每个 `*_minus_cond_hist` 块，
     校验 `|mean(A) − mean(B) − mean(D)| ≤ 1e-6`，且 A、B、D 来自**同一 gate**、`n` 相同。
  **B. 跨协议护栏（硬判据）**：任何 gate 的 `cond_hist` 必须取自**该 gate 自己**的记录；
     并列出「按协议分组」的基线值，便于人工核对。若某个 gate 的对照值等于**另一协议**的
     基线值，直接 FAIL。
  **C. 文档层混用扫描（启发式）**：扫描交付文档，找出「外来协议基线值」与「本协议配对差」
     出现在同一行的情形；**带否定标记**（如"另一套协议"/"不得"/"≠"/"不可"）的行视为**已声明**，
     不算违规 —— 全文警告与错误引用必须区分开。

用法（仓库根）：
    python CC_Compare/tools/check_pairing_identity.py \
        --data CC_Compare/results/a1_compare_table/dataset.json
退出码：0 = PASS；1 = FAIL；2 = 数据缺失。
"""
import argparse
import io
import json
import os
import re
import sys

import numpy as np

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

EPS = 1e-6
# 否定/声明标记：出现这些词的行 = 作者**已明确**指出该数字属另一协议 ⇒ 不算违规
NEG_MARKERS = ('另一套协议', '另一协议', '不得', '不可', '≠', '不构成', '不可与本', '勿与',
               '跨批次混用', '混用')
# 要扫描的交付文档（相对仓库根）
DOCS = [
    'CC_Compare/results/a1_compare_table/README.md',
    'CC_Compare/results/a1_compare_table/读表注意与已知限定.md',
    'CC_Compare/results/a1_compare_table/A-v1对比方法总表.md',
    'CC_Compare/P1A_强动态对照_预声明_2026-10-14.md',
    'C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md',
    'C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/结果与进度/执行进度表.md',
]


def num(s):
    try:
        return float(s)
    except Exception:                                        # noqa: BLE001
        return None


def _idx(data):
    """(gate, arm, p_c) → record。

    ⚠️ 2026-10-14 实测踩到：`main_comparison` 块**存在该 gate 的每一条记录上**
    （四个臂共享同一份 `{maskco_minus_cond_hist, warmstart_minus_...}`）。
    若直接拿"当前记录的臂效用"去配块里的差，就会把 cond_hist 自己的效用
    拿去解释 `maskco_minus_cond_hist` ⇒ 一次跑出 156 条**假失败**。
    ⇒ 必须**从块名解析出臂**（`<arm>_minus_cond_hist`）。
    """
    out = {}
    for r in data.get('main_model') or []:
        out[(r.get('gate'), r.get('arm'), r.get('p_c'))] = r
    return out


def _u(idx, gate, arm, pc):
    return ((idx.get((gate, arm, pc)) or {}).get('reported_utility') or {}).get('mean')


def _pairs(data):
    """产出 (gate, pc, arm, 配对块, 索引)。

    ⚠️ 必须**从该臂自己的记录**里读 `<arm>_minus_cond_hist`：`main_comparison` 在四条臂的
    记录上各存一份，若按 (gate,pc,arm) 去重后"取先遇到的那份"，就可能读到**未被改动**的副本
    ⇒ 篡改检测被静默绕过（2026-10-14 反例测试实测踩到）。
    """
    idx = _idx(data)
    for r in data.get('main_model') or []:
        arm = r.get('arm')
        if not arm:
            continue
        d = (r.get('main_comparison') or {}).get('%s_minus_cond_hist' % arm)
        if not isinstance(d, dict) or d.get('mean') is None:
            continue
        yield r.get('gate'), r.get('p_c'), arm, d, idx


def part_a(data):
    """数据层：逐块校验 A − B = D，且 A、B、D 同 gate 同档。"""
    fails, checked = [], 0
    seen = set()
    for gate, pc, arm, d, idx in _pairs(data):
        if (gate, pc, arm) in seen:
            continue
        seen.add((gate, pc, arm))
        checked += 1
        A, B = _u(idx, gate, arm, pc), _u(idx, gate, 'cond_hist', pc)
        if A is None or B is None:
            fails.append('A: gate=%s %s/%s 拿不到同 gate 同档的 %s 或 cond_hist 效用 ⇒ '
                         '无法证明三者同源' % (gate, pc, arm, arm))
            continue
        if abs((A - B) - d['mean']) > EPS:
            fails.append('A: gate=%s %s/%s 恒等式不成立：%.3f − %.3f = %.3f ≠ %.3f'
                         % (gate, pc, arm, A, B, A - B, d['mean']))
        na = ((idx.get((gate, arm, pc)) or {}).get('reported') or {}).get('n_days')
        nb = ((idx.get((gate, 'cond_hist', pc)) or {}).get('reported') or {}).get('n_days')
        if None not in (na, nb, d.get('n')) and not (na == nb == d.get('n')):
            fails.append('A: gate=%s %s/%s 天数不一致 n_a=%s n_b=%s n_diff=%s'
                         % (gate, pc, arm, na, nb, d.get('n')))
    return checked, fails


def part_b(data):
    """跨协议护栏：**隐含对照值**必须等于同 gate 自己的 cond_hist。

    `隐含对照 = A − D`。这是「同源」在数据层唯一可判定的形式：
      I3：456.317 − (−144.680) = **600.997** = 该 gate 自己的 cond_hist ✅
      若有人把 631.410 当作该配对差的对照，则 456.317 − 631.410 = −175.093，
      与落盘的 −144.680 不符 ⇒ 必然被这条判据挡下。
    """
    fails = []
    foreign = {}
    for k, v in (data.get('baseline_arms') or {}).items():
        m = ((v or {}).get('utility') or {}).get('mean')
        if m is not None:
            foreign[k] = m
    implied = {}
    seen = set()
    for gate, pc, arm, d, idx in _pairs(data):
        if (gate, pc, arm) in seen:
            continue
        seen.add((gate, pc, arm))
        A = _u(idx, gate, arm, pc)
        if A is None:
            continue
        b_impl = A - d['mean']
        own = _u(idx, gate, 'cond_hist', pc)
        implied['%s|%s|%s' % (gate, pc, arm)] = round(b_impl, 3)
        if own is None or abs(b_impl - own) > EPS:
            fails.append('B: gate=%s %s/%s 隐含对照 %.3f ≠ 同 gate cond_hist %s ⇒ '
                         '该配对差与所述对照**不同源**'
                         % (gate, pc, arm, b_impl, ('%.3f' % own) if own is not None else '缺失'))
    return foreign, implied, fails


def _stat_replica(x, seed):
    """**逐位复刻** `run_exp_reserve._stat` / `collect_compare_table._stat`。

    ⚠️ 必须用**同一个 rng 顺序**（一个 `default_rng(seed+777)` 顺序抽 2000 次），
    否则 CI 不会与落盘值逐位一致，就无法据以判"同源"。
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    n = len(x)
    r = np.random.default_rng(seed + 777)
    mm = np.array([x[r.integers(0, n, n)].mean() for _ in range(2000)])
    return {'mean': float(np.mean(x)), 'ci_lo': float(np.percentile(mm, 2.5)),
            'ci_hi': float(np.percentile(mm, 97.5)), 'n': int(n)}


def _gate_seed(g, gate_path):
    """按**确定性优先级**（而非事后择优）解析「该 gate 自己的」bootstrap 种子。

    优先级由**两个生产者的写入 schema** 决定（2026-10-14 只读核对源码）：

      ① `seeds['ran']` 恰为**单一 int** ⇒ 用该值。`run_a1_step3_compare.py:405` 与
         `run_a1_external_accept.py:411` 的 gate **只**写 `ran`、**不写** `seeds.gate`；
         且官方裁决器正是这样读的 —— `diag_a3_adjudicate.py:165` 断言
         `(doc['seeds']).get('ran') == [int(seed)]`，`diag_i3_online_adjudicate.py:45`
         直接 `doc['seeds']['ran']`。**这是本仓「gate 种子」的规范字段。**
      ② `seeds['gate']` 为 int ⇒ 用该值（`run_a1_step2_gate.py:390` 的 schema）。
      ③ 目录名中的 `_YYYYMMDD_` ⇒ 用该 8 位整数（与 ① 实测一致，仅作兜底）。
      ④ `seeds['dev']` 为 int ⇒ 用该值（最后兜底）。

    ⚠️ **冲突/歧义一律 fail-closed**：`ran` 非单一 int、或 `ran` 与 `gate` 同时存在且
    不等 ⇒ 返回 `None`，让该块落到 SKIP 并写明原因，**绝不**为了让 CI 对上而试遍种子
    （那是事后择优，会使 ④ 失去证据价值）。
    返回 `(seed|None, source_tag)`。
    """
    sd = g.get('seeds') or {}
    ran, gs = sd.get('ran'), sd.get('gate')
    gs_ok = isinstance(gs, int) and not isinstance(gs, bool)
    if ran is not None:
        ran_ok = (isinstance(ran, (list, tuple)) and len(ran) == 1
                  and isinstance(ran[0], int) and not isinstance(ran[0], bool))
        if not ran_ok:
            return None, 'ambiguous:seeds.ran=%r（非单一 int）' % (ran,)
        if gs_ok and int(gs) != int(ran[0]):
            return None, 'conflict:seeds.gate=%r≠seeds.ran=%r' % (gs, ran[0])
        return int(ran[0]), 'seeds.ran'
    if gs_ok:
        return int(gs), 'seeds.gate'
    m = re.search(r'_(20\d{6})_',
                  os.path.basename(os.path.dirname(os.path.abspath(gate_path))))
    if m:
        return int(m.group(1)), 'dir_date'
    dv = sd.get('dev')
    if isinstance(dv, int) and not isinstance(dv, bool):
        return int(dv), 'seeds.dev'
    return None, 'absent'


def _per_day_of(gate_path, pc):
    """从原始 gate 取**指定档位**的 `per_day` 与该 gate 自己的 **bootstrap 种子**。

    ⚠️ 必须按 `pc` 选块：一个 gate 的 `results[year]` 下同时有 `p_c=0` /
    `p_c=(5,10,15)` / `p_c=(10,20,30)` 三个块，各带自己的 `per_day`。
    2026-10-14 实测踩到：只要"第一个含 per_day 的块"，就会拿**别的档位**的逐日记录
    去配本档的配对差 ⇒ 一次跑出 148 条假失败。

    ⚠️ 还要取**该 gate 自己的** 种子（见 `_gate_seed`）：`a1_a3_online_density_*` 有三个
    不同的开发日集（20260926 / 20260930 / 20261001），**每个用各自的种子**。
    此前一律用 `generated_from.seed`(=20260926)，于是 0930/10001 的块 CI 对不上 ——
    那是**种子**差异，不是生产者差异。嵌套 schema 下 `results[<seed>]` 的键**本身就是**
    该块的种子，证据最强，优先采用。
    返回 `(per_day, seed|None, source_tag)`。
    """
    if not os.path.isfile(gate_path):
        return None, None, 'gate_missing'
    try:
        g = json.load(open(gate_path, encoding='utf-8'))
    except Exception:                                        # noqa: BLE001
        return None, None, 'gate_unreadable'
    gseed, src = _gate_seed(g, gate_path)
    cont = g.get('results') or g.get('gate') or {}
    for _k, yb in cont.items():
        if not isinstance(yb, dict):
            continue
        # 扁平 schema：gate[p_c]['per_day']
        if 'per_day' in yb and _k == pc:
            return yb['per_day'], gseed, src
        # 嵌套 schema：results[seed][p_c]['per_day'] —— 外层键即该块种子
        blk = yb.get(pc)
        if isinstance(blk, dict) and 'per_day' in blk:
            kseed = None
            if isinstance(_k, int) and not isinstance(_k, bool):
                kseed = int(_k)
            elif isinstance(_k, str) and _k.isdigit():
                kseed = int(_k)
            return blk['per_day'], (kseed if kseed is not None else gseed), \
                ('results_key' if kseed is not None else src)
    return None, None, 'per_day_absent'


def part_d(data, gates_root, ci_exact_tol=1e-6):
    """**日级证据**（2026-10-14 用户要求；含 CI 精确复现边界）。

    四项分别报告（用户 2026-10-14 复核要求拆开）：
      ① **相同 day_id 配对**：两侧原始记录各 40 条、ID 为整数且唯一、集合恰为 `0..39`。
         ⚠️ 必须在**构建字典之前**校验原始记录数/唯一性 —— 字典会覆盖重复 ID，
         只比较两个字典的键集合**不能**证明配对完整。
      ② **逐日差均值重算**：与落盘 `mean` 逐位一致。
      ③ **CI 排零方向一致**：落盘与重算的「是否排除 0」必须一致。
      ④ **CI 数值精确复现**：边界逐位一致（`ci_exact_tol`）。

    ④ 不通过**不得**输出笼统 PASS：调用方须显式标注 `CI_exact_reproduction=False`，
    并**保留原正式裁决与历史 CI**（不自动推翻 I3 负结果）。

    **采样顺序（2026-10-14 只读核对生产者源码后确定）**：主模型配对块的 `mean/ci_*`
    由 `run_a1_step2_gate.py::paired_all_days()` 产出，它按 **`a_rows` 的原始顺序**
    取对（`for r in a_rows`），而 `collect_compare_table._pair()` 按 **day_id 排序** 取对。
    两者用**同一个 `_stat`**（`from run_exp_reserve import _stat`），而 `_stat` 的 bootstrap
    是**按位置**抽样的 ⇒ **顺序不同 ⇒ CI 不同、均值相同**。
    ⇒ 本函数按**生产者顺序**复刻，使 CI 可数值复现；两种定义都保留记录，**不事后择优**。

    **两种定义已并列实测（2026-10-14，`tools/_probe_bootstrap_defs.py`，只读）**：

        P（生产者顺序）vs S（day_id 升序）在**全部 52 个配对块**上：
        顺序相同 = 52 / 顺序不同 = 0 / CI 上下界最大差 = **0.000000** /
        落盘值命中 P = 52、命中 S = 52、两者皆不命中 = 0

    ⇒ 本批 `per_day` 列表**本来就是 day_id 升序**，两种定义**数值上完全等同**，
    因此**不存在**"事后选更有利区间"的余地（也无人这样做）。⚠️ 但顺序敏感性是
    `_stat` 的**结构性**性质（按位置抽样），**换一批数据仍可能出现 P≠S** ——
    故保留按生产者顺序复刻的实现，并把 `src_tally` 与本节结论一并披露。

    **种子（2026-10-14 修正）**：每个块用**该 gate 自己的**种子，按 `_gate_seed` 的确定性
    优先级解析（`seeds.ran` ▸ `seeds.gate` ▸ 目录日期 ▸ `seeds.dev`；冲突/歧义 fail-closed）。
    ⚠️ 此前回落到 `generated_from.seed`(=20260926)，导致 0930/20261001 两个 gate 的
    18 个块 CI 对不上 —— 那是**种子**差异，不是生产者差异。解析不出的块**记 SKIP**，
    不回落到任何默认种子（否则 ④ 会静默变成假象）。返回的 `src_tally` 记录每个块
    实际用了哪一路证据，供报告逐条披露。
    """
    fails, warns, skipped, ci_inexact, checked = [], [], [], [], 0
    src_tally = {}
    for gate, pc, arm, d, _i in _pairs(data):
        gp = gate if os.path.isabs(gate) else os.path.join(gates_root, *str(gate).split('/'))
        pd, gseed, src = _per_day_of(gp, pc)
        if pd is None:
            skipped.append('%s %s/%s：原始 gate 不可得（%s）' % (gate, pc, arm, gp))
            continue
        if not isinstance(gseed, int):
            # fail-closed：**不**回落到 generated_from.seed（那正是此前 18 个块 CI 对不上的原因）
            skipped.append('%s %s/%s：gate 种子不可解析（%s，%s）'
                           % (gate, pc, arm, src, os.path.basename(os.path.dirname(gp))))
            continue
        src_tally[src] = src_tally.get(src, 0) + 1
        seed_use = gseed
        ra = pd.get(arm) if isinstance(pd, dict) else None
        rb = pd.get('cond_hist') if isinstance(pd, dict) else None
        if ra is None or rb is None:
            skipped.append('%s %s/%s：per_day 未按臂分键' % (gate, pc, arm))
            continue
        checked += 1
        # ① 原始记录校验（在任何 dict 化之前）
        bad_raw = []
        for tag, rows in (('A', ra), ('B', rb)):
            ids = [r.get('i') for r in rows]
            if len(rows) != 40:
                bad_raw.append('%s 记录数 %d≠40' % (tag, len(rows)))
                continue
            if any((not isinstance(v, int)) or isinstance(v, bool) for v in ids):
                bad_raw.append('%s 含非整数 day_id %s' % (tag, [v for v in ids
                                                               if not isinstance(v, int)][:4]))
                continue
            if len(set(ids)) != len(ids):
                bad_raw.append('%s day_id 重复' % tag)
                continue
            if sorted(ids) != list(range(40)):
                bad_raw.append('%s day_id 集合≠0..39' % tag)
        if bad_raw:
            fails.append('D①: %s %s/%s %s' % (gate, pc, arm, '；'.join(bad_raw)))
            continue
        # ② 按**生产者顺序**（a_rows 原始顺序）配对并重算
        mb = {r['i']: r.get('utility') for r in rb}
        diffs, bad_u = [], []
        for r in ra:
            ua, ub = r.get('utility'), mb.get(r['i'])
            if ua is None or ub is None or not np.isfinite(ua) or not np.isfinite(ub):
                bad_u.append(r['i'])
            else:
                diffs.append(ua - ub)
        if bad_u:
            fails.append('D②: %s %s/%s 含不可配对/非有限效用 day=%s'
                         % (gate, pc, arm, bad_u[:6]))
            continue
        rec = _stat_replica(diffs, seed_use)
        if rec is None or abs(rec['mean'] - d['mean']) > 1e-6:
            fails.append('D②: %s %s/%s 重算均值 %s ≠ 落盘 %.6f（**同源失败**）'
                         % (gate, pc, arm,
                            ('None' if rec is None else '%.6f' % rec['mean']), d['mean']))
            continue
        # ③ CI 排零方向一致
        if d.get('ci_lo') is not None and d.get('ci_hi') is not None:
            if (d['ci_lo'] > 0 or d['ci_hi'] < 0) != (rec['ci_lo'] > 0 or rec['ci_hi'] < 0):
                fails.append('D③: %s %s/%s **CI 排零方向不一致**：落盘 [%.3f, %.3f] '
                             'vs 重算 [%.3f, %.3f]'
                             % (gate, pc, arm, d['ci_lo'], d['ci_hi'],
                                rec['ci_lo'], rec['ci_hi']))
                continue
            # ④ CI 数值精确复现
            dev = max(abs(rec['ci_lo'] - d['ci_lo']), abs(rec['ci_hi'] - d['ci_hi']))
            if dev > ci_exact_tol:
                ci_inexact.append('%s %s/%s CI 偏差 %.6f（落盘 [%.3f, %.3f] '
                                  'vs 重算 [%.3f, %.3f]）'
                                  % (gate, pc, arm, dev, d['ci_lo'], d['ci_hi'],
                                     rec['ci_lo'], rec['ci_hi']))
        A = _u(_idx(data), gate, arm, pc)
        B = _u(_idx(data), gate, 'cond_hist', pc)
        if A is not None and B is not None and abs((A - B) - rec['mean']) > 1e-6:
            fails.append('D②: %s %s/%s 逐日差均值与"两臂效用之差"不一致' % (gate, pc, arm))
    return checked, skipped, fails, warns, ci_inexact, src_tally


def part_c(foreign_values, own_ctx_values):
    """文档层：外来协议基线值 与 本协议配对差 同行的启发式扫描。"""
    flags = []
    fv_pats = [re.compile(r'(?<![\d.])' + re.escape('%.1f' % v) + r'(?![\d])')
               for v in foreign_values]
    ctx_pats = [re.compile(r'(?<![\d.])' + re.escape(s) + r'(?![\d])')
                for s in own_ctx_values]
    for rel in DOCS:
        p = os.path.join(ROOT, *rel.split('/'))
        if not os.path.isfile(p):
            continue
        for i, line in enumerate(open(p, encoding='utf-8'), 1):
            if not any(f.search(line) for f in fv_pats):
                continue
            if not any(c.search(line) for c in ctx_pats):
                continue
            if any(mk in line for mk in NEG_MARKERS):
                continue                      # 已声明为另一协议 ⇒ 合法
            flags.append('C: %s:%d 同一行同时出现「外来协议基线值」与「本协议配对差」'
                         '且未见声明标记' % (rel, i))
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='CC_Compare/results/a1_compare_table/dataset.json')
    ap.add_argument('--own-context', default='144.680,144.7,456.317,456.3',
                    help='本协议上下文数字（与外来基线同现即告警）')
    ap.add_argument('--gates-root', default=os.path.join(
        ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem'),
        help='原始 gate.json 所在根（gate 路径以 results/ 开头，相对于扩展根）')
    args = ap.parse_args()
    p = args.data if os.path.isabs(args.data) else os.path.join(ROOT, *args.data.split('/'))
    if not os.path.isfile(p):
        print('FAIL: dataset 不存在: %s' % p)
        return 2
    data = json.load(open(p, encoding='utf-8'))

    checked, fa = part_a(data)
    foreign, implied, fb = part_b(data)
    fc = part_c(sorted(set(round(v, 3) for v in foreign.values())),
                [s.strip() for s in args.own_context.split(',') if s.strip()])

    print('A. 数据层恒等式：检查 %d 个配对块 → %d 条失败' % (checked, len(fa)))
    for x in fa[:12]:
        print('   ✗ %s' % x)
    print('B. 跨协议护栏：隐含对照值已逐个核对（%d 个）；外部协议另有 %d 个基线值'
          % (len(implied), len(foreign)))
    _i3 = {k: v for k, v in implied.items() if 'i3_online_recovery' in k and 'maskco' in k}
    if _i3:
        print('   I3 侧隐含对照（应 = 600.997）：%s' % _i3)
    for x in fb[:12]:
        print('   ✗ %s' % x)
    print('C. 文档层混用扫描（启发式，已声明行豁免）：%d 条' % len(fc))
    for x in fc[:12]:
        print('   ✗ %s' % x)

    checked_d, skipped_d, fd, wd, cid, src_tally = part_d(data, args.gates_root)
    # 用户要求：④ CI 数值精确复现必须**单独**报告，且不得与笼统 PASS 混淆
    _ci_txt = ('CI_exact_reproduction=True' if not cid
               else 'CI_exact_reproduction=**False**（%d 个块未逐位复现）' % len(cid))
    print('D. 日级证据：①相同 day_id / ②均值重算 / ③CI 排零方向 / ④CI 数值精确复现')
    print('   核对 %d 个配对块 → 硬失败 %d、CI 数值未复现 %d、SKIP %d'
          % (checked_d, len(fd), len(cid), len(skipped_d)))
    print('   每块所用 gate 种子来源（确定性优先级，未做事后择优）：%s'
          % ', '.join('%s=%d' % kv for kv in sorted(src_tally.items())))
    for x in fd[:12]:
        print('   ✗ %s' % x)
    if cid:
        print('   ⚠️ ④ %s（**保留原正式裁决与历史 CI**，不自动推翻 I3 负结果）：'
              % _ci_txt)
        for x in cid[:6]:
            print('      · %s' % x)
    else:
        print('   ✅ ④ %s' % _ci_txt)
    if skipped_d:
        print('   ⚠️ SKIP %d 个块（**不得**当作 PASS）：' % len(skipped_d))
        for x in skipped_d[:4]:
            print('      · %s' % x)

    if fa or fb or fc or fd:
        print('PAIRING_IDENTITY_FAIL: A=%d B=%d C=%d D=%d'
              % (len(fa), len(fb), len(fc), len(fd)))
        return 1
    if skipped_d:
        print('PAIRING_IDENTITY_PARTIAL: A/B/C 通过，但 D 段有 %d 个块因 gate 不可得而 SKIP '
              '⇒ **不构成完整通过**（请在持有原始 gate 的机器上重跑）' % len(skipped_d))
        return 3
    if cid:
        print('PAIRING_IDENTITY_PASS_WITH_CI_NOT_EXACT: ①②③ 通过；④ %s '
              '⇒ **不是**完整 PASS（原 CI 保留，须在 Release 元数据中标注）' % _ci_txt)
        return 4
    print('PAIRING_IDENTITY_PASS（①②③④ 全部通过，CI 逐位复现）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
