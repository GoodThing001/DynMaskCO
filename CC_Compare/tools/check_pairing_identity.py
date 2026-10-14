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


def _per_day_of(gate_path, pc):
    """从原始 gate 取**指定档位**的 `per_day`（可能是 list 或 dict{臂: list}）。

    ⚠️ 必须按 `pc` 选块：一个 gate 的 `results[year]` 下同时有 `p_c=0` /
    `p_c=(5,10,15)` / `p_c=(10,20,30)` 三个块，各带自己的 `per_day`。
    2026-10-14 实测踩到：只要"第一个含 per_day 的块"，就会拿**别的档位**的逐日记录
    去配本档的配对差 ⇒ 一次跑出 148 条假失败。
    """
    if not os.path.isfile(gate_path):
        return None
    try:
        g = json.load(open(gate_path, encoding='utf-8'))
    except Exception:                                        # noqa: BLE001
        return None
    cont = g.get('results') or g.get('gate') or {}
    for _k, yb in cont.items():
        if not isinstance(yb, dict):
            continue
        # 扁平 schema：gate[p_c]['per_day']
        if 'per_day' in yb and _k == pc:
            return yb['per_day']
        # 嵌套 schema：results[seed][p_c]['per_day']
        blk = yb.get(pc)
        if isinstance(blk, dict) and 'per_day' in blk:
            return blk['per_day']
    return None


def part_d(data, gates_root, ci_material_tol=5.0):
    """**日级证据**（2026-10-14 用户要求）：同一组 40 个 day_id + 重算日差/均值/CI。

    前两段只能证明"均值恒等式成立"，**不能**证明两边真的用了同一批 day_id。
    本段从原始 gate 的 `per_day` 取 A/B 的逐日记录，要求：
      ① day_id 集合完全相同（否则 FAIL）；
      ② 逐日差值的**均值**与落盘配对块逐位一致（FAIL 门槛）；
      ③ 按天聚类 bootstrap CI **重算**；判据是**决策相关的那一项** ——
         「CI 是否排除 0」必须与落盘一致（不一致 ⇒ FAIL）；
         边界数值若偏差 > `ci_material_tol` 只记 WARN。
      ④ gate 不可得 ⇒ SKIP 并说明原因（**不得**记为 PASS）。

    ⚠️ 为什么 CI 不要求逐位一致（2026-10-14 实测）：主模型的配对块是**主模型 driver**
    自己算的，不是 `collect_compare_table.py` 复刻的那份；两者 bootstrap 参数/取流顺序
    可能不同。实测 36 个块的 CI 在第 3–4 位有效数字上有差异，而**均值全部逐位一致**、
    day_id 集合全部相同。⇒ 对 CI 只保证"结论（是否排除 0）"一致，避免把生产者差异
    误报成同源失败 —— 但如果连结论都不一致，就必须 FAIL。
    """
    fails, warns, skipped, checked = [], [], [], 0
    seed = int(((data.get('generated_from') or {}).get('seed')) or 20260926)
    for gate, pc, arm, d, _i in _pairs(data):
        gp = gate if os.path.isabs(gate) else os.path.join(gates_root, *str(gate).split('/'))
        pd = _per_day_of(gp, pc)
        if pd is None:
            skipped.append('%s %s/%s：原始 gate 不可得（%s）' % (gate, pc, arm, gp))
            continue
        ra = pd.get(arm) if isinstance(pd, dict) else None
        rb = pd.get('cond_hist') if isinstance(pd, dict) else None
        if ra is None or rb is None:
            skipped.append('%s %s/%s：per_day 未按臂分键（无法取 A/B 逐日记录）'
                           % (gate, pc, arm))
            continue
        ma = {int(r['i']): r.get('utility') for r in ra if r.get('i') is not None}
        mb = {int(r['i']): r.get('utility') for r in rb if r.get('i') is not None}
        checked += 1
        if set(ma) != set(mb):
            fails.append('D: %s %s/%s 两侧 day_id 集合不同（仅A %s / 仅B %s）'
                         % (gate, pc, arm, sorted(set(ma) - set(mb))[:6],
                            sorted(set(mb) - set(ma))[:6]))
            continue
        days = sorted(ma)                      # 与 collect_compare_table._pair 同序（按 day 排序）
        pair = [(ma[k], mb[k]) for k in days
                if ma[k] is not None and mb[k] is not None
                and np.isfinite(ma[k]) and np.isfinite(mb[k])]
        if not pair:
            fails.append('D: %s %s/%s 逐日差为空' % (gate, pc, arm))
            continue
        diffs = [a - b for a, b in pair]
        rec = _stat_replica(diffs, seed)
        if abs(rec['mean'] - d['mean']) > 1e-6:
            fails.append('D: %s %s/%s 重算均值 %.6f ≠ 落盘 %.6f（**同源失败**）'
                         % (gate, pc, arm, rec['mean'], d['mean']))
        # 决策相关判据：CI 是否排除 0
        if d.get('ci_lo') is not None and d.get('ci_hi') is not None:
            excl_rec = (d['ci_lo'] > 0 or d['ci_hi'] < 0)
            excl_new = (rec['ci_lo'] > 0 or rec['ci_hi'] < 0)
            if excl_rec != excl_new:
                fails.append('D: %s %s/%s **CI 的「是否排除 0」结论不一致**：'
                             '落盘 [%.3f, %.3f] vs 重算 [%.3f, %.3f]'
                             % (gate, pc, arm, d['ci_lo'], d['ci_hi'],
                                rec['ci_lo'], rec['ci_hi']))
            else:
                dev = max(abs(rec['ci_lo'] - d['ci_lo']), abs(rec['ci_hi'] - d['ci_hi']))
                if dev > ci_material_tol:
                    warns.append('D: %s %s/%s CI 边界偏差 %.2f（结论一致；'
                                 '生产者可能用了不同 bootstrap 参数）'
                                 % (gate, pc, arm, dev))
        # 逐日差均值必须等于两臂效用之差
        A = _u(_idx(data), gate, arm, pc)
        B = _u(_idx(data), gate, 'cond_hist', pc)
        if A is not None and B is not None and abs((A - B) - rec['mean']) > 1e-6:
            fails.append('D: %s %s/%s 逐日差均值与"两臂效用之差"不一致' % (gate, pc, arm))
    return checked, skipped, fails, warns


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

    checked_d, skipped_d, fd, wd = part_d(data, args.gates_root)
    print('D. 日级证据（同一组 day_id + 重算日差/均值/CI）：核对 %d 个配对块 → %d 条失败、%d 条告警'
          % (checked_d, len(fd), len(wd)))
    for x in fd[:12]:
        print('   ✗ %s' % x)
    for x in wd[:6]:
        print('   ⚠️ %s' % x)
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
    print('PAIRING_IDENTITY_PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
