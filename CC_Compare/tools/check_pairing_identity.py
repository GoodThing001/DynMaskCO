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

    if fa or fb or fc:
        print('PAIRING_IDENTITY_FAIL: A=%d B=%d C=%d' % (len(fa), len(fb), len(fc)))
        return 1
    print('PAIRING_IDENTITY_PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
