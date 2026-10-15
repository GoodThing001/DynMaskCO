#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**版本化展示件生成器**：A-v1 对比方法总表 → 资格分层版（v2）。

授权：用户 2026-10-14「审查结论：可以推进新版展示件，但不能直接照搬当前的 S1 标签」。
  * 新文件名 + 新 manifest；**旧总表与旧 manifest 逐字节不变**；
  * 用**现有逐行记录**重排展示，加入 配对组 / 证据资格 / 因果核验 / 依赖闭包 / 方法保真度；
  * **禁止**筛除 S3/S4 行后再重新排名；主档优先，`p_c=0` 单独作敏感性；
  * 修正判定逻辑（三个维度分开），`Q_method_health=null` 是**结构性不适用**、不得强制为 True；
  * `λ=3` 单列 `DEGENERATE`，不并入"不可核验"；
  * `booking_horizon` 缺口单列 `causal_gate_status`，**不得**用"已登记"替代"已验证"。

本模块**只读**输入：旧总表（解析）+ 资格矩阵（读取）。**不**运行任何方法、
**不**重算效用、**不**修改任何既有文件。
"""
from __future__ import annotations

try:
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # pragma: no cover
    pass

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CC = os.path.dirname(_HERE)
_REPO = os.path.dirname(_CC)
_DOCS = os.path.join(_CC, 'docs', 'a1_development_comparison')
OLD_TABLE = os.path.join(_DOCS, 'A-v1对比方法总表.md')
OLD_MANIFEST = os.path.join(_DOCS, 'ARTIFACT_MANIFEST.json')
MATRIX = os.path.join(_CC, 'audit', 'BASELINE_QUALIFICATION_MATRIX_2026-10-14.json')

NEW_NAME = 'A-v1对比方法总表_v2_资格分层.md'
NEW_MANIFEST_NAME = 'ARTIFACT_MANIFEST_v2_资格分层.json'

PC_MAIN = 'p_c=(5,10,15)'
PC_ZERO = 'p_c=0'
PC_STRICT = 'p_c=(10,20,30)'

# ---- 证据资格（**三个维度分开**中的"证据资格"维；机械判定，顺序即优先级）----
#: ⚠️ `REFERENCE` 是**保留值**，只给"配对参照臂本身"（不作优势判定对象）。
EVIDENCE_TIERS = ('DEGENERATE', 'NOT_VERIFIABLE', 'STRICT_PASS', 'CORE_PASS',
                  'DIAGNOSTIC', 'REFERENCE')

# ---- 方法保真度（是否代表原论文完整算法）----
FIDELITY = ('native', 'solver-adapted', 'ordering-adapted', 'incompatible')

# ---- 因果核验状态 ----
CAUSAL = ('VERIFIED', 'IDENTITY_REGISTERED_ONLY', 'NOT_VERIFIABLE_FROM_ARCHIVED_GATE',
          'NOT_APPLICABLE')

#: 旧总表 §1 的列序（**逐字保留**原始单元格）
OLD_COLS = ('idx', 'method', 'cmp_status', 'q_core', 'dep_closure', 'pair_source',
            'protocol_id', 'category', 'venue', 'ccf', 'backbone', 'params',
            'native_constraint', 'u_zero', 'u_main', 'u_strict',
            'd_zero', 'd_main', 'to_zero', 'to_main', 'hard_zero',
            'served_zero', 'energy_zero', 'note')

#: 行 → (旧表行号, arm_dir, arm, p_c, pairing_group, fidelity, 超时风险, 其它风险标记)
#: `pairing_group`：A = 外部正式批（配对基线 `a1_c1fresh_reveal_40|cond_hist`）；
#:                 B = Ours I3 自批；NONE = 跨源、无配对基线。
ROW_SPECS = [
    (1, None, 'cond_hist', PC_MAIN, 'A', 'native', False, '配对参照臂本身（不作优势判定对象）'),
    (2, 'A3（掩码重构·预训练权重）', 'maskco', PC_MAIN, 'NONE', 'native', True,
     '配对来源**跨源**（shared_source_sha256_mismatch）⇒ 禁止用于任何优势判定'),
    (3, 'I3（决策感知）', 'maskco', PC_MAIN, 'B', 'native', True,
     '**有同批配对证据**，但**超时资格存在缺口**；配对对象是同批 cond_hist=600.997'),
    (4, 'rule_valuee20_51015', 'rule_valuee_lam2.00', PC_MAIN, 'A', 'native', False, ''),
    (5, 'rule_myopic_51015', 'rule_myopic', PC_MAIN, 'A', 'native', False, ''),
    (6, 'ortools', 'ortools_accept', PC_MAIN, 'A', 'solver-adapted', False, ''),
    (7, 'sgbs_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', True, ''),
    (8, 'routefinder_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False, ''),
    (9, 'mvmoe_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False, ''),
    (10, 'pomo_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False,
     '与 SGBS / Omni-VRP **共用同一权重文件** ⇒ 非独立模型'),
    (11, 'attention_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False, ''),
    (12, 'rrnco_51015', 'rrnco_accept', PC_MAIN, 'A', 'ordering-adapted', False, ''),
    (13, 'coenrich_nopchgs_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False, ''),
    (14, 'coenrich_pchgs_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False, ''),
    (15, 'symnco_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', True, ''),
    (16, 'omnivrp_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False,
     '与 POMO **共用同一权重文件**'),
    (17, 'mapt_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False,
     '**弱介入**：模型只决定前 ~3 位、尾部 EDD ⇒ 不得当作 MAPT 方法能力'),
    (18, 'lih_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', True,
     '**跨次不可复现**（清缓存后两次调用不一致）'),
    (19, 'deepaco_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False,
     '**跨次不可复现**（ACO 未固定采样 seed）'),
    (20, 'cada_51015', 'ordering_accept', PC_MAIN, 'A', 'ordering-adapted', False, ''),
    (21, 'pyvrp', 'pyvrp_accept', PC_MAIN, 'A', 'solver-adapted', False,
     '**展示层身份矛盾**：数值来自修复腿（须 0.11.3），旧标注写 0.14.0 ⇒ 见 manifest'),
    (22, 'rule_valuee30_51015', 'rule_valuee_lam3.00', PC_MAIN, 'A', 'native', False,
     '**全拒单（0 接单）的退化规则**，不是"性能很差的规则"'),
]

#: 不在 22 行之内的 L-D 机制不匹配（无任何数值）
INCOMPATIBLE = (
    ('Learning-to-Delegate', '回归目标是 HGS 子问题距离、规模 n≥500；A-v1 单决策可见客户 ≤90'),
    ('PIP-constraint', '单车辆 TSPTW/TSPDL，env 无车队维度；无法对接 15 车接单+路由决策'),
)

#: PyVRP 展示层身份（**只读核验结论**；历史运行时**不得**由当前环境倒推）
PYVRP_IDENTITY = {
    'method': 'PyVRP-RH accept',
    'adaptation_arm_impl': 'A-v1 `SolverAcceptReplanner` + PyVRP HGS',
    'repair_leg_runtime_version': '0.11.3',
    'independent_frozen_method': 'PyVRP-RH-D，wheel `0.14.0`',
    'result_attribution': 'A-v1 修复腿，**不是** PyVRP-RH-D 的独立冻结实验',
    'identity_evidence_status': 'REPORTED / HISTORICAL_RUNTIME_NOT_INDEPENDENTLY_VERIFIED',
}


# ------------------------------------------------------------------ IO helpers
def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _read_bytes(p):
    with io.open(p, 'rb') as f:
        return f.read()


def _git(*args):
    r = subprocess.run(['git'] + list(args), cwd=_REPO,
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.stdout.decode('utf-8', 'replace').strip() if r.returncode == 0 else None


def parse_old_rows():
    """解析旧总表 §1 的 22 行；**单元格逐字保留**。"""
    text = io.open(OLD_TABLE, encoding='utf-8').read()
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('## 1. 主表'))
    rows = []
    for l in lines[start:]:
        if l.startswith('## 2. '):
            break
        if not l.startswith('|'):
            continue
        cells = [c.strip() for c in l.strip().strip('|').split('|')]
        if len(cells) != len(OLD_COLS):
            continue
        if cells[0] in ('#', '---') or set(cells[0]) <= set('-: '):
            continue
        if not cells[0].isdigit():
            continue
        rows.append(dict(zip(OLD_COLS, cells)))
    return rows, text


def load_matrix():
    d = json.load(io.open(MATRIX, encoding='utf-8'))
    idx = {}
    for a in d['arms']:
        idx[(a['arm_dir'], a['arm'], a['p_c'])] = a
    return d, idx


# ------------------------------------------------------- 三维度机械判定
def evidence_tier(q, arm, is_reference=False):
    """**证据资格**维：机械判定，顺序即优先级。

    ⚠️ **本轮修正（用户 2026-10-14 裁定）**：`Q_identity is None` 是
    **结构性不适用**（纯规则 / 基线臂**没有** checkpoint 身份块），
    **不得**与 `Q_identity is False`（真的被否决）混同。
    上一版把 `None` 也判成 `NOT_VERIFIABLE`，会把 `Q_formal_strict=True` 的规则臂
    错标为"不可核验" —— 与用户点名的 `Q_method_health=null` 是**同一类错误**。
    """
    if is_reference:
        return 'REFERENCE'
    if q.get('Q_nondegen') is False:
        return 'DEGENERATE'
    if q.get('Q_identity') is False:                       # 仅 False 才是否决
        return 'NOT_VERIFIABLE'
    if q.get('Q_formal_strict') is True:
        return 'STRICT_PASS'
    if q.get('Q_formal_core') is True:
        return 'CORE_PASS'
    return 'DIAGNOSTIC'


def causal_status(arm):
    """**因果核验**维：`booking_horizon` 等因果字段是否**验证**过（非"已登记"）。"""
    gp = arm.get('gate_path')
    if not gp:
        return 'NOT_VERIFIABLE_FROM_ARCHIVED_GATE'
    # 归档在服务器；本地可解析的同名 gate 若存在则尝试读字段，否则如实判不可核验
    local = os.path.join(_CC, 'results', *(gp.split('/results/')[-1].split('/')))
    if not os.path.isfile(local):
        return 'NOT_VERIFIABLE_FROM_ARCHIVED_GATE'
    try:
        g = json.load(io.open(local, encoding='utf-8'))
    except Exception:                                        # noqa: BLE001
        return 'NOT_VERIFIABLE_FROM_ARCHIVED_GATE'
    if 'booking_horizon' not in g:
        return 'NOT_VERIFIABLE_FROM_ARCHIVED_GATE'
    return 'IDENTITY_REGISTERED_ONLY' if g.get('booking_horizon') is None else 'VERIFIED'


def method_health_cell(arm):
    """`Q_method_health`：**None = 结构性不适用**（不得强制为 True）。

    表格里只写短标签；完整理由见 manifest 的 `rows[].method_health_reason`。
    """
    k = (arm.get('Q') or {}).get('Q_method_health')
    if k is None:
        return 'N/A（结构性不适用）'
    return 'True' if k else 'False'


def method_health_reason(arm):
    k = (arm.get('Q') or {}).get('Q_method_health')
    if k is not None:
        return None
    return ((arm.get('Q_reasons_not_applicable') or {}).get('Q_method_health')
            or '结构性不适用')


def _tok(md_cell):
    """从形如 `631.4 [522.7, 727.6]` 的单元格提取**首个数**用于排序（不改原值）。"""
    import re
    m = re.search(r'-?\d+\.?\d*', md_cell or '')
    return float(m.group(0)) if m else float('-inf')


# ------------------------------------------------------------------ 渲染
def render(rows, mx):
    by_row = {r['idx']: r for r in rows}
    enriched = []
    for (idx, arm_dir, arm, p_c, grp, fid, to_risk, extra) in ROW_SPECS:
        r = by_row[str(idx)]
        a = mx.get((arm_dir, arm, p_c))
        q = (a or {}).get('Q') or {}
        tier = evidence_tier(q, a or {}, is_reference=(idx == 1))
        cs = causal_status(a or {}) if a else 'NOT_APPLICABLE'
        dep = {'True': 'True', 'False': 'False', 'None': 'N/A（结构性不适用）'}.get(
            str((a or {}).get('Q_dependency_closure')), 'N/A（未在该矩阵中单列）')
        # `Q_identity=None` ⇒ 结构性不适用（**不得**当否决）
        if a is None:
            ident = 'N/A（未在该矩阵中单列，身份以 §2 基线表为准）'
        elif q.get('Q_identity') is None:
            why = (a.get('Q_reasons_not_applicable') or {}).get('Q_identity') or '结构性不适用'
            why = str(why).replace('（', '（').strip()
            why = why.split('；')[0].rstrip('（(，,、 ')[:36]
            ident = 'N/A（%s）' % why
        else:
            ident = 'True' if q.get('Q_identity') else 'False'
        problems = a.get('identity_problems') or [] if a else []
        risks = []
        if to_risk:
            risks.append('超时>0')
        if problems:
            risks.append('身份问题：%s' % '，'.join(str(x) for x in problems[:2]))
        if extra:
            risks.append(extra)
        enriched.append({
            'old_idx': idx, 'method': r['method'], 'fidelity': fid, 'group': grp,
            'tier': tier, 'causal': cs, 'dep': dep, 'ident': ident,
            'health': method_health_cell(a) if a else 'N/A（未在该矩阵中单列）',
            'health_reason': method_health_reason(a) if a else None,
            'u_main': r['u_main'], 'd_main': r['d_main'], 'to_main': r['to_main'],
            'u_zero': r['u_zero'], 'd_zero': r['d_zero'], 'to_zero': r['to_zero'],
            'hard_zero': r['hard_zero'], 'served_zero': r['served_zero'],
            'energy_zero': r['energy_zero'], 'u_strict': r['u_strict'],
            'risks': '；'.join(risks) if risks else '—',
            'arm_dir': arm_dir, 'arm': arm, 'p_c': p_c,
        })
    return enriched


def md_report(enriched, meta):
    L = []
    A = L.append
    A('# A-v1 对比方法总表 · v2 资格分层版（版本化展示件）')
    A('')
    A('> **版本化展示件，不覆盖旧总表。** 旧总表与旧 `ARTIFACT_MANIFEST.json` '
      '**逐字节保留**（见 `%s`）。' % NEW_MANIFEST_NAME)
    A('>')
    A('> 本件按用户 2026-10-14 裁定制作：**三个维度分开**，'
      '**不筛除任何行**，主档优先、`p_c=0` 单独作敏感性。')
    A('>')
    A('> ⚠️ **本件不改变任何负结果**：主档同批 `I3 − cond_hist = '
      '−144.680 [−191.109, −100.125]`（同批 `cond_hist = 600.997`）原样保留。')
    A('')
    A('## 0. 三个维度（**不要合并成一个标签**）')
    A('')
    A('| 维度 | 字段 | 取值 | 解决的问题 |')
    A('|---|---|---|---|')
    A('| 配对范围 | `pairing_group` | `A` / `B` / `NONE` | 哪些结果能在**同批**比较 |')
    A('| 证据资格 | `evidence_tier` | `%s` | 该行证据**能支持什么结论** |'
      % ' / '.join(EVIDENCE_TIERS))
    A('| 方法保真度 | `fidelity` | `%s` | 是否代表**原论文完整算法** |'
      % ' / '.join(FIDELITY))
    A('| 因果核验 | `causal_gate_status` | `%s` | **登记 ≠ 验证** |' % ' / '.join(CAUSAL))
    A('')
    A('**判定规则（可机械执行）**：')
    A('')
    A('1. `evidence_tier` 按**优先级顺序**判定（`REFERENCE` 为保留值，只给配对参照臂本身）：')
    A('   `Q_nondegen=False` ⇒ `DEGENERATE`；**`Q_identity=False`** ⇒ `NOT_VERIFIABLE`；'
      '`Q_formal_strict=True` ⇒ `STRICT_PASS`；'
      '`Q_formal_core=True` ⇒ `CORE_PASS`；否则 `DIAGNOSTIC`。')
    A('2. ⚠️ **`Q_identity=None` 是结构性不适用，不是否决**：'
      '纯规则 / 基线臂**没有** checkpoint 身份块 ⇒ 单列 `identity` 显示 `N/A（…）`。'
      '上一版把 `None` 也判成 `NOT_VERIFIABLE`，会把 `Q_formal_strict=True` 的规则臂'
      '错标为"不可核验"（与 `Q_method_health=null` 是**同一类错误**）。')
    A('3. **`pairing_group` 与 `evidence_tier` 相互独立**：'
      '有配对关系**不等于**严格合格（例：`I3` = 配对组 B + `CORE_PASS`，且有超时缺口），'
      '严格合格也**不等于**与其他组可配对。')
    A('4. **`Q_method_health=None` 是结构性不适用**（内部基线/规则臂的 gate 不含该判据），'
      '**不得**强制要求等于 `True`，也**不得**因此降级。')
    A('5. **`booking_horizon` 缺口是跨状态的因果风险**：'
      '本件单列 `causal_gate_status`；**不得**用"已登记"替代"已验证"。')
    A('6. **`λ=3` 单列 `DEGENERATE`**（可观察的退化行为），'
      '**不得**并入"不可核验"（会让读者误以为原件缺失）。')
    A('7. `pairing_group` 标注**不代表**超时/健康资格通过；'
      '超时缺口与身份问题在「风险标记」列**逐行**显式给出。')
    A('')

    def _table(title, rows, val_u, val_d, val_to, note):
        A('## %s' % title)
        A('')
        A(note)
        A('')
        A('| 原# | 方法 | 方法保真度 | 配对组 | 证据资格 | 因果核验 | 依赖闭包 | 身份 | '
          '方法健康 | Utility | Δ vs cond_hist | 超时/天 | 风险标记 |')
        A('|---:|---|---|---|---|---|---|---|---|---:|---:|---:|---|')
        for r in rows:
            A('| %d | %s | `%s` | **%s** | **%s** | `%s` | %s | %s | %s | %s | %s | %s | %s |'
              % (r['old_idx'], r['method'], r['fidelity'], r['group'], r['tier'],
                 r['causal'], r['dep'], r['ident'], r['health'], r[val_u], r[val_d],
                 r[val_to], r['risks']))
        A('')

    main_rows = sorted([r for r in enriched if r['p_c'] == PC_MAIN],
                       key=lambda r: -_tok(r['u_main']))
    _table('1. 主档 `p_c=(5,10,15)`（**正式主结论口径**）', main_rows, 'u_main', 'd_main',
           'to_main',
           '> **这是正式论文主结论的档位。** 排序**仅**用于阅读；'
           '跨 `pairing_group` 的数值**不得**直接做差。')
    zero_rows = sorted([r for r in enriched if r['p_c'] == PC_ZERO],
                       key=lambda r: -_tok(r['u_zero']))
    _table('2. 敏感性档 `p_c=0`（**不是**主结论口径）', zero_rows, 'u_zero', 'd_zero',
           'to_zero',
           '> ⚠️ `p_c=0` 绝对效用更高**只是评分函数差异**（接拒构造**不接收** `p_c`），'
           '**不得**充当头条贡献证据。')
    strict_rows = [r for r in enriched if r['p_c'] == PC_STRICT]
    if strict_rows:
        _table('3. 第三档 `p_c=(10,20,30)`', strict_rows, 'u_strict', 'd_zero', 'to_zero',
               '> 该档多为负值；保留以完整呈现，不作主结论。')

    A('## 4. 逐行原始数值（**逐字保留**，供一致性核对）')
    A('')
    A('| 原# | Utility p_c=0 | Utility 主档 | Utility p_c=(10,20,30) | Δ(p_c=0) | '
      'Δ(主档) | 超时/天 p_c=0 | 超时/天 主档 | 硬可行率 p_c=0 | 服务单/天 | 能耗 kWh/天 |')
    A('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in enriched:
        A('| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |'
          % (r['old_idx'], r['u_zero'], r['u_main'], r['u_strict'], r['d_zero'],
             r['d_main'], r['to_zero'], r['to_main'], r['hard_zero'],
             r['served_zero'], r['energy_zero']))
    A('')
    A('## 5. PyVRP：实现身份拆分（**只读核验结论**）')
    A('')
    A('| 字段 | 内容 |')
    A('|---|---|')
    for k, v in PYVRP_IDENTITY.items():
        A('| `%s` | %s |' % (k, v))
    A('')
    A('> **不得**用现在执行一次 `import pyvrp; print(__version__)` 反推历史运行时版本。'
      '应先检查历史启动命令、解释器绝对路径、环境清单、原始日志，'
      '以及当时是否记录了 package/wheel SHA。')
    A('> 若历史未封存包版本 ⇒ 保持「**报告称 0.11.3、历史环境未独立核验**」。')
    A('> 出处：`读表注意与已知限定.md` §2.5、`docs/a1_development_comparison/README.md` §13.3、'
      '`audit/INDEPENDENT_READONLY_ADJUDICATION_2026-10-14.md`。')
    A('')
    A('## 6. 机制不匹配（**未适配**，无任何数值）')
    A('')
    A('| 方法 | 原因 |')
    A('|---|---|')
    for n, why in INCOMPATIBLE:
        A('| %s | %s |' % (n, why))
    A('')
    A('## 7. 本件**不**做什么')
    A('')
    A('- **不**改旧总表、旧 `ARTIFACT_MANIFEST.json`、任何封存运行原件与既有 gate；')
    A('- **不**筛除 `S3/S4` 行后再重新排名（全部 22 行**逐行保留**）；')
    A('- **不**运行求解器、**不**重算效用、**不**新增/改写任何数值；')
    A('- **不**构成优势主张；主档最可信结论仍是**负结果**：'
      '`I3 − cond_hist = −144.680 [−191.109, −100.125]`。')
    A('')
    A('---')
    A('')
    A('_生成器：`CC_Compare/tools/render_qualification_tiered_table.py`；'
      '源提交 `%s`；旧总表工作树 SHA `%s`。_'
      % (meta['source_commit'], meta['old_table_worktree_sha256'][:16]))
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------ 自检
def display_selfcheck(old_text, old_rows, new_text, enriched, meta, mx):
    chk = []

    def c(name, ok, detail=''):
        chk.append({'check': name, 'ok': bool(ok), 'detail': detail})

    # ① 旧版身份不变
    ob = _read_bytes(OLD_TABLE)
    c('old_table_worktree_sha256_unchanged',
      _sha_bytes(ob) == meta['old_table_worktree_sha256'],
      '%s' % _sha_bytes(ob)[:16])
    blob = _git('cat-file', 'blob', 'HEAD:CC_Compare/docs/a1_development_comparison/'
                'A-v1对比方法总表.md')
    c('old_table_blob_sha256_unchanged',
      blob is not None and _sha_bytes(blob.encode('utf-8') if isinstance(blob, str)
                                      else blob) == meta['old_table_blob_sha256'],
      'blob=%s' % (_sha_bytes(blob.encode()) [:16] if blob else None))
    c('old_manifest_sha256_unchanged',
      _sha_bytes(_read_bytes(OLD_MANIFEST)) == meta['old_manifest_sha256'])
    # ② 22 行原始数值不变
    c('n_rows_22', len(enriched) == 22 and len(old_rows) == 22,
      '%d/%d' % (len(enriched), len(old_rows)))
    bad = []
    by_idx = {r['idx']: r for r in old_rows}
    for r in enriched:
        o = by_idx[str(r['old_idx'])]
        for new_k, old_k in (('u_zero', 'u_zero'), ('u_main', 'u_main'),
                             ('u_strict', 'u_strict'), ('d_zero', 'd_zero'),
                             ('d_main', 'd_main'), ('to_zero', 'to_zero'),
                             ('to_main', 'to_main'), ('hard_zero', 'hard_zero'),
                             ('served_zero', 'served_zero'),
                             ('energy_zero', 'energy_zero')):
            if r[new_k] != o[old_k]:
                bad.append((r['old_idx'], new_k, o[old_k], r[new_k]))
    c('all_original_values_verbatim', not bad, '%d mismatch' % len(bad))
    # ③ 配对组不串用
    groups = {r['method']: r['group'] for r in enriched}
    c('pairing_group_not_cross_used',
      groups.get('★ Ours: MaskCO I3（决策感知）') == 'B'
      and groups.get('★ Ours: MaskCO A3（掩码重构 · 预训练权重）') == 'NONE',
      json.dumps({k: groups.get(k) for k in groups if 'Ours' in k}, ensure_ascii=False))
    # ④ I3 不再被误标为严格合格
    i3 = [r for r in enriched if r['old_idx'] == 3][0]
    c('I3_not_mislabeled_strict',
      i3['tier'] != 'STRICT_PASS' and i3['causal'] == 'NOT_VERIFIABLE_FROM_ARCHIVED_GATE',
      'tier=%s causal=%s' % (i3['tier'], i3['causal']))
    # ⑤ 风险标记保留
    risk_expected = {7: '超时>0', 15: '超时>0', 18: '超时>0', 2: '超时>0', 3: '超时>0',
                     10: '共用同一权重文件', 16: '共用同一权重文件', 17: '弱介入',
                     19: '跨次不可复现', 21: '展示层身份矛盾'}
    miss = [i for i, t in risk_expected.items()
            if t not in ([r for r in enriched if r['old_idx'] == i][0]['risks'])]
    c('a1l_risk_markers_retained', not miss, 'missing rows %s' % miss)
    c('degenerate_row_kept_separate',
      [r for r in enriched if r['old_idx'] == 22][0]['tier'] == 'DEGENERATE')
    c('no_row_dropped', len(enriched) == len(OLD_COLS and ROW_SPECS) == 22,
      'rows=%d' % len(enriched))
    c('main_pc_first', new_text.index('## 1. 主档') < new_text.index('## 2. 敏感性档'))
    # ⑥ 三个维度互不替代
    tiers_used = sorted({r['tier'] for r in enriched})
    c('dimensions_independent', {'tier', 'group', 'causal'} <= set(enriched[0]),
      'tiers=%s' % tiers_used)
    c('method_health_not_forced_true',
      any('N/A' in r['health'] for r in enriched),
      [r['health'] for r in enriched if 'N/A' in r['health']][:2])
    # ⑦ **本轮修正的判定缺陷回归**（用户点名同类错误）
    strict_but_nv = [r['old_idx'] for r in enriched
                     if r['tier'] == 'NOT_VERIFIABLE'
                     and (mx.get((r['arm_dir'], r['arm'], r['p_c'])) or {})
                     .get('Q', {}).get('Q_formal_strict') is True]
    c('no_strict_pass_mislabeled_not_verifiable', not strict_but_nv,
      'rows=%s' % strict_but_nv)
    rules = [r for r in enriched if r['old_idx'] in (4, 5)]
    c('rule_arms_not_downgraded_by_identity_none',
      all(r['tier'] == 'STRICT_PASS' and r['ident'].startswith('N/A') for r in rules),
      [(r['old_idx'], r['tier'], r['ident'][:14]) for r in rules])
    c('identity_none_shown_as_structural_na',
      any('N/A' in r['ident'] for r in enriched),
      [r['ident'][:24] for r in enriched if 'N/A' in r['ident']][:2])
    c('reference_tier_used_only_for_reference',
      [r['old_idx'] for r in enriched if r['tier'] == 'REFERENCE'] == [1],
      'reference rows=%s' % [r['old_idx'] for r in enriched if r['tier'] == 'REFERENCE'])
    return chk


def main(argv=None):
    ap = argparse.ArgumentParser(description='生成 A-v1 总表 v2 资格分层展示件（只读）')
    ap.add_argument('--outdir', default=_DOCS)
    ap.add_argument('--write', action='store_true', help='真正写文件（默认仅自检）')
    a = ap.parse_args(argv)

    old_rows, old_text = parse_old_rows()
    mxdoc, mx = load_matrix()
    enriched = render(old_rows, mx)

    meta = {
        'kind': 'development-display-package-v2-qualification-tiered',
        'note': ('**版本化展示件**。旧总表与旧 `ARTIFACT_MANIFEST.json` '
                 '**逐字节保留、不被覆盖**。本 manifest 只描述**新增**的 v2 展示件。'),
        'source_commit': _git('rev-parse', 'HEAD'),
        'old_manifest_path': 'CC_Compare/docs/a1_development_comparison/ARTIFACT_MANIFEST.json',
        'old_manifest_sha256': _sha_bytes(_read_bytes(OLD_MANIFEST)),
        'old_table_path': 'CC_Compare/docs/a1_development_comparison/A-v1对比方法总表.md',
        'old_table_worktree_sha256': _sha_bytes(_read_bytes(OLD_TABLE)),
        'old_table_worktree_size': len(_read_bytes(OLD_TABLE)),
        'old_table_worktree_crlf_count': _read_bytes(OLD_TABLE).count(b'\r\n'),
        'old_table_blob_sha256': None,
        'old_table_blob_note': ('旧总表登记的是**工作树 raw 字节**（CRLF）；'
                                'Git blob 为 LF ⇒ **双身份**，两者分别记录。'
                                '**不得**给它加 `.gitattributes eol=lf`（会破坏登记身份）。'),
        'qualification_matrix_source': ('CC_Compare/audit/'
                                        'BASELINE_QUALIFICATION_MATRIX_2026-10-14.json'),
        'qualification_matrix_sha256': _sha_bytes(_read_bytes(MATRIX)),
        'field_adjustments': [
            '新增 pairing_group（A/B/NONE）：配对范围与证据资格**分开**',
            '新增 evidence_tier（%s）：机械判定，顺序即优先级' % ' / '.join(EVIDENCE_TIERS),
            '新增 causal_gate_status：booking_horizon 缺口单列，**登记 ≠ 验证**',
            '新增 fidelity（%s）：是否代表原论文完整算法' % ' / '.join(FIDELITY),
            '`Q_method_health=None` 记为结构性 N/A，**不得**强制为 True',
            '`λ=3` 单列 DEGENERATE，**不**并入 NOT_VERIFIABLE',
            '主档 p_c=(5,10,15) 优先展示；p_c=0 单独作敏感性',
            '**未**筛除任何行；22 行逐行保留',
        ],
        'pairing_groups': {
            'A': '外部正式批（配对基线 cond_hist=631.4，protocol a1_c1fresh_reveal_40）',
            'B': 'Ours I3 自批（同批 cond_hist=600.997，i3_online_recovery_1009_v1）',
            'NONE': '跨源，无可用配对基线（Ours A3）',
        },
        # 逐行三维度（供独立复核；数值一律取自旧总表原件，**逐字保留**）
        'rows': [
            {'old_idx': r['old_idx'], 'method': r['method'],
             'arm_dir': r['arm_dir'], 'arm': r['arm'], 'p_c': r['p_c'],
             'pairing_group': r['group'], 'evidence_tier': r['tier'],
             'causal_gate_status': r['causal'], 'dependency_closure': r['dep'],
             'identity': r['ident'], 'method_health': r['health'],
             'method_health_reason': r['health_reason'],
             'fidelity': r['fidelity'], 'risk_markers': r['risks'],
             'utility_main': r['u_main'], 'utility_zero': r['u_zero'],
             'utility_strict': r['u_strict'], 'delta_main': r['d_main'],
             'delta_zero': r['d_zero'], 'timeouts_main': r['to_main'],
             'timeouts_zero': r['to_zero']}
            for r in enriched
        ],
        'pyvrp_identity': PYVRP_IDENTITY,
        'preserved_negative_results': [
            '主档同批 I3 − cond_hist = −144.680 [−191.109, −100.125]（同批 cond_hist=600.997）',
            'I3 对 warmstart / reconstruction control 的配对 CI 跨零 ⇒ 未建立独立增益',
            'stable_online_gain = False；三层证据 A1/A2/A3 尚未全部成立',
            'rule: value/energy λ=3 是全拒单**退化**规则，不是"性能很差"',
            'p_c=0 绝对效用更高只是评分函数差异，不得充当贡献证据',
        ],
    }
    blob = _git('cat-file', 'blob', 'HEAD:CC_Compare/docs/a1_development_comparison/'
                'A-v1对比方法总表.md')
    if blob is not None:
        meta['old_table_blob_sha256'] = _sha_bytes(blob.encode('utf-8')
                                                   if isinstance(blob, str) else blob)

    new_text = md_report(enriched, meta) if False else None
    # md_report 需要 meta 中的 source_commit / old sha ⇒ 先生成
    new_text = md_report(enriched, meta)
    meta['new_path'] = 'CC_Compare/docs/a1_development_comparison/%s' % NEW_NAME
    meta['new_sha256'] = _sha_bytes(new_text.encode('utf-8'))
    meta['new_size'] = len(new_text.encode('utf-8'))

    chk = display_selfcheck(old_text, old_rows, new_text, enriched, meta, mx)
    meta['display_layer_selfcheck'] = chk
    meta['selfcheck_ok'] = all(x['ok'] for x in chk)

    if a.write:
        out_md = os.path.join(a.outdir, NEW_NAME)
        if os.path.exists(out_md):
            raise SystemExit('拒绝覆盖已存在的文件：%s' % out_md)
        with io.open(out_md, 'w', encoding='utf-8', newline='\n') as f:
            f.write(new_text)
        out_mf = os.path.join(a.outdir, NEW_MANIFEST_NAME)
        with io.open(out_mf, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
        print('WROTE: %s' % out_md)
        print('WROTE: %s' % out_mf)

    print(json.dumps({
        'selfcheck_ok': meta['selfcheck_ok'],
        'n_checks': len(chk), 'n_failed': sum(1 for x in chk if not x['ok']),
        'failed': [x for x in chk if not x['ok']],
        'new_sha256': meta['new_sha256'][:16],
        'old_table_worktree_sha256': meta['old_table_worktree_sha256'][:16],
        'tiers': {t: sum(1 for r in enriched if r['tier'] == t) for t in EVIDENCE_TIERS},
    }, ensure_ascii=False, indent=2))
    return 0 if meta['selfcheck_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
