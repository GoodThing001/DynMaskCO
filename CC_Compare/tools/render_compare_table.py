"""A-v1 对比方法总表渲染器（Markdown + Excel）。

输入 = `collect_compare_table.py --json` 产出的数据集；输出：
  - `A-v1对比方法总表.md`（Markdown，主表 + 分解 + 未适配清单）
  - `A-v1对比方法总表.xlsx`（多 sheet，AEGIS-CD 风格）

用法：
    python CC_Compare/tools/render_compare_table.py --data <json> --outdir <dir>
"""
import argparse
import glob
import json
import os
import sys

HERE_R = os.path.dirname(os.path.abspath(__file__))
CC_ROOT = os.path.dirname(HERE_R)


def _gate_key(p):
    """gate 路径 → 稳定键 `a1_ext_formal2/mapt_0` 形式（供与审计矩阵对齐）。

    与 `audit_qualification_matrix.py` 用**同一规则**（取 `gate.json` 所在目录的最后两级），
    否则两边对不上、资格列会静默变 `—`。
    """
    parts = str(p or '').replace('\\', '/').rstrip('/').split('/')
    if parts and parts[-1] == 'gate.json':
        parts = parts[:-1]
    return '/'.join(parts[-2:])


def load_audit_matrix():
    """读 P0 独立资格矩阵（若存在）→ {(gate_key, p_c): row}。

    **单一事实来源**：资格列一律来自这份只读审计产物，不在渲染器里重算，
    也不采信运行器自报的 `formal_adjudication`。矩阵缺失时资格列渲染为 `—`
    （宁可显示"未知"，也不假装合格）。
    """
    cands = sorted(glob.glob(os.path.join(CC_ROOT, 'audit',
                                          'BASELINE_QUALIFICATION_MATRIX_*.json')))
    if not cands:
        return {}, None
    try:
        d = json.load(open(cands[-1], encoding='utf-8'))
    except Exception:                                        # noqa: BLE001
        return {}, None
    out = {}
    for r in d.get('arms') or []:
        # ⚠️ 只索引**方法臂 / Ours 行**：矩阵里同一个 gate 还含 3 个内部基线臂
        # （cond_hist / uncond_hist / explicit_feat）。若按 (gate, p_c) 无差别索引，
        # 后写入的基线行会覆盖方法行 ⇒ 表里会出现"方法臂 Q_core=否"的**假失败**
        # （2026-10-14 实测踩到：SGBS/I3 都被渲染成 DIAGNOSTIC）。
        if r.get('role') not in ('method', 'ours'):
            continue
        out[(_gate_key(r.get('gate_path')), r.get('p_c'))] = r
    return out, os.path.basename(cands[-1])

MAIN_ARM_DIRS = ('cada_0_f', 'cada_51015_f')
BASELINE_ORDER = ['cond_hist', 'uncond_hist', 'explicit_feat']
# 进 §0/§3 的主模型侧臂。
# 2026-10-13 增补 `a1_rollout`：它来自 `results/a1_a1_formal_40_1004_171907`（实测
# `seeds.gate=20260926`、`source_stable=True`、contract=e25a12104b20c3ea 即新基线同源），
# p_c=0 util=**1312.6**、**0 超时/天** —— 比任何外部方法都高，此前却因**目录名不含
# `20260926`** 被整 gate 静默丢掉（见 `_is_dev_gate`）。属于「信息不该丢」的那一类。
# **有意不加 `a1_consensus`**：同 gate 实测 **42.35–46.40 次超时/天**（远超 10 s 决策预算），
# 按 README §9 该 gate 数字只能算受污染诊断值，不能进 Ours 行；它仍会出现在 §0 顶部的
# 「未纳入」告警表里（带超时列），因此不是静默排除。
MAIN_MODEL_ARMS = ('maskco', 'warmstart', 'reconstruction_control', 'a1_rollout',
                   'cond_hist', 'explicit_feat', 'uncond_hist')

# ⚠️ **控制点只有一个**（2026-10-14 重构后）：
#   - `MAIN_MODEL_ARMS`    → 控制**哪些臂进 §0 Ours 摘要 / §3 主模型与内部臂 / Excel
#                            「主模型与内部臂」sheet**（过滤函数 `filter_main_model`）；
#   - `OURS_TABLE_SPECS`（见下方）→ 控制**交付主表（§1 / Excel 性能总表）里出现哪 1–2 行 Ours**。
# 旧版还有第三个开关 `OURS_BLOCK_ARMS`（把主模型记录铺成「表尾 Ours 块」）——
# 那正是用户 2026-10-14 指出的问题（30 行堆在表尾、不像「一个最终结果」），**已删除**。
# 现在主表里 Ours 与基线**同表**：仅 A3 / I3 两行，档位并成列。


DEV_SEED = 20260926
DEV_SEED_STR = str(DEV_SEED)
# 结论口径档位（2026-10-14 用户要求：正式论文主结论以主档为准）
TIER_MAIN = 'p_c=(5,10,15)'

# 交付主表里允许出现的 Ours 行（**只有这两个配置**）：
#   (显示名, 用于在 main_model 里筛选 gate 的子串, 是否钉在冻结开发集那一次运行上)
# 其余主模型记录一律进「主模型与内部臂」附录 sheet。2026-10-14 用户明确要求：
# 「我们的模型应该有**一个**最终结果，而不是那么多行」。
OURS_TABLE_SPECS = (
    ('Ours: MaskCO A3（掩码重构 · 预训练权重）', 'a1_a3_online_density_', True),
    ('Ours: MaskCO I3（决策感知）', 'i3_online_recovery', False),
)


def _is_dev_gate(r):
    """该 main_model 条目是否属于 20260926 开发集。

    2026-10-13 修复（实测踩到）：原先只判**目录名**含 `20260926`：
        if '20260926' not in g and 'online_20260926' not in g: continue
    于是 `results/a1_a1_formal_40_1004_171907/gate.json` 被**整个 gate 静默丢掉** ——
    尽管它文件里写着 `seeds={'train':20260925,'gate':20260926,...}`、`source_stable=True`、
    contract 正是新基线的 `e25a12104b20c3ea`，臂里还有 `a1_rollout`（p_c=0 util=1312.6、
    **0 超时**，比任何外部方法都高）。被丢的原因是**目录命名不含该 seed**，与实质无关。
    现在优先用采集器记录的 `dev_seed`；旧 dataset 无该字段时回退到目录名判定。
    """
    ds = r.get('dev_seed')
    if ds is not None:
        return int(ds) == DEV_SEED
    g = r['gate']
    return ('20260926' in g) or ('online_20260926' in g)


def filter_main_model(data):
    """主模型节：只保 20260926 开发日 gate（正式开发集），并只保留主模型/基线臂。"""
    out = []
    for r in data.get('main_model', []):
        if not _is_dev_gate(r):
            continue
        if r['arm'] not in MAIN_MODEL_ARMS:
            continue
        out.append(r)
    order = {'maskco': 0, 'warmstart': 1, 'reconstruction_control': 2, 'a1_rollout': 3,
             'cond_hist': 4, 'explicit_feat': 5, 'uncond_hist': 6}
    out.sort(key=lambda r: (r['gate'], r['p_c'], order.get(r['arm'], 9)))
    return out


def fnum(x, nd=2):
    if x is None:
        return '—'
    return ('%.' + str(nd) + 'f') % x


def fci(d, nd=1):
    if not d or d.get('mean') is None:
        return '—'
    fmt = '%%.%df [%%.%df, %%.%df]' % (nd, nd, nd)
    return fmt % (d['mean'], d['ci_lo'], d['ci_hi'])


def fpc(x):
    if x is None:
        return '—'
    return '%.1f%%' % (100.0 * x)


def pick_arm(data, arm_dir, p_c):
    for r in data['arms']:
        if r['arm_dir'] == arm_dir and r['p_c'] == p_c:
            return r
    return None


def pct_fallback(rec):
    sm = rec.get('solver_meta') or {}
    n, f = sm.get('n_solves'), sm.get('n_fail')
    if n:
        return float(f or 0) / float(n)
    return None


def build_ours_rows(data):
    """【已废弃，2026-10-14】旧版把主模型记录铺成「表尾 Ours 块」，现由 `canonical_ours` 取代。

    保留函数体只为让历史调用点报错时看得懂来源；**新代码不要用**：
    它会为**每一条** (gate, 档位) 记录产出一行（3 配置 × 3 次运行 × 档位 = 27 行 + 内部三臂），
    正是用户反馈的「我们模型怎么有那么多行」。
    """
    raise RuntimeError('build_ours_rows 已废弃：请用 canonical_ours / delivered_rows '
                       '（2026-10-14 起交付主表只出 1–2 行 Ours）')


def _row_from_records(recs, label, cls, native, venue='本工作'):
    """把**同一个 gate 的多个档位记录**压成**一行**（档位作列）。

    这是「每个方法一行」的关键：主模型的 gate 里 `p_c=0` / 主档 / `p_c=(10,20,30)`
    是三条独立记录，必须合并成一行，否则就会出现「一个配置铺三行、每行只填一档」的空洞。
    """
    by_pc = {}
    for r in recs:
        by_pc.setdefault(r['p_c'], r)

    def U(pc):
        v = ((by_pc.get(pc) or {}).get('reported_utility') or {})
        return v if v.get('mean') is not None else None

    def D(pc):
        rec = by_pc.get(pc) or {}
        d = (rec.get('main_comparison') or {}).get('%s_minus_cond_hist' % rec.get('arm'))
        return d if (d or {}).get('mean') is not None else None

    def R(pc, k):
        return ((by_pc.get(pc) or {}).get('reported') or {}).get(k)

    def M(pc):
        return ((by_pc.get(pc) or {}).get('solver_meta') or {}).get('max_decide_s')

    gate = str((recs[0] or {}).get('gate') or '')
    return dict(
        key=label, label=label, cls=cls, venue=venue,
        gate=gate, arm=(recs[0] or {}).get('arm'),
        # 交付 Ours 行的**资格事实**（2026-10-14 独立复核发现其必须随表披露）：
        # 主模型 gate 自身也有 mean_timeouts 与 _pairing_identity 裁决。
        pairing_identity=(recs[0] or {}).get('pairing_identity') or {},
        source_stable=(recs[0] or {}).get('identity_source_stable'),
        backbone=gate.replace('results/', '').split('/')[0] or '—',
        native=native, note='gate `%s`' % gate,
        ckpt_mb=None, params_m=None,
        util0=U('p_c=0'), utilm=U('p_c=(5,10,15)'), util3=U('p_c=(10,20,30)'),
        diff0=D('p_c=0'), diffm=D('p_c=(5,10,15)'), diff3=D('p_c=(10,20,30)'),
        hfr0=R('p_c=0', 'hard_feasible_rate'), hfrm=R('p_c=(5,10,15)', 'hard_feasible_rate'),
        fb0=None, fbm=None,
        to0=R('p_c=0', 'mean_timeouts'), tom=R('p_c=(5,10,15)', 'mean_timeouts'),
        md0=M('p_c=0'), mdm=M('p_c=(5,10,15)'),
        served0=R('p_c=0', 'mean_served'), servedm=R('p_c=(5,10,15)', 'mean_served'),
        energy0=R('p_c=0', 'mean_energy_kwh'),
        is_ours=True,
    )


def canonical_ours(data):
    """交付主表里**只留两行 Ours**：A3（掩码重构 · 预训练权重）与 I3（决策感知）。

    2026-10-14 用户明确要求后重构。此前 `build_ours_rows()` 把**每一条主模型 gate 记录**
    都渲染成一行 ⇒ 3 个配置 × 3 次重复运行 × 3 个档位 = 27 行，再加内部三臂 = **30 行**
    铺在表尾。「我们的模型有那么多行」不是因为有 30 个结果，而是**渲染粒度错了**：
    那是消融配置与重复运行，不是「最终结果」。

    ⇒ 现在：主表只出**最终结果**（两个并列配置各一行、档位并成列），
    其余记录进「主模型与内部臂」附录 sheet（见 `md_main_model` 与 Excel 同名 sheet）。
    """
    mm = [r for r in data.get('main_model', [])
          if _is_dev_gate(r) and r.get('arm') == 'maskco']
    out = []
    for label, sub, pin_dev in OURS_TABLE_SPECS:
        cand = [r for r in mm if sub in str(r.get('gate'))]
        if pin_dev and cand:
            # 钉在**冻结开发集**（seed 20260926）那一次运行上：同名配置在 0930/10001 还有
            # 重复运行（数值略不同，如 B 组 1582.0 vs 1577.9）。交付数字必须来自冻结开发集，
            # 否则同一次交付里会混进不同日期的运行。
            pinned = [r for r in cand if DEV_SEED_STR in str(r.get('gate'))]
            cand = pinned or cand
        if not cand:
            continue
        gates = {}
        for r in cand:
            gates.setdefault(str(r['gate']), []).append(r)

        def score(recs):
            u0 = ((next((x for x in recs if x['p_c'] == 'p_c=0'), None) or {})
                  .get('reported_utility') or {}).get('mean')
            um = ((next((x for x in recs if x['p_c'] == 'p_c=(5,10,15)'), None) or {})
                  .get('reported_utility') or {}).get('mean')
            return (u0 if u0 is not None else -1e18,
                    um if um is not None else -1e18)

        # A3 组按 p_c=0 选最强配置（B/C/D 三个训练配置里选 B）；
        # I3 组只有一个 gate，按主档选即可。
        best = max(gates, key=lambda g: score(gates[g])[0] if pin_dev else score(gates[g])[1])
        out.append(_row_from_records(gates[best], label, '主模型/内部臂',
                                     '接/拒 + 掩码重构'))
    return out


def canonical_c1(data):
    """C1 基线（cond_hist）也占**同一张主表的一行**，让表自洽可读（不必去翻别处找基线）。"""
    b = data.get('baseline_arms') or {}

    def U(pc):
        v = ((b.get('%s|cond_hist' % pc) or {}).get('utility') or {})
        return v if v.get('mean') is not None else None

    if U('p_c=0') is None and U('p_c=(5,10,15)') is None:
        return None
    base = str((data.get('generated_from') or {}).get('baseline') or '')
    zero = {'mean': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0}
    return dict(
        key='cond_hist', label='C1 cond_hist（SAA 强基线）', cls='内部基线',
        venue='本工作', backbone=os.path.basename(os.path.dirname(base.replace('\\', '/'))) or '—',
        native='场景采样 SAA', note='Δ 相对自身 ⇒ 0',
        ckpt_mb=None, params_m=None,
        util0=U('p_c=0'), utilm=U('p_c=(5,10,15)'), util3=U('p_c=(10,20,30)'),
        diff0=zero, diffm=zero, diff3=None,
        hfr0=None, hfrm=None, fb0=None, fbm=None, to0=None, tom=None,
        md0=None, mdm=None, served0=None, servedm=None, energy0=None,
        is_baseline=True,
    )


def delivered_rows(data):
    """交付主表的**唯一**行序列：C1 基线 + 19 个对比方法 + 恰好 2 行 Ours，全部同表按 p_c=0 排序。

    ⇒ 主模型与基线**在同一张表里**（用户要求），而不是单独成块铺在表尾；
    ⇒ 每个方法恰好一行（档位作列），大幅消除空格。
    """
    out = []
    c1 = canonical_c1(data)
    if c1:
        out.append(c1)
    methods = build_rows(data)
    ours = canonical_ours(data)

    def u0(r):
        return (r.get('util0') or {}).get('mean')

    methods.sort(key=lambda r: -(u0(r) if u0(r) is not None else -1e18))
    # Ours 行内部：有 p_c=0 的排在前（A3），没有的（I3 只有主档）紧随其后。
    ours.sort(key=lambda r: -(u0(r) if u0(r) is not None else -1e18))
    out.extend(methods)

    # 把**整块 Ours** 插到「首行 Ours 按其 p_c=0 应在的位置」，而不是让它散落在表尾：
    # 之前 I3 因为**没有 p_c=0 档**被排到全表最后一行，看起来像被孤立 —— 那正是用户抱怨的
    # 「Ours 不和基线放在一起」。现在 A3 紧跟在 C1 之下，I3 紧贴 A3。
    anchor = u0(ours[0]) if ours else None
    idx = len(out)
    if anchor is not None:
        idx = sum(1 for r in out if u0(r) is not None and u0(r) > anchor)
    ordered = out[:idx] + ours + out[idx:]
    audit, audit_name = load_audit_matrix()
    for r in ordered:
        annotate_qualification(r, audit, audit_name)
    return ordered


def annotate_qualification(r, audit, audit_name=None):
    """给每一行贴上**资格分层**字段（2026-10-14 用户要求：资格必须写在行上，不能只藏脚注）。

    字段语义（`comparison_status` 的构词法）：
        <资格层> / <相对 C1 的方向>
    资格层 ∈ {QUALIFIED, QUALIFIED-TIMEOUT, DEGENERATE, DIAGNOSTIC, UNKNOWN}
    方向   ∈ {NEGATIVE, ...}（本表中全部对比方法都低于 C1 基线；Ours I3 亦为 NEGATIVE）

    ⚠️ 取**主档** `p_c=(5,10,15)` 优先（正式论文的主结论口径），无主档才回落到 `p_c=0`。
    """
    r['protocol_id'] = '—'
    r['Q_core'] = None
    r['Q_dep_closure'] = None
    r['paired_source_identity'] = '—'
    r['comparison_status'] = '—'
    r['audit_source'] = audit_name
    if not audit:
        return r

    src = r.get('gate') or ''
    if r.get('a0') is not None:
        src = (r['a0'] or {}).get('gate_path') or src
    elif r.get('am') is not None:
        src = (r['am'] or {}).get('gate_path') or src
    gk = _gate_key(src)
    hit = audit.get((gk, 'p_c=(5,10,15)')) or audit.get((gk, 'p_c=0'))
    if not hit:
        return r

    q = hit.get('Q') or {}
    core = q.get('Q_formal_core')
    r['Q_core'] = core
    # 独立字段（2026-10-14 用户要求）：`Q_core=True` **不能**读成"复现身份已闭合"
    r['Q_dep_closure'] = hit.get('Q_dependency_closure')
    ident = hit.get('identity_verified')
    probs = hit.get('identity_problems') or []
    if q.get('Q_identity') is None:
        r['paired_source_identity'] = '不适用（纯规则臂）'
    elif ident is True:
        r['paired_source_identity'] = '同源'
    elif ident is False:
        r['paired_source_identity'] = '跨源（%s）' % ('、'.join(probs) or 'identity_failed')
    else:
        r['paired_source_identity'] = '—'

    if r.get('is_ours'):
        r['protocol_id'] = gk.replace('results/', '')
    else:
        _bf = ''
        for _rec in (r.get('a0'), r.get('am')):
            if _rec and _rec.get('baseline_from'):
                _bf = str(_rec['baseline_from'])
                break
        r['protocol_id'] = (os.path.basename(os.path.dirname(_bf.replace('\\', '/')))
                            if _bf else '—')

    to = r.get('tom') if r.get('tom') is not None else r.get('to0')
    nd = (hit.get('independent') or {}).get('sum_n_served')
    if nd == 0:
        layer = 'DEGENERATE'
    elif core is True:
        layer = 'QUALIFIED-TIMEOUT' if (to or 0) > 0 else 'QUALIFIED'
    elif core is False:
        # 身份未过（跨源）⇒ 诊断层；用户指定写法见 goal (B)
        layer = 'DIAGNOSTIC' if ident is False else 'DIAGNOSTIC'
    else:
        layer = 'UNKNOWN'
    if r.get('is_ours') and ident is False:
        r['comparison_status'] = 'DIAGNOSTIC / CROSS-SOURCE'
    elif r.get('is_ours'):
        r['comparison_status'] = '%s / NEGATIVE' % (
            'QUALIFIED DEVELOPMENT' if core is True else layer)
    else:
        r['comparison_status'] = '%s / NEGATIVE' % layer
    return r



def build_rows(data):
    """每个方法一行：{label, cls, ..., arm0, armmain}（只用终版臂）"""
    by_method = {}
    for r in data['arms']:
        if not r.get('formal'):
            continue
        by_method.setdefault(r['method'], {}).setdefault(r['p_c'], []).append(r)

    def decide(rec):
        sm = (rec or {}).get('solver_meta') or {}
        return sm.get('max_decide_s')

    rows = []
    GROUP_RANK = {'a1_ext_formal2': 0, 'a1_ext_a05e': 1, 'a1_ext_extra': 2}
    for key, per in sorted(by_method.items()):
        def choose(pc):
            cands = per.get(pc) or []
            if not cands:
                return None
            return sorted(cands, key=lambda c: (GROUP_RANK.get(c['group'], 9),
                                                not c.get('formal')))[0]
        a0, am, a3 = choose('p_c=0'), choose('p_c=(5,10,15)'), choose('p_c=(10,20,30)')
        base = a0 or am or a3
        meta = (base or {}).get('meta') or {}
        label = meta.get('label') or (base or {}).get('arm_dir') or key

        def diff(rec, ban='cond_hist'):
            if rec is None:
                return None
            return (rec.get('repaired_vs_baseline') or {}).get(ban)

        rows.append(dict(
            key=key, label=label, cls=meta.get('cls') or '—',
            venue=meta.get('venue') or '—', backbone=meta.get('backbone') or '—',
            native=meta.get('native') or '—', note=meta.get('note') or '',
            ckpt_mb=meta.get('ckpt_mb'),
            params_m=meta.get('params_m'),
            util0=(a0 or {}).get('reported', {}).get('utility') if a0 else None,
            utilm=(am or {}).get('reported', {}).get('utility') if am else None,
            util3=(a3 or {}).get('reported', {}).get('utility') if a3 else None,
            diff0=diff(a0), diffm=diff(am), diff3=diff(a3),
            hfr0=(a0 or {}).get('reported', {}).get('hard_feasible_rate') if a0 else None,
            hfrm=(am or {}).get('reported', {}).get('hard_feasible_rate') if am else None,
            fb0=pct_fallback(a0) if a0 else None,
            fbm=pct_fallback(am) if am else None,
            to0=(a0 or {}).get('reported', {}).get('mean_timeouts') if a0 else None,
            tom=(am or {}).get('reported', {}).get('mean_timeouts') if am else None,
            md0=decide(a0), mdm=decide(am),
            a0=a0, am=am,
            served0=(a0 or {}).get('reported', {}).get('mean_served') if a0 else None,
            servedm=(am or {}).get('reported', {}).get('mean_served') if am else None,
            rej0=(a0 or {}).get('reported', {}).get('mean_rejected') if a0 else None,
            energy0=(a0 or {}).get('reported', {}).get('mean_energy_kwh') if a0 else None,
            revenue0=(a0 or {}).get('reported', {}).get('revenue') if a0 else None,
            fuel0=(a0 or {}).get('reported', {}).get('fuel_cost') if a0 else None,
            rejectloss0=(a0 or {}).get('reported', {}).get('reject_loss') if a0 else None,
        ))
    rows.sort(key=lambda r: -(r['util0']['mean'] if r['util0'] else -1e18))
    return rows


def md_main_table(data, rows):
    """§1 主表：**C1 基线 + 19 个对比方法 + 2 行 Ours，同一张表**（按 p_c=0 降序）。

    2026-10-14 重构（用户反馈三点：Ours 不该单独成块铺 30 行、主模型该只有一个结果、
    空格太多）：列也一并精简 —— 去掉填充率极低的 `fallback%` / `最大单决策(s)` /
    `权重(MB)`，只留每次比对都要读的字段。
    """
    L = []
    L.append('| # | 方法 | 比较状态 | Q_core | 依赖闭包 | 配对来源身份 | protocol_id | 类别 | 期刊/会议 | '
             'Backbone / 权重 | 参数量(M) | 原生约束 | '
             'Utility p_c=0 ↑ | Utility 主档 p_c=(5,10,15) | Utility p_c=(10,20,30) | '
             'Δ vs cond_hist (p_c=0) | Δ vs cond_hist (主档) | '
             '超时/天 p_c=0 | 超时/天 主档 | 硬可行率 p_c=0 | 服务单/天 p_c=0 | 能耗 kWh/天 p_c=0 | 备注 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for i, r in enumerate(rows, 1):
        tag = '★ ' if r.get('is_ours') else ('◆ ' if r.get('is_baseline') else '')
        _qc = r.get('Q_core')
        _dc = r.get('Q_dep_closure')
        L.append('| %d | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            i, tag, r['label'],
            r.get('comparison_status') or '—',
            '—' if _qc is None else ('是' if _qc else '**否**'),
            '—' if _dc is None else ('是' if _dc else '**否**'),
            r.get('paired_source_identity') or '—',
            r.get('protocol_id') or '—',
            r['cls'], r['venue'], r['backbone'],
            fnum(r.get('params_m'), 3), r['native'],
            fci(r['util0']), fci(r['utilm']), fci(r['util3']),
            fci(r['diff0']), fci(r['diffm']),
            fnum(r['to0'], 2), fnum(r['tom'], 2),
            fnum(r['hfr0'], 3),
            fnum(r['served0'], 1), fnum(r['energy0'], 1),
            r.get('note') or ''))
    return '\n'.join(L)


def md_main_model(data):
    """附录：**全部**主模型 / 内部臂记录（含消融与重复运行）。

    ⚠️ 这里与 §1 主表的分工（2026-10-14 用户要求后明确）：§1 每个方法**只出一行**最终结果
    （主模型只出 A3 / I3 两行）；本附录保留其余记录，并在第一列标出哪些是交付行、
    哪些是**消融 / 重复运行 / 内部臂** —— 证据不丢，但不再混进主表。
    """
    delivered = {(r.get('gate'), r.get('arm')) for r in canonical_ours(data)}
    L = []
    L.append('| 交付行？ | gate | 档位 | 臂 | utility | 超时/天 | 身份块 | source_stable | future_policy | standby | workers | 配对身份 |')
    L.append('|---|---|---|---|---:|---:|---|---|---|---|---|---|')
    for r in filter_main_model(data):
        u = r.get('reported_utility') or {}
        sm = r.get('reported') or {}
        mark = '★ 是' if (r['gate'], r['arm']) in delivered else '否（消融/重复/内部臂）'
        L.append('| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            mark, r['gate'], r['p_c'], r['arm'],
            fci(u) if u.get('mean') is not None else '—',
            fnum(sm.get('mean_timeouts'), 2),
            '有' if r['identity_present'] else '无',
            r['identity_source_stable'],
            (r['config'] or {}).get('future_policy'),
            (r['config'] or {}).get('standby_orders'),
            (r['config'] or {}).get('workers'),
            (r['pairing_identity'] or {}).get('identity_verified', '—')))
    return '\n'.join(L)


def ours_same_batch_pair(data, rows=None):
    """交付 Ours 行的**同 gate 三元组**：本臂效用 / 同批 cond_hist 效用 / 配对差(+CI)。

    为什么必须有这个函数（2026-10-14 用户指出的真实错误）：
    我曾在正文里把 **I3 gate 内的配对差 −144.680** 与 **另一套协议**的 C1 基线值
    `a1_c1fresh_reveal_40|cond_hist` = **631.41** 并列 ⇒ 三个数**不同源**：
        456.317 − 600.997 = −144.680  ✅ 同批成立
        456.317 − 631.410 = −175.093  ✗ 与 −144.680 不符
    ⇒ 本函数**只从同一个 gate 记录**里取三个数，并在 `md_table_notes` / Excel 注里
    用恒等式自证；跨协议的数字一律另行标注来源。
    """
    ours = canonical_ours(data)
    if not ours:
        return None
    # ⚠️ `canonical_ours()` 的行**还没有** `Q_core`（那是 `annotate_qualification` 事后加的）
    # ⇒ 必须自己查审计矩阵，否则会退化成"永远选第一行"（2026-10-14 实测：错选了 A3）。
    audit, _ = load_audit_matrix()

    def qcore(r):
        hit = (audit.get((_gate_key(r.get('gate')), TIER_MAIN)) or
               audit.get((_gate_key(r.get('gate')), 'p_c=0')) or {})
        return (hit.get('Q') or {}).get('Q_formal_core')

    pick = (next((r for r in ours if qcore(r) is True), None)
            or next((r for r in ours if qcore(r) is not None), None)
            or ours[0])
    gate = pick.get('gate')
    recs = [r for r in (data.get('main_model') or []) if r.get('gate') == gate]

    def U(arm):
        """**只取主档** —— 结论口径是主档；漏掉这个过滤会抓到 p_c=(10,20,30) 的记录
        （2026-10-14 实测：一度把 −704.061 / −459.904 当成主档三元组）。"""
        for r in recs:
            if r.get('arm') == arm and r.get('p_c') == TIER_MAIN:
                u = (r.get('reported_utility') or {})
                if u.get('mean') is not None:
                    return u
        return None

    diff = None
    for r in recs:
        if r.get('p_c') != TIER_MAIN:
            continue
        mc = (r.get('main_comparison') or {}).get('%s_minus_cond_hist' % (r.get('arm') or ''))
        if mc and mc.get('mean') is not None:
            diff = mc
            break
    a, b = U('maskco'), U('cond_hist')
    ident = None
    if a and b and diff:
        ident = abs((a['mean'] - b['mean']) - diff['mean']) <= 1e-6
    return {'label': pick.get('label'), 'gate': gate, 'tier': TIER_MAIN,
            'arm_util': a, 'cond_util': b, 'diff': diff, 'identity_holds': ident}


def md_table_notes(rows, data=None):
    """主表下方的「读表注记」：**数据驱动**，只对表内真实出现的方法加注。

    为什么放在渲染器里而不是只写在文档里：这些限定（全拒单规则、弱介入、不可复现、
    共用权重）必须**长在表上**，否则「重出表」这一步很容易把脚注丢掉。
    刻意**不写任何数字/哈希**，避免与 `check_table_consistency.py` 的 MD↔Excel 数值比对混淆；
    引用出处统一指向 `读表注意与已知限定.md`。
    """
    labels = ' | '.join((r.get('label') or '') for r in rows)
    notes = []
    if 'value/energy λ=3' in labels:
        notes.append('- ※ `rule: value/energy λ=3`：该规则**全拒单**（选择不服务任何订单），'
                     '**不是「性能很差的规则」**。依据见 `读表注意与已知限定.md` §2.1。')
    if 'MAPT' in labels:
        notes.append('- ※ `MAPT`：**弱介入**——模型只决定排序**前 ~3 位**、其余尾部由 EDD 补全，'
                     '不得当作「MAPT 方法能力」的证据。见 `读表注意与已知限定.md` §2.2。')
    if ('DeepACO' in labels) or ('Learn-Improvement-Heuristics' in labels):
        notes.append('- ※ `LIH` / `DeepACO`：**跨次不可复现**（清缓存后两次调用不一致），'
                     '不得与确定性方法并列宣称「同 seed 逐位复现」。见 §2.3。')
    if ('POMO' in labels) or ('SGBS' in labels) or ('Omni-VRP' in labels):
        notes.append('- ※ `POMO` / `SGBS` / `Omni-VRP`：**共用同一权重文件**'
                     '（三者 checkpoint 的 sha256 完全相同），是同一 checkpoint 的三种'
                     '解码/搜索用法，**不是三个独立训练的模型**。见 §2.4。')
    notes.append('- ※ 引用任何数字时**必须同时报告** `mean_timeouts`、`hard_feasible_rate` 与'
                 '设备/解释器（同一方法在 cpu/cuda 下速度差异可达数量级，直接影响超时）。')
    # ---- 2026-10-14 独立审计新增的两条**关键限定**（见 CC_Compare/audit/）----
    # 1) p_c 不进入决策规则：源码实测 run_a1_external_accept.py:143-167 每天只跑一次 env.run(i)，
    #    再用同一组 acc/rej 对每个罚分档重新评分；OrderingAcceptReplanner.__init__ 根本没有 p_c 形参。
    notes.append('- ※ **主档列是「同一策略的惩罚重评分」，不是「策略对惩罚的响应」**：'
                 '接拒 replanner 的构造**不接收 `p_c`**（`run_a1_external_accept.py:127-130`、'
                 '`rrnco_accept_replanner.py:48-49`），故档间差异**不可能**来自策略调整，'
                 '只能来自评分函数。（本批各档是**分别运行**的两次，因此 `LIH`/`DeepACO` 这类'
                 '跨次不可复现的方法还可能混入运行噪声。）'
                 '若要真正的惩罚敏感性，需让决策规则使用 `p_c` 并逐档重跑（属新协议，不得追认）。'
                 '依据：`CC_Compare/audit/INDEPENDENT_READONLY_ADJUDICATION_2026-10-14.md` §二-问题 2。')
    # 3) **Ours 行自身的资格事实**（2026-10-14 独立复核新增，**数据驱动**、每次渲染自证）：
    #    主模型 gate 也有 mean_timeouts 与 _pairing_identity；此前表里只给外部臂打 †，
    #    会让读者以为 Ours 行干净 —— 实测恰恰相反。
    _ours = [r for r in rows if r.get('is_ours')]
    if _ours:
        _bits = []
        for r in _ours:
            _pi = (r.get('pairing_identity') or {})
            _pb = ('，问题=%s' % '、'.join(_pi.get('problems') or [])) if _pi.get('problems') else ''
            _bits.append('**%s**（超时/天 = %s，配对身份 = %s%s）'
                         % (r['label'],
                            fnum(r.get('to0') if r.get('to0') is not None else r.get('tom'), 2),
                            _pi.get('identity_verified'), _pb))
        notes.append('- ※ **Ours 行同样受 CPU 争用影响，且其中一行配对身份未通过 —— '
                     '引用主模型结果时必须与外部臂一样带限定**：' + '；'.join(_bits)
                     + '。作为对比，对比方法臂的正式批次中**除上表 † 标出的 6 个之外'
                     '全部 `超时/天 = 0.0`**。（注意「方法数」与「臂数」不同：'
                     '方法按行计，臂按 (方法 × 档位) 计。）'
                     '依据：`CC_Compare/audit/BASELINE_QUALIFICATION_MATRIX_2026-10-14.json`。')
    # 5) **结论口径**：正式论文的主结论以**主档**为准（2026-10-14 用户要求）
    #    三元组**从同一 gate 现算**，避免再次出现跨批次混用（见 ours_same_batch_pair 注释）。
    _p = ours_same_batch_pair(data, rows) if data else None
    if _p and _p.get('arm_util') and _p.get('cond_util') and _p.get('diff'):
        _d = _p['diff']
        notes.append('- ※ **结论口径：正式论文的主结论以主档 `p_c=(5,10,15)` 为准**；'
                     '`p_c=0` 是**敏感性结果**，不得因为它的绝对效用更高就充当头条贡献证据。'
                     '本表按 p_c=0 排序仅为呈现习惯，不代表主结论口径。'
                     '**主档同批三元组**（同一 gate `%s`）：%s = **%.3f**、同批 `cond_hist` = **%.3f**、'
                     '配对差 = **%+.3f** [%.3f, %.3f]，恒等式校验 **%s**。'
                     '⇒ **主档上主模型未超过 C1 强基线**。'
                     '⚠️ 解释该配对差时**必须用同批的 %.3f**；`631.4` 属**另一套协议**'
                     '（外部臂配对基线 `a1_c1fresh_reveal_40|cond_hist`），**不得**与本行并列。'
                     % (_p['gate'], _p['label'] or 'Ours', _p['arm_util']['mean'],
                        _p['cond_util']['mean'], _d['mean'], _d['ci_lo'], _d['ci_hi'],
                        '通过' if _p.get('identity_holds') else '**不通过**',
                        _p['cond_util']['mean']))
    # 4) future_policy 不同 ⇒ 本表不是同协议方法归因
    notes.append('- ※ **本表各行的 `future_policy` 并不一致**：Ours 两行取自 `density` 配置'
                 '（gate 名含 `density` / `i3_online_recovery`），而 19 个对比方法的配对基线由'
                 '`--future-policy reveal` 生成（`scenario_saa.py:268` 的默认值）。'
                 '**A2 证据已表明该杠杆会显著改变 `cond_hist−uncond_hist` 的相对差** ⇒ '
                 '本表的效用排序只可读作**开发集能力对照**，**不构成**同协议下的方法归因。'
                 '依据：同上 §二-问题 7。')
    return '\n\n' + '\n'.join(notes)


def unlisted_main_model_arms(data):
    """20260926 开发集里出现、但**没被渲染进 Ours/内部臂节**的主模型臂，附原因。

    两类沉默失败（都在 2026-10-13 排查中实测到）：
      A. 臂名不在 `MAIN_MODEL_ARMS`（`filter_main_model` 第 28 行静默剔除）——
         用户改了主模型、新臂换名（`policy_v2` 之类）时就会发生；
      B. gate 被开发集判定挡掉（原先按**目录名**判，见 `_is_dev_gate` 的说明）。
    这里把两类都收集出来并**附实测数字与超时**，渲染进 §0 顶部：既不再静默，
    也不至于无条件塞进 Ours 行（有些臂是超时污染的，例如 `a1_consensus` 实测 42–46 次超时/天，
    按 README §9 只能作诊断、不能当结果）。
    """
    out = {}
    for r in data.get('main_model', []):
        if not _is_dev_gate(r):
            continue
        if r['arm'] in MAIN_MODEL_ARMS:
            continue
        u = (r.get('reported_utility') or {}).get('mean')
        to = (r.get('reported') or {}).get('mean_timeouts')
        out.setdefault(r['arm'], []).append((r['p_c'], u, to, r['gate']))
    return out


def md_ours_summary(data, rows):
    """Ours 摘要：主模型各臂 vs 最强外部方法 vs C1 基线（同 40 天开发集）。"""
    mm = filter_main_model(data)
    L = ['| 档位 | 行 | 臂 / 方法 | utility | 相对 C1 cond_hist | 来源 gate |',
         '|---|---|---|---:|---:|---|']
    # 先给「未纳入 Ours 块的主模型臂」告警（若有）—— 这是「用户改了主模型但新臂没进表」
    # 这一沉默失败模式的唯一可见出口，因此放在 §0 最前面，并**附上数字与超时**，
    # 让读者能自己判断「这是有意的超时污染排除」还是「新臂忘了登记」。
    _un = unlisted_main_model_arms(data)
    if _un:
        _lines = ['> ⚠️ **以下主模型臂未纳入本节的 Ours 块**（`filter_main_model` 只认 '
                  '`MAIN_MODEL_ARMS`，未登记即静默剔除）。附实测数字与 `超时/天`，请自行判断是'
                  '「有意排除」还是「新臂忘了登记」：', '>',
                  '> | 臂 | 档位 | utility | 超时/天 | 来源 gate |', '> |---|---|---:|---:|---|']
        for arm in sorted(_un):
            for (pc, u, to, g) in sorted(_un[arm], key=lambda x: x[0]):
                _flag = ' ← **超时污染，只能作诊断**' if (to or 0) > 0 else ''
                _lines.append('> | `%s` | %s | %s | %s%s | `%s` |' % (
                    arm, pc, fnum(u, 1), fnum(to, 2), _flag, g))
        _lines += ['>',
                   '> 若这是你**新改的**主模型臂，请把臂名加入 `CC_Compare/tools/render_compare_table.py`',
                   '> 的 `MAIN_MODEL_ARMS`（必要时也加进其下方 `order` 字典控制排序）后重渲染 ——',
                   '> 否则总表会**只显示旧臂**，看起来像「改进没生效」。若属有意排除（如上表超时污染），',
                   '> 忽略本告警即可。', '']
        L = _lines + L
    tiers = ['p_c=0', 'p_c=(5,10,15)']
    for tier in tiers:
        sub = [r for r in mm if r['p_c'] == tier]
        base = next((r for r in sub if r['arm'] == 'cond_hist'), None)
        bu = ((base or {}).get('reported_utility') or {}).get('mean')
        for r in sub:
            if r['arm'] in ('explicit_feat', 'uncond_hist'):
                continue
            u = (r.get('reported_utility') or {}).get('mean')
            star = '★ ' if r['arm'] == 'maskco' else ''
            L.append('| %s | 主模型/内部臂 | %s%s | %s | %s | `%s` |' % (
                tier, star, r['arm'], fnum(u, 1),
                fnum(u - bu, 1) if (u is not None and bu is not None) else '—',
                r['gate']))
        ext = [x for x in rows if x['util0'] if True]
        best = None
        for x in rows:
            u = (x['util0'] if tier == 'p_c=0' else x['utilm']) or {}
            if u.get('mean') is None:
                continue
            if best is None or u['mean'] > best[1]:
                best = (x['label'], u['mean'])
        if best:
            L.append('| %s | 最强外部方法 | %s | %s | — | 本表 §1 |' % (
                tier, best[0], fnum(best[1], 1)))
    return '\n'.join(L)


def md_baseline_table(data):
    L = []
    L.append('| 基线臂 | 档位 | Utility | 服务单/天 | 拒绝单/天 | 收入/天 | 燃油/天 | 拒绝损失/天 | 能耗 kWh/天 |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for k, v in sorted(data['baseline_arms'].items()):
        pname, an = k.split('|')
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            an, pname, fci(v['utility']), fnum(v['mean_served'], 1),
            fnum(v['mean_rejected'], 1),
            fnum((v['revenue'] or {}).get('mean'), 1),
            fnum((v['fuel_cost'] or {}).get('mean'), 2),
            fnum((v['reject_loss'] or {}).get('mean'), 1),
            fnum(v['mean_energy_kwh'], 1)))
    return '\n'.join(L)


def md_arm_detail(data):
    L = []
    L.append('| 臂目录 | 方法 | 档位 | 正式/诊断 | 天数 | 可裁决 | 身份块 | source_stable | 记录配对身份 | gate 路径 |')
    L.append('|---|---|---|---|---:|---|---|---|---|---|')
    for r in sorted(data['arms'], key=lambda x: (x['method'], x['p_c'], x['arm_dir'])):
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | `%s` |' % (
            r['arm_dir'], r['arm'] or '—', r['p_c'],
            '正式' if r['formal'] else '诊断',
            (r.get('reported') or {}).get('n_days', '—'),
            (r.get('reported') or {}).get('adjudicable', '—'),
            '有' if r['identity_present'] else '无',
            r['identity_source_stable'],
            (r['pairing_identity_recorded'] or {}).get('identity_verified', '—'),
            r['gate_path']))
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--md-only', action='store_true',
                    help='只出 Markdown（服务器端无 openpyxl 时用）')
    args = ap.parse_args()

    data = json.load(open(args.data, encoding='utf-8'))
    rows = build_rows(data)                 # 只含 19 个对比方法（§0 的「最强外部」用它）
    delivered = delivered_rows(data)        # §1 主表与 Excel 性能总表用这张：基线 + 方法 + 2 行 Ours
    os.makedirs(args.outdir, exist_ok=True)

    gen = data['generated_from']
    head = [
        '# A-v1 动态冷链接单协议 · 对比方法总表',
        '',
        '> 生成器：`CC_Compare/tools/collect_compare_table.py` + `render_compare_table.py`。',
        '> 协议：动态 pickup-to-depot；揭示即接/拒不可撤销；同容量/车辆数/预算/温区/时间窗；',
        '> 决策时限 10s；**同一天 40 天开发集（seed %d）**；效用 = 收入 − 燃油 − 拒绝损失。' % gen['seed'],
        '> 统计 = 按天聚类 bootstrap（seed+777，2000 次重采样，2.5/97.5 分位），逐位复刻项目 `_stat`。',
        '> 基线 = `%s`（身份块：%s，source_stable=%s）。' % (
            os.path.basename(os.path.dirname(gen['baseline'])),
            '有' if gen['baseline_identity_present'] else '无',
            gen.get('baseline_source_stable')),
    ]
    # 正式裁决口径：**数据驱动**，不要写死。
    #   若主行全部来自 v2 正式批次 `a1_ext_formal2` 且落盘 `identity_verified=true`
    #   → 写「已达成正式裁决」；
    #   否则（=旧表状态：基线 c1rr2 自身 source_stable=False，只能事后重配对）
    #   → 保留「重配对证据、不等于正式裁决」的限定。
    # 这样表格无论渲染哪一批，说的都是当时的事实。
    _v2 = [r for r in data.get('arms', [])
           if r.get('formal') and r.get('group') == 'a1_ext_formal2']
    # 2026-10-13 修复（成因经源码核实）：**纯规则臂结构性没有配对身份块**。
    # `rule_myopic/valuee20/valuee30` 由 `run_a1_downstream_ablation.py` 产出，而该脚本
    # **从不 import / 调用 `pairing_identity_check`**（grep 已核：文件里 0 处出现），
    # 因此它们的 gate 里没有 `_pairing_identity`。原先判据写成「全部 formal2 臂都
    # identity_verified=true」⇒ **永远不可能成立** ⇒ ①本标题永远停在「重配对证据／正在补齐」
    # 分支，v2 批次花十几小时换来的「落盘正式裁决」永远不会显示；②`render_v2_summary.py`
    # 的同款判据会把总判定永久打成 FAIL。⇒ 按「配对身份要求」把臂分成两类，**并把排除写明**。
    _NO_PAIR_ID = {'myopic', 'valuee20', 'valuee30'}   # 纯规则臂：无权重、无身份块（by design）
    _v2_rule = [r for r in _v2 if r.get('method') in _NO_PAIR_ID]
    _v2_req = [r for r in _v2 if r.get('method') not in _NO_PAIR_ID]
    _v2_ok = [r for r in _v2_req
              if (r.get('pairing_identity_recorded') or {}).get('identity_verified') is True]
    if _v2_req and len(_v2_ok) == len(_v2_req):
        # 该臂落盘时实际配对的基线（v2 = 同源新基线 a1_c1fresh_reveal_40）
        # 注意：config.baseline_from 里是 Windows 反斜杠（`results\a1_c1fresh_reveal_40\gate.json`），
        # 而渲染器在 Windows（本地）与 Linux（服务端）都会跑 —— 先统一分隔符再取目录名，
        # 否则在 Linux 上 dirname 会把整串当文件名、基线上名渲染成空串。
        _bf = sorted({os.path.basename(os.path.dirname(str(r.get('baseline_from')).replace('\\', '/')))
                      for r in _v2 if r.get('baseline_from')})
        _bf_txt = ('、'.join('`%s`' % x for x in _bf)) if _bf else '（gate config 未记录）'
        _Δ_stable = bool(gen.get('baseline_source_stable'))
        head += [
            '> **正式裁决：已达成（臂侧落盘）**。主行取自 v2 正式批次 `a1_ext_formal2` 中**需要有配对身份的 '
            '%d 个臂**（全部 `identity_verified=true`），各臂在**自身 gate 内**以基线 %s'
            % (len(_v2_req), _bf_txt),
            '> 完成配对并落盘 `formal_adjudication=true`（3 档）—— 这是本批次相对上一版的关键升级。',
        ]
        if _v2_rule:
            head += [
                '> 另有 **%d 个纯规则臂**（%s）**不参与上述裁决声明**：它们没有学习权重，'
                % (len(_v2_rule), '、'.join(sorted({r['method'] for r in _v2_rule}))),
                '> 产出脚本 `run_a1_downstream_ablation.py` **从不调用** `pairing_identity_check`'
                '（源码已核：0 处出现），',
                '> 故其 gate 结构性没有 `_pairing_identity` 块 —— 这是设计如此，**不是缺陷**；'
                '它们的 Δ 列同样由本表以身份合格基线的 per-day 重配对给出。',
            ]
        if _Δ_stable:
            head += ['> 本表 Δ 列使用的基线 `%s` 自身 `source_stable=True`，'
                     '故 **Δ 列同属正式裁决**。'
                     % os.path.basename(os.path.dirname(gen['baseline']))]
        else:
            head += [
                '> ⚠️ **但本表的 Δ 列是另行重配对的，不等于上述裁决**：为与上一版逐位可比，Δ 列统一改用',
                '> `%s` 的 per-day 重算，而该基线在自身记录里 `source_stable=False`'
                % os.path.basename(os.path.dirname(gen['baseline'])),
                '> （见上一行「基线 = …」）。⇒ **Δ 列本身仍属「重配对证据」**，并因新基线更优',
                '> （超时更少）而系统性偏移：相对新基线的差比 Δ 列小约 2–7。引用 Δ 时必须一并说明；',
                '> 臂侧的 `formal_adjudication=true` 不受影响。详见 `v2_baseline_verdict.md`。',
            ]
    else:
        _tail = ('（v2 批次尚未纳入本表。）' if not _v2_req
                 else '（v2 批次正在补齐：%d/%d 个需配对的正式臂已合格。）'
                      % (len(_v2_ok), len(_v2_req)))
        head += [
            '> **配对口径说明**：正式臂落盘时把基线臂从旧 `a1_step2_gate_c1_20260926`（无身份块）导入，',
            '> 故落盘 `_pairing_identity.identity_verified=false`。本表改用身份合格的 `a1_c1rr2_reveal_40`',
            '> per-day 重算配对差（两者 contract/data/meta/budget/seed 逐项相同）。',
            '> **正式裁决限制**：按 `run_identity.pairing_identity_check`，配对要求双侧 `source_stable=True`',
            '> 且 `shared_source_sha256` 全等；`a1_c1rr2_reveal_40` 记录 `source_stable=False`，',
            '> 因此本表为**重配对证据**，不等于已获正式裁决。' + _tail,
        ]
    head += ['']
    md = '\n'.join(head)
    md += '\n## 0. Ours（主模型）摘要\n\n' + md_ours_summary(data, rows)
    md += '\n\n## 1. 主表（C1 基线 + 全部对比方法 + Ours，同一张表按 p_c=0 效用降序）\n\n' + md_main_table(data, delivered) + md_table_notes(delivered, data)
    md += '\n\n## 2. 身份合格的 C1 基线臂（配对参照）\n\n' + md_baseline_table(data)
    md += '\n\n## 3. 主模型 / 内部臂（**不同 future_policy / standby / 基线，不与上表同尺度比较**）\n\n' + md_main_model(data)
    md += '\n\n## 4. 臂明细与身份\n\n' + md_arm_detail(data)
    md += '\n\n## 5. 不可适配 / 未适配方法\n\n'
    md += '> 只列**机制层面**接不上的方法；其资产（权重/代码）本身是可加载的，缺的是可迁移到 A-v1\n'
    md += '> 「接单 + 车队路由」决策的语义。MAPT 与 CO-enriched-ML **不在此列**：二者已适配并产出\n'
    md += '> v2 正式臂（见 §1 主表；MAPT 的弱介入代价见 README §8）。\n\n'
    md += '| 方法 | 不可适配的具体原因 | 依据文档 |\n|---|---|---|\n'
    for x in data['incomparable_or_unadapted']:
        md += '| %s | %s | `%s` |\n' % (x['method'], x.get('reason', '—'), x['path'])

    mdp = os.path.join(args.outdir, 'A-v1对比方法总表.md')
    with open(mdp, 'w', encoding='utf-8') as f:
        f.write(md)
    print('wrote', mdp)

    if args.md_only:
        print('md-only: 跳过 Excel')
        return

    # ---------- Excel ----------
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    thin = Side(style='thin', color='BFBFBF')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    hdr_fill = PatternFill('solid', fgColor='1F4E79')
    ours_fill = PatternFill('solid', fgColor='FFF2CC')
    hdr_font = Font(color='FFFFFF', bold=True, size=10)
    wrap = Alignment(wrap_text=True, vertical='center')
    ctr = Alignment(horizontal='center', vertical='center')

    def style_header(ws, row=1, ncol=None):
        ncol = ncol or ws.max_column
        for c in range(1, ncol + 1):
            cell = ws.cell(row=row, column=c)
            cell.fill = hdr_fill
            cell.font = hdr_font
            cell.alignment = Alignment(wrap_text=True, vertical='center',
                                       horizontal='center')
            cell.border = border

    # Sheet 1
    ws = wb.active
    ws.title = '性能总表'
    cols = ['#', '方法', '比较状态', 'Q_core', '依赖闭包', '配对来源身份', 'protocol_id',
            '类别', '期刊/会议', 'Backbone/权重', '参数量(M)', '原生约束支持',
            'Utility p_c=0', 'CI下界(p_c=0)', 'CI上界(p_c=0)',
            'Utility 主档', 'CI下界(主档)', 'CI上界(主档)',
            'Utility p_c=10/20/30', 'CI下界(p=10/20/30)', 'CI上界(p=10/20/30)',
            'Δ vs cond p_c=0', 'ΔCI下界', 'ΔCI上界',
            'Δ vs cond 主档', 'Δ主档CI下界', 'Δ主档CI上界',
            '超时/天 p_c=0', '超时/天 主档', '硬可行率 p_c=0',
            '服务单/天 p_c=0', '能耗kWh/天 p_c=0', '备注']
    # 2026-10-14：按用户要求把**资格分层**提到行首（`比较状态 / Q_core / 配对来源身份 /
    # protocol_id`）—— 读者不必翻到表下脚注就能看到 Ours 行的资格缺口。取值来自
    # `CC_Compare/audit/BASELINE_QUALIFICATION_MATRIX_*.json`（单一事实来源）。
    # 2026-10-14 精简（用户反馈「空了那么多空没填」）：删掉填充率极低的
    # `权重(MB)`(27%) / `fallback%`(90% 但语义弱) / `最大单决策s`(31%)，
    # 并把三处重名的 `CI下界/CI上界` 改成带档位的名字 —— 重名会让读者不知道该列属于哪一档。
    # ⚠️ 四个**被 check_table_consistency.py 按表头名定位**的列必须原样保留：
    # `Utility p_c=0` / `Utility 主档` / `Utility p_c=10/20/30` / `Δ vs cond p_c=0`。
    ws.append(cols)
    # ⚠️ 这里必须用 `delivered`（= C1 基线 + 19 方法 + 2 行 Ours），**不是** `rows`（只有 19 个方法）。
    # 2026-10-14 重构：主模型不再是「表尾单独的 Ours 块」，而是与基线同表的一行。
    for i, r in enumerate(delivered, 1):
        u0, um, u3 = r['util0'] or {}, r['utilm'] or {}, r['util3'] or {}
        d0, dm = r['diff0'] or {}, r['diffm'] or {}
        tag = '★ ' if r.get('is_ours') else ('◆ ' if r.get('is_baseline') else '')
        _qc = r.get('Q_core')
        _dc = r.get('Q_dep_closure')
        ws.append([i, tag + r['label'],
                   r.get('comparison_status') or '—',
                   '—' if _qc is None else ('是' if _qc else '否'),
                   '—' if _dc is None else ('是' if _dc else '否'),
                   r.get('paired_source_identity') or '—',
                   r.get('protocol_id') or '—',
                   r['cls'], r['venue'], r['backbone'],
                   r.get('params_m'), r['native'],
                   u0.get('mean'), u0.get('ci_lo'), u0.get('ci_hi'),
                   um.get('mean'), um.get('ci_lo'), um.get('ci_hi'),
                   u3.get('mean'), u3.get('ci_lo'), u3.get('ci_hi'),
                   d0.get('mean'), d0.get('ci_lo'), d0.get('ci_hi'),
                   dm.get('mean'), dm.get('ci_lo'), dm.get('ci_hi'),
                   r['to0'], r['tom'], r['hfr0'],
                   r['served0'], r['energy0'], r.get('note') or ''])
    style_header(ws)
    # ---- Ours / C1 行的视觉强调 ----
    # 2026-10-14 起 Ours **不再单独成块铺在表尾**，而是与基线同表（按 p_c=0 排序）；
    # 这里只把 `★ Ours` / `◆ C1` 行加底色，便于一眼找到主模型。
    for ri, r in enumerate(delivered, 2):
        if r.get('is_ours') or r.get('is_baseline'):
            for c in ws[ri]:
                c.fill = ours_fill
            ws.cell(row=ri, column=2).font = Font(bold=True)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for c in row:
            c.border = border
            if 13 <= c.column <= 32:          # 数值列（前 12 列是文字/资格列）
                c.number_format = '0.00'
        row[0].alignment = ctr
        row[1].alignment = wrap
    # ---- 表下**关键限定**（与 MD 的读表注记同源，2026-10-14 独立审计新增）----
    _o = [r for r in delivered if r.get('is_ours')]
    _o_note = ''
    if _o:
        _bits = []
        for r in _o:
            _pi = (r.get('pairing_identity') or {})
            _pb = ('，问题=%s' % '、'.join(_pi.get('problems') or [])) if _pi.get('problems') else ''
            _bits.append('%s [超时/天=%s, 配对身份=%s%s]'
                         % (r['label'],
                            fnum(r.get('to0') if r.get('to0') is not None else r.get('tom'), 2),
                            _pi.get('identity_verified'), _pb))
        _o_note = ('※ Ours 行同样受 CPU 争用影响，且其中一行配对身份未通过 ⇒ 引用主模型结果'
                   '必须与外部臂一样带限定（外部 19 个正式臂超时/天=0.0）：' + '；'.join(_bits))
    _p2 = ours_same_batch_pair(data, delivered)
    _tri = ''
    if _p2 and _p2.get('arm_util') and _p2.get('cond_util') and _p2.get('diff'):
        _tri = ('※ 结论口径：正式论文主结论以【主档 p_c=(5,10,15)】为准；p_c=0 是敏感性结果。'
                '主档**同批**三元组（gate %s）：%s=%.3f、同批 cond_hist=%.3f、配对差=%+.3f '
                '[%.3f, %.3f]，恒等式校验 %s ⇒ 主档上主模型未超过 C1。'
                '⚠️ 必须用同批的 %.3f；631.4 属另一套协议（a1_c1fresh_reveal_40|cond_hist），'
                '不得与本行并列。'
                % (_p2['gate'], _p2['label'] or 'Ours', _p2['arm_util']['mean'],
                   _p2['cond_util']['mean'], _p2['diff']['mean'], _p2['diff']['ci_lo'],
                   _p2['diff']['ci_hi'],
                   '通过' if _p2.get('identity_holds') else '不通过',
                   _p2['cond_util']['mean']))
    for _t in (
        _o_note,
        '※ 主档列 = 同一策略的惩罚「重新评分」，不是策略对惩罚的「响应」：'
        'run_a1_external_accept.py 每天只跑一次 env.run(i)，再用同一组接/拒决定在各罚分档重算；'
        '接拒 replanner 的构造不接收 p_c。两档差异全部来自评分函数。',
        '※ 本表各行 future_policy 不一致：Ours 两行为 density，19 个对比方法的配对基线为 reveal；'
        '该杠杆会显著改变 cond_hist−uncond_hist 的相对差 ⇒ 本表效用排序只可读作开发集能力对照，'
        '不构成同协议方法归因。详见 CC_Compare/audit/INDEPENDENT_READONLY_ADJUDICATION_2026-10-14.md',
        _tri,
    ):
        if not _t:
            continue
        # ⚠️ 不要用 `ws.append([])` 来"换行" —— openpyxl 里空列表是 **no-op**，`max_row` 不前进，
        # 于是紧接的 `ws.cell(row=ws.max_row, ...)` 会**覆盖最后一条数据行**（本处曾把
        # `rule: value/energy λ=3` 那一行的方法名冲掉，被 `check_table_consistency.py` 抓出
        # 「MD 22 行 vs Excel 21 行，仅在 MD: rule: value/energy λ=3」）。显式取 max_row+1。
        _r = ws.max_row + 1
        ws.cell(row=_r, column=1, value='')
        ws.cell(row=_r, column=2, value=_t).font = Font(italic=True, color='9C0006')
        ws.cell(row=_r, column=2).alignment = wrap
    ws.column_dimensions['B'].width = 58
    ws.freeze_panes = 'C2'
    widths = [4, 34, 16, 18, 24, 9, 24] + [11] * 21 + [10, 12, 26]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    # Sheet 2: baseline arms
    ws2 = wb.create_sheet('C1基线臂')
    ws2.append(['基线臂', '档位', 'Utility', 'CI下界', 'CI上界', '服务单/天',
                '拒绝单/天', '收入/天', '燃油/天', '拒绝损失/天', '能耗kWh/天'])
    for k, v in sorted(data['baseline_arms'].items()):
        pname, an = k.split('|')
        u = v['utility'] or {}
        ws2.append([an, pname, u.get('mean'), u.get('ci_lo'), u.get('ci_hi'),
                    v['mean_served'], v['mean_rejected'],
                    (v['revenue'] or {}).get('mean'), (v['fuel_cost'] or {}).get('mean'),
                    (v['reject_loss'] or {}).get('mean'), v['mean_energy_kwh']])
    style_header(ws2)
    for row in ws2.iter_rows(min_row=2, max_row=ws2.max_row):
        for c in row:
            c.border = border
            if c.column >= 3:
                c.number_format = '0.00'
    for i, w in enumerate([20, 16, 12, 12, 12, 12, 12, 12, 12, 13, 13], 1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    # Sheet 3: arm detail
    ws3 = wb.create_sheet('臂明细与身份')
    ws3.append(['臂目录', '方法', '档位', '正式/诊断', '天数', '可裁决', '身份块',
                'source_stable', '落盘配对身份', 'gate 路径'])
    for r in sorted(data['arms'], key=lambda x: (x['method'], x['p_c'], x['arm_dir'])):
        rep = r.get('reported') or {}
        ws3.append([r['arm_dir'], r['arm'], r['p_c'],
                    '正式' if r['formal'] else '诊断', rep.get('n_days'),
                    rep.get('adjudicable'), '有' if r['identity_present'] else '无',
                    r['identity_source_stable'],
                    (r['pairing_identity_recorded'] or {}).get('identity_verified', '—'),
                    r['gate_path']])
    style_header(ws3)
    for row in ws3.iter_rows(min_row=2, max_row=ws3.max_row):
        for c in row:
            c.border = border
    for i, w in enumerate([20, 24, 16, 12, 8, 9, 8, 14, 15, 60], 1):
        ws3.column_dimensions[get_column_letter(i)].width = w

    # Sheet 4: main model / internal arms —— 这里**故意收录全部**主模型记录：
    # 交付主表（Sheet 1）只留 1–2 行最终结果，其余（3 配置 × 多次重复运行 × 档位 + 内部三臂）
    # 属于**消融/重复运行**，用户要求「移入附录 sheet 并标明」，故用第 1 列显式标注。
    ws5 = wb.create_sheet('主模型与内部臂')
    _delivered = {(r.get('gate'), r.get('arm')) for r in canonical_ours(data)}
    ws5.append(['交付主表？', 'gate', '档位', '臂', 'Utility', 'CI下界', 'CI上界', '超时/天',
                '身份块', 'source_stable', 'future_policy', 'standby_orders',
                'workers', '配对身份'])
    for r in filter_main_model(data):
        u = r.get('reported_utility') or {}
        sm = r.get('reported') or {}
        _in = '★ 是（主表 Ours 行）' if (r['gate'], r['arm']) in _delivered \
            else '否（消融 / 重复运行 / 内部臂）'
        ws5.append([_in, r['gate'], r['p_c'], r['arm'], u.get('mean'), u.get('ci_lo'),
                    u.get('ci_hi'), sm.get('mean_timeouts'),
                    '有' if r['identity_present'] else '无',
                    r['identity_source_stable'],
                    (r['config'] or {}).get('future_policy'),
                    (r['config'] or {}).get('standby_orders'),
                    (r['config'] or {}).get('workers'),
                    (r['pairing_identity'] or {}).get('identity_verified', '—')])
    style_header(ws5)
    for row in ws5.iter_rows(min_row=2, max_row=ws5.max_row):
        for c in row:
            c.border = border
    # 高亮真正的交付行
    for ri in range(2, ws5.max_row + 1):
        if str(ws5.cell(row=ri, column=1).value or '').startswith('★'):
            for c in ws5[ri]:
                c.fill = ours_fill
                c.font = Font(bold=True)
    for i, w in enumerate([24, 46, 16, 24, 12, 12, 12, 10, 8, 14, 14, 14, 9, 12], 1):
        ws5.column_dimensions[get_column_letter(i)].width = w

    # Sheet 5: unadapted
    ws4 = wb.create_sheet('未适配方法')
    ws4.append(['方法', '不可适配的具体原因', '依据文档'])
    for x in data['incomparable_or_unadapted']:
        ws4.append([x['method'], x.get('reason', '—'), x['path']])
    style_header(ws4)
    ws4.column_dimensions['A'].width = 26
    ws4.column_dimensions['B'].width = 78
    ws4.column_dimensions['C'].width = 52

    xp = os.path.join(args.outdir, 'A-v1对比方法总表.xlsx')
    wb.save(xp)
    print('wrote', xp)


if __name__ == '__main__':
    main()
