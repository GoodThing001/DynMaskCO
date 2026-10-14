"""A-v1 对比方法总表渲染器（Markdown + Excel）。

输入 = `collect_compare_table.py --json` 产出的数据集；输出：
  - `A-v1对比方法总表.md`（Markdown，主表 + 分解 + 未适配清单）
  - `A-v1对比方法总表.xlsx`（多 sheet，AEGIS-CD 风格）

用法：
    python CC_Compare/tools/render_compare_table.py --data <json> --outdir <dir>
"""
import argparse
import json
import os
import sys

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

# ⚠️ **两个开关，别只改一个**（2026-10-13 明确，此前只有 `MAIN_MODEL_ARMS` 被文档提到）：
#   - `MAIN_MODEL_ARMS`   → 控制**哪些臂出现在 §0 Ours 摘要 / §3 主模型与内部臂 / Excel
#                           「主模型与内部臂」sheet**（过滤函数 `filter_main_model`）；
#   - `OURS_BLOCK_ARMS`   → 控制**Excel「性能总表」末尾那几行 `Ours: …`**（`build_ours_rows`）。
# 两者是**不同的展示位**：前者含内部基线（cond_hist 等）与对照臂，后者只放代表本工作的臂
# （标签写成 `Ours: <arm> [tag]`），因此不能混为一谈、也不要求集合相同。
# 若你把新臂加进 `MAIN_MODEL_ARMS` 却发现 Excel 的 `Ours:` 行没变 —— 那是**预期**，
# 需要在 `OURS_BLOCK_ARMS` 里再登记一次。
OURS_BLOCK_ARMS = ('maskco', 'warmstart', 'reconstruction_control')


DEV_SEED = 20260926


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
    """主模型（Ours）行，放进性能总表末尾（对齐参考表的「Ours 占最后一行」格式）。

    口径提醒：这些行来自**主模型侧自己的 gate**（不同 driver/standby 设置），
    `Δ` 列给的是**该 gate 内的配对差**（maskco − cond_hist），不是对 c1rr2 的配对差。
    """
    ours = []
    for r in filter_main_model(data):
        if r['arm'] not in OURS_BLOCK_ARMS:
            continue
        cfg = r['config'] or {}
        gate_short = r['gate'].replace('results/', '').split('/')[0]
        cfg2 = (r['config'] or {}).get('arm')
        model = (cfg.get('model_bin') or '').split('/')[-2] if cfg.get('model_bin') else ''
        tag = 'A3-%s' % (model or 'a3') if 'a3_' in str(cfg.get('model_bin')) else (
            'I3-%s' % (model or 'i3'))
        ours.append(dict(
            label='Ours: %s [%s]' % (r['arm'], tag),
            cls='主模型/内部臂', venue='本工作', backbone=gate_short,
            native='接/拒 + 掩码重构',
            p_c=r['p_c'],
            util=r.get('reported_utility'),
            paired=(r.get('main_comparison') or {}).get('%s_minus_cond_hist' % r['arm']),
            timeouts=(r.get('reported') or {}).get('mean_timeouts'),
            source_stable=r.get('identity_source_stable'),
            gate=r['gate'],
        ))
    # 按档位、再按 utility 降序
    order = {'p_c=0': 0, 'p_c=(5,10,15)': 1}
    ours.sort(key=lambda x: (order.get(x['p_c'], 9),
                             -(x['util'] or {}).get('mean', -1e18)))
    return ours


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
    L = []
    L.append('| # | 方法 | 类别 | 来源 | Backbone / 权重 | 参数量(M) | 权重(MB) | 原生约束 | Utility p_c=0 ↑ | Utility 主档 p_c=(5,10,15) | '
             'Utility p_c=(10,20,30) | Δ vs cond_hist (p_c=0) | Δ vs cond_hist (主档) | '
             '硬可行率 | fallback% | 超时/天 | 最大单决策(s) |')
    L.append('|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for i, r in enumerate(rows, 1):
        L.append('| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            i, r['label'], r['cls'], r['venue'], r['backbone'],
            fnum(r.get('params_m'), 3), fnum(r.get('ckpt_mb'), 2), r['native'],
            fci(r['util0']), fci(r['utilm']), fci(r['util3']),
            fci(r['diff0']), fci(r['diffm']),
            fnum(r['hfr0'], 3),
            fpc(r['fb0']),
            fnum(r['to0'], 2),
            fnum(r['md0'], 2)))
    ours = build_ours_rows(data)
    if ours:
        L.append('| | **—— Ours / 主模型侧（来自主模型自己的 gate；Δ 为该 gate 内配对差）——** | | | | | | | | | | | | | | | |')
        for r in ours:
            u = r['util'] or {}
            pr = r['paired'] or {}
            L.append('| ★ | %s | %s | %s | %s | — | — | %s | %s | — | — | %s | — | — | — | %s | — |' % (
                r['label'], r['cls'], r['venue'], r['backbone'], r['native'],
                fci(u), fci(pr) if pr.get('mean') is not None else '—',
                fnum(r['timeouts'], 2)))
    return '\n'.join(L)


def md_main_model(data):
    L = []
    L.append('| gate | 档位 | 臂 | utility | 超时/天 | 身份块 | source_stable | future_policy | standby | workers | 配对身份 |')
    L.append('|---|---|---|---:|---:|---|---|---|---|---|---|')
    for r in filter_main_model(data):
        u = r.get('reported_utility') or {}
        sm = r.get('reported') or {}
        L.append('| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['gate'], r['p_c'], r['arm'],
            fci(u) if u.get('mean') is not None else '—',
            fnum(sm.get('mean_timeouts'), 2),
            '有' if r['identity_present'] else '无',
            r['identity_source_stable'],
            (r['config'] or {}).get('future_policy'),
            (r['config'] or {}).get('standby_orders'),
            (r['config'] or {}).get('workers'),
            (r['pairing_identity'] or {}).get('identity_verified', '—')))
    return '\n'.join(L)


def md_table_notes(rows):
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
    rows = build_rows(data)
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
    md += '\n\n## 1. 主表（对比方法，按 p_c=0 效用降序）\n\n' + md_main_table(data, rows) + md_table_notes(rows)
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
    cols = ['#', '方法', '类别', '来源/年份', 'Backbone/权重', '参数量(M)', '权重(MB)', '原生约束支持',
            'Utility p_c=0', 'CI下界', 'CI上界',
            'Utility 主档', 'CI下界 ', 'CI上界 ',
            'Utility p_c=10/20/30', 'CI下界  ', 'CI上界  ',
            'Δ vs cond p_c=0', 'Δ CI下界', 'Δ CI上界',
            'Δ vs cond 主档', 'Δ主档 CI下界', 'Δ主档 CI上界',
            '硬可行率 p_c=0', 'fallback% p_c=0',
            '超时/天 p_c=0', '超时/天 主档', '最大单决策s p_c=0', '最大单决策s 主档',
            '服务单/天 p_c=0', '能耗kWh/天 p_c=0', '备注']
    ws.append(cols)
    for i, r in enumerate(rows, 1):
        u0, um, u3 = r['util0'] or {}, r['utilm'] or {}, r['util3'] or {}
        d0, dm = r['diff0'] or {}, r['diffm'] or {}
        ws.append([i, r['label'], r['cls'], r['venue'], r['backbone'],
                   r.get('params_m'), r.get('ckpt_mb'), r['native'],
                   u0.get('mean'), u0.get('ci_lo'), u0.get('ci_hi'),
                   um.get('mean'), um.get('ci_lo'), um.get('ci_hi'),
                   u3.get('mean'), u3.get('ci_lo'), u3.get('ci_hi'),
                   d0.get('mean'), d0.get('ci_lo'), d0.get('ci_hi'),
                   dm.get('mean'), dm.get('ci_lo'), dm.get('ci_hi'),
                   r['hfr0'], r['fb0'],
                   r['to0'], r['tom'], r['md0'], r['mdm'],
                   r['served0'], r['energy0'], r['note']])
    style_header(ws)
    # ---- Ours（主模型）块：与参考表一致，占表末 ----
    ours = build_ours_rows(data)
    if ours:
        sep = ws.max_row + 2
        ws.cell(row=sep, column=1,
                value='—— Ours / 主模型侧（来自主模型自己的 gate，Δ 为该 gate 内配对差）——')
        ws.cell(row=sep, column=1).font = Font(bold=True, color='7F6000')
        for r in ours:
            u = r['util'] or {}
            pr = r['paired'] or {}
            ws.append(['★', r['label'], r['cls'], '%s @ %s' % (r['venue'], r['p_c']),
                       r['backbone'], None, None, r['native'],
                       u.get('mean'), u.get('ci_lo'), u.get('ci_hi'),
                       None, None, None, None, None, None,
                       pr.get('mean'), pr.get('ci_lo'), pr.get('ci_hi'),
                       None, None, None, None,
                       r['timeouts'], None, None, None,
                       None, None, r['gate']])
            for c in ws[ws.max_row]:
                c.fill = ours_fill
                c.border = border
                if 8 <= c.column <= 21:
                    c.number_format = '0.00'
            ws.cell(row=ws.max_row, column=2).alignment = wrap
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for c in row:
            c.border = border
            if 6 <= c.column <= 30:
                c.number_format = '0.00'
        row[0].alignment = ctr
        row[1].alignment = wrap
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

    # Sheet 4: main model / internal arms
    ws5 = wb.create_sheet('主模型与内部臂')
    ws5.append(['gate', '档位', '臂', 'Utility', 'CI下界', 'CI上界', '超时/天',
                '身份块', 'source_stable', 'future_policy', 'standby_orders',
                'workers', '配对身份'])
    for r in filter_main_model(data):
        u = r.get('reported_utility') or {}
        sm = r.get('reported') or {}
        ws5.append([r['gate'], r['p_c'], r['arm'], u.get('mean'), u.get('ci_lo'),
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
    for i, w in enumerate([46, 16, 24, 12, 12, 12, 10, 8, 14, 14, 14, 9, 12], 1):
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
