"""生成 v2 正式批次的结果摘要（`V2批次结果.md`）——由收口脚本自动调用。

内容：新臂清单（效用/CI/超时/硬可行率/身份裁决）+ 基线超时对账 + 源码冻结漂移。
即使没有人工介入，最终交付目录里也会有一份可直接引用的 v2 结论。

用法：
    python CC_Compare/tools/render_v2_summary.py \
        --data CC_Compare/results/a1_compare_table/dataset.json \
        --outdir CC_Compare/results/a1_compare_table
"""
import argparse
import glob
import json
import os
import time

EXT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..',
                                   'C-VRP_Cold-chainVehicleRoutingProblem'))


def load(p):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--baseline-gate',
                    default=os.path.join(EXT, 'results', 'a1_c1fresh_reveal_40',
                                         'gate.json'))
    ap.add_argument('--ref-gate',
                    default=os.path.join(EXT, 'results', 'a1_c1rr2_reveal_40',
                                         'gate.json'))
    args = ap.parse_args()

    d = load(args.data) or {}
    v2 = [r for r in d.get('arms', []) if r.get('group') == 'a1_ext_formal2']
    L = []
    L.append('# v2 正式批次结果摘要（自动生成）')
    L.append('')
    L.append('> 生成时间：%s（由 `tools/render_v2_summary.py` 在收口链末尾自动写出）'
             % time.strftime('%Y-%m-%d %H:%M:%S'))
    L.append('> 口径：40 天开发集 seed 20260926；效用 = 收入 − 燃油 − 拒绝损失；'
             '统计 = 按天聚类 bootstrap（seed+777，2000 次）。')
    L.append('')

    L.append('## 1. 新批次臂清单（group = `a1_ext_formal2`）')
    L.append('')
    if not v2:
        L.append('**（空：v2 臂尚未采集到 `dataset.json`，或批次未完成）**')
    else:
        L.append('| 臂 | 档位 | 效用 | CI 下界 | CI 上界 | 超时/天 | 硬可行率 | '
                 '落盘配对身份 | Δ vs cond_hist | 配对可裁决 |')
        L.append('|---|---|---:|---:|---:|---:|---|---:|---:|---|')
        for r in sorted(v2, key=lambda x: (x['method'], x['p_c'])):
            rep = r.get('reported') or {}
            u = rep.get('utility') or {}
            pid = r.get('pairing_identity_recorded') or {}
            pd = (r.get('repaired_vs_baseline') or {}).get('cond_hist') or {}
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                r['arm_dir'], r['p_c'],
                ('%.1f' % u['mean']) if u.get('mean') is not None else '—',
                ('%.1f' % u['ci_lo']) if u.get('ci_lo') is not None else '—',
                ('%.1f' % u['ci_hi']) if u.get('ci_hi') is not None else '—',
                rep.get('mean_timeouts'), rep.get('hard_feasible_rate'),
                pid.get('identity_verified'),
                ('%.1f' % pd['mean']) if pd.get('mean') is not None else '—',
                pd.get('adjudicable')))
    L.append('')

    L.append('## 2. 基线超时对账（负载污染判据）')
    L.append('')
    new, ref = load(args.baseline_gate), load(args.ref_gate)
    if not new:
        L.append('**（新基线 gate 未落盘）**')
    else:
        ni = new.get('identity') or {}
        L.append('- 新基线 `source_stable` = **%s**，identity 键数 = %d，'
                 'contract = `%s`，data = `%s`，B = %s'
                 % (ni.get('source_stable'), len(ni),
                    str(ni.get('contract_sha256'))[:16],
                    str(ni.get('data_sha256'))[:16],
                    new.get('budget', {}).get('B')))
        L.append('')
        L.append('| 档位 | 臂 | 新基线 效用 | 超时/天 | c1rr2 效用 | c1rr2 超时/天 | 超时比 |')
        L.append('|---|---|---:|---:|---:|---:|---:|')
        for pname in sorted(new.get('gate', {})):
            for arm in sorted(new['gate'][pname]['arms']):
                a = new['gate'][pname]['arms'][arm]
                b = ((ref or {}).get('gate', {}).get(pname, {})
                     .get('arms', {}).get(arm, {}))
                ta, tb = a.get('mean_timeouts'), b.get('mean_timeouts')
                ratio = ('%.2fx' % (ta / tb)) if (ta and tb) else '—'
                L.append('| %s | %s | %.1f | %.2f | %.1f | %.2f | %s |' % (
                    pname, arm,
                    a['utility_all_days']['mean'], ta or 0,
                    b.get('utility_all_days', {}).get('mean', float('nan')), tb or 0,
                    ratio))
        L.append('')
        L.append('> 判据：任一臂超时超过 c1rr2 同档位 **2 倍** ⇒ 判受负载污染，'
                 '须在低负载下重跑 C1 基线。')

    L.append('')
    L.append('## 3. 源码冻结漂移（启动 vs 结束）')
    L.append('')
    s = load(os.path.join(args.outdir, 'freeze_v2_start.json'))
    e = load(os.path.join(args.outdir, 'freeze_v2_end.json'))
    allowed = {'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/'
               'ordering_providers.py'}
    if not s or not e:
        L.append('**（缺少 freeze_v2_start/end.json）**')
        drift = None
    else:
        # 2026-10-13 修复：原先只遍历 start 的键，只出现在 end 的文件
        # （= 批次中途往 `record_freeze_v2.py::FILES` 加的新文件）会被**完全静默忽略**。
        # 这正是 R9（两个 v2-only provider 不在冻结清单里）不能靠「事后补进清单」解决的原因
        # —— 补了也看不见。现按**键的并集**分类：changed / added / removed。
        keys = sorted(set(s) | set(e))
        changed = [k for k in keys if (k in s and k in e and s[k] != e.get(k))]
        added = [k for k in keys if k not in s]
        removed = [k for k in keys if k not in e]
        unexpected = [k for k in changed if k not in allowed]
        drift = unexpected
        L.append('- 冻结文件数：start %d / end %d / 并集 %d；**两侧都有且哈希变化：%d**'
                 '（预期内接线：`ordering_providers.py`；**意外漂移 %d 个**）；'
                 '只在 end：%d；只在 start：%d'
                 % (len(s), len(e), len(keys), len(changed), len(unexpected),
                    len(added), len(removed)))
        for k in changed:
            L.append('  - `%s`%s' % (k, '（预期：追加 provider 接线）'
                                    if k in allowed else '  ← **意外**'))
        for k in added:
            L.append('  - `%s` ← **只在 end 快照里**（批次中途往 FILES 加的新文件；'
                     'start 无基线 ⇒ 本次**无法判定它是否被改过**）' % k)
        for k in removed:
            L.append('  - `%s` ← **只在 start 快照里**（文件被删或从 FILES 移除）' % k)
        L.append('')
        L.append('> 配对要求双侧 `source_stable=True` 且 `shared_source_sha256` 全等；'
                 '共享计算文件若有意外漂移，须在解读前说明影响。')
        if added or removed:
            L.append('> ⚠️ 冻结清单本身在批次期间变过（见上表 added/removed）：'
                     '**下一批**应在启动前把清单定稿（README §12 R9）。')

    # ---------------- 4. 验收判定（README §11 的机器可判部分） ----------------
    L.append('')
    L.append('## 4. 验收判定（自动；对应 README §11）')
    L.append('')
    checks = []

    def add(name, ok, detail):
        checks.append((name, ok, detail))

    if not v2:
        add('v2 臂已采集', False, 'dataset 中无 a1_ext_formal2 臂')
    else:
        bad_days = [r['arm_dir'] for r in v2
                    if (r.get('reported') or {}).get('n_days') != 40
                    or (r.get('reported') or {}).get('missing_day_ids')]
        add('天数完整性（每臂 40 天、无缺日）', not bad_days,
            '%d/%d 臂通过' % (len(v2) - len(bad_days), len(v2))
            + ('；异常：%s' % bad_days[:5] if bad_days else ''))
        not_adj = [r['arm_dir'] for r in v2
                   if not (r.get('reported') or {}).get('adjudicable')]
        add('可裁决性（adjudicable）', not not_adj,
            '%d/%d 臂通过' % (len(v2) - len(not_adj), len(v2)))
        # 2026-10-13 修复（成因经源码核实，与 render_compare_table 同一处）：**纯规则臂
        # 结构性没有身份块** —— `rule_myopic/valuee20/valuee30` 由
        # `run_a1_downstream_ablation.py` 产出，该脚本从不调用 `pairing_identity_check`
        # （grep 已核：0 处出现），故其 gate 里既没有 `_pairing_identity`，也没有 `identity`。
        # 原判据「全部 v2 臂 source_stable=True / identity_verified=True」因此
        # **永远不可能通过** ⇒ 总判定被永久打成 FAIL，且 v2 批次换来的「落盘正式裁决」
        # 永远显示不出来。现按「是否需要配对身份」分子集核验，并把排除**写明**（不静默）。
        _NO_PAIR_ID = {'myopic', 'valuee20', 'valuee30'}
        v2_req = [r for r in v2 if r.get('method') not in _NO_PAIR_ID]
        v2_rule = [r for r in v2 if r.get('method') in _NO_PAIR_ID]
        _rule_note = ('；另有 %d 个纯规则臂无身份块（设计如此，见上）' % len(v2_rule)
                      if v2_rule else '')
        no_stable = [r['arm_dir'] for r in v2_req
                     if r.get('identity_source_stable') is not True]
        add('臂 source_stable=True（仅需配对身份的臂）', not no_stable,
            '%d/%d 臂通过' % (len(v2_req) - len(no_stable), len(v2_req))
            + ('；异常：%s' % no_stable[:5] if no_stable else '') + _rule_note)
        no_pair = [r['arm_dir'] for r in v2_req
                   if not (r.get('pairing_identity_recorded') or {}
                           ).get('identity_verified')]
        add('落盘配对 identity_verified（仅需配对身份的臂）', not no_pair,
            '%d/%d 臂通过' % (len(v2_req) - len(no_pair), len(v2_req))
            + ('；未过：%s' % no_pair[:5] if no_pair else '') + _rule_note)
        bad_hfr = [r['arm_dir'] for r in v2
                   if (r.get('reported') or {}).get('hard_feasible_rate') != 1.0]
        add('硬可行率 = 1.0', not bad_hfr,
            '%d/%d 臂通过' % (len(v2) - len(bad_hfr), len(v2)))
        fails = {r['arm_dir']: (r.get('reported') or {}).get('fail_counts')
                 for r in v2 if (r.get('reported') or {}).get('fail_counts')}
        add('无硬失败（fail_counts 空）', not fails, str(fails) if fails else '全部为空')

    if drift is None:
        add('冻结漂移（仅允许 ordering_providers.py）', False, '缺 freeze 快照')
    else:
        add('冻结漂移（仅允许 ordering_providers.py）', not drift,
            '意外漂移 %d 个%s' % (len(drift), ('：%s' % drift[:3]) if drift else ''))

    # 方法覆盖（目标第 (1)/(4) 项）：既不能有方法被静默丢掉，也不能同一方法
    # 既在结果行又在「未适配」表（后者正是 2026-10-13 修掉的旧缺陷）。
    # 完整版守卫（含目录级正向覆盖 + 依据文档存在性）见 check_method_coverage.py。
    _EXPECT = ('ortools', 'pyvrp', 'rrnco', 'routefinder', 'cada', 'mvmoe', 'pomo',
               'symnco', 'sgbs', 'deepaco', 'attention', 'omnivrp', 'lih',
               'coenrich_nopchgs', 'coenrich_pchgs', 'mapt',
               'myopic', 'valuee20', 'valuee30')
    _arms = {a.get('method') for a in (d.get('arms') or []) if a.get('method')}
    _un = {u.get('method') for u in (d.get('incomparable_or_unadapted') or [])
           if u.get('method')}
    _both = sorted(_arms & _un)
    _miss = [k for k in _EXPECT if k not in (_arms | _un)]
    add('方法覆盖（19 key 齐 且 结果行 ∩ 未适配 = ∅）', not _both and not _miss,
        ('冲突：%s；' % _both if _both else '')
        + ('缺 key：%s；' % _miss if _miss else '')
        + ('结果行 %d / 未适配 %d / 预期 %d' % (len(_arms), len(_un), len(_EXPECT))
           if not _both and not _miss else ''))

    L.append('| # | 检查 | 判定 | 说明 |')
    L.append('|---|---|---|---|')
    for i, (name, ok, detail) in enumerate(checks, 1):
        L.append('| %d | %s | **%s** | %s |' % (i, name, 'PASS' if ok else 'FAIL', detail))
    L.append('')
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    L.append('**总判定：%s**（%d/%d 项通过）'
             % ('PASS —— 可进入论文引用' if n_fail == 0 else 'FAIL —— 见上表',
                len(checks) - n_fail, len(checks)))
    L.append('')
    L.append('> 仍需人工确认的项（README §11）：MD 与 Excel 数字一致、README 头条数字一致、'
             '负载污染判定（§2 超时比）、新臂并入 19 方法总表、文档同步。')

    out = os.path.join(args.outdir, 'V2批次结果.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', out, '(%d v2 arms)' % len(v2))


if __name__ == '__main__':
    main()
