"""从 `dataset.json` 打印 README §2「主结果」所需的**成品文本**，供收口后直接粘贴。

为什么要有它：验收清单 §3 第 5 项（文档同步）是**人工**步骤，也是整条链上唯一
「会话必须还活着」才能完成的环节（表格/配图/摘要/门禁都由 `_local_finalize.ps1` 自动产出）。
把「从数据算出§2 该写什么」机械化后，即使会话提前结束，重出 §2 也只是复制粘贴。

输出四块：
  A. C1 基线行（`baseline_arms` 的 cond_hist 两档）
  B. 主表行（每个方法取**最优批次**的臂，按 p_c=0 降序；列式与 README §2 一致）
  C. 三个派生差值（Ours−最好外部 / Ours−最好规则 / Ours−C1）与方法条数
  D. 供核对的三条「人工判断项」（方法条数、最好外部是谁、Ours 行取自哪个 gate）

用法（仓库根）：
    python CC_Compare/tools/print_readme_section2.py \
        --data CC_Compare/results/a1_compare_table/dataset.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)

from collect_compare_table import METHOD_META, PARAMS_M  # noqa: E402

RANK = {'a1_ext_formal2': 0, 'a1_ext_a05e': 1, 'a1_ext_extra': 2}
RULE_METHODS = {'myopic', 'valuee20', 'valuee30'}
TIERS = ('p_c=0', 'p_c=(5,10,15)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',
                    default='CC_Compare/results/a1_compare_table/dataset.json')
    args = ap.parse_args()
    p = args.data if os.path.isabs(args.data) else os.path.join(ROOT, *args.data.split('/'))
    d = json.load(open(p, encoding='utf-8'))

    per = {}
    for r in d['arms']:
        if not r.get('formal'):
            continue
        per.setdefault(r['method'], {}).setdefault(r['p_c'], []).append(r)

    def pick(m, pc):
        c = per.get(m, {}).get(pc) or []
        if not c:
            return None
        return sorted(c, key=lambda x: RANK.get(x['group'], 9))[0]

    def util(m, pc):
        rec = pick(m, pc)
        return ((rec or {}).get('reported') or {}).get('utility') or {}

    b = d['baseline_arms']
    base0 = b['p_c=0|cond_hist']['utility']['mean']
    basem = b['p_c=(5,10,15)|cond_hist']['utility']['mean']

    print('== A. C1 基线行 ==')
    print('C1 基线（身份合格 gate `%s`）：**p_c=0 = %.1f**，**主档 = %.1f**。'
          % (d['generated_from']['baseline'].split('/')[-2], base0, basem))

    print()
    print('== B. 主表行（按 p_c=0 降序；直接粘贴到 README §2 表体）==')
    rows = []
    for m in per:
        u0 = util(m, 'p_c=0').get('mean')
        um = util(m, 'p_c=(5,10,15)').get('mean')
        if u0 is None and um is None:
            continue
        rows.append((m, u0, um))
    rows.sort(key=lambda x: (x[1] is None, -(x[1] if x[1] is not None else 0)))
    for i, (m, u0, um) in enumerate(rows, 1):
        lab = METHOD_META.get(m, {}).get('label', m)
        note = ''
        if m == 'valuee30':
            note = '（0 接单，判负留档）'
        print('| %d | %s%s | %s | %s |' % (
            i, lab, note,
            ('%.1f' % u0) if u0 is not None else '—',
            ('%.1f' % um) if um is not None else '—'))

    print()
    print('== C. 派生量（脚本算，写进 §2 读点）==')
    # ⚠️ 2026-10-13 两处口径修正（首版跑出来会**给出错的数**，若直接粘贴会污染 README）：
    #  ① 必须把**经典求解器**（OR-Tools / PyVRP）与**学习型**外部方法分开：README §2 读点写的是
    #     「高出最好的**学习型**外部方法（RouteFinder 1266.2）」⇒ Ours−RouteFinder = +311.7；
    #     若把 ortools(1300.7) 也算进「外部」，会算成 +277.3 —— 两个都不假，但**混用就是错的**。
    #  ② 「Ours − C1」在 README 里用的是**该 gate 内**的配对差（`main_comparison.
    #     maskco_minus_cond_hist`，对照该 gate 自己的 cond_hist 1694.7 ⇒ −116.7），
    #     **不是**「Ours − C1 基线 1703.7 = −125.8」。两者是不同量，必须分开标注。
    SOLVERS = {'ortools', 'pyvrp'}
    learned = [m for m in per if m not in RULE_METHODS and m not in SOLVERS]
    externals = [m for m in per if m not in RULE_METHODS]
    rules = [m for m in per if m in RULE_METHODS]
    print('学习型外部方法条数：%d ；经典求解器：%d ；规则臂：%d'
          % (len(learned), len(SOLVERS & set(per)), len(rules)))
    print('§2 读点里「N 个外部方法」= 学习型 %d + 经典 %d = **%d**（排除 3 个规则臂）'
          % (len(learned), len(SOLVERS & set(per)), len(externals)))
    bl = max(((m, util(m, 'p_c=0').get('mean')) for m in learned
              if util(m, 'p_c=0').get('mean') is not None),
             key=lambda x: x[1], default=(None, None))
    ba = max(((m, util(m, 'p_c=0').get('mean')) for m in externals
              if util(m, 'p_c=0').get('mean') is not None),
             key=lambda x: x[1], default=(None, None))
    br = max(((m, util(m, 'p_c=0').get('mean')) for m in rules
              if util(m, 'p_c=0').get('mean') is not None),
             key=lambda x: x[1], default=(None, None))
    print('最好**学习型**外部（p_c=0）：%s = %s' % (bl[0], ('%.1f' % bl[1]) if bl[1] is not None else '—'))
    print('最好外部（含经典求解器，p_c=0）：%s = %s' % (ba[0], ('%.1f' % ba[1]) if ba[1] is not None else '—'))
    print('最好规则（p_c=0）：%s = %s' % (br[0], ('%.1f' % br[1]) if br[1] is not None else '—'))
    mm = [r for r in d.get('main_model', [])
          if r['arm'] == 'maskco' and r['p_c'] == 'p_c=0'
          and 'pretrained' in str((r.get('config') or {}).get('model_bin'))]
    our0 = ((mm[0].get('reported_utility') or {}).get('mean') if mm else None)
    print('Ours（MaskCO A3 pretrained，p_c=0）= %s' % (('%.1f' % our0) if our0 is not None else '—'))
    if our0 is not None and bl[1] is not None:
        print('  Ours − 最好**学习型**外部 = %+.1f   ← §2 读点用的就是这个口径' % (our0 - bl[1]))
    if our0 is not None and ba[1] is not None:
        print('  Ours − 最好外部（含 classic）= %+.1f' % (our0 - ba[1]))
    if our0 is not None and br[1] is not None:
        print('  Ours − 最好规则 = %+.1f' % (our0 - br[1]))
    if mm:
        mc = (mm[0].get('main_comparison') or {}).get('maskco_minus_cond_hist') or {}
        if mc.get('mean') is not None:
            print('  Ours − **该 gate 内** cond_hist = %+.1f [%.1f, %.1f]   ← §2 读点 2 的口径'
                  % (mc['mean'], mc.get('ci_lo', float('nan')), mc.get('ci_hi', float('nan'))))
    if our0 is not None:
        print('  Ours − **C1 基线**（%s，另一种口径，勿与上一条混用）= %+.1f'
              % (d['generated_from']['baseline'].split('/')[-2], our0 - base0))

    print()
    print('== D. 必须人工判断/核对的项 ==')
    print('1) §2 读点里的「N 个外部方法」= %d（学习型 %d + 经典 %d；**排除 3 个规则臂**）'
          % (len(externals), len(learned), len(SOLVERS & set(per))))
    print('2) 「最好的学习型外部方法」= %s（%s）—— 探针值不能当结论，必须用本表正式臂'
          % (bl[0], ('%.1f' % bl[1]) if bl[1] is not None else '—'))
    print('   注意：若把经典求解器也算进去，「最好外部」会变成 %s（%s）—— '
          'README 用的是**学习型**口径，勿混。'
          % (ba[0], ('%.1f' % ba[1]) if ba[1] is not None else '—'))
    print('3) Ours 行来源 gate 见 dataset.main_model 的 config.model_bin；'
          '若主模型侧出现新臂，先看总表 §0 顶部告警表')
    print('4) CO-enriched / MAPT 进表后，上面 B 表会多出行、C 的条数会变 —— 重跑本脚本即可')


if __name__ == '__main__':
    main()
