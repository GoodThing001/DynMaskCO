"""方法覆盖守卫：确认「没有任何对比方法被静默丢掉」，也没有方法自相矛盾。

对应目标第 (1) 项（审计并对齐**全部**对比方法）与第 (4) 项（如实记录不可适配项及原因）。
这两项的前提是覆盖完整 —— 本守卫把「覆盖完整」变成机器可查，而不是靠叙述：

A. **正向覆盖**：`CC_Compare/` 下每个非基础设施目录都必须有归宿
     - 结果行（`dataset['arms'][*]['method']`），或
     - 「未适配」清单（`dataset['incomparable_or_unadapted'][*]['method']`）。
   两者都没有 ⇒ FAIL（有方法目录没被审计进交付物）。
B. **不自相矛盾**：出现在 `incomparable_or_unadapted` 的方法**不得**同时有结果行
   （否则同一方法既在结果表又在「未适配」表 —— 这正是 2026-10-13 修掉的那类缺陷）。
C. **19 个预期方法 key 全部出现**：经典求解器 2（ortools/pyvrp）+ 学习型 14
   （rrnco/routefinder/cada/mvmoe/pomo/symnco/sgbs/deepaco/attention/omnivrp/lih/
   coenrich_nopchgs/coenrich_pchgs/mapt）+ 规则 3（myopic/valuee20/valuee30）。
D. **内部三臂在位**：`baseline_arms` 必须含 cond_hist / uncond_hist / explicit_feat。
E. **需配对身份的正式臂必须落盘 `identity_verified`**：v2 正式批次 `a1_ext_formal2` 里，
   凡**非纯规则**的臂都必须 `_pairing_identity.identity_verified=True`；
   纯规则臂（myopic / valuee20 / valuee30）**豁免** —— 它们由
   `run_a1_downstream_ablation.py` 产出，该脚本从不调用 `pairing_identity_check`（源码已核），
   故结构性没有身份块，这是设计如此而不是缺陷。
   （E 的作用：把「学习型臂静默丢了配对身份」和「规则臂按设计没有身份」区分开 ——
   前者就是 2026-10-13 缺陷 F 那一类，必须报错。）
F. **每个学习型方法都要有落盘的参与度证据** `participation/<name>.json`，
   且凡通用 C2 只能报 `NOT_SUPPORTED` 的方法（纯 NumPy 实现）必须另有**专用 C2** 产物。
   为什么要机器查：2026-10-14 实测发现 `CO-enriched-ML` 是 13 个学习型方法里**唯一**
   没有 `participation/*.json` 的 —— 台账里它写着「强」，但那是**散文断言**而非产物；
   根因是 `COEnrichedMLProvider` **没有 `_load()`**（在 `__init__` 里就加载完），
   通用 harness 一调用就 `AttributeError`。这种缺口**靠人读台账发现不了**，必须机器查。

用法（仓库根，收口后跑）：
    python CC_Compare/tools/check_method_coverage.py \
        --data CC_Compare/results/a1_compare_table/dataset.json

退出码：0 = PASS；1 = FAIL（逐条打印原因，不静默）。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
CC = os.path.join(ROOT, 'CC_Compare')

# 非方法目录（基础设施 / 共享代码 / 产物），不参与方法覆盖判定。
INFRA_DIRS = {'common', 'results', 'tools', 'a1_accept_common', '_upload'}

# 纯规则臂：没有学习权重，产出脚本 `run_a1_downstream_ablation.py` 从不调用
# `pairing_identity_check` ⇒ 结构性没有身份块。判据里必须**豁免**它们，
# 否则「全部正式臂都有配对身份」永远不可能成立（详见 §E）。
RULE_METHODS = {'myopic', 'valuee20', 'valuee30'}

# 经典求解器：无学习参数，「权重是否参与决策」不成立 ⇒ 不要求参与度产物（见台账）。
SOLVERS = {'ortools', 'pyvrp'}

# 方法 key → `participation/` 文件基名（CO-enriched 两个模式共用一份）。
PARTICIPATION_ALIAS = {'coenrich_nopchgs': 'coenrich', 'coenrich_pchgs': 'coenrich'}

# 通用 C2（torch 随机权重）对纯 NumPy 实现只能报 `NOT_SUPPORTED` ⇒
# 这些方法必须另有专用 C2 产物，否则「权重因果」实际上没被测过。
EXTRA_C2 = {'coenrich': 'coenrich_c2_swap.json'}

# 19 个预期方法 key（与 collect_compare_table.METHOD_META 一一对应）。
EXPECTED_KEYS = [
    'ortools', 'pyvrp', 'rrnco',
    'routefinder', 'cada', 'mvmoe', 'pomo', 'symnco', 'sgbs', 'deepaco',
    'attention', 'omnivrp', 'lih', 'coenrich_nopchgs', 'coenrich_pchgs', 'mapt',
    'myopic', 'valuee20', 'valuee30',
]

# CC_Compare 方法目录 → 归宿（方法 key 列表；[] = 归「未适配」清单）。
# 'internal' 是项目内部三臂（基线，不是对比方法），单独按 D 项核。
DIR_HOME = {
    'AttentionModel': ['attention'],
    'CaDA': ['cada'],
    'CO-enriched-ML': ['coenrich_nopchgs', 'coenrich_pchgs'],
    'DeepACO': ['deepaco'],
    'Learn-Improvement-Heuristics': ['lih'],
    'Learning-to-Delegate': [],
    'MAPT': ['mapt'],
    'MAPT_code': ['mapt'],
    'MVMoE': ['mvmoe'],
    'Omni-VRP': ['omnivrp'],
    'OR-Tools': ['ortools'],
    'PIP-constraint': [],
    'POMO': ['pomo'],
    'PyVRP': ['pyvrp'],
    'RRNCO': ['rrnco'],
    'RouteFinder': ['routefinder'],
    'SGBS': ['sgbs'],
    'Sym-NCO': ['symnco'],
    'internal': ['cond_hist', 'uncond_hist', 'explicit_feat'],
}

# 归「未适配」清单的目录，其依据文档必须真实存在（表里引用的路径不能是死链）。
UNADAPTED_DOC = {
    'Learning-to-Delegate': 'CC_Compare/Learning-to-Delegate/A1_ASSESSMENT.md',
    'PIP-constraint': 'CC_Compare/PIP-constraint/A1_ASSESSMENT.md',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='CC_Compare/results/a1_compare_table/dataset.json')
    args = ap.parse_args()

    path = args.data if os.path.isabs(args.data) else os.path.join(ROOT, *args.data.split('/'))
    if not os.path.isfile(path):
        print('FAIL: dataset 不存在: %s' % path)
        return 1
    with open(path, encoding='utf-8') as f:
        d = json.load(f)

    arm_methods = set()
    for a in d.get('arms') or []:
        if a.get('method'):
            arm_methods.add(a['method'])
    unadapted = [u.get('method') for u in (d.get('incomparable_or_unadapted') or [])]
    unadapted_set = set(m for m in unadapted if m)
    baseline_keys = set((d.get('baseline_arms') or {}).keys())

    fails = []
    warns = []

    # ---- A. 正向覆盖：每个方法目录都要有归宿 ----
    dirs = sorted(n for n in os.listdir(CC)
                  if os.path.isdir(os.path.join(CC, n)) and n not in INFRA_DIRS)
    unknown = [n for n in dirs if n not in DIR_HOME]
    if unknown:
        fails.append('A: CC_Compare 下有未登记归宿的方法目录（可能被漏审计）: %s'
                     % ', '.join(unknown))
    for n in dirs:
        if n not in DIR_HOME:
            continue
        home = DIR_HOME[n]
        if home == []:
            if n not in unadapted_set:
                fails.append('A: 目录 %s 归「未适配」，但清单里没有它' % n)
        elif n == 'internal':
            miss = [k for k in home if not any(k in bk for bk in baseline_keys)]
            if miss:
                fails.append('D: baseline_arms 缺内部臂 %s' % ', '.join(miss))
        else:
            miss = [k for k in home if k not in arm_methods]
            if miss:
                fails.append('A: 目录 %s 的方法 key 无结果行: %s' % (n, ', '.join(miss)))

    # ---- B. 不自相矛盾 ----
    both = sorted(arm_methods & unadapted_set)
    if both:
        fails.append('B: 同时出现在结果行与「未适配」表（自相矛盾）: %s' % ', '.join(both))

    # ---- C. 19 个预期 key 全部出现 ----
    seen = arm_methods | unadapted_set
    missing = [k for k in EXPECTED_KEYS if k not in seen]
    if missing:
        fails.append('C: 预期方法 key 未出现在交付物中: %s' % ', '.join(missing))

    # ---- 附带：未适配项的依据文档必须存在 ----
    for m, rel in UNADAPTED_DOC.items():
        if m in unadapted_set and not os.path.isfile(os.path.join(ROOT, *rel.split('/'))):
            fails.append('依据文档缺失: %s' % rel)
        if m in unadapted_set:
            u = [x for x in d['incomparable_or_unadapted'] if x.get('method') == m][0]
            if not (u.get('reason') or '').strip():
                fails.append('未适配项 %s 未写具体原因' % m)

    # ---- E. 需配对身份的正式臂必须落盘 identity_verified（规则臂豁免） ----
    formal = [a for a in (d.get('arms') or [])
              if a.get('formal') and a.get('group') == 'a1_ext_formal2']
    req = [a for a in formal if a.get('method') not in RULE_METHODS]
    exempt = [a for a in formal if a.get('method') in RULE_METHODS]
    noid = [a.get('arm_dir') for a in req
            if ((a.get('pairing_identity_recorded') or {})
                .get('identity_verified') is not True)]
    if noid:
        fails.append('E: 需配对身份的正式臂缺 identity_verified（学习/求解器臂不允许缺）: %s'
                     % ', '.join(str(x) for x in noid[:8])
                     + (' …' if len(noid) > 8 else ''))

    # ---- F. 学习型方法必须有落盘的参与度证据（含专用 C2） ----
    partdir = os.path.join(CC, 'results', 'a1_compare_table', 'participation')
    learned = [k for k in EXPECTED_KEYS if k not in RULE_METHODS and k not in SOLVERS]
    bases = sorted({PARTICIPATION_ALIAS.get(k, k) for k in learned})
    missing_part = [b for b in bases
                    if not os.path.isfile(os.path.join(partdir, b + '.json'))]
    if missing_part:
        fails.append('F: 学习型方法缺 participation/<name>.json 落盘证据（台账里写「强」不算）: %s'
                     % ', '.join(missing_part))
    missing_c2 = [f for b, f in sorted(EXTRA_C2.items())
                  if os.path.isfile(os.path.join(partdir, b + '.json'))
                  and not os.path.isfile(os.path.join(partdir, f))]
    if missing_c2:
        fails.append('F: 通用 C2 为 NOT_SUPPORTED 的方法缺**专用** C2 产物: %s'
                     % ', '.join(missing_c2))

    # ---- 报告 ----
    print('方法目录 %d 个（非基础设施）| 结果行 method key %d 个 | 未适配 %d 项 | 内部臂 %d 个'
          % (len(dirs), len(arm_methods), len(unadapted_set), len(baseline_keys)))
    print('  结果行: %s' % ', '.join(sorted(arm_methods)))
    print('  未适配: %s' % ', '.join(sorted(unadapted_set)))
    print('  正式臂(需配对身份) %d 个 → 缺 identity_verified %d 个；纯规则臂豁免 %d 个'
          % (len(req), len(noid), len(exempt)))
    print('  学习型方法参与度产物 %d/%d 份在位%s'
          % (len(bases) - len(missing_part), len(bases),
             '；专用 C2 齐备' if not missing_c2 else ''))
    for w in warns:
        print('  WARN %s' % w)
    if fails:
        print('FAIL 共 %d 条:' % len(fails))
        for x in fails:
            print('  - %s' % x)
        return 1
    print('PASS 方法覆盖完整（无静默丢弃、无自相矛盾、19 key 齐、内部三臂在位、'
          '学习型参与度产物齐备）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
