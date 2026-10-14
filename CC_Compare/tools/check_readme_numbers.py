"""核对 README §2「主结果」里引用的数字是否与 dataset.json 一致。

用途：防止文档与数据悄悄漂移（我改了很多次文档）。
用法：python CC_Compare/tools/check_readme_numbers.py
"""
import json
import os
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = r"CC_Compare\results\a1_compare_table"
data = json.load(open(os.path.join(BASE, "dataset.json"), encoding="utf-8"))
readme = open(os.path.join(BASE, "README.md"), encoding="utf-8").read()

# 从 dataset 取「终版」臂，按方法挑每组最优（与渲染器同规则）
RANK = {'a1_ext_formal2': 0, 'a1_ext_a05e': 1, 'a1_ext_extra': 2}
per = {}
for r in data['arms']:
    if not r.get('formal'):
        continue
    per.setdefault(r['method'], {}).setdefault(r['p_c'], []).append(r)


def pick(method, pc):
    c = per.get(method, {}).get(pc) or []
    if not c:
        return None
    return sorted(c, key=lambda x: RANK.get(x['group'], 9))[0]


def u(rec):
    return ((rec or {}).get('reported') or {}).get('utility') or {}


b = data['baseline_arms']
checks = [
    ("C1 cond_hist p_c=0", b['p_c=0|cond_hist']['utility']['mean'], 1703.7),
    ("C1 cond_hist main", b['p_c=(5,10,15)|cond_hist']['utility']['mean'], 631.4),
    ("RouteFinder p_c=0", u(pick('routefinder', 'p_c=0'))['mean'], 1266.2),
    ("RouteFinder main", u(pick('routefinder', 'p_c=(5,10,15)'))['mean'], -26.1),
    ("OR-Tools p_c=0", u(pick('ortools', 'p_c=0'))['mean'], 1300.7),
    ("OR-Tools main", u(pick('ortools', 'p_c=(5,10,15)'))['mean'], 20.9),
    ("PyVRP p_c=0", u(pick('pyvrp', 'p_c=0'))['mean'], 909.2),
    ("rule l2 p_c=0", u(pick('valuee20', 'p_c=0'))['mean'], 1403.3),
    ("rule l2 main", u(pick('valuee20', 'p_c=(5,10,15)'))['mean'], 180.6),
    ("CaDA p_c=0", u(pick('cada', 'p_c=0'))['mean'], 1157.0),
    ("MVMoE p_c=0", u(pick('mvmoe', 'p_c=0'))['mean'], 1257.7),
    # ---- 2026-10-13 扩充：把 §2 主表**剩下的**数字也纳入机器核对 ----
    # 动机：§2 是有约 30 个数字的头条表，而原先只核了 12 个 ⇒ 其余（myopic/POMO/SGBS/
    # Attention/RRNCO/Sym-NCO/Omni-VRP/DeepACO/LIH/λ3 的各档值）**可以静默变旧**。
    # 一句话：文档里的数字要么被核，要么就会漂。
    ("rule myopic p_c=0", u(pick('myopic', 'p_c=0'))['mean'], 1305.2),
    ("rule myopic main", u(pick('myopic', 'p_c=(5,10,15)'))['mean'], 30.5),
    ("rule l3 p_c=0", u(pick('valuee30', 'p_c=0'))['mean'], 0.0),
    ("rule l3 main", u(pick('valuee30', 'p_c=(5,10,15)'))['mean'], -1943.1),
    ("POMO p_c=0", u(pick('pomo', 'p_c=0'))['mean'], 1238.8),
    ("POMO main", u(pick('pomo', 'p_c=(5,10,15)'))['mean'], -68.1),
    # ---- 2026-10-14：v2 正式批次定稿，下列 8 个数字按新数据更新 ----
    # ⚠️ 这是**有意的口径更新**，不是"让门禁变绿"：v2 换了新 C1 基线 `a1_c1fresh_reveal_40`，
    # 且其中 3 个方法带 CPU 争用超时（下表标 †）。更新前先读 `TIMEOUT_SCAN.md` §2.8 / R10。
    #   SGBS †  （12.0/9.1 次超时每天）  1236.5→1266.5 / −72.5→−52.8
    #   Sym-NCO †（0.08/0.23）          1229.2→1229.3 / −82.6→−76.1
    #   LIH †   （25.4/**54.2**，最长单决策 50.79 s）1213.8→1215.0 / −105.0→**−335.1**（偏 v1 +219%）
    #   DeepACO（**无超时**）            1221.2→1213.6 / −113.4→−100.0
    ("SGBS p_c=0", u(pick('sgbs', 'p_c=0'))['mean'], 1266.5),
    ("SGBS main", u(pick('sgbs', 'p_c=(5,10,15)'))['mean'], -52.8),
    ("AttentionModel p_c=0", u(pick('attention', 'p_c=0'))['mean'], 1236.2),
    ("AttentionModel main", u(pick('attention', 'p_c=(5,10,15)'))['mean'], -72.9),
    ("RRNCO p_c=0", u(pick('rrnco', 'p_c=0'))['mean'], 1235.4),
    ("RRNCO main", u(pick('rrnco', 'p_c=(5,10,15)'))['mean'], -73.2),
    # ---- 2026-10-14 新增：本批新进表的两个方法（CO-enriched-ML 两模式 + MAPT）----
    ("CO-enriched nopchgs p_c=0", u(pick('coenrich_nopchgs', 'p_c=0'))['mean'], 1234.5),
    ("CO-enriched nopchgs main", u(pick('coenrich_nopchgs', 'p_c=(5,10,15)'))['mean'], -74.3),
    ("CO-enriched pchgs p_c=0", u(pick('coenrich_pchgs', 'p_c=0'))['mean'], 1230.5),
    ("CO-enriched pchgs main", u(pick('coenrich_pchgs', 'p_c=(5,10,15)'))['mean'], -80.5),
    ("MAPT p_c=0", u(pick('mapt', 'p_c=0'))['mean'], 1215.4),
    ("MAPT main", u(pick('mapt', 'p_c=(5,10,15)'))['mean'], -102.2),
    ("Sym-NCO p_c=0", u(pick('symnco', 'p_c=0'))['mean'], 1229.3),
    ("Sym-NCO main", u(pick('symnco', 'p_c=(5,10,15)'))['mean'], -76.1),
    ("Omni-VRP p_c=0", u(pick('omnivrp', 'p_c=0'))['mean'], 1226.9),
    ("Omni-VRP main", u(pick('omnivrp', 'p_c=(5,10,15)'))['mean'], -85.7),
    ("DeepACO p_c=0", u(pick('deepaco', 'p_c=0'))['mean'], 1213.6),
    ("DeepACO main", u(pick('deepaco', 'p_c=(5,10,15)'))['mean'], -100.0),
    ("LIH p_c=0", u(pick('lih', 'p_c=0'))['mean'], 1215.0),
    ("LIH main", u(pick('lih', 'p_c=(5,10,15)'))['mean'], -335.1),
    ("CaDA main", u(pick('cada', 'p_c=(5,10,15)'))['mean'], -190.3),
    ("MVMoE main", u(pick('mvmoe', 'p_c=(5,10,15)'))['mean'], -38.6),
    ("PyVRP main", u(pick('pyvrp', 'p_c=(5,10,15)'))['mean'], -569.8),
]
bad = 0
print('%-24s %12s %12s %s' % ('条目', '数据', '§2 期望', '一致'))
for name, val, cited in checks:
    ok = val is not None and abs(val - cited) <= 0.06
    bad += 0 if ok else 1
    print('%-24s %12.2f %12.2f %s' % (name, val if val is not None else float('nan'),
                                      cited, 'OK' if ok else 'MISMATCH'))

# Ours 行
mm = [r for r in data.get('main_model', [])
      if 'maskco' == r['arm'] and r['p_c'] == 'p_c=0'
      and 'pretrained' in str((r.get('config') or {}).get('model_bin'))]
if mm:
    m = (mm[0].get('reported_utility') or {}).get('mean')
    ok = m is not None and abs(m - 1577.9) <= 0.06
    bad += 0 if ok else 1
    print('%-24s %12.2f %12.2f %s' % ('Ours MaskCO A3-B p_c=0', m, 1577.9,
                                      'OK' if ok else 'MISMATCH'))

print('\n不一致条目:', bad)
for pat in ('1577.9', '1266.2', '1703.7', '1403.3', '909.2'):
    print('README 是否出现 %s: %s' % (pat, bool(re.search(re.escape(pat), readme))))
sys.exit(1 if bad else 0)
