"""检查总表主行「每个方法每一档」实际选中的臂来自哪个批次目录。

动机：`render_compare_table.build_rows` 用 GROUP_RANK 优先 `a1_ext_formal2`（v2 正式批次），
但**不保证**每一档都命中 v2。若某方法的 v2 臂缺失（例如 3080 GPU 腿尚未回传），它会静默
回落到 `a1_ext_extra`（v1 批次）。这一点在本次改动里是**实质风险**：v1 的 CaDA/MVMoE 是
`--provider-device cpu` 口径（CaDA 每天超时 30.4/130.95 次），v2 改为 cuda，两者不可混用。
本检查把这种「回落」显式报出来，作为收口前的一道门禁。

用法：python CC_Compare/tools/check_arm_provenance.py [--data <dataset.json>]
      （`--data` 供**收口前排演**用：指向临时渲染出的 dataset，避免为了试跑而覆盖交付物；
        省略时读交付目录的 dataset.json，与 `_local_finalize.ps1` 的行为一致。）
退出码：0 = 每档都命中 v2；1 = 存在回落（或 v2 臂缺失）。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, '..', 'results', 'a1_compare_table')
GROUP_RANK = {'a1_ext_formal2': 0, 'a1_ext_a05e': 1, 'a1_ext_extra': 2}
TIERS = ('p_c=0', 'p_c=(5,10,15)', 'p_c=(10,20,30)')

# 两个都算「正式批次」：a1_ext_formal2 = v2 全量重跑；a1_ext_a05e = OR-Tools/PyVRP 的正式臂。
FORMAL_GROUPS = {'a1_ext_formal2', 'a1_ext_a05e'}

# 有意只保留 v1 臂（a1_ext_extra）的方法。写进这里 = 已知并接受，理由必须写清楚。
# rule_* 是纯 NumPy 确定性规则，与 provider-device 无关，故设备口径修正不影响它们。
ALLOW_V1_ONLY = {
    'myopic': 'rule_* 纯 NumPy 规则，无 provider-device 概念',
    'valuee20': 'rule_* 纯 NumPy 规则，无 provider-device 概念',
    'valuee30': 'rule_* 纯 NumPy 规则，无 provider-device 概念',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(TAB, 'dataset.json'))
    args = ap.parse_args()
    path = args.data
    if not os.path.isabs(path):
        # 相对路径按**仓库根**解析（`HERE` 是 CC_Compare/tools，要上跳两层）。
        # 2026-10-13 修复：原先只上跳一层 → `CC_Compare/CC_Compare/results/...`，
        # 于是排演时 `--data CC_Compare/results/...` 直接报 MISSING（首轮排演实测踩到）。
        path = os.path.join(os.path.dirname(os.path.dirname(HERE)), path)
    if not os.path.exists(path):
        print('MISSING %s（先跑 collect_compare_table.py）' % path)
        return 1
    data = json.load(open(path, encoding='utf-8'))

    by_method = {}
    for r in data['arms']:
        if not r.get('formal'):
            continue
        by_method.setdefault(r['method'], {}).setdefault(r['p_c'], []).append(r)

    bad = []
    for method, per in sorted(by_method.items()):
        cells = []
        for pc in TIERS:
            cands = per.get(pc) or []
            if not cands:
                cells.append('%-14s=-' % pc)
                continue
            best = sorted(cands, key=lambda c: (GROUP_RANK.get(c['group'], 9),
                                                not c.get('formal')))[0]
            cells.append('%-14s=%s/%s' % (pc, best['group'], best['arm_dir']))
            if best['group'] not in FORMAL_GROUPS and method not in ALLOW_V1_ONLY:
                bad.append((method, pc, best['group'], best['arm_dir']))
        print('%-22s %s' % (method, '  '.join(cells)))

    print()
    if bad:
        print('FAIL: %d 档仍落在 v1 批次 a1_ext_extra（总表会混入旧设备/旧口径数字）：' % len(bad))
        for m, pc, g, d in bad:
            print('  - %-22s %-14s -> %s/%s' % (m, pc, g, d))
        print()
        print('  若确属有意保留，把方法名加入 ALLOW_V1_ONLY 并写明理由。')
        return 1
    print('PASS: 每档都命中正式批次（a1_ext_formal2 / a1_ext_a05e），或属于已声明的 v1-only 规则臂')
    print('      已声明 v1-only：%s' % (', '.join(sorted(ALLOW_V1_ONLY)) or '（无）'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
