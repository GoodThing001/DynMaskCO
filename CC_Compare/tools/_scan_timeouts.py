"""只读扫描：列出所有 v2 正式臂的**必须随数字一起报告的量**（超时/天、最大单决策）。

动机：2026-10-13 发现 `symnco_51015` 因 CPU 争用把单次 order() 推高 14×、产生 9 天超时
（见 `读表注意与已知限定.md` §2.8）。单臂异常已记录，但**是否还有别的臂也被争用推高了**？
本脚本把全部正式臂的 `mean_timeouts` / `max_decide_s` / `fallback%` 一次列出来，
便于在收口前判断「哪些行的数字必须带超时限定」。

用法（1003 扩展根下）：
    python ../CC_Compare/tools/_scan_timeouts.py            # 扫 results/a1_ext_formal2
"""
import glob
import json
import os
import sys

ROOT = 'results/a1_ext_formal2'
BASELINES = {'cond_hist', 'uncond_hist', 'explicit_feat'}


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, '*', 'gate.json'))):
        arm_dir = os.path.basename(os.path.dirname(p))
        try:
            g = json.load(open(p, encoding='utf-8'))
        except Exception as e:                                   # noqa: BLE001
            print('  %-22s **坏 gate.json**: %s' % (arm_dir, e))
            continue
        res = g.get('results') or g.get('gate') or {}
        for y, blk in res.items():
            if y.startswith('_') or not isinstance(blk, dict):
                continue
            for pc, b2 in blk.items():
                if not isinstance(b2, dict):
                    continue
                arms = b2.get('arms') or {}
                sm = b2.get('solver_meta') or {}
                for an, ab in arms.items():
                    if an in BASELINES:
                        continue
                    u = (ab.get('utility_all_days') or {}).get('mean')
                    rows.append((arm_dir, pc, an, u, ab.get('mean_timeouts'),
                                 sm.get('max_decide_s'), ab.get('hard_feasible_rate'),
                                 ab.get('fail_counts') or {}))
    print('%-22s %-16s %-16s %9s %7s %11s %5s %s' %
          ('arm_dir', 'tier', 'arm', 'util', 'to/day', 'max_decide_s', 'hfr', 'fail'))
    if not rows:
        # 2026-10-13 加固：**「一个都没扫到」绝不能报成「0 个需要限定」**。
        # 起因：本脚本的 ROOT 是**相对路径** `results/a1_ext_formal2`，在仓库根下不存在
        # （真实路径在扩展根下）⇒ 本地误跑时它会打印空表 + 「0 个」，看起来像**全部干净**。
        # 这种「空扫描 = 假全清」必须响亮失败。
        print('  **未找到任何 gate.json（ROOT=%s，cwd=%s）**' % (ROOT, os.getcwd()))
        print('  ⇒ 请确认在**扩展根**下运行：cd C-VRP_Cold-chainVehicleRoutingProblem && '
              'python ../CC_Compare/tools/_scan_timeouts.py')
        print('== 扫描无效（0 个 gate），**不得**据此认为「没有臂需要超时限定」==')
        return 2
    worst = []
    for r in sorted(rows, key=lambda x: -(x[4] or 0)):
        print('%-22s %-16s %-16s %9s %7s %11s %5s %s' %
              (r[0], r[1], r[2], ('%.1f' % r[3]) if r[3] is not None else 'n/a',
               r[4], ('%.2f' % r[5]) if isinstance(r[5], (int, float)) else '-',
               r[6], r[7] if r[7] else ''))
        if (r[4] or 0) > 0 or (isinstance(r[5], (int, float)) and r[5] > 10.0):
            worst.append(r)
    print()
    print('== 需要带超时/超预算限定的臂：%d 个 ==' % len(worst))
    for r in worst:
        print('   %-22s %-16s to=%-6s max_decide_s=%s' %
              (r[0], r[1], r[4], ('%.2f' % r[5]) if isinstance(r[5], (int, float)) else '-'))


if __name__ == '__main__':
    sys.exit(main())
