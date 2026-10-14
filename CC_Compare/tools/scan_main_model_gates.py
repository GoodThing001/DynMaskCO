"""只读扫描：找出最近产生、且含主模型臂（maskco 等）的 40 天在线 gate。

用途：让总表的 Ours 行跟随用户主模型线的最新正式臂（而不是停在 10-09 的 I3 gate）。
用法（在 C-VRP_Cold-chainVehicleRoutingProblem 下）：
    python ../CC_Compare/tools/scan_main_model_gates.py [--since 2026-10-09] [--days 40]
"""
import argparse
import json
import os
import sys
import time

EXT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..',
                                   'C-VRP_Cold-chainVehicleRoutingProblem'))
MAIN_ARMS = ('maskco', 'reconstruction_control', 'warmstart', 'a1_rollout',
             'a1_consensus', 'cond_hist')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2026-10-09')
    ap.add_argument('--days', type=int, default=40)
    args = ap.parse_args()

    cutoff = time.mktime(time.strptime(args.since, '%Y-%m-%d'))
    hits = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(EXT, 'results')):
        if 'gate.json' not in filenames:
            continue
        p = os.path.join(dirpath, 'gate.json')
        try:
            if os.path.getmtime(p) < cutoff:
                continue
            d = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        cont = d.get('results') or d.get('gate') or {}
        rows = []
        for k, v in cont.items():
            if not isinstance(v, dict):
                continue
            if 'arms' in v:
                blocks = [(k, v)]
            else:
                blocks = [(k2, v2) for k2, v2 in v.items() if isinstance(v2, dict)]
            for pname, blk in blocks:
                for an, ab in (blk.get('arms') or {}).items():
                    if an not in MAIN_ARMS:
                        continue
                    u = ab.get('utility_all_days') or {}
                    if ab.get('n_days') != args.days:
                        continue
                    rows.append((pname, an, u.get('mean'), ab.get('mean_timeouts')))
        if rows:
            hits.append((os.path.relpath(p, EXT), os.path.getmtime(p), rows))

    hits.sort(key=lambda x: -x[1])
    print('== %d 个含主模型臂的 %d 天 gate（since %s）==' % (len(hits), args.days, args.since))
    for rel, mt, rows in hits:
        print('\n%s   (mtime %s)' % (rel, time.strftime('%Y-%m-%d %H:%M', time.localtime(mt))))
        for pname, an, mean, to in sorted(rows):
            print('   %-18s %-24s util=%-10s timeouts=%s' % (
                pname, an, ('%.2f' % mean) if isinstance(mean, float) else mean, to))


if __name__ == '__main__':
    main()
