# -*- coding: utf-8 -*-
"""修订版两批的身份与超时分布核查。"""
import json
import sys

import numpy as np


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path, encoding='utf-8'))
        i = d.get('identity') or {}
        print('==', path.split('/')[-2])
        print('  identity keys:', sorted(i.keys()))
        print('  source_stable:', i.get('source_stable'), 'contract:',
              (str(i.get('contract_sha256'))[:12] if i.get('contract_sha256') else 'MISSING'))
        print('  shared_config:', i.get('shared_config'))
        for pn in ('p_c=(5,10,15)',):
            g = d['gate'][pn]
            for arm in ('cond_hist', 'uncond_hist'):
                rows = g['per_day'][arm]
                tos = [r['timeouts'] for r in rows]
                print('  %s %s: timeout days=%d max=%d mean=%.1f served=%.1f'
                      % (pn, arm, sum(1 for t in tos if t > 0), max(tos),
                         float(np.mean(tos)),
                         float(np.mean([r['n_served'] for r in rows]))))
            c = g['per_day']['cond_hist']
            u = g['per_day']['uncond_hist']
            dto = [c[k]['timeouts'] - u[k]['timeouts'] for k in range(len(c))]
            print('  cond-uncond timeout diff: mean=%.1f (cond 超时比 uncond 多的部分)'
                  % float(np.mean(dto)))


if __name__ == '__main__':
    main()
