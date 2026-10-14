#!/usr/bin/env python
"""Read-only probe: what does each gate record in `seeds`, and what does the
directory name say?  Used only to explain the part_d CI-reproduction boundary.
"""
import glob
import json
import os
import re

ROOT = 'results'


def main():
    pats = [
        'a1_a3_online_density_B_*/gate.json',
        'a1_i3_online_*/gate.json',
        '*online_baseline*_B_*/gate.json',
    ]
    seen = set()
    rows = []
    for pat in pats:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            if p in seen:
                continue
            seen.add(p)
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    d = json.load(f)
            except Exception as exc:  # pragma: no cover
                rows.append((p, 'READ_ERROR', str(exc)))
                continue
            name = os.path.dirname(p)
            m = re.search(r'_(20\d{6})_', name)
            date_in_name = m.group(1) if m else None
            seeds = d.get('seeds')
            gen = d.get('generated_from') if isinstance(d.get('generated_from'), dict) else {}
            rows.append((name, date_in_name, seeds,
                         None if not gen else gen.get('seed'),
                         sorted(d.keys())))
    for r in rows:
        print(r[0])
        print('   date_in_name=%s  seeds=%s  generated_from.seed=%s' % (r[1], r[2], r[3]))
        print('   top_keys=%s' % (r[4],))
    print('total gates: %d' % len(rows))


if __name__ == '__main__':
    main()
