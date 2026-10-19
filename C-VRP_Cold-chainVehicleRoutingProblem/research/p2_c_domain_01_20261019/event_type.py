"""P2-C-DOMAIN-01: event-type census of the three replan regimes (READ-ONLY).

Question it answers: are R15/R14 reveal events or not? The test is purely artifact-based:
a phase is a reveal event iff its clock equals some order's reveal time in the saved
public_view. No engine call, no controller, no rollout.

Usage: python event_type.py --out event_type_census.json    (refuses to overwrite)
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARMS = {'SAA': ROOT / 'research/p2_c_opp_01_20261019' / 'phases.json',
        'MYOPIC': ROOT / 'research/p2_c_ctrl_01_20261019' / 'phases.json'}


def regime_of(pv):
    n_replan = len(set(pv['replan_ids'] or ()))
    n_committed = sum(1 for v in pv['vehicles'] if v['status'] == 'committed')
    return n_replan, n_committed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'event_type_census.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('refuse overwrite')

    payload = {'schema': 'p2_c_domain_01_event_type_census_v1',
               'reveal_test': 'clock == some order.reveal (1e-6) in the saved public_view',
               'per_arm': {}}
    for arm, path in ARMS.items():
        phases = json.loads(path.read_text(encoding='utf-8'))
        cross = defaultdict(Counter)
        multi_order_pools = Counter()
        reveal_phases = 0
        reveal_pool_ge2 = 0
        for ph in phases:
            pv = ph['public_view']
            clock = float(ph['clock'])
            n_replan, n_committed = regime_of(pv)
            is_reveal = any(abs(float(o['reveal']) - clock) < 1e-6
                            for o in pv['orders'] if float(o['reveal']) > 0)
            tails = sum(1 for v in pv['vehicles'] if (v['mutable_suffix'] or ()))
            key = '|replan|=%d,committed=%d' % (n_replan, n_committed)
            cross[key][('%s/%s' % ('reveal' if is_reveal else 'nonreveal',
                                   'tails>0' if tails else 'tails=0'))] += 1
            # pool = pending orders on replannable vehicles (same predicate as audit.py)
            from_types = None  # placeholder to keep the import surface minimal
            allowed = {int(v['vehicle_id']) for v in pv['vehicles']
                       if v['status'] in ('idle', 'ready')
                       and int(v['vehicle_id']) in set(int(x) for x in (pv['replan_ids'] or ()))
                       and int(v['vehicle_id']) not in set(int(x) for x in (pv['protected'] or ()))}
            accepted = {int(x) for x in pv['accepted']}
            served = {int(x) for x in pv['served']}
            pool = sum(1 for v in pv['vehicles'] if int(v['vehicle_id']) in allowed
                       for c in (v['mutable_suffix'] or ())
                       if int(c) in accepted and int(c) not in served)
            if is_reveal:
                reveal_phases += 1
                if pool >= 2:
                    reveal_pool_ge2 += 1
            if pool >= 2:
                multi_order_pools['reveal' if is_reveal else 'nonreveal'] += 1
        payload['per_arm'][arm] = {
            'phases': len(phases),
            'regime_x_eventtype': {k: dict(v) for k, v in sorted(cross.items())},
            'reveal_phases': reveal_phases,
            'reveal_phases_with_pool_ge2': reveal_pool_ge2,
            'phases_with_pool_ge2_by_eventtype': dict(multi_order_pools),
        }
        print(arm, 'phases', len(phases), 'reveal', reveal_phases,
              'reveal_with_pool>=2', reveal_pool_ge2,
              'pool>=2 by eventtype', dict(multi_order_pools))

    a.out.open('x', encoding='utf-8', newline='\n').write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print('WROTE', a.out.name)


if __name__ == '__main__':
    main()
