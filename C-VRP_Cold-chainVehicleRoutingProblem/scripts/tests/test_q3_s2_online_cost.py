#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 S2 CONTROL ROWS -- online cost of the S1 route bracket (DIAGNOSTIC, not production).

Rows S2-1..S2-7 and S2-N1, frozen in
`scripts/q3_qualification/q3_s1_s2_preregistration_2026-10-17.md` E4.2 before this file existed.

The cost question is answered three ways and all three are checked:
  * a CLOSED FORM derived from the plan shape (so a campaign can be sized before it runs);
  * an EXTERNAL count obtained by patching the enclosure functions in the modules the certificate
    looks them up in (so the count is measured on the running code, not self-reported);
  * an ISOLATED wall clock against the pre-declared 40 ms ceiling.

Usage: python scripts/tests/test_q3_s2_online_cost.py --out <json path>
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts/coldchain'))
sys.path.insert(0, str(EXT / 'scripts/simulation'))

from coldchain_contract import ColdChainContractV2                       # noqa: E402
import q3_s1_route_bracket_experimental as S1                            # noqa: E402
import q3_s2_online_cost_experimental as S2                              # noqa: E402

CONTRACT = ColdChainContractV2()
ZTH, N_LN_CONTRACT = S1.zone_thresholds_exact(CONTRACT)
N_ZONES = len(ZTH)
TAUS = tuple(float(ZTH[z]['tau']) for z in range(N_ZONES))
THETA_HI = tuple(float(ZTH[z]['theta_hi']) for z in range(N_ZONES))
ROWS = []
SEED = 20261017
LIMITS = ('synthetic instances with a fixed seed, one machine, no GPU; the wall clock is a LOCAL '
          'diagnostic and never a cross-platform guarantee; a passing row is a DIAGNOSTIC and never '
          'a production certificate')


def ck(cid, name, ok, detail=None, conclusion='', limits=LIMITS):
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'detail': detail or {},
                 'conclusion': conclusion, 'limits': limits})
    print('[%s] %-6s %s' % ('PASS' if ok else 'FAIL', cid, name[:96]))


def make_instance(*, n_nodes, seed, service, cluster):
    """Deterministic synthetic instance (same generator as the S1 controls, different seeds)."""
    import random
    rng = random.Random(seed)
    coords = [(rng.uniform(0.0, cluster), rng.uniform(0.0, cluster)) for _ in range(n_nodes)]
    coords[0] = (0.0, 0.0)
    svc = [0.0] + [round(rng.uniform(service[0], service[1]), 6) for _ in range(n_nodes - 1)]
    cm = np.asarray(coords, dtype=np.float32)
    d = np.sqrt(((cm[:, None, :] - cm[None, :, :]) ** 2).sum(axis=-1)).astype(np.float32)
    return {'service_time': np.asarray(svc, dtype=np.float32),
            'tw_start': np.zeros(n_nodes, dtype=np.float32),
            'temp_class': np.asarray([0] + [rng.randrange(N_ZONES) for _ in range(n_nodes - 1)],
                                     dtype=np.int32),
            'reveal_time': np.zeros(n_nodes, dtype=np.float32),
            'dist_mat': d, 'demands': np.asarray([0] + [1] * (n_nodes - 1), dtype=np.float32),
            'tw_speed': 1.0}


def fresh_entry():
    return (tuple(TAUS), tuple(TAUS))


def run_plan(inst, plans, entry, *, uncached=False, clock=0.0):
    """`uncached=True` recomputes the per-contract Theta thresholds on every call."""
    kw = {} if uncached else {'thresholds': ZTH}
    return S1.check_route_certificate(
        contract=CONTRACT, contract_identity=CONTRACT.contract_hash, instance=inst, clock=clock,
        plans_by_vehicle=plans, entry_state_by_vehicle=entry,
        committed_order_ids=sum_suffix(plans), state_identity='S',
        expected_state_identity='S', source_identity='SYNTH', **kw)


def sum_suffix(plans):
    out = set()
    for rec in plans.values():
        for n in rec.get('suffix') or ():
            if int(n) != 0:
                out.add(int(n))
    return out


def fleet(inst, *, n_vehicles, per_vehicle, n_nodes):
    """Split `n_vehicles * per_vehicle` consecutive nodes into routes ending with the depot."""
    plans, nxt = {}, 1
    for v in range(n_vehicles):
        stops = []
        for _ in range(per_vehicle):
            if nxt >= n_nodes:
                break
            stops.append(nxt)
            nxt += 1
        plans[v] = {'current_node': 0, 'ready_time': 0.0, 'prefix': [], 'suffix': stops + [0]}
    return plans


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(EXT / 'results' / 'q3_s2_online_cost' / 'controls.json'))
    args = ap.parse_args(argv)

    import hashlib
    src = {}
    for rel in ('scripts/coldchain/q3_s1_route_bracket_experimental.py',
                'scripts/coldchain/q3_s2_online_cost_experimental.py',
                'scripts/coldchain/q3_strict_float_experimental.py',
                'scripts/coldchain/coldchain_state.py',
                'scripts/coldchain/coldchain_contract.py',
                'scripts/tests/test_q3_s2_online_cost.py'):
        src[rel] = hashlib.sha256((EXT / rel).read_bytes()).hexdigest()

    inst_long = make_instance(n_nodes=41, seed=SEED, service=(0.15, 0.60), cluster=10.0)
    inst_tight = make_instance(n_nodes=41, seed=SEED + 2, service=(0.004, 0.010), cluster=0.02)

    # ============================================================ S2-1 closed form == measured
    cases, mismatches, totals = 100, [], {'exp': 0, 'ln': 0, 'rows': 0}
    n_closed_ok = 0
    for i in range(cases):
        inst = inst_long if i % 2 == 0 else inst_tight
        n_stops = 2 + (i % 11)
        n_veh = 1 + (i % 3)
        plans = fleet(inst, n_vehicles=n_veh, per_vehicle=n_stops, n_nodes=41)
        if not any(p['suffix'][:-1] for p in plans.values()):
            continue
        with S2.count_transcendentals() as cnt:
            rec = run_plan(inst, plans, {v: fresh_entry() for v in plans})
        shape = S2.plan_shape(rec)
        cf = S2.closed_form(n_doors=shape['n_doors'], n_short_gaps=shape['n_short_gaps'],
                            n_vehicles=shape['n_vehicles'], n_zones=shape['n_zones'])
        # INDEPENDENT collapse predicate, recomputed here from the recorded per-gap entry state and
        # the contract thresholds: a gap costs an enclosure iff the zone is above the floor AND is
        # not provably returned by the cached threshold.  This checks that the ENACTED eval count
        # equals the SPECIFIED predicate; the specification itself is S1's lemma.
        static_short = 0
        for z in range(N_ZONES):
            for prop in ((rec['per_zone_detail'].get(str(z)) or {}).get('per_vehicle')
                         or {}).values():
                for g in prop.get('gaps') or []:
                    hi_prev = F(g['entry_upper'])
                    if hi_prev > ZTH[z]['tau'] and not (hi_prev <= ZTH[z]['band_high']
                                                        and F(g['delta']) >= ZTH[z]['theta_hi']):
                        static_short += 1
        totals['exp'] += cnt['exp']
        totals['ln'] += cnt['ln']
        totals['rows'] += 1
        ok = (cnt['exp'] == cf['n_exp_exact'] == shape['n_exp_reported']
              and cnt['ln'] == 0 and static_short == cf['n_exp_exact'])
        if ok:
            n_closed_ok += 1
        else:
            mismatches.append({'case': i, 'measured_exp': cnt['exp'], 'measured_ln': cnt['ln'],
                               'closed_form_exp': cf['n_exp_exact'],
                               'reported_exp': shape['n_exp_reported'],
                               'static_short': static_short, 'shape': shape})
    ck('S2-1', 'the EXTERNAL count of rigorous `exp` enclosures equals the closed form and the '
       'certificate\u2019s own report on 100 plans (mixed long-gap and short-gap fleets), and an '
       'INDEPENDENT static prediction derived only from the gap list and the Theta thresholds '
       'agrees too; the cached-threshold path evaluates ZERO `ln` enclosures',
       n_closed_ok == totals['rows'] and totals['rows'] >= 90 and not mismatches,
       {'n_cases': totals['rows'], 'n_agreeing': n_closed_ok,
        'measured_exp_total': totals['exp'], 'measured_ln_total': totals['ln'],
        'mismatches': mismatches[:6],
        'unit_note': 'the counted object is a RIGOROUS TRANSCENDENTAL ENCLOSURE (exact-rational '
                     'series), not a `math.exp` call: neither the S1 decision path nor the ln '
                     'threshold path calls `math.exp` at all (see S2-3).  This is the operational '
                     'reading of the frozen S2-1 wording -- report ERRATA-1'},
       conclusion='the cost model is exact, not an estimate: the number of expensive evaluations is '
                  'a closed-form function of the plan shape')

    # ============================================================ S2-2 collapse path
    plans_long = fleet(inst_long, n_vehicles=8, per_vehicle=5, n_nodes=41)
    with S2.count_transcendentals() as cnt_c:
        rec_long = run_plan(inst_long, plans_long, {v: fresh_entry() for v in plans_long})
    with S2.count_transcendentals() as cnt_u:
        rec_long_u = run_plan(inst_long, plans_long, {v: fresh_entry() for v in plans_long},
                              uncached=True)
    sh_long = S2.plan_shape(rec_long)
    ck('S2-2', 'the COLLAPSE path costs nothing: on a fleet whose every gap exceeds the threshold the '
       'certificate evaluates zero `exp` enclosures, and the one-off Theta cost is exactly one `ln` '
       'enclosure per zone when the thresholds are computed and zero when they are reused',
       cnt_c['exp'] == 0 and sh_long['n_exp_reported'] == 0 and cnt_c['ln'] == 0
       and cnt_u['ln'] == N_LN_CONTRACT and N_LN_CONTRACT == N_ZONES
       and sh_long['n_short_gaps'] == 0 and (rec_long['verdict'] == S1.PROVED_CONDITIONAL),
       {'n_doors': sh_long['n_doors'], 'n_short_gaps': sh_long['n_short_gaps'],
        'measured_exp_cached': cnt_c['exp'], 'measured_ln_cached': cnt_c['ln'],
        'measured_ln_uncached': cnt_u['ln'], 'ln_per_zone': N_LN_CONTRACT,
        'verdict': rec_long['verdict']},
       conclusion='the ordinary certification is free; only a genuinely short gap costs a '
                  'transcendental evaluation')

    # ============================================================ S2-3 mechanical AST audit
    audit = S2.audit_decision_path(
        [EXT / 'scripts/coldchain/q3_s1_route_bracket_experimental.py',
         EXT / 'scripts/coldchain/q3_s2_online_cost_experimental.py'],
        sf_path=EXT / 'scripts/coldchain/q3_strict_float_experimental.py')
    ck('S2-3', 'the DECISION path calls NO libm transcendental at all: an AST scan of every function '
       'in the S1/S2 modules (and of the audited dependency functions) finds no math.exp/log/pow '
       'outside the two deliberately-excluded, non-bound-affecting uses',
       audit['ok'],
       {'findings': audit['findings'], 'allowlist': audit['allowlist'],
        'forbidden': audit['forbidden'], 'n_scanned': len(audit['scanned']),
        'reading': audit['reading']},
       conclusion='the bound does not rest on the platform libm, which is what makes this an R-level '
                  'certificate rather than a replay')

    # ============================================================ S2-4 exact rational decisions
    types = {}
    types['band_high'] = all(isinstance(ZTH[z]['band_high'], F) for z in range(N_ZONES))
    types['theta_hi'] = all(isinstance(ZTH[z]['theta_hi'], F) for z in range(N_ZONES))
    types['theta_lo'] = all(isinstance(ZTH[z]['theta_lo'], F) for z in range(N_ZONES))
    sup_ok, n_sup = True, 0
    for z in range(N_ZONES):
        for prop in ((rec_long['per_zone_detail'].get(str(z)) or {}).get('per_vehicle')
                     or {}).values():
            n_sup += 1
            sup_ok = sup_ok and isinstance(prop['sup_upper'], F)
    types['sup_upper_is_Fraction'] = sup_ok
    ck('S2-4', 'every operand of the verdict comparison is an exact rational (the band threshold, '
       'both Theta endpoints, and the propagated supremum): no float ever decides a verdict',
       all(types.values()),
       {'types': types, 'n_sup_operands': n_sup,
        'note': 'the FLOAT values in the receipt are for reading only; the comparison that sets '
                'band_ok uses Fractions'},
       conclusion='the decision is exact, so it cannot be flipped by a sub-ulp difference')

    # ============================================================ S2-5 budget gate
    entry_long = {v: fresh_entry() for v in plans_long}
    t_long = S2.isolated_timing(
        lambda: run_plan(inst_long, plans_long, entry_long),
        warmup=20, repeat=200)
    plans_tight = fleet(inst_tight, n_vehicles=8, per_vehicle=5, n_nodes=41)
    entry_tight = {v: fresh_entry() for v in plans_tight}
    rec_tight = run_plan(inst_tight, plans_tight, entry_tight)
    sh_tight = S2.plan_shape(rec_tight)
    t_tight = S2.isolated_timing(
        lambda: run_plan(inst_tight, plans_tight, entry_tight),
        warmup=5, repeat=40)
    ck('S2-5', 'one whole-fleet certificate fits the pre-declared online budget '
       '(B_dec = 10.0 s, C_admissible = 250 drafts per decision, t_max = 40 ms) in BOTH the ordinary '
       'case and the adversarial all-short-gap case',
       t_long['within_budget'] and t_tight['within_budget'],
       {'ordinary_case': dict(t_long, n_doors=sh_long['n_doors'],
                              n_short_gaps=sh_long['n_short_gaps']),
        'adversarial_all_short': dict(t_tight, n_doors=sh_tight['n_doors'],
                                      n_short_gaps=sh_tight['n_short_gaps']),
        'budget': {'B_dec_seconds': S2.B_DEC_SECONDS, 'C_admissible': S2.C_ADMISSIBLE,
                   't_max_seconds': S2.T_MAX_SECONDS},
        'reading': 'the ordinary case is free because every gap exceeds Theta; the adversarial case '
                   'pays one exact-rational Taylor enclosure per short gap, about 0.7 ms each'},
       conclusion='the certificate is affordable inside the decision budget on this machine, with '
                  'the measured headroom stated rather than assumed')

    # ============================================================ S2-6 linear in n_stops
    def bench(per_vehicle, n_vehicles):
        plans = fleet(inst_long, n_vehicles=n_vehicles, per_vehicle=per_vehicle, n_nodes=41)
        ent = {v: fresh_entry() for v in plans}
        return S2.isolated_timing(lambda: run_plan(inst_long, plans, ent),
                                  warmup=20, repeat=120), S2.plan_shape(
            run_plan(inst_long, plans, ent))

    t10, sh10 = bench(2, 5)
    t40, sh40 = bench(8, 5)
    ratio = (t40['median_s'] / t10['median_s']) if t10['median_s'] > 0 else float('inf')

    # enclosure-DOMINATED regime: everything so tight that every post-first gap needs an enclosure,
    # so the work scales with the ENCLOSURE count and a per-enclosure cost can be checked directly
    inst_ultra = make_instance(n_nodes=41, seed=SEED + 3, service=(0.0004, 0.0010),
                               cluster=0.0005)

    def bench_ultra(per_vehicle, n_vehicles):
        plans = fleet(inst_ultra, n_vehicles=n_vehicles, per_vehicle=per_vehicle, n_nodes=41)
        ent = {v: fresh_entry() for v in plans}
        return S2.isolated_timing(lambda: run_plan(inst_ultra, plans, ent), warmup=5, repeat=40), \
            S2.plan_shape(run_plan(inst_ultra, plans, ent))

    tu10, shu10 = bench_ultra(2, 5)
    tu40, shu40 = bench_ultra(8, 5)
    per_exp_10 = (tu10['median_s'] / shu10['n_exp_reported']) if shu10['n_exp_reported'] else None
    per_exp_40 = (tu40['median_s'] / shu40['n_exp_reported']) if shu40['n_exp_reported'] else None
    per_exp_ratio = (per_exp_40 / per_exp_10) if per_exp_10 else float('inf')
    ck('S2-6', 'the cost is LINEAR: quadrupling the door count (10 -> 40) on the ordinary instance '
       'raises the measured median by at most 6x, the closed-form door count scales exactly by 4, '
       'and in the enclosure-DOMINATED regime the measured cost per enclosure is stable within a '
       'factor of 2.5 as the enclosure count grows from 5 to 35',
       ratio <= 6.0 and sh40['n_doors'] == 4 * sh10['n_doors']
       and shu10['n_short_gaps'] > 0 and shu40['n_short_gaps'] > 0
       and 0.4 <= per_exp_ratio <= 2.5,
       {'ordinary_instance': {'t_10_doors_s': t10['median_s'], 't_40_doors_s': t40['median_s'],
                              'ratio': ratio, 'n_doors_10': sh10['n_doors'],
                              'n_doors_40': sh40['n_doors'],
                              'n_short_gaps_10': sh10['n_short_gaps'],
                              'n_short_gaps_40': sh40['n_short_gaps'],
                              'reading': 'fixed receipt overhead dominates here, so the ratio is well '
                                         'BELOW 4 -- the gate is therefore an upper bound'},
        'enclosure_dominated': {
            't_10_doors_s': tu10['median_s'], 't_40_doors_s': tu40['median_s'],
            'n_doors_10': shu10['n_doors'], 'n_doors_40': shu40['n_doors'],
            'n_exp_10': shu10['n_exp_reported'], 'n_exp_40': shu40['n_exp_reported'],
            'seconds_per_enclosure_10': per_exp_10, 'seconds_per_enclosure_40': per_exp_40,
            'per_enclosure_ratio': per_exp_ratio,
            'reading': 'this is where the exact-rational Taylor cost is visible; it is the number to '
                       'quote for the worst case'},
        'closed_form_10': S2.closed_form(n_doors=sh10['n_doors'],
                                         n_short_gaps=sh10['n_short_gaps'],
                                         n_vehicles=sh10['n_vehicles'], n_zones=N_ZONES),
        'closed_form_40': S2.closed_form(n_doors=sh40['n_doors'],
                                         n_short_gaps=sh40['n_short_gaps'],
                                         n_vehicles=sh40['n_vehicles'], n_zones=N_ZONES),
        'limit_note': 'no per-point monotonicity is required, so measurement noise cannot produce a '
                      'false FAIL'},
       conclusion='the closed form is the right scaling law, so a campaign can be sized before it '
                  'runs')

    # ============================================================ S2-7 worst-case bound
    with S2.count_transcendentals() as cnt_w:
        rec_w = run_plan(inst_tight, plans_tight, entry_tight)
    sh_w = S2.plan_shape(rec_w)
    cf_w = S2.closed_form(n_doors=sh_w['n_doors'], n_short_gaps=sh_w['n_short_gaps'],
                          n_vehicles=sh_w['n_vehicles'], n_zones=sh_w['n_zones'])
    ck('S2-7', 'the adversarial worst case is measured and bounded: with most post-first-door gaps '
       'short the certificate evaluates exactly ONE enclosure per evaluated gap, so '
       'n_exp == n_short_gaps <= n_doors - n_active_vehicles <= n_doors (the first gap on a zone is '
       'always free because the compartment starts on its floor)',
       cnt_w['exp'] == cf_w['n_exp_exact'] == sh_w['n_short_gaps']
       and sh_w['n_short_gaps'] <= cf_w['n_exp_upper_bound_tighter']
       and cnt_w['exp'] <= cf_w['n_exp_upper_bound']
       and cnt_w['exp'] <= cf_w['n_exp_pre_registered_formula'],
       {'measured_exp': cnt_w['exp'], 'n_short_gaps': sh_w['n_short_gaps'],
        'n_doors': sh_w['n_doors'], 'n_active_vehicles': sh_w['n_vehicles'],
        'closed_form': cf_w,
        'pre_registered_formula_n_doors_plus_z_times_A': cf_w['n_exp_pre_registered_formula'],
        'prereg_note': 'the frozen S2-7 wrote the worst case as `n_doors + Z*A`; that DOUBLE-COUNTS '
                       'the entry interval, because the implementation merges the entry gap into the '
                       'first door gap (one gap row per door).  The measured cost is therefore '
                       'strictly BELOW the frozen formula, which remains a valid upper bound, and '
                       'the corrected statement above is tighter than the frozen one.  Report '
                       'ERRATA-2'},
       conclusion='the worst case is a declared, closed-form quantity, and the frozen formula remains '
                  'a valid upper bound')

    # ============================================================ S2-N1 the count is not self-certifying
    tampered = [S2.closed_form(n_doors=sh_w['n_doors'], n_short_gaps=sh_w['n_short_gaps'] + 1,
                               n_vehicles=sh_w['n_vehicles'], n_zones=sh_w['n_zones'])]
    rep = S2.cost_report(rec_w, measured_exp=cnt_w['exp'], measured_ln=0)
    rep_bad = S2.cost_report(rec_w, closed=tampered[0], measured_exp=cnt_w['exp'], measured_ln=0)
    ck('S2-N1', 'the counting row is not self-certifying: the correct closed form passes the report, '
       'and perturbing ONE constant of the closed form makes the same report FAIL',
       rep['ok'] is True and rep_bad['ok'] is False
       and [c for c in rep_bad['checks'] if not c['ok']][0]['id'] == 'exp_equals_closed_form',
       {'checks_ok': rep['checks'], 'bad_checks': [c for c in rep_bad['checks'] if not c['ok']]},
       conclusion='the equality is a real test of the implementation, not a tautology')

    payload = {
        'schema': 'q3_s2_online_cost_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'source_sha256': src,
        'contract_hash': CONTRACT.contract_hash,
        'seed': SEED,
        'budget': {'B_dec_seconds': S2.B_DEC_SECONDS, 'C_admissible': S2.C_ADMISSIBLE,
                   't_max_seconds': S2.T_MAX_SECONDS,
                   'source': 'pre-registered in E5.3 before any measurement'},
        'rows': ROWS, 'n_rows': len(ROWS),
        'n_pass': sum(1 for r in ROWS if r['ok']),
        'n_fail': sum(1 for r in ROWS if not r['ok']),
        'verdict': ('S2_CONTROLS_PASS' if all(r['ok'] for r in ROWS) else 'S2_CONTROLS_FAIL'),
        'labels': {'DECIDED': 'S2-1/2/7 closed form + external count',
                   'MECHANICAL': 'S2-3 AST audit, S2-4 exact-rational operand types',
                   'LOCAL_DIAGNOSTIC': 'S2-5/6 wall clock on one machine'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL -> %s ==='
          % (len(ROWS), payload['n_pass'], payload['n_fail'], payload['verdict']))
    print('WROTE', out)
    return 0 if payload['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
