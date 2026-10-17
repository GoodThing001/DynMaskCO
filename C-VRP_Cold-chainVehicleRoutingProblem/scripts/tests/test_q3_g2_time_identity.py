#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G-2 TIME-PARAMETER IDENTITY AND STRICT-ONLINE SOURCE QUALIFICATION -- controls (DIAGNOSTIC).

This file implements the controls registered in
`scripts/q3_qualification/q3_g2_preregistration_2026-10-17.md` BEFORE any G-2 code was written:

    C1   primary control: the public substitution is never looser than the current parameters
    C2   the four time parameters each get {field, source, visibility, admissibility}, no gap
    C3   the horizon witness is mechanical (max(tw_end) == tw_end[0] per instance)
    C4   the advance bound comes from a static call-graph census, not from a constant
    C5   the inherited reading `n = max_iter` is REFUSED
    C6   the iteration bound has two sides and reports which one binds
    C7   hours_per_time_unit: certifiable only at exactly 1.0, and identity-bound
    C8   the three tolerances are registered, with why only one is in the slack budget
    C9   the identity digest covers every field and changes when one changes
    C10  nine negative controls, each with its expected refusal
    C11  the audited sources are unchanged by this round

Nothing here trains, uses a GPU, runs a formal rollout, or touches a final-test seed.  The one
execution is a 5-node synthetic episode used ONLY to count advances (no model, no scoring, no
replanner search); it is labelled DIAGNOSTIC and is not a rollout.

Usage: python scripts/tests/test_q3_g2_time_identity.py --out <json path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(EXT / 'scripts'))
sys.path.insert(0, str(EXT / 'scripts' / 'coldchain'))
sys.path.insert(0, str(EXT / 'scripts' / 'm1b_adapter'))
sys.path.insert(0, str(EXT / 'scripts' / 'simulation'))

import coldchain_contract as CC                                            # noqa: E402
import m1b_contract_identity as CI                                         # noqa: E402
import m1b_quality_certificate as QC                                       # noqa: E402
import q3_g2_time_identity_experimental as G2                              # noqa: E402

ENV_SRC = EXT / 'scripts' / 'simulation' / 'strict_online_env.py'
CERT_REL = 'scripts/m1b_adapter/m1b_q2_certifier.py'
PIN_REPORT = (EXT / 'docs' / '当前规划' / '决策与审计'
              / 'GTMIN交付后独立勘误_对象保留与复现性_2026-10-17.md')
SPLIT_GLOB = 'data/m0_scale/*.npz'
ROWS = []
LIMITS = ('read-only static analysis + one 5-node synthetic episode; no training, no GPU, no formal '
          'rollout, no final-test seed, no production certifier change')


def ck(cid, name, ok, detail=None, conclusion='', status=None):
    st = status or ('PASS' if ok else 'FAIL')
    ROWS.append({'id': cid, 'name': name, 'ok': bool(ok), 'status': st, 'detail': detail or {},
                 'conclusion': conclusion, 'limits': LIMITS})
    print('[%s] %-7s %s' % (st, cid, name[:92]))


def ck_skip(cid, name, reason, detail=None):
    d = dict(detail or {})
    d['skip_reason'] = reason
    ROWS.append({'id': cid, 'name': name, 'ok': True, 'status': 'SKIP', 'detail': d,
                 'conclusion': 'NOT EVALUATED in this checkout', 'limits': LIMITS})
    print('[SKIP ] %-7s %s' % (cid, name[:92]))


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pinned_certifier_digest():
    """The G-TMIN final identity, parsed from the errata pin block (digests live in reports)."""
    if not PIN_REPORT.exists():
        return None
    rx = re.compile(r'^\s*(C-VRP_Cold-chainVehicleRoutingProblem/scripts/m1b_adapter/'
                    r'm1b_q2_certifier\.py)\s*=\s*([0-9a-f]{64})\s*$')
    out = None
    for line in PIN_REPORT.read_text(encoding='utf-8').splitlines():
        m = rx.match(line)
        if m:
            out = m.group(2)
    return out


def git_status_scoped():
    """Porcelain status of the extension root, with non-ASCII bytes left unescaped."""
    raw = subprocess.run(['git', '-c', 'core.quotePath=false', 'status', '--porcelain', '--', '.'],
                         cwd=str(EXT), capture_output=True, text=True,
                         encoding='utf-8', errors='replace').stdout.splitlines()
    prefix = EXT.name + '/'
    out = []
    for ln in raw:
        path = ln[3:].strip().replace('\\', '/').strip('"')
        out.append((ln[:2], path[len(prefix):] if path.startswith(prefix) else path))
    return out


def load_splits():
    import numpy as np
    out = []
    for path in sorted(EXT.glob(SPLIT_GLOB)):
        d = np.load(path, allow_pickle=True)
        out.append((path.name, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-synthetic', action='store_true')
    args = ap.parse_args()

    contract = CC.default_pilot_contract()
    source = G2.source_identity(EXT, repo_root=EXT.parent, rel_prefix=EXT.name + '/')
    census = G2.call_graph_census(ENV_SRC.read_text(encoding='utf-8'))
    splits = load_splits()

    # the registered witness of the first instance anchors the domain-level identity
    w0 = G2.classify_horizon(splits[0][1]['tw_end'][0],
                             splits[0][1]['reveal_time'][0] if 'reveal_time' in splits[0][1] else None)
    n_inst = sum(int(d['tw_end'].shape[0]) for _n, d in splits)
    domain = {'dataset_glob': SPLIT_GLOB, 'n_instances': n_inst,
              'splits': [n for n, _ in splits], 'horizon_value': w0['depot_tw_end']}
    ident = G2.build_identity(source=source, census=census, contract_hash=contract.contract_hash,
                              contract=contract, horizon_witness=w0,
                              max_iter=census['max_iter'], domain=domain)

    # ============================================================ C11 sources unchanged
    audited = list(G2.AUDITED_FILES)
    dirty = [p for st, p in git_status_scoped() if st not in ('??', '!!') and p in audited]
    pinned = pinned_certifier_digest()
    cert_now = source['files'][CERT_REL]
    if pinned is None:
        pin_ok, pin_note = True, 'the errata pin block is not present in this checkout'
    else:
        pin_ok = (cert_now == pinned)
        pin_note = 'the certifier blob equals the G-TMIN registered final identity'
    ck('C11', 'the audited sources are unchanged by this round: no audited file is modified in the '
       'worktree, and the certifier blob still equals the identity registered by the G-TMIN errata',
       not dirty and pin_ok,
       {'basis': source['basis'], 'audited': {p: source['files'][p] for p in audited},
        'modified_audited': dirty, 'certifier_blob': cert_now,
        'gtmin_registered': pinned, 'pin_note': pin_note},
       conclusion='G-2 read the corpus and changed none of it')

    # ============================================================ C4 census from the call graph
    sites = census['advance_call_sites']
    commit_sites = sorted(s['line'] for s in sites.get('_commit', []))
    fleet_sites = sorted(s['line'] for s in sites.get('_advance_fleet', []))
    ok4 = (census['pre_loop'] == 1 and census['per_iteration'] == 2 and census['termination'] == 2
           and census['n_max_per_vehicle'] == 1 + 2 * census['max_iter'] + 2
           and commit_sites == [272, 286] and fleet_sites == [304]
           and census['max_iter'] == 10000 and census['max_iter_line'] == 481)
    ck('C4', 'the advance-count bound is DERIVED from a static call-graph census of '
       'strict_online_env.py: `n_max = 1 + 2*max_iter + 2 = 20003`, with the three advance sites '
       '(:272, :286 inside _commit; :304 inside _advance_fleet) all found',
       ok4,
       {'n_max_per_vehicle': census['n_max_per_vehicle'], 'formula': census['n_max_formula'],
        'per_iteration_call_sites': census['per_iteration_call_sites'],
        'once_call_sites': census['once_call_sites'],
        'advance_call_sites': sites, 'multiplicity': census['multiplicity'],
        'max_iter': census['max_iter'], 'max_iter_line': census['max_iter_line'],
        'inherited_reading': 'max_iter = 10000 was previously taken as the advance bound',
        'census_digest': census['census_digest']},
       conclusion='the bound is a derived upper bound, not a constant')

    # ============================================================ C4b the derivation is live
    src_text = ENV_SRC.read_text(encoding='utf-8')
    tampered_calls = src_text.replace(
        '            self._advance_coldchain_to(v, clock)\n            if v.status ==',
        '            self._advance_coldchain_to(v, clock)\n'
        '            self._advance_coldchain_to(v, clock)\n            if v.status ==', 1)
    tampered_return = src_text.replace('            return\n        # customer',
                                       '            pass\n        # customer', 1)
    grew = {}
    for label, text in (('extra advance site', tampered_calls),
                        ('return-branch witness removed', tampered_return)):
        grew[label] = G2.call_graph_census(text)
    ok4b = (grew['extra advance site']['n_max_per_vehicle']
            > census['n_max_per_vehicle']
            and grew['return-branch witness removed']['n_max_per_vehicle']
            > census['n_max_per_vehicle']
            and grew['extra advance site']['census_digest'] != census['census_digest']
            and tampered_calls != src_text and tampered_return != src_text)
    ck('C4b', 'the census reacts to the code: adding an advance site or deleting the return-branch '
       'witness both increase the bound and change the census digest, so the number cannot be a '
       'hard-coded constant that silently survives an edit',
       ok4b,
       {'baseline_n_max': census['n_max_per_vehicle'],
        'with_extra_advance_site': grew['extra advance site']['n_max_per_vehicle'],
        'with_witness_removed': grew['return-branch witness removed']['n_max_per_vehicle'],
        'replacement_applied': [tampered_calls != src_text, tampered_return != src_text]},
       conclusion='the derivation is mechanical and refutable')

    # ============================================================ C6 iteration bound, two sides
    it_small = G2.iteration_bound(1e-3, census['max_iter'])
    it_room = G2.iteration_bound(24.0, census['max_iter'])
    it_big = G2.iteration_bound(1e6, census['max_iter'])
    ok6 = (it_small['binding'] == 'time' and it_small['bound'] == 1001
           and it_room['binding'] == 'max_iter' and it_room['bound'] == 10000
           and it_big['binding'] == 'max_iter')
    ck('C6', 'the iteration bound has two sides -- max_iter and the clock advancing more than '
       'eps_event per iteration -- and reports which side binds: a small horizon is time-bound '
       '(1e-3 -> 1001), the registered horizon is max_iter-bound (24 -> 10000)',
       ok6, {'T=1e-3': it_small, 'T=24': it_room, 'T=1e6': it_big},
       conclusion='max_iter is a cap, not the bound, and can be loose for small horizons')

    # ============================================================ C5 the inherited reading is refused
    refused_old, detail_old = False, {}
    try:
        G2.advance_budget(24.0, census, declared_n=census['max_iter'])
    except G2.Refusal as exc:
        refused_old, detail_old = (exc.code == 'INSUFFICIENT_COUNT_BOUND'), exc.detail
    budget = G2.advance_budget(24.0, census)
    slack_new = G2.slack_budget(24.0, budget['n_max_per_vehicle'], 1.0)
    slack_old = G2.slack_budget(24.0, census['max_iter'], 1.0)
    ck('C5', 'the inherited reading `n = max_iter` is REFUSED by the budget gate, and the corrected '
       'bound strictly enlarges the rounding term (the old budget under-estimated it by ~2x)',
       refused_old and slack_new['total_slack'] > slack_old['total_slack'],
       {'refused_with': detail_old, 'n_derived': budget['n_max_per_vehicle'],
        'slack_old': slack_old['total_slack'], 'slack_new': slack_new['total_slack'],
        'skip_term': slack_new['skip_term'], 'rounding_term_new': slack_new['rounding_term'],
        'old_constant_2eps': 2e-9,
        'magnitude_note': ('both slack values are ~1e-9 h, four orders of magnitude below the '
                           'inter-threshold Delta*, so the numerical conclusion of L1 survives; '
                           'what fails is the DERIVATION, not the number')},
       conclusion='the correction changes the derivation and not the conclusion')

    # ============================================================ C8 tolerances registered
    tol = ident['tolerances']
    # eps_skip is registered as the EXACT rational 1/10^9; the source literal `1e-9` is the nearest
    # double.  Comparing Fractions is the only comparison that can be made without a float rounding
    # of the comparison itself, so the test pins the rational and records the double separately.
    env_eps = {'skip': Fraction(1, 10 ** 9), 'backward': 1e-6, 'event': 1e-6}
    parsed = {}
    for m in re.finditer(r'(1e-(?:9|6))', ENV_SRC.read_text(encoding='utf-8')):
        parsed[m.group(1)] = parsed.get(m.group(1), 0) + 1
    ok8 = (Fraction(tol['eps_skip']) == env_eps['skip']
           and tol['eps_backward'] == env_eps['backward']
           and tol['eps_event'] == env_eps['event']
           and tol['in_slack_budget'] == ['eps_skip']
           and set(tol['not_in_slack_budget']) == {'eps_backward', 'eps_event'}
           and '1e-9' in parsed and '1e-6' in parsed)
    ck('C8', 'the three tolerances are registered with their role: only eps_skip=1e-9 enters the '
       'slack budget; eps_backward and eps_event are the same literal 1e-6 and both act on event '
       'timing, not on the telescoping sum of durations',
       ok8, {'identity_tolerances': tol, 'literal_counts_in_env': parsed,
             'why_not_in_budget': tol['why_not']},
       conclusion='no uncatalogued epsilon participates in the L1 bound')

    # ============================================================ C3 horizon witness over the domain
    witness_rows, unequal, leak = [], [], []
    n_inst = 0
    for name, data in splits:
        for inst in range(int(data['tw_end'].shape[0])):
            tw = data['tw_end'][inst]
            rtv = data['reveal_time'][inst] if 'reveal_time' in data else None
            w = G2.classify_horizon(tw, rtv)
            n_inst += 1
            if not w['witness_equal']:
                unequal.append({'split': name, 'inst': inst, 'leak': w['leak_magnitude']})
            leak.append(w['hidden_customer_max'] or 0.0)
            if inst < 1:
                witness_rows.append({'split': name, 'inst': inst, **w})
    ok3 = (n_inst == 96 and not unequal)
    ck('C3', 'the horizon witness holds for every registered instance: max(tw_end) == tw_end[0], so '
       'the value the simulator derives from the FULL array is identical to the depot deadline the '
       'strict-online decider already reads (96/96); the leak is empty in this data domain, and that '
       'is a WITNESSED fact, not an assumption',
       ok3,
       {'n_instances': n_inst, 'n_unequal': len(unequal), 'unequal': unequal[:5],
        'worst_hidden_customer_deadline': max(leak) if leak else None,
        'example': witness_rows,
        'grading': 'PROVED_CONDITIONAL: it depends on the generator invariant plus the per-instance '
                   'witness; it is not an unconditional theorem'},
       conclusion='the horizon is admissible here only because, and only while, the witness holds')

    # ============================================================ C1 primary control
    per_class, pcverd, pcdet = CI.check_quality_classes_are_certifiable(
        contract, epsilon_init=QC.EPSILON_INIT)
    per_class_ok = (pcverd == CI.CONTRACT_IDENTITY_OK)
    # The default pilot contract is a v1 `ColdChainContract`, and the Q2 quality bound requires
    # `QualityConfigV2`, so gate 7 cannot be evaluated against it at all
    # (verdict CONTRACT_QUALITY_CONFIG_UNSUPPORTED).  That is a fact about the chain, not a detail to
    # paper over: the q_lb arm below therefore uses a DECLARED SYNTHETIC constant set to show that the
    # bound propagates H_max unchanged, and the finding is registered in the report.
    # `rate_per_hour` is the documented mullet-fish zero-order rate (0.01883/h), so the demonstration
    # lands on the same 8.548... lower bound the G-TMIN record reports instead of saturating at 0.
    synthetic_pc = {'fish4': {'in_scope': True, 'dynamics_order': 0, 'threshold_type': 'absolute',
                              'initial_value': 9.0, 'q0_safe': 9.0, 'rate_per_hour': 0.01883,
                              'threshold': 5.0}}
    qc_pc = per_class if per_class_ok else synthetic_pc
    arm_a, arm_b, looser = [], [], []
    for name, data in splits:
        for inst in range(int(data['tw_end'].shape[0])):
            w = G2.classify_horizon(data['tw_end'][inst],
                                    data['reveal_time'][inst] if 'reveal_time' in data else None)
            a = QC.horizon_hours(w['tw_end_max'], 0.0, contract.units.hours_per_time_unit)[0]
            b = QC.horizon_hours(w['depot_tw_end'], 0.0, contract.units.hours_per_time_unit)[0]
            qa = QC.certify_quality_lb(qc_pc, T_end=w['tw_end_max'], t_min=0.0,
                                       hours_per_time_unit=contract.units.hours_per_time_unit)[0]
            qb = QC.certify_quality_lb(qc_pc, T_end=w['depot_tw_end'], t_min=0.0,
                                       hours_per_time_unit=contract.units.hours_per_time_unit)[0]
            # Arm A is the CURRENT parameterisation: the full-array maximum and n = max_iter.
            # It is spelled out here directly (not through advance_budget) because the gate in
            # advance_budget REFUSES that reading -- which is control C5; C1 compares the two
            # parameterisations, it does not re-litigate their admissibility.
            na = census['max_iter']
            nb = G2.advance_budget(w['depot_tw_end'], census)
            sa = G2.slack_budget(w['tw_end_max'], na)
            sb = G2.slack_budget(w['depot_tw_end'], nb['n_max_per_vehicle'])
            arm_a.append({'H': a, 'n': na, 'slack': sa['total_slack']})
            arm_b.append({'H': b, 'n': nb['n_max_per_vehicle'], 'slack': sb['total_slack']})
            bad = []
            if not (b >= a):
                bad.append('H_max decreased')
            if not (nb['n_max_per_vehicle'] >= na):
                bad.append('n decreased')
            if not (sb['total_slack'] >= sa['total_slack']):
                bad.append('slack decreased')
            if qa is not None and qb is not None:
                for c in qa:
                    if qa[c].get('in_scope') and qb[c].get('in_scope'):
                        if qb[c]['q_lb'] > qa[c]['q_lb']:
                            bad.append('q_lb raised for class %s' % c)
                        if qb[c]['q_lb'] != qa[c]['q_lb']:
                            bad.append('q_lb moved for class %s (H_max was expected to be equal)' % c)
            if bad:
                looser.append({'split': name, 'inst': inst, 'problems': bad})
    ok1 = (not looser and n_inst == 96)
    ck('C1', 'PRIMARY CONTROL -- the public substitution is never looser than the current '
       'parameters: over all 96 registered instances H_max does not shrink, q_lb is not raised, the '
       'advance bound does not shrink and the slack does not shrink, when (tw_end.max, max_iter) is '
       'replaced by (tw_end[0], derived bound)',
       ok1,
       {'n_instances': n_inst, 'n_looser': len(looser), 'looser': looser[:5],
        'H_max_both_arms': sorted({r['H'] for r in arm_a} | {r['H'] for r in arm_b}),
        'n_arm_a': sorted({r['n'] for r in arm_a}), 'n_arm_b': sorted({r['n'] for r in arm_b}),
        'slack_arm_a': sorted({r['slack'] for r in arm_a}),
        'slack_arm_b': sorted({r['slack'] for r in arm_b}),
        'per_class_constants_from_contract': per_class_ok,
        'per_class_verdict': pcverd,
        'per_class_detail': pcdet,
        'q_lb_constants_used': ('contract per-class constants' if per_class_ok else
                                'DECLARED SYNTHETIC constants: the pilot contract is a v1 '
                                'ColdChainContract and gate 7 requires QualityConfigV2, so the '
                                'contract cannot feed certify_quality_lb at all'),
        'q_lb_arm_a': qa, 'q_lb_arm_b': qb,
        'reading': ('A = tw_end.max() with n = max_iter (the current, unaudited parameters); '
                    'B = tw_end[0] with the derived bound.  The public arm is at least as strict, '
                    'so the hidden-derived value is NOT load-bearing for the certificate.')},
       conclusion='the certificate does not depend on information it is not allowed to read')

    # ============================================================ C9 identity digest
    ident_again = G2.build_identity(source=source, census=census,
                                   contract_hash=contract.contract_hash, contract=contract,
                                   horizon_witness=w0, max_iter=census['max_iter'],
                                   domain=domain)
    mutated = []
    m1 = json.loads(json.dumps(ident))
    m1['advance_budget']['max_iter'] = census['max_iter'] + 1
    mutated.append(G2.sha256_json({k: v for k, v in m1.items() if k != 'identity_digest'}))
    m2 = json.loads(json.dumps(ident))
    m2['horizon']['value'] = float(m2['horizon']['value']) + 1.0
    mutated.append(G2.sha256_json({k: v for k, v in m2.items() if k != 'identity_digest'}))
    m3 = json.loads(json.dumps(ident))
    m3['code']['files'][G2.AUDITED_FILES[0]] = '0' * 64
    mutated.append(G2.sha256_json({k: v for k, v in m3.items() if k != 'identity_digest'}))
    ok9 = (ident['identity_digest'] == ident_again['identity_digest']
           and all(d != ident['identity_digest'] for d in mutated)
           and ident['advance_budget']['n_max_per_vehicle_at_max_iter'] == 20003
           and ident['horizon']['witness_equal'] is True
           and ident['tolerances']['eps_skip'] == str(G2.EPS_SKIP))
    ck('C9', 'the identity digest is reproducible and total: rebuilding yields the same digest, and '
       'changing max_iter, the horizon value, or one source digest each changes it; the object '
       'binds contract, code, unit scale, public horizon, event/advance budget, tolerances and domain',
       ok9,
       {'identity_digest': ident['identity_digest'], 'rebuilt_equal': True,
        'digests_after_each_single_field_change': mutated,
        'fields': sorted(ident.keys()), 'advance_budget': ident['advance_budget'],
        'binding_note': 'a certificate that cites this digest cannot silently swap any of the six '
                        'bound quantities'},
       conclusion='the identity fails closed on drift')

    # ============================================================ C7 unit scale
    tight = CC.default_pilot_contract()
    hpu_ok = (tight.units.hours_per_time_unit == 1.0)
    q_units = [G2.qualify(identity=ident, horizon_witness=w0, horizon_value=w0['depot_tw_end'],
                          census=census, contract=contract)['verdict']]
    class NonUnit(object):
        class units(object):
            hours_per_time_unit = 2.0
            distance_km_per_unit = 1.0
        contract_hash = 'non-unit-probe'

    class NonUnitContract(object):
        units = NonUnit.units
        contract_hash = 'non-unit-probe'

    q_nonunit = G2.qualify(identity=ident, horizon_witness=w0, horizon_value=w0['depot_tw_end'],
                           census=census, contract=NonUnitContract())
    codes = [r['code'] for r in q_nonunit['refusals'] + q_nonunit['not_certifiable']]
    ok7 = (hpu_ok and q_units[0] == G2.PROVED_CONDITIONAL
           and q_nonunit['verdict'] == G2.REFUSED
           and 'TIME_UNIT_CONVERSION_NOT_BUDGETED' in codes
           and 'IDENTITY_MISMATCH' in codes)
    ck('C7', 'the unit scale is certifiable only at exactly 1.0 (a non-unit scale adds a rounding '
       'per advance that the budget does not cover), and because hours_per_time_unit sits inside '
       'contract_hash, pairing a non-unit contract with the registered identity is an '
       'IDENTITY_MISMATCH as well as a NOT_CERTIFIABLE unit',
       ok7,
       {'contract_hpu': tight.units.hours_per_time_unit,
        'verdict_unit_contract': q_units[0], 'verdict_non_unit': q_nonunit['verdict'],
        'codes': codes, 'refusals': q_nonunit['refusals'],
        'not_certifiable': q_nonunit['not_certifiable']},
       conclusion='a non-unit contract can never inherit a unit-scale certificate')

    # ============================================================ C2 parameter matrix
    matrix = G2.parameter_matrix()
    need = {'field', 'defined_source', 'visibility', 'admissibility'}
    ok2 = (len(matrix) == 7
           and all(need <= set(r) for r in matrix)
           and all(r['visibility'] in G2.VISIBILITY_LAYERS for r in matrix)
           and {r['visibility'] for r in matrix} == set(G2.VISIBILITY_LAYERS)
           and any(r['carries_unrevealed_info'] for r in matrix)
           and sum(1 for r in matrix if r['carries_unrevealed_info']) == 1)
    ck('C2', 'all four parameter families are registered in the matrix with {field, defined source, '
       'visibility, admissibility}; the three visibility layers are each occupied (the horizon in '
       'the simulator layer, its admitted substitute in the online layer, the budget/unit/tolerances '
       'in the certifier layer); and exactly one parameter carries unrevealed information',
       ok2, {'n_rows': len(matrix), 'rows': matrix, 'layers': list(G2.VISIBILITY_LAYERS)},
       conclusion='no time parameter is left unclassified')

    # ============================================================ C10 negative controls
    def verdict_of(**kw):
        base = dict(identity=ident, horizon_witness=w0, horizon_value=w0['depot_tw_end'],
                    census=census, contract=contract)
        base.update(kw)
        return G2.qualify(**base)

    nc = {}

    # N-01 horizon that does not cover the episode
    q = verdict_of(observed_max_clock=float(w0['depot_tw_end']) + 1.0)
    nc['N-01'] = {'expect': 'HORIZON_NOT_AN_UPPER_BOUND', 'verdict': q['verdict'],
                  'codes': [r['code'] for r in q['refusals']]}

    # N-02 a horizon that carries an unrevealed order's deadline
    hidden = G2.classify_horizon([24.0, 20.0, 30.0], [0.0, 1.0, 2.0])
    q = verdict_of(horizon_witness=hidden, horizon_value=hidden['tw_end_max'])
    nc['N-02'] = {'expect': 'HIDDEN_DERIVED_HORIZON', 'verdict': q['verdict'],
                  'codes': [r['code'] for r in q['refusals'] + q['not_certifiable']],
                  'witness': {'tw_end_max': hidden['tw_end_max'],
                              'depot': hidden['depot_tw_end'],
                              'leak': hidden['leak_magnitude']}}

    # N-03 non-unit scale
    q = verdict_of(contract=NonUnitContract())
    nc['N-03'] = {'expect': 'TIME_UNIT_CONVERSION_NOT_BUDGETED', 'verdict': q['verdict'],
                  'codes': [r['code'] for r in q['refusals'] + q['not_certifiable']]}

    # N-04 a resume between the two doors
    q = verdict_of(resumed_between_doors=True)
    nc['N-04'] = {'expect': 'SNAPSHOT_RESTORE_INSIDE_INTERVAL', 'verdict': q['verdict'],
                  'codes': [r['code'] for r in q['refusals']]}

    # N-05 an advance count above the derived bound
    q = verdict_of(observed_advances=budget['n_max_per_vehicle'] + 1)
    nc['N-05'] = {'expect': 'COUNT_BOUND_VIOLATED', 'verdict': q['verdict'],
                  'codes': [r['code'] for r in q['refusals'] + q['not_certifiable']]}

    # N-06 tampered identity
    bad_census = dict(census)
    bad_census['max_iter'] = census['max_iter'] + 1
    bad_census['census_digest'] = G2.sha256_json(
        {k: v for k, v in bad_census.items() if k != 'census_digest'})
    q = verdict_of(recomputed_census=bad_census)
    nc['N-06'] = {'expect': 'IDENTITY_MISMATCH', 'verdict': q['verdict'],
                  'codes': [r['code'] for r in q['refusals']]}

    # N-07 the inherited count reading
    q = verdict_of(n_declared=census['max_iter'])
    nc['N-07'] = {'expect': 'INSUFFICIENT_COUNT_BOUND', 'verdict': q['verdict'],
                  'codes': [r['code'] for r in q['refusals']]}

    # N-08 malformed horizons
    codes_n8, verdicts_n8 = [], []
    for bad in (float('nan'), -1.0, 0.0):
        q = verdict_of(horizon_value=bad)
        verdicts_n8.append(q['verdict'])
        codes_n8.extend(r['code'] for r in q['refusals'])
    nc['N-08'] = {'expect': 'HORIZON_INVALID',
                  'verdict': verdicts_n8[0] if len(set(verdicts_n8)) == 1 else verdicts_n8,
                  'codes': codes_n8}

    # N-09 non-unit contract paired with the registered identity
    q = verdict_of(contract=NonUnitContract())
    nc['N-09'] = {'expect': 'CONTRACT_IDENTITY_MISMATCH', 'verdict': q['verdict'],
                  'codes': [p.get('field') for r in q['refusals']
                            for p in (r.get('detail', {}).get('problems') or [])]}

    def hit(key):
        v = nc[key]
        codes = ' '.join(str(c) for c in v['codes'])
        if v['verdict'] not in (G2.REFUSED, G2.NOT_CERTIFIABLE):
            return False
        if key == 'N-09':
            return 'contract.contract_hash' in codes
        return v['expect'] in codes

    for key in sorted(nc):
        ok = hit(key)
        ck('C10/' + key, 'negative control %s: %s -> %s expected %s'
           % (key, 'injected', 'refused' if ok else 'NOT refused', nc[key]['expect']),
           ok, nc[key], 'the gate fires on the injected condition')

    # ============================================================ C10b synthetic episode (DIAGNOSTIC)
    if args.skip_synthetic:
        ck_skip('G2-SYN', 'synthetic 5-node episode: per-vehicle advance count vs the derived bound',
                '--skip-synthetic')
    else:
        import numpy as np
        n_nodes = 5
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
                          dtype=np.float32)[None, ...]
        data = {
            'coords': coords,
            'demands': np.array([[0.0, 5.0, 5.0, 5.0, 5.0]], dtype=np.float32),
            'tw_start': np.zeros((1, n_nodes), dtype=np.float32),
            'tw_end': np.array([[24.0, 12.0, 14.0, 16.0, 18.0]], dtype=np.float32),
            'service_time': np.full((1, n_nodes), 0.5, dtype=np.float32),
            'reveal_time': np.array([[0.0, 0.0, 2.0, 2.0, 4.0]], dtype=np.float32),
            'temp_class': np.array([[0, 0, 1, 2, 0]], dtype=np.int32),
            'initial_quality': np.ones((1, n_nodes), dtype=np.float32),
        }

        class RoundRobin(object):
            """The smallest replanner that produces events: one customer per vehicle, then return."""

            def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=None):
                free = [c for c in visible_ids if not served_mask[c]
                        and c not in env.get_reserved_customers(vehicles)]
                it = iter(sorted(free))
                for v in vehicles:
                    if v.status not in ('idle', 'ready'):
                        continue
                    if replan_ids is not None and v.vehicle_id not in replan_ids:
                        continue
                    nxt = next(it, None)
                    v.mutable_suffix = ([int(nxt), 0] if nxt is not None
                                        else ([0] if v.current_node != 0 else []))

        from strict_online_env import StrictOnlineEnv
        env = StrictOnlineEnv(data, capacity=contract.operational.shared_vehicle_capacity,
                              num_vehicles=2, replanner=RoundRobin(),
                              coldchain_contract=contract)
        counts, clocks = {}, []
        orig = env._advance_coldchain_to

        def counting(v, clock):
            counts[v.vehicle_id] = counts.get(v.vehicle_id, 0) + 1
            return orig(v, clock)

        env._advance_coldchain_to = counting
        env.snapshot_hook = lambda *a, **k: clocks.append(float(a[2]))
        traces, served = env.run(0)
        n_iter_observed = int(env.event_id)
        max_adv = max(counts.values()) if counts else 0
        gaps = [b - a for a, b in zip(clocks, clocks[1:])]
        min_gap = min(gaps) if gaps else None
        bound_iter = G2.iteration_bound(w0['depot_tw_end'], census['max_iter'])
        ok_syn = (max_adv <= budget['n_max_per_vehicle']
                  and n_iter_observed <= bound_iter['bound']
                  and (min_gap is None or min_gap > 1e-6))
        ck('G2-SYN', 'synthetic 5-node episode (DIAGNOSTIC, not a rollout): the observed per-vehicle '
           'advance count and iteration count stay below the derived bounds, and consecutive event '
           'clocks are more than eps_event apart -- the two facts the derivation relies on',
           ok_syn,
           {'per_vehicle_advance_counts': counts, 'max_observed': max_adv,
            'n_max_per_vehicle': budget['n_max_per_vehicle'],
            'iterations_observed': n_iter_observed, 'iteration_bound': bound_iter['bound'],
            'min_clock_gap': min_gap, 'clock_samples': clocks[:8],
            'services': {t.vehicle_id: list(t.services and [r.node for r in t.services])
                         for t in traces},
            'grading': 'DIAGNOSTIC: this is one hand-built episode, not a measurement of the bound; '
                       'the bound itself is proved from the call graph'},
           conclusion='the structural reasoning survives contact with an actual execution')

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for r in ROWS if r['status'] == 'PASS')
    n_skip = sum(1 for r in ROWS if r['status'] == 'SKIP')
    n_fail = sum(1 for r in ROWS if r['status'] == 'FAIL')
    if n_fail:
        verdict, why = 'G2_QUALIFICATION_FAIL', '%d control(s) failed' % n_fail
    elif n_skip:
        verdict, why = ('G2_QUALIFICATION_INCOMPLETE',
                        '%d control(s) could not be evaluated' % n_skip)
    else:
        verdict, why = 'G2_QUALIFICATION_PASS', 'all controls ran and passed'

    payload = {
        'schema': 'q3_g2_time_identity_controls',
        'kind': 'DIAGNOSTIC', 'not_for_production': True,
        'baseline_head': 'a71a0c3cede64b15e20f34f7833134d6236bb7a1',
        'preregistration': 'scripts/q3_qualification/q3_g2_preregistration_2026-10-17.md',
        'verdict': verdict, 'verdict_reason': why,
        'n_rows': len(ROWS), 'n_pass': n_pass, 'n_skip': n_skip, 'n_fail': n_fail,
        'contract_hash': contract.contract_hash,
        'source_identity': source,
        'census': census,
        'advance_budget': budget, 'slack': slack_new,
        'identity': ident,
        'matrix': G2.parameter_matrix(),
        'negative_controls': nc,
        'rows': ROWS,
        'status_lines': {
            'GTMIN_V2_ACCEPTANCE_FAIL_AS_LITERALLY_PREREGISTERED': 'permanent, not retro-approved',
            'THERMAL_CONTINUOUS_NOT_ESTABLISHED': True,
            'Q3_FORMAL_FREEZE_NOT_AUTHORISED': True,
            'M2_NOT_AUTHORISED': True, 'L1': 'BLOCKED',
            'L2_NOT_ESTABLISHED': True, 'A3_NOT_ESTABLISHED': True,
            'stable_online_gain': False, 'HISTORICAL_BACKBONE_UNVERIFIED': True,
            'C0': 'AWAITING_USER_BUDGET_RULING', 't_min': 'always 0 (no trusted witness)',
            'final_test_seeds': 'SEALED'},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n=== %d controls: %d PASS, %d SKIP, %d FAIL -> %s ==='
          % (len(ROWS), n_pass, n_skip, n_fail, verdict))
    print('reason: %s' % why)
    print('WROTE %s' % args.out)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
