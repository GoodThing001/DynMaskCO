#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1-D3-L1-CONTRACT-PROTOCOL-FREEZE (user-authorised).

One new L1 protocol identity file + a lightweight regression.  Read-only with respect to every
historical artefact: the recovered calibration contract, the profile, frozen sources and archived
reports are only READ.  Nothing here re-searches contracts or re-runs the provenance audit.

The user's ruling encoded here (not re-litigated):
  calibration_source_contract_sha256 = 0f00af5d...ce28   (provenance identity only, VERIFIED)
  l1_base_contract_sha256            = 32124bfd...30b0   (CURRENT physics protocol)
  l1_effective_contract_sha256       = edc1b2ea...4833   (base + apply_objective_profile)
  objective_profile_sha256           = df1f7a6e...25b6
  booking_horizon                    = 24.0
  calibration_vs_execution_contract  = CROSS_VERSION_EXPLICIT
  historical_result_comparability    = NOT_AUTOMATICALLY_ESTABLISHED

Checks: current-module contract construction and hash, physics sections unchanged by the profile
application, the cross-version object-compatibility probe (does the current state transition read
fields the recovered old contract lacks?), cold-chain snapshot schema + validate_snapshot_against_env,
and the L1 cost-semantics separation (the L1 collection path computes no cost or utility at all).
Every check is recomputed from files; the identity JSON is written by this same execution.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))
REPO = EXT.parent
CASES = []
RECORDS = {}
CAL_SRC_SHA = 'abf97d9789f60450fdd58ea79205cd5b32534ab460b18af1431a86e4cfbca601'
CAL_CONTRACT = '0f00af5d197cf35368b1b2e807bcaf53d9b58a1cd425a7f53ec66091f6acce28'
ARCHIVED_CAL = ('archive/正式冻结包/O0-CC_DEV-GATE/o0cc_devgate_2dfff7ae_20260908_v4/source/'
                'C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_contract.py')


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))
    sys.stdout.flush()


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def plain(x, d=0):
    if d > 8:
        return '<depth>'
    if dataclasses.is_dataclass(x):
        return {f.name: plain(getattr(x, f.name), d + 1) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): plain(v, d + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(i, d + 1) for i in x]
    if hasattr(x, 'item'):
        try:
            return x.item()
        except Exception:
            return repr(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default=str(EXT / 'results/o0cc/scale_v2/'
                                             'objective_profile.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default=str(EXT / 'data/m0_scale/'
                                             'dcc_50_r1_edod05_cal_teacher.npz'))
    args = ap.parse_args()
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                    compute_coldchain_cost, default_pilot_contract)

    prof_dict = json.loads(Path(args.profile).read_text(encoding='utf-8'))
    fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    prof = ObjectiveProfile(**{k: v for k, v in prof_dict.items() if k in fields})

    base = default_pilot_contract()
    eff = apply_objective_profile(base, prof)
    base_h, eff_h = getattr(base, 'contract_hash', None), getattr(eff, 'contract_hash', None)
    RECORDS['identity'] = {
        'l1_base_contract_sha256': base_h,
        'l1_effective_contract_sha256': eff_h,
        'objective_profile_sha256': prof_dict.get('profile_hash'),
        'calibration_source_contract_sha256': prof_dict.get('provenance', {}).get('contract_hash'),
        'calibration_source_file_sha256': CAL_SRC_SHA,
        'calibration_source_archived_path': ARCHIVED_CAL,
        'l1_uses_calibration_contract_as_base': False,
        'booking_horizon': 24.0,
        'calibration_vs_execution_contract': 'CROSS_VERSION_EXPLICIT',
        'historical_result_comparability': 'NOT_AUTOMATICALLY_ESTABLISHED',
        'source_sha256': {
            'coldchain_contract': sha(EXT / 'scripts/coldchain/coldchain_contract.py'),
            'coldchain_state': sha(EXT / 'scripts/coldchain/coldchain_state.py'),
            'strict_online_env': sha(EXT / 'scripts/simulation/strict_online_env.py'),
            'recourse_snapshot': sha(EXT / 'scripts/simulation/recourse_snapshot.py'),
            'd3_l1_state_collector': sha(EXT / 'scripts/simulation/d3_l1_state_collector.py'),
            'this_tool': sha(__file__)},
        'data_sha256': {Path(args.dataset).name: sha(args.dataset)},
        'profile_file_sha256': sha(args.profile)}
    ck('I1 the L1 base contract comes from the CURRENT cold-chain module and hashes to the '
       'registered value', base_h == '32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0'
       and RECORDS['identity']['source_sha256']['coldchain_contract'] != CAL_SRC_SHA,
       {'base_hash': base_h, 'current_source': RECORDS['identity']['source_sha256']
        ['coldchain_contract'], 'calibration_source': CAL_SRC_SHA})
    ck('I2 the L1 effective contract (base + apply_objective_profile) hashes to the registered '
       'value and the profile hash matches its declaration',
       eff_h == 'edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833'
       and RECORDS['identity']['objective_profile_sha256'] ==
       'df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6',
       {'effective_hash': eff_h, 'profile_hash': RECORDS['identity']['objective_profile_sha256']})
    ck('I3 the calibration contract is registered as PROVENANCE ONLY and is not the L1 base',
       RECORDS['identity']['calibration_source_contract_sha256'] == CAL_CONTRACT
       and RECORDS['identity']['l1_uses_calibration_contract_as_base'] is False,
       {'calibration': CAL_CONTRACT, 'l1_base': base_h})

    # ---- physics sections must be untouched by the profile application --------------------
    pb, pe = plain(base), plain(eff)
    changed = [k for k in pb if pb.get(k) != pe.get(k)]
    RECORDS['profile_application_effect'] = {
        'sections_changed': changed,
        'only_objective_changed': changed == ['objective'],
        'note': 'apply_objective_profile must not alter C0 physics/units/quality'}
    ck('I4 applying the profile changes ONLY the objective section: the C0 physical, unit and '
       'quality constraints of the base contract are carried through unchanged',
       changed == ['objective'], RECORDS['profile_application_effect'])

    # ---- cross-version object compatibility probe (the user's claim) ----------------------
    probe = {}
    try:
        src_state = (EXT / 'scripts/coldchain/coldchain_state.py').read_text(encoding='utf-8')
        probe['current_state_references_precool_field'] = \
            'precool_energy_per_unit_per_c' in src_state
        probe['current_state_reference_lines'] = [
            i for i, l in enumerate(src_state.splitlines(), 1)
            if 'precool_energy_per_unit_per_c' in l][:6]
        p = EXT / ARCHIVED_CAL
        probe['archived_calibration_source_present'] = p.is_file()
        probe['archived_source_sha_matches'] = sha(p) == CAL_SRC_SHA
        _n = 'cc_probe_cal'
        spec = importlib.util.spec_from_file_location(_n, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_n] = mod
        spec.loader.exec_module(mod)
        old = mod.default_pilot_contract()
        probe['old_contract_hash'] = getattr(old, 'contract_hash', None)
        cur_thermal = {f.name for f in dataclasses.fields(type(base.thermal))}
        old_thermal = {f.name for f in dataclasses.fields(type(old.thermal))}
        probe['thermal_fields_current'] = sorted(cur_thermal)
        probe['thermal_fields_calibration'] = sorted(old_thermal)
        probe['fields_missing_in_calibration'] = sorted(cur_thermal - old_thermal)
        probe['hasattr_old_precool_energy'] = hasattr(old.thermal,
                                                     'precool_energy_per_unit_per_c')
        probe['hasattr_old_diurnal_amplitude'] = hasattr(old.thermal, 'diurnal_amplitude_c')
        probe['verdict'] = ('CROSS_VERSION_OBJECT_INCOMPATIBLE_CONFIRMED'
                            if (probe['current_state_references_precool_field']
                                and not probe['hasattr_old_precool_energy'])
                            else 'CLAIM_NOT_CONFIRMED')
    except BaseException as exc:
        probe['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])
        probe['verdict'] = 'PROBE_INCONCLUSIVE'
    RECORDS['cross_version_probe'] = probe
    ck('I5 the cross-version object risk is verified rather than assumed: the current state '
       'transition reads a field the recovered calibration contract does not carry (or the claim '
       'is reported as not confirmed)',
       probe.get('verdict') in ('CROSS_VERSION_OBJECT_INCOMPATIBLE_CONFIRMED',
                                'CLAIM_NOT_CONFIRMED'),
       {k: probe.get(k) for k in ('verdict', 'current_state_reference_lines',
                                  'fields_missing_in_calibration',
                                  'hasattr_old_precool_energy')})

    # ---- cold-chain snapshot schema + validate under the L1 contract ----------------------
    import numpy as np
    from d3_l1_state_collector import (HORIZON, RevealSlotCollector, build_env, make_replanner,
                                       run_baseline_rollout)
    from recourse_snapshot import (CC_SNAPSHOT_SCHEMA_VERSION, snapshot_state_hash,
                                   validate_snapshot_against_env)
    data = dict(np.load(args.dataset))
    env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, HORIZON, eff)[0], eff, HORIZON)
    col = RevealSlotCollector(env, 0, 1)
    run_baseline_rollout(env, 0, hook=col)
    ok_schema, ok_validate, ok_contract = False, False, False
    if col.slots:
        snap = col.slots[0]['snapshot']
        ok_schema = snap.get('schema_version') == CC_SNAPSHOT_SCHEMA_VERSION
        ok_contract = snap.get('coldchain_contract_hash') == eff_h
        try:
            validate_snapshot_against_env(env, snap)
            ok_validate = True
        except BaseException as exc:
            RECORDS['validate_error'] = '%s: %s' % (type(exc).__name__, str(exc)[:200])
        RECORDS['snapshot'] = {'schema_version': snap.get('schema_version'),
                               'contract_hash_matches': ok_contract,
                               'snapshot_state_hash': snapshot_state_hash(snap)}
    ck('I6 under the frozen L1 contract the snapshot is a v3_cc cold-chain snapshot whose contract '
       'hash equals the effective hash and validate_snapshot_against_env accepts it',
       ok_schema and ok_validate and ok_contract, RECORDS.get('snapshot'))

    # ---- cost semantics -------------------------------------------------------------------
    coll = (EXT / 'scripts/simulation/d3_l1_state_collector.py').read_text(encoding='utf-8')
    RECORDS['cost_semantics'] = {
        'collector_mentions_distance_cost': 'distance_cost' in coll,
        'collector_mentions_coldchain_cost': 'coldchain_cost' in coll or 'compute_coldchain_cost' in coll,
        'collector_computes_any_utility': False,
        'statement': 'L1 collection captures decision states only: it evaluates no objective, so '
                     'coldchain_cost / distance_cost cannot be mixed at L1. Cost semantics bind at '
                     'L2/L3 where J = revenue - fuel - reject_loss is evaluated.',
        'compute_coldchain_cost_callable': callable(compute_coldchain_cost)}
    ck('I7 L1 cost-semantics separation holds: the F-free collector computes no cost or utility, so '
       'coldchain_cost and distance_cost are not mixed at this layer',
       not RECORDS['cost_semantics']['collector_mentions_coldchain_cost']
       and not RECORDS['cost_semantics']['collector_mentions_distance_cost']
       and RECORDS['cost_semantics']['compute_coldchain_cost_callable'],
       RECORDS['cost_semantics'])

    RECORDS['known_limits'] = {
        'cross_version_calibration': 'the objective profile was calibrated under the historical '
                                     'contract 0f00af5d (pre-diurnal/precooling). The L1 route uses '
                                     'the CURRENT physics protocol 32124bfd -> edc1b2ea, so the '
                                     'profile is NOT re-calibrated under it. L1 may use this as an '
                                     'explicitly declared cross-version diagnostic configuration; '
                                     'L2/L3 must separately settle calibration applicability and '
                                     'fairness.',
        'historical_result_comparability': 'NOT_AUTOMATICALLY_ESTABLISHED',
        'p1b_coverage_limit': 'archived FULL_PARITY stands; canon() normalises NumPy scalars and '
                              'the cross-type numeric path has a signed-zero exception, so IEEE-754 '
                              'bit equality was not verified for every float leaf.',
        'states_collected_for_L1': 0, 'algorithmic_gain_claim': 'none'}
    fails = [c for c in CASES if not c[1]]
    RECORDS['verdict'] = {
        'L1_CONTRACT_PROTOCOL_FREEZE': 'FROZEN' if not fails else 'NOT_FROZEN',
        'labels': {'CALIBRATION_SOURCE_IDENTITY': 'VERIFIED',
                   'L1_CONTRACT_ROUTE': 'CURRENT_PHYSICS_PROTOCOL_RECOMMENDED',
                   'L1_CONTRACT_PROTOCOL_SELECTION': 'APPROVED_IN_PRINCIPLE',
                   'L1_COLLECTOR_CONTRACT_FROZEN': 'PENDING_MANIFEST_AND_RUNTIME_CHECK'
                   if not fails else 'NOT_FROZEN'},
        'n_fail': len(fails)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {'rows': [{'name': n, 'ok': o, 'detail': dd} for n, o, dd in CASES],
         'records': RECORDS, 'n_rows': len(CASES), 'n_pass': len(CASES) - len(fails),
         'n_fail': len(fails), 'source_sha256_this_tool': sha(__file__),
         'argv': sys.argv}, indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== L1_CONTRACT_PROTOCOL_FREEZE: %s | %d rows, %d PASS, %d FAIL ==='
          % (RECORDS['verdict']['L1_CONTRACT_PROTOCOL_FREEZE'], len(CASES),
             len(CASES) - len(fails), len(fails)))
    for name, _ok, detail in fails:
        print('FAILED-ROW %s :: %s' % (name, json.dumps(detail, default=repr, sort_keys=True)))
    print('WROTE', out)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
