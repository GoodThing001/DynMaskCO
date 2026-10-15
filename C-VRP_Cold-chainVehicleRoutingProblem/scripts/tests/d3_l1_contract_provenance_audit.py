#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 CONTRACT-PROVENANCE CLOSURE (read-only; user-authorised).

Goal: recover the historical CALIBRATION contract `0f00af5d...` and state the correct identity
chain, instead of judging it by equality with the effective contract.

Erratum this tool exists to support: the previous round's A3 assertion compared
`profile.provenance.contract_hash` (the contract the profile was CALIBRATED under, written at
freeze_coldchain_scale.py:251) with the contract obtained by applying that profile
(`apply_objective_profile`).  Those are different STAGES and may legitimately differ, so the
equality test was the wrong criterion.  This tool therefore establishes three identities:

    C_calibration        0f00af5d...   contract the profile was calibrated under
    C_base_current       32124bfd...   current default_pilot_contract()
    C_effective_current  edc1b2ea...   C_base_current + apply_objective_profile(profile)

Recovery routes tried (all read-only, nothing written outside --outdir):
  R1 a stored contract manifest (validate_contract_manifest and/or a 'contract' section inside
     the frozen manifests of the scale_v2 / dev_gate_v4 line)
  R2 the calibration-time SOURCE of coldchain_contract.py, located by the SHA recorded in
     profile.provenance.code_sha256['coldchain_contract'] = abf97d97..., then calling that
     module's default_pilot_contract() in isolation

If a route yields a contract whose recomputed hash equals 0f00af5d..., the field-by-field
difference against C_base_current is reported by section (physical/units/quality/objective), the
profile is applied to the recovered calibration contract and the full chain
C_calibration -> profile -> C_effective is recorded.  No guessed configuration is ever presented
as the historical contract; if nothing matches, the verdict is CONTRACT_NOT_RECOVERED.
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
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data'):
    sys.path.insert(0, str(EXT / p))
REPO = EXT.parent
TARGET = '0f00af5d197cf35368b1b2e807bcaf53d9b58a1cd425a7f53ec66091f6acce28'
CAL_SRC = 'abf97d9789f60450fdd58ea79205cd5b32534ab460b18af1431a86e4cfbca601'
RECORDS = {}


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def as_plain(x, depth=0):
    if depth > 8:
        return '<depth>'
    if dataclasses.is_dataclass(x):
        return {f.name: as_plain(getattr(x, f.name), depth + 1) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): as_plain(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [as_plain(i, depth + 1) for i in x]
    if hasattr(x, 'item'):
        try:
            return x.item()
        except Exception:
            return repr(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def section_diff(a, b, path='C'):
    """All leaf differences between two plain structures."""
    out = []

    def walk(x, y, p):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y), key=str):
                if k not in x or k not in y:
                    out.append({'path': '%s.%s' % (p, k), 'kind': 'missing',
                                'a': k in x, 'b': k in y})
                else:
                    walk(x[k], y[k], '%s.%s' % (p, k))
            return
        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                out.append({'path': '%s.len' % p, 'kind': 'length', 'a': len(x), 'b': len(y)})
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, '%s[%d]' % (p, i))
            return
        if x != y:
            out.append({'path': p, 'kind': 'value', 'a': repr(x)[:120], 'b': repr(y)[:120]})

    walk(a, b, path)
    return out


def try_manifest_routes():
    found = []
    cands = []
    for pat in ('results/o0cc/scale_v2/manifest.json', 'results/o0cc/service_audit/manifest.json',
                'results/o0cc/dev_gate_v4_official/pre_run_manifest.json'):
        p = EXT / pat
        if p.is_file():
            cands.append(p)
    cands += sorted((EXT / 'results/o0cc/dev_gate_v4_official').glob('*/summary.json'))[:2] \
        if (EXT / 'results/o0cc/dev_gate_v4_official').is_dir() else []
    for p in cands:
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:
            found.append({'file': str(p.relative_to(EXT)), 'error': str(exc)})
            continue
        hits = [k for k in d if 'contract' in k.lower()]
        found.append({'file': str(p.relative_to(EXT)), 'sha256': sha(p), 'contract_keys': hits,
                      'has_full_contract_section': any(isinstance(d[k], dict) and
                                                       len(d[k]) > 5 for k in hits)})
    return found


def try_source_route():
    """Find a coldchain_contract.py whose sha equals the calibration SHA, then call its
    default_pilot_contract() in isolation."""
    hits, tried = [], []
    roots = [EXT / 'scripts', EXT / 'archive', EXT / 'negative_result_paper',
             EXT / '03_负结果线', EXT / 'results']
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob('coldchain_contract.py'):
            sp = str(p)
            if '_bverify' in sp or 'shadow' in sp or sp in seen:
                continue
            seen.add(sp)
            h = sha(p)
            if h == CAL_SRC:
                hits.append(p)
    for p in hits[:3]:
        try:
            _n = 'cc_cal_%d' % (tried.__len__())
            spec = importlib.util.spec_from_file_location(_n, p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[_n] = mod            # dataclasses resolve cls.__module__ via sys.modules
            spec.loader.exec_module(mod)
            c = mod.default_pilot_contract()
            tried.append({'source': str(p.relative_to(EXT)), 'route': 'archived_module',
                          'recomputed_hash': getattr(c, 'contract_hash', None),
                          'matches_target': getattr(c, 'contract_hash', None) == TARGET})
        except BaseException as exc:
            tried.append({'source': str(p.relative_to(EXT)), 'route': 'archived_module',
                          'error': '%s: %s' % (type(exc).__name__, str(exc)[:300])})
    return {'files_with_calibration_sha': [str(p.relative_to(EXT)) for p in hits],
            'attempts': tried}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default=str(EXT / 'results/o0cc/scale_v2/'
                                             'objective_profile.json'))
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile, default_pilot_contract,
                                    validate_contract_manifest)

    prof_dict = json.loads(Path(args.profile).read_text(encoding='utf-8'))
    fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    prof = ObjectiveProfile(**{k: v for k, v in prof_dict.items() if k in fields})

    cur = default_pilot_contract()
    RECORDS['identities'] = {
        'C_calibration_declared': prof_dict.get('provenance', {}).get('contract_hash'),
        'C_base_current': getattr(cur, 'contract_hash', None),
        'C_base_current_definition': 'default_pilot_contract() of the CURRENT '
                                     'scripts/coldchain/coldchain_contract.py',
        'current_source_sha256': sha(EXT / 'scripts/coldchain/coldchain_contract.py'),
        'calibration_source_sha256_declared': (prof_dict.get('provenance', {})
                                              .get('code_sha256', {})
                                              .get('coldchain_contract')),
        'target_hash': TARGET,
        'stages_are_distinct_by_definition': True}
    print('C_calibration=%s' % str(RECORDS['identities']['C_calibration_declared'])[:16])
    print('C_base_current=%s (source %s)' % (str(RECORDS['identities']['C_base_current'])[:16],
                                             str(RECORDS['identities']
                                                 ['current_source_sha256'])[:16]))
    print('calibration source sha declared=%s'
          % str(RECORDS['identities']['calibration_source_sha256_declared'])[:16])

    RECORDS['route_R1_manifests'] = try_manifest_routes()
    RECORDS['route_R2_archived_source'] = try_source_route()

    recovered = None
    for at in RECORDS['route_R2_archived_source']['attempts']:
        if at.get('matches_target'):
            recovered = at['source']
    RECORDS['recovery'] = {'recovered_calibration_source_file': recovered,
                           'verdict': ('CONTRACT_RECOVERED' if recovered
                                       else 'CONTRACT_NOT_RECOVERED')}
    if recovered:
        path = EXT / recovered
        spec = importlib.util.spec_from_file_location('cc_cal_final', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['cc_cal_final'] = mod
        spec.loader.exec_module(mod)
        cal = mod.default_pilot_contract()
        RECORDS['field_diff_calibration_vs_base_current'] = section_diff(
            as_plain(cal), as_plain(cur), 'C')
        eff_cal = apply_objective_profile(cal, prof)
        eff_cur = apply_objective_profile(cur, prof)
        RECORDS['chain'] = {
            'C_calibration_hash': getattr(cal, 'contract_hash', None),
            'profile_hash_declared': prof_dict.get('profile_hash'),
            'C_effective_from_calibration': getattr(eff_cal, 'contract_hash', None),
            'C_effective_from_base_current': getattr(eff_cur, 'contract_hash', None),
            'effective_hashes_equal': (getattr(eff_cal, 'contract_hash', None)
                                       == getattr(eff_cur, 'contract_hash', None))}
    else:
        RECORDS['chain'] = None

    # correct criterion, stated mechanically
    RECORDS['corrected_criterion'] = {
        'wrong_previous_criterion': 'C_calibration == C_effective_current',
        'why_wrong': 'C_calibration is the contract the profile was CALIBRATED under '
                     '(freeze_coldchain_scale.py:200-201 loads it, :251 writes its hash into '
                     'provenance); C_effective_current is that profile APPLIED to a contract. '
                     'They are different stages and may legitimately differ.',
        'correct_criteria': [
            'the calibration source must be independently recoverable and hash-verifiable',
            'the profile application relation must be recomputable: C_base + profile -> '
            'C_effective with a recorded hash',
            'the L1 protocol must NAME which base contract it uses, and must not silently mix '
            'the calibration source with the current default'],
        'a3_original_failure_preserved': True,
        'verdict': 'IDENTITY_PARTIAL' if not recovered else 'CALIBRATION_SOURCE_RECOVERED'}

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RECORDS, indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('VERDICT:', RECORDS['recovery']['verdict'], '|',
          RECORDS['corrected_criterion']['verdict'])
    if RECORDS.get('chain'):
        print('chain:', json.dumps(RECORDS['chain'])[:300])
    print('field differences:', len(RECORDS.get('field_diff_calibration_vs_base_current') or []))
    print('WROTE', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
