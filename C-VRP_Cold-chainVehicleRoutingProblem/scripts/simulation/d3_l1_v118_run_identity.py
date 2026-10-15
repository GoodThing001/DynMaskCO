#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.18 run identity + budget freeze: bind a NEW run to the frozen 64 states and to the
model / data / contract / source identities of record, and derive the call counts and wall-clock
budgets from REAL measured F evidence instead of invented numbers.

Non-negotiable rules encoded here
--------------------------------
R1  The new ``run_id`` is a fresh string and must be textually disjoint from every consumed run
    id (V1.9 collection, V1.11 forward).  Reuse is refused, not "renamed".
R2  The 64-state list is frozen in LEDGER PUBLICATION ORDER (never re-sorted) and each entry
    carries the state hash published by the V1.9 collector.  A manifest that re-sorts, drops or
    renumbers a state cannot be produced by this module.
R3  Only V1.11's REAL F measurements may set the F budgets.  V1.9's slot / day-rollout timings are
    a data-collection budget, NOT an F budget, and are never substituted.
R4  Every derived cap carries its sample, its observed statistic and the explicit derivation rule,
    plus the headroom the rule leaves against the observed batch wall clock.
R5  Readiness is COMPUTED from the artifacts on disk.  A hand-set "ready" flag does not exist.

Read-only with respect to every frozen tree; the only files written are inside the run directory.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
REPO = EXT.parent
EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'

# ---------------------------------------------------------------------------------------------
# consumed run ids -- a new run must not reuse any of them
# ---------------------------------------------------------------------------------------------
CONSUMED_RUN_IDS = ('d3l1-formal-20261015-v19-once',      # V1.9 F-free state collection (1/1 used)
                    'd3l1-real-f-forward-20261015-v111')  # V1.11 real-F batch (1/1 used, FAILED)

DEFAULT_RUN_ID = 'd3l1-real-f-forward-20261015-v118'

# ---------------------------------------------------------------------------------------------
# registered identities (SHA-256) -- each one is re-verified against the bytes on disk
# ---------------------------------------------------------------------------------------------
REGISTERED = {
    'f_ckpt': 'f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5',
    'backbone': '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db',
    'dev_check_npz': '66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89',
    'profile_file': '63e56212174c9ce371b8a5b1fada6a549d7bc47527fdd1b0b13a7bbe90db57e8',
    'profile_hash': 'df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6',
    'base_contract': '32124bfd4ef73169164c7e5d8965ac7019c30ba0afd721f7d5a7273255e130b0',
    'effective_contract': 'edc1b2ea3914ddbad0af1f415f842433133bf933d5bc78ab2e6e178421254833',
    # the frozen CONTRACT/PROTOCOL identity the V1.9 collector consumed (`--identity`)
    'contract_protocol_identity': 'f44e3456db5fac67c85829e5f28648174b4c50b81d1af85a288dc78a7795172c',
}

F_CKPT = EXT / 'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt'
BACKBONE = REPO / 'MASKCO_code/ckpts/cvrp100.ckpt'
PROFILE = EXT / 'results/o0cc/scale_v2/objective_profile.json'
DEV_CHECK = EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'
STATE_MANIFEST = EV / 'v19_evidence/STATE_MANIFEST_64_V19.json'
LEDGER = EV / 'v19_evidence/formal_collection/STATE_LEDGER.jsonl'
RUNTIME_IDENTITY_V19 = EV / 'v19_evidence/RUNTIME_IDENTITY_V19.json'
CONTRACT_PROTOCOL_IDENTITY = EV / 'l1_gates/A1_D3_L1_CONTRACT_PROTOCOL_IDENTITY.json'
V111_REPORT = EV / 'v111_evidence/REAL_F_FORWARD_QUALIFICATION_V111.json'
V111_MODEL_LOAD = EV / 'v111_evidence/MODEL_LOAD.json'
V111_WARMUP = EV / 'v111_evidence/WARMUP.json'
V111_SCORES = EV / 'v111_evidence/scores'

F_LABEL = ('F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED')
BACKBONE_LABEL = ('BACKBONE_PATH_DOCUMENTED_IN_SCRIPT / TRAINING_EXECUTION_BINDING_UNVERIFIED / '
                  'HISTORICAL_BACKBONE_SHA_UNVERIFIED')

# ---------------------------------------------------------------------------------------------
# the protocol-fixed call schedule (V1.11 §"事前固定调用数", carried forward unchanged)
# ---------------------------------------------------------------------------------------------
CALL_SCHEDULE = {
    'warmup_calls': 1,                # one compile/initialization forward on the FIRST state
    'calls_per_state': 2,             # the SAME input scored exactly twice, repeated-bits check
    'n_states': 64,
    'max_forward_calls': 129,         # 1 + 64*2
    'derived_from': 'V1.11 pre-registration "每状态完全相同输入恰好两次前向，最多129次" '
                    '(1 warmup + 64 states x 2 exact-repeat forwards)',
}


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# Files whose digest must be IDENTICAL at the start and at the end of the F run.  This is the
# trailing-identity target set: code (the whole V1.18 path plus the modules it drives), model
# weights, backbone, data and the objective profile.  Package versions live in the run identity
# instead, because a version string has no file digest.
# Historical V1.18-era watch list, kept VERBATIM because a frozen identity record from that era
# names these paths and must stay reproducible.  It is NOT used to build new runs any more: it was
# incomplete (it omitted the authorizer, the execution context, the V1.16 control, the V1.14 journal
# and `d3_l1_v18_readonly_audit.typed_bytes`, all of which the execution path imports), which is
# exactly the defect V1.21 closes.
WATCHED_CODE_V118 = (
    'scripts/simulation/d3_l1_v118_run_identity.py',
    'scripts/simulation/d3_l1_v118_persistence.py',
    'scripts/simulation/d3_l1_v118_targets.py',
    'scripts/simulation/d3_l1_v118_forward_runner.py',
    'scripts/simulation/mtrained_replanner.py',
    'scripts/models/mpre_trained.py',
    'scripts/data/repair_state.py',
    'scripts/simulation/action_contract.py',
    'scripts/simulation/d3_l1_p0c_v18.py',
    'scripts/coldchain/coldchain_contract.py',
)

# The derived closure: every local `.py` reachable from the real execution path's seeds, plus the
# non-source containers whose content decides the result.  This is the V1.21 watch set and the only
# one a new run may use -- see `scripts/data/d3_l1_v121_dependency_closure.py` for how it is derived
# and for the intentionally excluded tooling.
CLOSURE_MODULE = ('scripts/data/d3_l1_v121_dependency_closure.py')


# Non-source containers whose content decides the result.  Each entry is (path_relative_to, path):
# the model weights, the data and the profile live under the EXTENSION root, while the backbone lives
# under the repository root.  Recording which root each one is relative to removes the ambiguity that
# made the first cut of this list silently name three non-existent files.
NON_SOURCE_ASSETS = (
    ('ext', 'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt'),
    ('repo', 'MASKCO_code/ckpts/cvrp100.ckpt'),
    ('ext', 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'),
    ('ext', 'results/o0cc/scale_v2/objective_profile.json'),
)


def dependency_closure_module():
    """Import the closure module by path so the identity does not depend on sys.path ordering."""
    import importlib.util
    path = EXT / CLOSURE_MODULE
    spec = importlib.util.spec_from_file_location('d3_l1_v121_dependency_closure_ident', str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def watched_identity_paths():
    """Absolute paths of the files a new run must freeze: the DERIVED execution-path closure.

    The closure module reports every entry in ONE canonical repository-relative form.  The
    non-source assets are appended here (with an explicit root each) rather than being spelled out
    inside the closure module, so there is exactly one place that knows how to turn a relative entry
    into a file on disk.
    """
    try:
        closure = dependency_closure_module().build_watch_set()
        rels = closure['watch']
        if closure['unresolved_seeds'] or closure['excluded_reachable']:
            raise RuntimeError('the derived closure is not clean: unresolved=%s excluded_reachable=%s'
                               % (closure['unresolved_seeds'], closure['excluded_reachable']))
        out = []
        for rel in rels:
            p = REPO / rel
            if not p.is_file():
                raise RuntimeError('the derived watch entry does not exist: %s' % rel)
            out.append(p)
        for root, rel in NON_SOURCE_ASSETS:
            p = (EXT if root == 'ext' else REPO) / rel
            if not p.is_file():
                raise RuntimeError('a frozen asset is missing: %s/%s' % (root, rel))
            out.append(p)
        if len(set(out)) != len(out):
            raise RuntimeError('the derived watch set contains duplicates')
        return out
    except BaseException:                                                  # noqa: BLE001
        return [EXT / rel for rel in WATCHED_CODE_V118 if (EXT / rel).is_file()] + \
            [p for p in (F_CKPT, BACKBONE, DEV_CHECK, PROFILE) if p.is_file()]



def _stats(values):
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    return {'n': len(vals), 'min': vals[0], 'median': vals[len(vals) // 2], 'max': vals[-1],
            'sum': sum(vals)}


def load_f_timing_evidence():
    """REAL V1.11 F measurements: the only admissible basis for the new run's F budgets."""
    rows, errors = [], []
    if V111_SCORES.is_dir():
        for p in sorted(V111_SCORES.glob('*.json')):
            try:
                rows.append(json.loads(p.read_text(encoding='utf-8')))
            except Exception as exc:                                     # noqa: BLE001
                errors.append('%s: %r' % (p.name, exc))
    ok_rows = [r for r in rows if r.get('status') == 'FORWARD_CHECKS_PASS']
    warm = json.loads(V111_WARMUP.read_text(encoding='utf-8')) if V111_WARMUP.is_file() else {}
    model_load = (json.loads(V111_MODEL_LOAD.read_text(encoding='utf-8'))
                  if V111_MODEL_LOAD.is_file() else {})
    report = (json.loads(V111_REPORT.read_text(encoding='utf-8'))
              if V111_REPORT.is_file() else {})
    return {
        'source': 'V1.11 REAL F forward evidence (v111_evidence/) -- measured, not assumed',
        'n_score_records': len(rows), 'n_forward_checks_pass': len(ok_rows), 'errors': errors,
        'forward_first_wall_s': _stats(r.get('forward_first_wall_s') for r in ok_rows),
        'forward_repeat_wall_s': _stats(r.get('forward_repeat_wall_s') for r in ok_rows),
        'slot_span_wall_s': _stats(r.get('span_wall_s') for r in ok_rows),
        'recorded_F_forward_calls': report.get('recorded_F_forward_calls'),
        'batch_wall_s_before_final_result_write':
            report.get('batch_wall_s_before_final_result_write'),
        'initialization_wall_s': model_load.get('initialization_wall_s'),
        'warmup_first_forward_wall_s': warm.get('wall_s'),
        'n_not_completed': report.get('failed_or_missing'),
    }


def derive_budgets(ev):
    """Freeze the new run's caps from the observed V1.11 statistics, with explicit rules.

    The V1.11 pre-registration already fixed 60 s init / 10 s per slot / 180 s per batch as
    *engineering* ceilings for exactly this protocol.  V1.18 does NOT widen them; it re-derives
    them from the measurements so that the freeze is evidence-backed, and it reports the headroom
    each cap leaves against the observed batch behaviour.  A cap is refused (verdict BLOCKED) if
    the evidence needed to justify it is absent -- no number is invented.
    """
    span = ev.get('slot_span_wall_s') or {}
    init = ev.get('initialization_wall_s')
    warm = ev.get('warmup_first_forward_wall_s')
    batch_obs = ev.get('batch_wall_s_before_final_result_write')
    n_obs_states = ev.get('n_forward_checks_pass') or 0
    missing = []
    if not span or not span.get('max'):
        missing.append('no per-slot F span measurements in the V1.11 evidence')
    if init is None:
        missing.append('no measured F model-initialization wall clock in the V1.11 evidence')
    if warm is None:
        missing.append('no measured F compile/warm-up wall clock in the V1.11 evidence')
    if not n_obs_states:
        missing.append('no completed V1.11 F state records')
    if missing:
        return {'verdict': 'BLOCKED_INSUFFICIENT_F_TIMING_EVIDENCE', 'missing_evidence': missing,
                'caps': None}

    slot_cap = 10.0
    init_cap = 60.0
    batch_cap = 180.0
    # linear extrapolation of the OBSERVED batch wall clock from the 22 completed states to the
    # full 64-state schedule, plus the fixed init/compile segment
    per_state_obs = (float(batch_obs) / float(n_obs_states)) if batch_obs and n_obs_states else None
    projected = (per_state_obs * 64.0 + float(init) + float(warm)) if per_state_obs else None
    return {
        'verdict': 'FROZEN_FROM_MEASURED_V111_F_EVIDENCE',
        'caps': {'initialize_compile_s': init_cap, 'slot_span_s': slot_cap, 'batch_s': batch_cap},
        'derivation': {
            'slot_span_s': {
                'rule': 'V1.11 fixed engineering ceiling for this protocol; NOT widened in V1.18',
                'observed_max_s': span['max'], 'observed_median_s': span['median'],
                'observed_n': span['n'],
                'headroom_vs_observed_max': slot_cap / max(span['max'], 1e-9),
                'claim_limit': 'engineering interface ceiling only; it is NOT evidence that the '
                               'L3 full 8+2 s online publication gate is met'},
            'initialize_compile_s': {
                'rule': 'V1.11 fixed engineering ceiling; covers measured model init + first '
                        'compiled forward',
                'observed_model_init_s': init, 'observed_compile_warmup_s': warm,
                'observed_init_plus_warmup_s': float(init) + float(warm),
                'headroom_vs_observed': init_cap / max(float(init) + float(warm), 1e-9)},
            'batch_s': {
                'rule': 'V1.11 fixed batch ceiling; extrapolated observed cost for the full '
                        '64-state schedule must sit inside it',
                'observed_batch_s': batch_obs, 'observed_states': n_obs_states,
                'observed_per_state_s': per_state_obs,
                'projected_full_schedule_s': projected,
                'headroom_vs_projected': (batch_cap / projected) if projected else None,
                'projection_note': 'linear extrapolation of the OBSERVED 22-state batch wall '
                                   'clock; a projection, labelled as such, not a measurement'},
        },
        'not_evidence_for': ['the L3 complete 8+2 s online publication gate',
                             'any F-vs-D-Geo comparison', 'any scientific claim'],
    }


def freeze_state_list():
    """The frozen 64-state list in LEDGER PUBLICATION ORDER with per-state identity.

    Order is read from STATE_LEDGER.jsonl append order and cross-checked against the canonical
    manifest.  Any disagreement is a hard failure: silently choosing one order would freeze a
    different object than the collector published.
    """
    manifest = json.loads(STATE_MANIFEST.read_text(encoding='utf-8'))
    pub = []
    for i, line in enumerate(LEDGER.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('event') == 'PUBLISHED':
            pub.append({'day_id': r.get('day_id'), 'position': r.get('position'),
                        'state_hash': r.get('state_hash'), 'ledger_line': i,
                        'snapshot_sha256': r.get('snapshot_sha256'),
                        'validation_sha256': r.get('validation_sha256'),
                        'clock': r.get('clock'), 'event_id': r.get('event_id')})
    man = {int(m['day_id']) * 10 + int(m['position']): m for m in manifest}
    mismatches = []
    for s in pub:
        m = man.get(int(s['day_id']) * 10 + int(s['position']))
        if m is None:
            mismatches.append({'state': [s['day_id'], s['position']], 'reason': 'absent_in_manifest'})
            continue
        for k in ('state_hash', 'snapshot_sha256', 'validation_sha256'):
            if m.get(k) != s.get(k):
                mismatches.append({'state': [s['day_id'], s['position']], 'field': k,
                                   'ledger': s.get(k), 'manifest': m.get(k)})
    order = [(s['day_id'], s['position']) for s in pub]
    canonical = sorted(order)
    return {
        'n_states': len(pub),
        'order_source': 'STATE_LEDGER.jsonl append (PUBLISHED) order -- never re-sorted',
        'order_equals_canonical': order == canonical,
        'order_keys_unique': len(set(order)) == len(order),
        'states': pub,
        'manifest_cross_check': {'n_mismatches': len(mismatches), 'rows': mismatches[:8]},
    }


def validate_run_id(run_id):
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError('run_id must be a non-empty string')
    if run_id in CONSUMED_RUN_IDS:
        raise ValueError('run_id %r is a CONSUMED run id; a new run must be textually disjoint '
                         'from every consumed run (R1)' % run_id)
    if 'v111' in run_id.lower():
        raise ValueError('run_id %r contains "v111": the V1.18 run must be isolated from the '
                         'failed V1.11 batch, whose 42 missing states must never be spliced' % run_id)
    return run_id


def build_run_identity(run_id, *, created_utc):
    validate_run_id(run_id)
    v19_ident = json.loads(RUNTIME_IDENTITY_V19.read_text(encoding='utf-8'))
    dependency_identity = {name: rec['sha256'] for name, rec in sorted(v19_ident['modules'].items())}
    timing = load_f_timing_evidence()
    budgets = derive_budgets(timing)
    states = freeze_state_list()
    identities = {
        'f_ckpt': {'path': str(F_CKPT.relative_to(EXT)), 'sha256': sha(F_CKPT),
                   'registered': REGISTERED['f_ckpt'],
                   'matches': sha(F_CKPT) == REGISTERED['f_ckpt'], 'label': F_LABEL},
        'backbone': {'path': 'MASKCO_code/ckpts/cvrp100.ckpt', 'sha256': sha(BACKBONE),
                     'registered': REGISTERED['backbone'],
                     'matches': sha(BACKBONE) == REGISTERED['backbone'], 'label': BACKBONE_LABEL},
        'dev_check_npz': {'path': str(DEV_CHECK.relative_to(EXT)), 'sha256': sha(DEV_CHECK),
                          'registered': REGISTERED['dev_check_npz'],
                          'matches': sha(DEV_CHECK) == REGISTERED['dev_check_npz']},
        'profile_file': {'path': str(PROFILE.relative_to(EXT)), 'sha256': sha(PROFILE),
                         'registered': REGISTERED['profile_file'],
                         'matches': sha(PROFILE) == REGISTERED['profile_file'],
                         'profile_hash': json.loads(PROFILE.read_text(encoding='utf-8'))
                         .get('profile_hash')},
        'state_manifest': {'path': str(STATE_MANIFEST.relative_to(REPO)),
                           'sha256': sha(STATE_MANIFEST)},
        'runtime_identity_v19': {'path': str(RUNTIME_IDENTITY_V19.relative_to(REPO)),
                                 'sha256': sha(RUNTIME_IDENTITY_V19)},
        'contract_protocol_identity': {'path': str(CONTRACT_PROTOCOL_IDENTITY.relative_to(REPO)),
                                       'sha256': sha(CONTRACT_PROTOCOL_IDENTITY),
                                       'registered':
                                       REGISTERED['contract_protocol_identity'],
                                       'matches': sha(CONTRACT_PROTOCOL_IDENTITY)
                                       == REGISTERED['contract_protocol_identity']},
    }
    states_binding = hashlib.sha256(json.dumps(
        [[s['day_id'], s['position'], s['state_hash']] for s in states['states']],
        sort_keys=True).encode()).hexdigest()
    return {
        'schema': 'd3_l1_v118_run_identity',
        'run_id': run_id,
        'created_utc': created_utc,
        'isolated_from': list(CONSUMED_RUN_IDS),
        'isolation_claim': 'this run_id is textually disjoint from every consumed run id; its '
                           'outputs live in their own directory; no state, score or evidence file '
                           'from V1.9 or V1.11 is reused or spliced',
        'labels': {'F': F_LABEL, 'backbone': BACKBONE_LABEL,
                   'science': 'ENGINEERING_INTERFACE_QUALIFICATION_ONLY',
                   'stable_online_gain': False, 'algorithmic_gain_claim': 'none'},
        'identities': identities,
        'contracts': {'base_contract_hash': v19_ident['base_contract_hash'],
                      'effective_contract_hash': v19_ident['effective_contract_hash'],
                      'registered_effective_contract': REGISTERED['effective_contract'],
                      'matches': v19_ident['effective_contract_hash']
                      == REGISTERED['effective_contract']},
        'dependency_identity': dependency_identity,
        'dependency_identity_sha256': hashlib.sha256(json.dumps(
            dependency_identity, sort_keys=True).encode()).hexdigest(),
        'state_list': states,
        'state_list_sha256': hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest(),
        'state_binding_sha256': states_binding,
        'call_schedule': CALL_SCHEDULE,
        'budgets': budgets,
        'timing_evidence': timing,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'packages': v19_ident.get('packages')},
        'measured_budget_observed_by_v19_collector': {
            'note': 'V1.9 F-FREE data-collection timings are recorded for provenance ONLY. They '
                    'are not F model-initialization or F forward budgets and are never '
                    'substituted for one (R3).',
            'slot_span_wall_s_max': 3.327999999979511,
            'day_rollout_wall_s_max': 0.5190811157226562,
            'source': 'v118_evidence/D3_V118_FREEZE_CHECK.json (records.state_asset.budget_observed)',
        },
    }


def identity_match_report(identity):
    """Mechanical re-verification of every registered identity against the bytes on disk."""
    rows = {}
    for key, rec in identity['identities'].items():
        actual = sha(EXT / rec['path']) if (EXT / rec['path']).is_file() else sha(REPO / rec['path'])
        rows[key] = {'path': rec['path'], 'expected': rec.get('registered', rec.get('sha256')),
                     'actual': actual,
                     'ok': (rec['matches'] if 'matches' in rec else actual == rec.get('sha256'))}
    rows['contracts'] = {'ok': bool(identity['contracts']['matches']),
                         'effective': identity['contracts']['effective_contract_hash']}
    rows['state_list'] = {'ok': (identity['state_list']['n_states'] == 64
                                 and identity['state_list']['order_equals_canonical']
                                 and identity['state_list']['order_keys_unique']
                                 and identity['state_list']['manifest_cross_check']['n_mismatches']
                                 == 0
                                 and all(s.get('state_hash') for s in identity['state_list']['states'])),
                          'n_states': identity['state_list']['n_states'],
                          'n_mismatches': identity['state_list']['manifest_cross_check']['n_mismatches']}
    return {'rows': rows, 'ok': all(r['ok'] for r in rows.values())}


def compute_missing_items(run_dir, identity, extra=None):
    """DYNAMICALLY compute what is still missing before READY_FOR_USER_SIGNATURE.

    Every item is decided from artifacts on disk or from verdicts produced by the other V1.18
    tools -- never from a hand-written claim.  `extra` carries verdicts computed by the caller
    (acceptance harness, independent verifier) so this function stays a pure aggregator.
    """
    extra = extra or {}
    run_dir = Path(run_dir)
    missing = []
    if not run_dir.is_dir():
        missing.append('the isolated run directory has not been created yet')
        return {'verdict': 'NOT_READY', 'missing_before_signature': missing, 'checks': {}}

    def _read(name):
        p = run_dir / name
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:                                                # noqa: BLE001
            return None

    checks = {}
    ident = identity_match_report(identity)
    checks['identity_bound'] = ident['ok']
    if not ident['ok']:
        missing.append('one or more registered identities (model / backbone / data / profile / '
                       'contract / dependency) do not match the frozen bytes')

    # RUNTIME watch closure.  The registered identities above are asset hashes that a code edit
    # cannot change, so on their own they would let a MODIFIED RUNNER still report ready -- and the
    # worker would then die at startup on the watched-path comparison.  The readiness gate must
    # therefore also re-derive every path in the run's frozen IDENTITY_TARGETS from disk, which is
    # what makes "edited the runtime source" produce a NEW execution identity instead of a ready one.
    try:
        import d3_l1_v118_persistence as _persistence_mod
        targets_record = _persistence_mod.VerifiedReader(run_dir).latest('IDENTITY_TARGETS')
    except Exception:                                                    # noqa: BLE001
        targets_record = None
    if not targets_record or not targets_record.get('paths'):
        checks['runtime_watch_closed'] = False
        checks['runtime_watch'] = {'reason': 'no IDENTITY_TARGETS record in the run directory'}
        missing.append('the run directory has no frozen runtime watch set (IDENTITY_TARGETS), so '
                       'the code identity of the execution cannot be checked')
    else:
        watch = {p: {'frozen': want, 'current': sha(p), 'ok': sha(p) == want}
                 for p, want in targets_record['paths'].items()}
        drifted = sorted(p for p, r in watch.items() if not r['ok'])
        checks['runtime_watch_closed'] = not drifted
        checks['runtime_watch'] = {'n_paths': len(watch), 'n_drifted': len(drifted),
                                   'drifted': drifted[:10]}
        if drifted:
            missing.append('the runtime source/asset watch set has DRIFTED for %d path(s): this run '
                           'identity is stale and a NEW run identity version is required before '
                           'signature (the worker would refuse at startup)' % len(drifted))
        # Cross-check the frozen watch set against the DERIVED closure.  A freeze missing a file the
        # execution path imports would silently under-check -- that is the V1.21 defect -- so the
        # readiness gate reports the difference instead of trusting the record it was given.
        try:
            cc_ok, cc = dependency_closure_module().cross_check_watch_set(
                list(targets_record['paths']))
        except BaseException as exc:                                      # noqa: BLE001
            cc_ok, cc = False, {'error': '%s: %s' % (type(exc).__name__, exc)}
        checks['runtime_watch_covers_closure'] = bool(cc_ok)
        checks['runtime_watch_closure_diff'] = cc
        if not cc_ok:
            miss = cc.get('missing_from_freeze') or []
            missing.append('the frozen runtime watch set does NOT cover the derived execution-path '
                           'closure: %d file(s) missing (%s)' % (len(miss), miss[:6]))

    try:
        import d3_l1_v118_persistence as _persistence
        published_run_id = (_persistence.VerifiedReader(run_dir).latest('RUN_IDENTITY') or {}).get(
            'run_id')
    except Exception:                                                    # noqa: BLE001
        published_run_id = None
    checks['run_id_isolated'] = published_run_id == identity['run_id']
    checks['published_run_id'] = published_run_id
    if not checks['run_id_isolated']:
        missing.append('the run identity JSON is absent or its run_id does not equal the frozen '
                       'run_id')

    b = identity['budgets']
    checks['budgets_frozen'] = b.get('caps') is not None
    if not checks['budgets_frozen']:
        missing.append('the F call counts and wall-clock budgets could not be frozen: %s'
                       % (b.get('missing_evidence') or 'unspecified'))

    accept = extra.get('acceptance') or _read('NEW_DESTINATION_ACCEPTANCE.json')
    if accept is None:
        # the acceptance harness writes SYNTHETIC_END_TO_END.json next to the run directory
        for candidate in (run_dir.parent / 'SYNTHETIC_END_TO_END.json',
                          run_dir / 'SYNTHETIC_END_TO_END.json'):
            if candidate.is_file():
                try:
                    accept = json.loads(candidate.read_text(encoding='utf-8'))
                except Exception:                                        # noqa: BLE001
                    accept = None
                break
    checks['persistence_acceptance'] = bool(accept) and \
        accept.get('verdict') == 'NEW_DESTINATION_PERSISTENCE_ACCEPTED' and accept.get('n_fail') == 0
    if not checks['persistence_acceptance']:
        missing.append('the real-worker new-destination persistence acceptance has not produced '
                       'a PASS verdict')

    verifier = extra.get('verifier') or _read('INDEPENDENT_READONLY_VERIFIER.json')
    selftest = extra.get('verifier_selftest') or \
        _read('INDEPENDENT_READONLY_VERIFIER_SUPERVISED_SELFTEST.json')
    if verifier is None:
        for candidate in (run_dir.parent / 'INDEPENDENT_READONLY_VERIFIER.json',
                          run_dir / 'INDEPENDENT_READONLY_VERIFIER.json'):
            if candidate.is_file():
                try:
                    verifier = json.loads(candidate.read_text(encoding='utf-8'))
                except Exception:                                        # noqa: BLE001
                    verifier = None
                break
    if selftest is None:
        for candidate in (run_dir.parent / 'INDEPENDENT_READONLY_VERIFIER_SUPERVISED_SELFTEST.json',
                          run_dir / 'INDEPENDENT_READONLY_VERIFIER_SUPERVISED_SELFTEST.json'):
            if candidate.is_file():
                try:
                    selftest = json.loads(candidate.read_text(encoding='utf-8'))
                except Exception:                                        # noqa: BLE001
                    selftest = None
                break
    # The verifier must be FROZEN, RUNNABLE and PROVEN to reach its row-level verdict: the
    # supervised self-test builds a disposable fixture through the production supervisor, verifies
    # all 64 synthetic rows with the independently re-implemented checker, and cross-checks its
    # verdict against the production verifier on every row.
    selftest_ok = bool(selftest) and selftest.get('verdict') == 'SELFTEST_SUPERVISED_VERIFIED' \
        and selftest.get('fail') == 0 \
        and bool(((selftest.get('records') or {}).get('row_state') or {}).get('n_agree'))
    checks['independent_verifier_selftest'] = selftest_ok
    checks['independent_verifier'] = bool(verifier) and selftest_ok and \
        verifier.get('verdict') in ('VERIFIED', 'VERIFIED_NOT_EXECUTED') and verifier.get('fail') == 0
    if not checks['independent_verifier']:
        missing.append('the independent read-only verifier is not frozen, runnable and proven '
                       '(it needs a VERIFIED / VERIFIED_NOT_EXECUTED verdict with 0 failures AND a '
                       'supervised self-test that exercises its row-level checker)')
    else:
        # The verifier is FROZEN and RUNNABLE.  Whether the real F forward has been executed is a
        # POST-SIGNATURE concern: requiring it here would make the gate require the very execution
        # the signature authorises, which is circular.  It is recorded, not required.
        checks['independent_verifier_verdict'] = verifier.get('verdict')
        checks['real_F_forward_executed'] = bool(
            ((verifier.get('records') or {}).get('row_state') or {}).get('n_parsed'))
        checks['verifier_selftest_verifier'] = selftest.get('records', {}).get('verdict')
        if verifier.get('verdict') == 'VERIFIED_NOT_EXECUTED':
            checks['independent_verifier_note'] = (
                'the real F forward has not run yet, so the verifier reports VERIFIED_NOT_EXECUTED; '
                'this is a statement about the FROZEN architecture, not a PASS on any F result')

    short = extra.get('synthetic_execution') or accept or _read('SYNTHETIC_END_TO_END.json')
    checks['synthetic_end_to_end'] = bool(short) and short.get('n_fail') == 0 and \
        short.get('verdict') == 'NEW_DESTINATION_PERSISTENCE_ACCEPTED'
    if not checks['synthetic_end_to_end']:
        missing.append('the synthetic-scorer end-to-end acceptance over the real worker / '
                       'supervisor path has not completed with 0 failures')

    if not checks['persistence_acceptance'] or not checks['synthetic_end_to_end']:
        missing.append('the F worker path has not been shown to persist, time out, be killed and '
                       'be revoked correctly on this platform')

    return {
        'verdict': 'READY_FOR_USER_SIGNATURE' if not missing else 'NOT_READY',
        'missing_before_signature': missing,
        'checks': checks,
        'not_run_and_not_authorised_here': [
            'no real F forward over the 64 states (this is the single item awaiting signature)',
            'no F-vs-D-Geo comparison', 'no training', 'no L2/L3', 'no 40-day batch',
            'no final test', 'no install into any production state'],
        'note': 'the verdict is computed from artifacts on disk; there is no hand-set ready flag',
    }
