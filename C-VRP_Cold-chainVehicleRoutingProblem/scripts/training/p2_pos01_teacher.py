"""P2-POS-01A private continuation plumbing, NOT an authorized label collector.

Implements event-keyed randomness, immutable capture identity, strict-engine
KEEP/no-op resume with explicit public booking_horizon, and failure-preserving
paired-label arithmetic. The production SAA mask intervention / future generator
are NOT connected or qualified. There is no real-day collection or training CLI.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import configure_imports
configure_imports(('experimental', 'simulation', 'coldchain', 'evaluation'))
from recourse_snapshot import snapshot_state_hash, validate_snapshot_against_env
from strict_online_env import StrictOnlineEnv
from p2_pos01_reference import identity, finite

PHASE = 'POST_ADMISSION_PRE_REPAIR'


def dataset_digest(dataset):
    digest = hashlib.sha256()
    for key in sorted(dataset):
        array = np.ascontiguousarray(dataset[key])
        digest.update(str((key, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class EventRNG:
    """Independent draw for each external event key, never a branch call counter.

    Caller must use external scenario/event identity; no action or branch ID.
    A different key may be drawn first without shifting subsequent event draws.
    """
    def __init__(self, scenario_identity, seed):
        self.scenario_identity, self.seed = str(scenario_identity), int(seed)

    def normal(self, external_event_key, size):
        key = identity([self.scenario_identity, self.seed, external_event_key])
        words = np.frombuffer(bytes.fromhex(key), dtype='<u4').copy()
        return np.random.default_rng(np.random.SeedSequence(words)).normal(size=size)


def seal_capture(snapshot, *, phase, public_view, continuation_identity, source_dataset):
    if phase != PHASE or not continuation_identity:
        raise ValueError('capture phase/continuation identity required')
    snap = copy.deepcopy(snapshot)
    if snap.get('state_hash') != snapshot_state_hash(snap):
        raise ValueError('unsealed engine snapshot')
    ledger = snap.get('replanner_state')
    if not isinstance(ledger, dict) or 'accepted' not in ledger or 'rejected' not in ledger:
        raise ValueError('complete acceptance ledger required')
    if set(ledger['accepted']) != set(public_view.accepted) or set(ledger['rejected']) != set(public_view.rejected):
        raise ValueError('public/private ledger mismatch')
    protocol = {'booking_horizon': public_view.booking_horizon, 'capacity': public_view.capacity,
                'tw_speed': public_view.tw_speed, 'contract_hash': public_view.contract.contract_hash}
    source = dataset_digest(source_dataset)
    return {'phase': phase, 'public_identity': identity(public_view), 'snapshot': snap,
            'protocol': protocol, 'dataset_identity': source,
            'continuation_identity': continuation_identity,
            'scope': 'KEEP_REPLAY_ONLY_NOT_MASK_SAA',
            'capture_identity': identity([phase, identity(public_view), snap['state_hash'],
                                          continuation_identity, protocol, source])}


def resume_keep(capture, dataset, policy_factory, contract, *, capacity, tw_speed,
                booking_horizon, continuation_identity):
    """Real existing StrictOnlineEnv replay; private teacher side only.

    No legacy resume factory and no old future-reading force_suffix function.
    This is KEEP plumbing, not the qualified mask+SAA continuation required by B.
    """
    if capture['phase'] != PHASE or capture['continuation_identity'] != continuation_identity:
        raise ValueError('continuation/phase changed')
    snap = copy.deepcopy(capture['snapshot'])
    if snap['state_hash'] != snapshot_state_hash(snap):
        raise ValueError('engine snapshot identity changed')
    protocol = {'booking_horizon': booking_horizon, 'capacity': capacity, 'tw_speed': tw_speed,
                'contract_hash': contract.contract_hash}
    if protocol != capture['protocol'] or dataset_digest(dataset) != capture['dataset_identity']:
        raise ValueError('public protocol/dataset changed during recovery')
    expected = identity([capture['phase'], capture['public_identity'], snap['state_hash'],
                         continuation_identity, protocol, capture['dataset_identity']])
    if expected != capture['capture_identity'] or not np.isfinite(booking_horizon):
        raise ValueError('capture/public booking identity changed')
    before = dataset_digest(dataset)
    policy = policy_factory()
    env = StrictOnlineEnv(copy.deepcopy(dataset), capacity=capacity, tw_speed=tw_speed,
        num_vehicles=int(snap['num_vehicles']), replanner=policy,
        coldchain_contract=contract, booking_horizon=booking_horizon)
    validate_snapshot_against_env(env, snap)
    traces, served = env.run_resumed(snap)
    after = dataset_digest(dataset)
    if after != before or snapshot_state_hash(capture['snapshot']) != snap['state_hash']:
        raise RuntimeError('private replay mutated source')
    return traces, served, policy.export_state()


def paired_label(candidate, keep):
    """No finite reward invented for failures. Caller retains all input rows."""
    valid = all(x.get('status') == 'OK' and x.get('hard_feasible') is True and
                x.get('utility') is not None and finite(x['utility']) for x in (candidate, keep))
    if (candidate.get('scenario_identity') != keep.get('scenario_identity') or
            candidate.get('continuation_identity') != keep.get('continuation_identity') or
            not candidate.get('scenario_identity') or not candidate.get('continuation_identity')):
        raise ValueError('unpaired future/continuation')
    return {'delta_utility': candidate['utility']-keep['utility'] if valid else None,
            'valid': valid, 'candidate_status': candidate.get('status'),
            'keep_status': keep.get('status'), 'training_weight': 1.0 if valid else 0.0}


def collect_real_labels(*_args, **_kwargs):
    raise NotImplementedError('P2-POS-01B NOT AUTHORISED; SAA mask replay/future generator NOT QUALIFIED')
