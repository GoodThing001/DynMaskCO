"""Diagnostic D-GUIDE-v1: where does the guide term damage the executable reconstruction?

Question (from the falsified D-EQ8-v1 panel): under an equal mask schedule the
zero-guide reconstructor (H) reached 105.23 while the pretrained guide (P) reached
114.70 and a random same-architecture guide (R) reached 111.66. The guide is
therefore harmful through this adapter. This diagnostic localizes the damage.

Method. It reads the *verified* D-EQ8-v1 raw rows produced on the server, takes
each recorded decision state (instance, arm, step) as given, recomputes the model
logits locally with the published weights, and re-runs the identical regret-2
insertion decoder under a 2x2 factorization of how the guide may enter:

    regret key   in {delta, combined}     -- which customer is inserted next
    position key in {delta, combined}     -- where in that customer's route it goes

    (combined, combined) = shipped adapter, i.e. what the panel actually ran
    (delta,    combined) = guide ranks positions only
    (combined, delta)    = guide ranks customers only
    (delta,    delta)    = no guide anywhere (the H rule)

Every rule sees the identical recorded state, so a difference between rules is
attributable to the decision rule rather than to trajectory divergence.

Scope and honesty limits:
  * This is a mechanism diagnosis, not a new performance arm. It has no wall
    budget and makes no claim of online gain.
  * Local JAX (0.5.0) differs from the server stack (0.6.2) and XLA reassociation
    changes float32 reduction order, so locally recomputed logits are NOT
    bit-identical to the server's. Model *parameter* identity is verified equal
    against the recorded identity; logits digests are reported, not asserted.
  * Agreement between the shipped rule and the recorded candidates is measured and
    reported, so the numerical-sensitivity bound is visible to the reader.
  * Only recorded states whose reconstruction ran to completion are used; states
    the server's wall budget truncated are counted and excluded, not silently
    recomputed under a budget the panel did not have.

Usage:
    python delivery_guide_misalignment.py --rows ROWS.jsonl --checkpoint CKPT \
        --upstream MASKCO_ROOT --recorded-identity IDENTITY.json --out NEWDIR \
        [--max-instances N]
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ('delivery', 'coldchain', 'research_20261010'):
    sys.path.insert(0, str(SCRIPTS / sub))

from delivery_baselines import matrices, route_ok
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
from delivery_equal_schedule import parameter_digest, reconstruct as shipped_reconstruct
from delivery_maskco import MaskCORouteProvider, partial_adjacency

RULES = (('combined', 'combined'), ('delta', 'combined'),
         ('combined', 'delta'), ('delta', 'delta'))
RULE_NAMES = {('combined', 'combined'): 'shipped',
              ('delta', 'combined'): 'guide_position_only',
              ('combined', 'delta'): 'guide_regret_only',
              ('delta', 'delta'): 'no_guide'}

PROTOCOL = {
    'version': 'D-GUIDE-v1',
    'kind': 'mechanism_diagnosis_not_performance_arm',
    'parent_panel': 'D-EQ8-v1',
    'parent_raw_progress_sha256': 'd47241d0fbd110f804407a8a88cde9450c67789ebadfbef68dfca5e2da2d3a41',
    'state_source': 'recorded_server_decision_states_taken_as_given',
    'rules': [RULE_NAMES[r] for r in RULES],
    'bootstrap_seed': 2026101088,
    'bootstrap_repetitions': 2000,
    'wall_budget': None,
    'claims_online_gain': False,
    'claims_maskco_weight_effect': False,
    'numerical_note': 'local_jax_stack_differs_from_server_logits_not_bitwise_equal',
    'all_failures_retained': True,
}


def reconstruct_probe(instance, incumbent, contract, removed, adjacency, logits,
                      guide_weight, regret_key, position_key):
    """Generalized copy of the shipped reconstruct() with a 2x2 guide factorization.

    With regret_key == position_key == 'combined' this must be bit-identical to
    delivery_equal_schedule.reconstruct, checked by verify_reconstruction_equivalence.
    """
    dist = matrices(instance)
    off = ~np.eye(len(dist), dtype=bool)
    dscale = max(1e-9, float(dist[off].mean()))
    lscale = 1.0 if logits is None else max(1e-8, float(np.std(logits[off])))
    if logits is None:
        guide_weight = 0.
    candidate = [[i for i in route if i not in removed] for route in incumbent]
    candidate += [[] for _ in range(contract.vehicles - len(candidate))]
    remaining = sorted(removed)
    while remaining:
        choices = []
        for i in remaining:
            options = []
            for k, route in enumerate(candidate):
                for at in range(len(route) + 1):
                    a, b = (route[at - 1] if at else 0), (route[at] if at < len(route) else 0)
                    if adjacency[a, b] > 0:
                        continue
                    if not route_ok(route[:at] + [i] + route[at:], instance, contract, dist, 0.):
                        continue
                    delta = dist[a, i] + dist[i, b] - dist[a, b]
                    gain = 0. if logits is None else (
                        logits[a, i] + logits[i, b] - (0. if a == b else logits[a, b])) / lscale
                    options.append({'delta': float(delta / dscale),
                                    'combined': float(delta / dscale - guide_weight * gain),
                                    'k': k, 'at': at})
            if options:
                ordered = sorted(options, key=lambda o: (o[position_key], o['k'], o['at']))
                regret_values = sorted(o[regret_key] for o in options)
                regret = ((regret_values[1] - regret_values[0]) if len(regret_values) > 1
                          else float('inf'))
                choices.append((regret, i, ordered[0]))
        if not choices:
            return candidate, remaining, 'no_feasible_insertion'
        _, i, best = max(choices, key=lambda row: (row[0], -row[1]))
        candidate[best['k']].insert(best['at'], i)
        remaining.remove(i)
    return candidate, [], 'complete'


def verify_reconstruction_equivalence(instance, incumbent, contract, removed, adjacency,
                                      logits, guide_weight):
    """The generalized probe must equal the shipped reconstructor on the shipped rule."""
    ours, _, _ = reconstruct_probe(instance, incumbent, contract, removed, adjacency,
                                   logits, guide_weight, 'combined', 'combined')
    theirs, _, _, _ = shipped_reconstruct(instance, incumbent, contract, removed, adjacency,
                                          logits, guide_weight, float('inf'), time.perf_counter)
    if ours != theirs:
        raise AssertionError('probe reconstructor diverges from shipped reconstructor')
    return True


def paired_bootstrap(base_by_instance, other_by_instance, seed, reps):
    """Paired instance-level bootstrap of mean(other - base)."""
    shared = sorted(set(base_by_instance) & set(other_by_instance))
    if len(shared) < 2:
        return None
    diffs = np.array([other_by_instance[i] - base_by_instance[i] for i in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {'n_instances': len(shared), 'mean_difference': float(diffs.mean()),
            'descriptive_95_CI': [float(np.percentile(means, 2.5)),
                                  float(np.percentile(means, 97.5))],
            'instances_other_lower': int(sum(d < 0 for d in diffs)),
            'instances_other_higher': int(sum(d > 0 for d in diffs)),
            'instances_tied': int(sum(d == 0 for d in diffs))}


def main(args):
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    checkpoint = Path(args.checkpoint)
    weight_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    recorded_identity = json.loads(Path(args.recorded_identity).read_text(encoding='utf-8'))
    expected_param = recorded_identity['model_parameters']
    if weight_sha != recorded_identity['checkpoint_sha256']:
        raise ValueError('Checkpoint does not match the recorded panel identity')

    providers = {'P': MaskCORouteProvider(str(checkpoint), args.upstream),
                 'R': MaskCORouteProvider(str(checkpoint), args.upstream, random_seed=42)}
    local_param = {arm: parameter_digest(providers[arm]) for arm in ('P', 'R')}
    for arm in ('P', 'R'):
        if local_param[arm] != expected_param[arm]:
            raise ValueError('Local model parameters differ from recorded identity: ' + arm)
        providers[arm].prewarm(51, contract.capacity)

    import jax
    import flax
    identity = {
        'protocol': PROTOCOL,
        'checkpoint_sha256': weight_sha,
        'local_model_parameters': local_param,
        'recorded_model_parameters': expected_param,
        'parameters_match_recorded': True,
        'local_jax_version': jax.__version__,
        'local_flax_version': flax.__version__,
        'local_numpy_version': np.__version__,
        'server_jax_version': '0.6.2',
        'threads': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'logits_bitwise_equal_to_server': False,
        'claim_scope': 'mechanism_diagnosis_of_recorded_states',
    }
    with (out / 'PROTOCOL.json').open('x', encoding='utf-8') as f:
        json.dump(PROTOCOL, f, indent=2)

    states = 0
    skipped_truncated = 0
    records = []
    equivalence_checked = False
    shipped_agreement = {'P': [0, 0], 'R': [0, 0], 'H': [0, 0]}
    no_guide_reproduction = [0, 0]
    logits_identity = {'P': [0, 0], 'R': [0, 0]}

    with open(args.rows, encoding='utf-8') as rows_file, \
            (out / 'per_state.jsonl').open('x', encoding='utf-8') as sink:
        for line in rows_file:
            row = json.loads(line)
            instance_id = row['instance_id']
            if args.max_instances is not None and instance_id >= args.max_instances:
                continue
            data = row['data']
            schedule = row['schedule']
            nodes = len(data['demands'])

            # V1: all three arms must share the identical step-1 decision state.
            s1 = {arm: row['arms'][arm]['attempts'][0] for arm in ('P', 'R', 'H')}
            if not (s1['P']['incumbent_before'] == s1['R']['incumbent_before'] ==
                    s1['H']['incumbent_before'] and
                    s1['P']['removed'] == s1['R']['removed'] == s1['H']['removed'] and
                    s1['P']['adjacency_sha256'] == s1['R']['adjacency_sha256'] ==
                    s1['H']['adjacency_sha256']):
                raise RuntimeError('Step-1 arms do not share a decision state; diagnosis invalid')

            features = {arm: providers[arm].features(data, contract.capacity)
                        for arm in ('P', 'R')}
            for arm in ('P', 'R'):
                if hasattr(features[arm], 'block_until_ready'):
                    features[arm].block_until_ready()

            for arm in ('P', 'R', 'H'):
                for attempt, mask in zip(row['arms'][arm]['attempts'], schedule):
                    if attempt['status'] != 'complete' or not attempt['candidate_complete']:
                        skipped_truncated += 1
                        continue
                    incumbent = attempt['incumbent_before']
                    removed = set(mask)
                    adjacency = partial_adjacency(incumbent, removed, nodes)
                    if hashlib.sha256(adjacency.tobytes()).hexdigest() != attempt['adjacency_sha256']:
                        raise RuntimeError('Recorded adjacency digest not reproduced')
                    logits = None
                    if arm in ('P', 'R'):
                        logits = providers[arm].logits(features[arm], adjacency)
                        logits_identity[arm][1] += 1
                        if attempt.get('logits_sha256') == hashlib.sha256(
                                np.ascontiguousarray(logits).tobytes()).hexdigest():
                            logits_identity[arm][0] += 1
                    if not equivalence_checked:
                        for probe_arm in ('P', 'R'):
                            verify_reconstruction_equivalence(
                                data, incumbent, contract, removed, adjacency,
                                providers[probe_arm].logits(features[probe_arm], adjacency), 1.)
                        verify_reconstruction_equivalence(data, incumbent, contract, removed,
                                                          adjacency, None, 1.)
                        equivalence_checked = True

                    before = float(attempt['cost_before'])
                    recorded_candidate = attempt['candidate_routes']
                    for rule in RULES:
                        name = RULE_NAMES[rule]
                        candidate, remaining, status = reconstruct_probe(
                            data, incumbent, contract, removed, adjacency, logits,
                            1., rule[0], rule[1])
                        record = {'instance_id': instance_id, 'arm': arm,
                                  'step': attempt['step'], 'rule': name,
                                  'reconstruct_status': status,
                                  'candidate_complete': not remaining,
                                  'cost_before': before,
                                  'matches_recorded_candidate': candidate == recorded_candidate}
                        if not remaining:
                            checked = evaluate(data, candidate, contract)
                            record['candidate_hard'] = bool(checked['hard'])
                            record['candidate_cost'] = checked['cost'] if checked['hard'] else None
                            record['candidate_gain'] = (before - checked['cost']
                                                        if checked['hard'] else None)
                            record['accepted'] = bool(checked['hard'] and checked['cost'] < before)
                        else:
                            record.update(candidate_hard=False, candidate_cost=None,
                                          candidate_gain=None, accepted=False)
                        sink.write(json.dumps(record, allow_nan=False) + '\n')
                        records.append(record)
                        states += 1

                    # Reproduction checks against the recorded outcome.
                    if arm == 'H':
                        reproduced, _, _ = reconstruct_probe(
                            data, incumbent, contract, removed, adjacency, None, 0.,
                            'combined', 'combined')
                        no_guide_reproduction[1] += 1
                        no_guide_reproduction[0] += int(reproduced == recorded_candidate)
                    else:
                        reproduced, _, _ = reconstruct_probe(
                            data, incumbent, contract, removed, adjacency, logits, 1.,
                            'combined', 'combined')
                        shipped_agreement[arm][1] += 1
                        shipped_agreement[arm][0] += int(reproduced == recorded_candidate)

    by_rule = {RULE_NAMES[rule]: [r for r in records if r['rule'] == RULE_NAMES[rule]]
               for rule in RULES}
    table = {}
    for name, rows_ in by_rule.items():
        hard = [r for r in rows_ if r['candidate_hard']]
        table[name] = {
            'n_states': len(rows_),
            'n_complete': int(sum(r['reconstruct_status'] == 'complete' for r in rows_)),
            'n_hard': len(hard),
            'n_accepted': int(sum(r['accepted'] for r in rows_)),
            'mean_candidate_cost_where_hard': (statistics.fmean([r['candidate_cost'] for r in hard])
                                               if hard else None),
            'mean_gain_where_hard': (statistics.fmean([r['candidate_gain'] for r in hard])
                                     if hard else None),
            'matches_recorded_candidate': int(sum(r['matches_recorded_candidate'] for r in rows_)),
        }

    # Paired per-instance comparison of mean candidate cost against the no-guide
    # rule, restricted to states where BOTH rules produced a hard candidate.
    base_name = 'no_guide'
    base_index = {(r['instance_id'], r['arm'], r['step']): r
                  for r in by_rule[base_name] if r['candidate_hard']}
    paired = {}
    for name, rows_ in by_rule.items():
        if name == base_name:
            continue
        per_instance = {}
        for r in rows_:
            key = (r['instance_id'], r['arm'], r['step'])
            if not r['candidate_hard'] or key not in base_index:
                continue
            per_instance.setdefault(r['instance_id'], []).append(
                r['candidate_cost'] - base_index[key]['candidate_cost'])
        means = {i: statistics.fmean(v) for i, v in per_instance.items()}
        stats = paired_bootstrap({i: 0.0 for i in means}, means,
                                 PROTOCOL['bootstrap_seed'], PROTOCOL['bootstrap_repetitions'])
        if stats is not None:
            paired[name + '_minus_' + base_name] = stats

    checks = [
        {'name': 'checkpoint_matches_recorded_identity', 'pass': True},
        {'name': 'model_parameters_match_recorded_identity', 'pass': True},
        {'name': 'step1_arms_share_decision_state', 'pass': True},
        {'name': 'recorded_adjacency_digests_reproduced', 'pass': True},
        {'name': 'probe_reconstructor_equals_shipped_reconstructor',
         'pass': bool(equivalence_checked)},
    ]
    result = {
        'identity': identity,
        'protocol': PROTOCOL,
        'checks': checks,
        'all_checks_pass': all(c['pass'] for c in checks),
        'n_states': states,
        'n_states_skipped_budget_truncated': skipped_truncated,
        'n_instances': len({r['instance_id'] for r in records}),
        'rule_table': table,
        'paired_vs_no_guide': paired,
        'reproduction': {
            'no_guide_rule_reproduces_recorded_H_candidate': no_guide_reproduction,
            'shipped_rule_reproduces_recorded_arm_candidate': shipped_agreement,
            'local_logits_digest_matches_server': logits_identity,
        },
        'interpretation': 'mechanism_diagnosis_only_no_online_gain_no_weight_claim',
        'wall_s': time.perf_counter() - started,
    }
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'states': states, 'instances': result['n_instances'],
                      'skipped_truncated': skipped_truncated,
                      'wall_s': round(result['wall_s'], 1),
                      'rule_table': table,
                      'no_guide_reproduction': no_guide_reproduction,
                      'shipped_agreement': shipped_agreement,
                      'logits_identity': logits_identity,
                      'paired': paired}, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--recorded-identity', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-instances', type=int, default=None)
    main(parser.parse_args())
