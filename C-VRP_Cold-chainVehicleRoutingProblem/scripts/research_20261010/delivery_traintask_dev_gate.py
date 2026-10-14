"""D-TRAINTASK-v1 development gate: the frozen trained guide against the registered controls.

Runs the four registered arms on the frozen DEV split, over the same instances, the same initial
routes and the same eight-mask schedule per instance, and reports the paired instance-level
relative full-cost reduction with a bootstrap interval.

Arm -> decoder calls, stated honestly because they are NOT all equal:
  * `trained`          - delivery-specific trained guide: 8 decode calls per instance
  * `zeroshot_guide`   - frozen zero-shot checkpoint:      8 decode calls per instance
  * `discard_control`  - pays 8 decode calls and discards:  8 decode calls per instance
  * `regret2_nolearn`  - no guide at all:                   0 decode calls, BY CONSTRUCTION

The intervention is the pairing that matters: `trained` vs `discard_control` is compute-matched and
therefore isolates the VALUE OF THE INFORMATION, while `trained` vs `regret2_nolearn` is the
non-learning reference the protocol names as the arm to beat and is explicitly NOT compute-matched.
Asserting all four arms are compute-matched would be false, so the run asserts only that the three
decode-consuming arms agree, and records the fourth explicitly.

Development only. Passing these gates selects at most one candidate and confers no alpha and no
confirmatory status; the program MAIN gate additionally requires the same >= 1 percent against
every registered random/nonlearned control AND a qualified strong rolling controller at identical
online information, legal actions and total budget. A CI spanning zero means the difference is not
established, never equivalence.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def bootstrap(scripts):
    scripts = Path(scripts).resolve()
    for p in (scripts / 'delivery', scripts / 'research_20261010'):
        sys.path.insert(0, str(p))
    return scripts


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class DiscardProvider:
    """Pays exactly one decode call per step, then returns no logits at all."""

    def __init__(self, base):
        self.base = base
        self.identity = dict(base.identity)
        self.identity['discards_output'] = True

    def features(self, instance, capacity):
        return self.base.features(instance, capacity)

    def logits(self, features, adjacency):
        self.base.logits(features, adjacency)      # pay the call
        return None                                # and throw the information away


def main(args):
    started = time.perf_counter()
    scripts = bootstrap(args.scripts)
    from delivery_equal_schedule import generate, DeliveryContract, run_arm
    from delivery_maskco import MaskCORouteProvider
    from delivery_trained_guide import DeliveryTrainedGuide

    proto_path = Path(args.protocol)
    proto = json.loads(proto_path.read_text(encoding='utf-8'))
    if proto.get('version') != 'D-TRAINTASK-v1':
        raise SystemExit('unexpected protocol version')
    draft_path = proto_path.with_name('delivery_traintask_protocol_DRAFT.json')
    recorded_draft = (proto.get('frozen_protocol_sha256') or {}).get('draft_sha256')
    if recorded_draft and file_sha(draft_path) != recorded_draft:
        raise SystemExit('DRAFT sha mismatch: the registration no longer matches its draft')
    seeds = proto['data_plan']['registered_seeds']
    dev_seed = int(seeds['dev_seed'])
    mask_seed = int(seeds['mask_seed_dev'])
    budget = float(proto['matched_resources']['wall_budget_s'])
    positions = int(proto['matched_resources']['planned_positions_per_instance'])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()

    # ---- arms -----------------------------------------------------------------------
    trained = DeliveryTrainedGuide(args.checkpoint, args.upstream_root)
    probe, _ = generate(dev_seed, 0)
    feats_probe = trained.features(probe, contract.capacity)
    if hasattr(feats_probe, 'block_until_ready'):
        feats_probe.block_until_ready()
    dim = int(np.asarray(feats_probe).shape[-1])
    trained.load_head(args.head, dim)
    zeroshot = MaskCORouteProvider(checkpoint=args.checkpoint, upstream_root=args.upstream_root)
    discard = DiscardProvider(trained)
    arms = {'trained': trained, 'zeroshot_guide': zeroshot, 'discard_control': discard,
            'regret2_nolearn': None}

    costs = {k: [] for k in arms}
    decoded = {k: 0 for k in arms}
    per_instance = []
    for inst in range(args.instances):
        data, initial = generate(dev_seed, inst)
        rng = np.random.default_rng(np.random.SeedSequence([mask_seed, inst]))
        nodes = len(data['demands'])
        size = max(1, int(math.ceil(args.mask_fraction * (nodes - 1))))
        schedule = [sorted(int(v) for v in rng.choice(np.arange(1, nodes), size=size,
                                                      replace=False))
                    for _ in range(positions)]
        row = {'instance': inst}
        for name, prov in arms.items():
            r = run_arm(data, initial, contract, schedule, provider=prov,
                        time_limit_s=budget, guide_weight=args.guide_weight)
            final = float(r['final_endpoint_cost'])
            costs[name].append(final)
            row[name] = final
            decoded[name] += int(r['decode_calls'])
            if name == 'trained':
                row['timeout'] = bool(r['timeout'])
                row['fixed_schedule_qualified'] = bool(r['fixed_schedule_qualified'])
        per_instance.append(row)
        print(json.dumps(row), flush=True)

    n = len(per_instance)
    per_arm_calls = {k: decoded[k] for k in arms}
    decode_consuming = ['trained', 'zeroshot_guide', 'discard_control']
    matched = len(set(per_arm_calls[k] for k in decode_consuming)) == 1

    def paired(control):
        d = np.array([(costs[control][i] - costs['trained'][i]) / costs[control][i]
                      for i in range(n)], dtype=np.float64)
        rng = np.random.default_rng(args.bootstrap_seed)
        draws = d[rng.integers(0, n, size=(args.bootstrap_repetitions, n))].mean(axis=1)
        return {'n_pairs': int(n), 'mean_relative_cost_reduction': float(d.mean()),
                'median': float(np.median(d)),
                'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                'n_improved': int((d > 0).sum()), 'n_tied': int((d == 0).sum()),
                'n_worse': int((d < 0).sum()),
                'passes_dev_1pct_gate': bool(d.mean() >= 0.01
                                             and np.percentile(draws, 2.5) > 0)}

    contrasts = {('trained_minus_' + c): paired(c) for c in
                 ('regret2_nolearn', 'zeroshot_guide', 'discard_control')}
    result = {
        'kind': 'delivery_traintask_development_gate',
        'protocol_version': 'D-TRAINTASK-v1',
        'protocol_sha256': file_sha(proto_path),
        'recipe_sha256': file_sha(args.recipe),
        'dev_seed': dev_seed, 'mask_seed_dev': mask_seed,
        'instances': args.instances, 'positions_per_instance': positions,
        'wall_budget_s': budget, 'guide_weight': args.guide_weight,
        'trained_head_sha256': file_sha(args.head),
        'mean_final_cost': {k: float(np.mean(costs[k])) for k in arms},
        'actual_decode_calls': per_arm_calls,
        'decode_calls_matched_across_decode_consuming_arms': bool(matched),
        'decode_call_matching_scope_note': ('regret2_nolearn performs zero decode calls by '
                                            'construction, so it is deliberately NOT '
                                            'compute-matched; the matched-compute contrast is '
                                            'trained vs discard_control'),
        'paired_contrasts': contrasts,
        'gates': {k: v['passes_dev_1pct_gate'] for k, v in contrasts.items()},
        'program_main_gate_restated': ('additionally requires >= 1 percent against EVERY '
                                       'registered random/nonlearned control AND a qualified '
                                       'strong rolling controller, at identical online '
                                       'information, legal actions and total budget; this panel '
                                       'has neither the random-model arm nor the rolling control, '
                                       'so it cannot address the main gate'),
        'per_instance': per_instance,
        'bootstrap_seed': args.bootstrap_seed,
        'bootstrap_repetitions': args.bootstrap_repetitions,
        'what_this_is': ['the registered DEVELOPMENT gate on the frozen DEV split'],
        'what_this_is_not': [
            'no confirmation: one candidate, no alpha, no pre-registered cohort',
            'not the program main gate: no random-model arm and no strong rolling control',
            'a CI spanning zero means the difference is not established',
            'no stable online gain, no MaskCO attribution, no impossibility, no equivalence'],
        'stable_online_gain_established': False, 'maskco_attributable': False,
        'equivalence_established': False,
        'jax_backend': __import__('jax').default_backend(),
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('mean_final_cost', 'actual_decode_calls',
                       'decode_calls_matched_across_decode_consuming_arms',
                       'paired_contrasts', 'gates', 'wall_s')}, indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--head', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--upstream-root', required=True)
    ap.add_argument('--instances', type=int, default=32)
    ap.add_argument('--mask-fraction', type=float, default=0.5)
    ap.add_argument('--guide-weight', type=float, default=1.0)
    ap.add_argument('--bootstrap-seed', type=int, default=2026120106)
    ap.add_argument('--bootstrap-repetitions', type=int, default=2000)
    main(ap.parse_args())
