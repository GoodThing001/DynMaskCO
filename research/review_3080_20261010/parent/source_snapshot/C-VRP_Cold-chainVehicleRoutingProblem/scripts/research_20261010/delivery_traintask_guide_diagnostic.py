"""Why is the delivery guide harmful? A per-decision attribution diagnostic.

The DEV gate left a specific, unexplained fact: no guide at all is best (mean final cost 80.731),
the frozen zero-shot guide is worse (90.139) and the trained guide is worst (106.230), with both
intervals excluding zero on the negative side. "The guide hurts" is not a mechanism. This module
measures the mechanism directly.

At every insertion decision it re-enumerates the option set exactly as the frozen reconstruct does
(fidelity to that enumeration was already established at match rate 1.0 by the trainer), then asks:
given the SAME state, which option does the distance-only rule pick and which option does the
guide pick, and how much more does the guide's choice cost in raw insertion delta?

  * `mean_extra_delta_per_decision` > 0 means the guide systematically chooses more expensive
    insertions than the distance-only rule at the very same state, which would explain the harm
    mechanically rather than statistically.
  * `corr_logits_distance` measures whether the guide's scores are even aligned with distance:
    a strong POSITIVE correlation with distance means high logits are placed on far-apart pairs, so
    the guide actively prefers expensive edges.

Descriptive diagnostic only. No arm is compared for a performance claim here, no interval is a
registered test, and nothing about impossibility or equivalence is claimed.
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


def correlate(logits, dist):
    """Pearson and Spearman over off-diagonal entries."""
    n = len(dist)
    off = ~np.eye(n, dtype=bool)
    x = np.asarray(logits)[off].astype(np.float64)
    y = np.asarray(dist)[off].astype(np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return {'pearson': None, 'spearman': None}
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    spear = float(np.corrcoef(rx, ry)[0, 1])
    return {'pearson': pear, 'spearman': spear}


def trace_decisions(instance, incumbent, contract, removed, adjacency, dist, logits,
                    guide_weight, enumerate_options):
    """Follow the guide's own trajectory and price its choice against the distance-only choice."""
    nodes = len(dist)
    off = ~np.eye(nodes, dtype=bool)
    dscale = max(1e-9, float(dist[off].mean()))
    lscale = max(1e-8, float(np.std(np.asarray(logits)[off]))) if logits is not None else 1.0
    candidate = [[i for i in r if i not in removed] for r in incumbent]
    candidate += [[] for _ in range(contract.vehicles - len(candidate))]
    remaining = sorted(removed)
    rows = []
    while remaining:
        choices = []
        for i in remaining:
            opts = enumerate_options(i, candidate, instance, contract, dist, adjacency)
            if not opts:
                continue
            scored = []
            for (delta, k, at, a, b) in opts:
                gain = 0.0 if logits is None else (
                    float(logits[a, i]) + float(logits[i, b])
                    - (0.0 if a == b else float(logits[a, b]))) / lscale
                scored.append((delta / dscale - guide_weight * gain, delta, k, at))
            scored.sort(key=lambda r: (r[0], r[2], r[3]))
            regret = (scored[1][0] - scored[0][0]) if len(scored) > 1 else float('inf')
            choices.append((regret, i, scored))
        if not choices:
            break
        _, i, scored = max(choices, key=lambda r: (r[0], -r[1]))
        by_dist = min(scored, key=lambda r: (r[1], r[2], r[3]))
        chosen = scored[0]
        rows.append({'customer': i, 'guide_delta': float(chosen[1]),
                     'distance_delta': float(by_dist[1]),
                     'extra_delta': float(chosen[1] - by_dist[1]),
                     'guide_option': [int(chosen[2]), int(chosen[3])],
                     'distance_option': [int(by_dist[2]), int(by_dist[3])],
                     'agrees': bool(chosen[2] == by_dist[2] and chosen[3] == by_dist[3])})
        candidate[int(chosen[2])].insert(int(chosen[3]), i)
        remaining.remove(i)
    return rows


def main(args):
    started = time.perf_counter()
    scripts = bootstrap(args.scripts)
    from delivery_equal_schedule import generate, DeliveryContract
    from delivery_baselines import matrices
    from delivery_maskco import MaskCORouteProvider, partial_adjacency
    from delivery_trained_guide import DeliveryTrainedGuide
    from delivery_traintask_train import enumerate_options

    proto = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    seeds = proto['data_plan']['registered_seeds']
    dev_seed = int(seeds['dev_seed'])
    mask_seed = int(seeds['mask_seed_dev'])
    contract = DeliveryContract()

    guide = DeliveryTrainedGuide(args.checkpoint, args.upstream_root)
    zeroshot = MaskCORouteProvider(checkpoint=args.checkpoint, upstream_root=args.upstream_root)
    data0, init0 = generate(dev_seed, 0)
    f0 = guide.features(data0, contract.capacity)
    if hasattr(f0, 'block_until_ready'):
        f0.block_until_ready()
    guide.load_head(args.head, int(np.asarray(f0).shape[-1]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    per_arm = {'trained': [], 'zeroshot_guide': []}
    corrs = {'trained': [], 'zeroshot_guide': []}
    per_instance = []
    for inst in range(args.instances):
        data, initial = generate(dev_seed, inst)
        dist = matrices(data)
        nodes = len(dist)
        feats = guide.features(data, contract.capacity)
        if hasattr(feats, 'block_until_ready'):
            feats.block_until_ready()
        rng = np.random.default_rng(np.random.SeedSequence([mask_seed, inst]))
        size = max(1, int(math.ceil(args.mask_fraction * (nodes - 1))))
        row = {'instance': inst, 'arms': {}}
        for arm in ('trained', 'zeroshot_guide'):
            all_rows = []
            for _ in range(args.masks_per_instance):
                removed = sorted(int(v) for v in rng.choice(np.arange(1, nodes), size=size,
                                                            replace=False))
                adjacency = partial_adjacency(initial, removed, nodes)
                if arm == 'trained':
                    logits = guide.logits(feats, adjacency)
                else:
                    logits = zeroshot.logits(feats, adjacency)
                corrs[arm].append(correlate(logits, dist))
                all_rows.extend(trace_decisions(data, initial, contract, removed, adjacency,
                                                dist, logits, args.guide_weight,
                                                enumerate_options))
            extra = np.array([r['extra_delta'] for r in all_rows], dtype=np.float64)
            agree = np.array([r['agrees'] for r in all_rows], dtype=bool)
            per_arm[arm].append(extra)
            row['arms'][arm] = {
                'n_decisions': int(len(extra)),
                'mean_extra_delta': float(extra.mean()) if len(extra) else None,
                'n_extra_positive': int((extra > 1e-9).sum()),
                'n_extra_zero': int((np.abs(extra) <= 1e-9).sum()),
                'n_extra_negative': int((extra < -1e-9).sum()),
                'agreement_with_distance_ordering': float(agree.mean()) if len(agree) else None,
            }
        per_instance.append(row)
        print(json.dumps(row), flush=True)

    summary = {}
    for arm in ('trained', 'zeroshot_guide'):
        e = np.concatenate([x for x in per_arm[arm]]) if per_arm[arm] else np.array([])
        c = corrs[arm]
        pe = [x['pearson'] for x in c if x['pearson'] is not None]
        sp = [x['spearman'] for x in c if x['spearman'] is not None]
        summary[arm] = {
            'n_decisions': int(len(e)),
            'mean_extra_delta': float(e.mean()) if len(e) else None,
            'median_extra_delta': float(np.median(e)) if len(e) else None,
            'n_extra_positive': int((e > 1e-9).sum()),
            'n_extra_zero': int((np.abs(e) <= 1e-9).sum()),
            'n_extra_negative': int((e < -1e-9).sum()),
            'corr_logits_distance_pearson_mean': float(np.mean(pe)) if pe else None,
            'corr_logits_distance_spearman_mean': float(np.mean(sp)) if sp else None,
            'corr_per_mask_pearson': pe,
        }
    result = {
        'kind': 'delivery_guide_harm_mechanism_diagnostic',
        'dev_seed': dev_seed, 'mask_seed_dev': mask_seed,
        'instances': args.instances, 'masks_per_instance': args.masks_per_instance,
        'mask_fraction': args.mask_fraction, 'guide_weight': args.guide_weight,
        'trained_head_sha256': file_sha(args.head),
        'summary': summary,
        'per_instance': per_instance,
        'reading': ('mean_extra_delta is the extra RAW insertion delta the guide pays at each '
                    'decision relative to the distance-only choice available at that SAME state. '
                    'A positive mean means the guide is mechanically more expensive, not merely '
                    'unlucky. corr_logits_distance measures whether the guide scores expensive '
                    'pairs highly: a positive correlation is a direct mechanism for harm.'),
        'what_this_is_not': [
            'not a performance comparison: no final plan is scored here',
            'descriptive only; no interval is a registered test and no alpha applies',
            'no impossibility, equivalence, attribution or stable-gain claim'],
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({'summary': summary, 'wall_s': result['wall_s']}, indent=2, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--head', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--upstream-root', required=True)
    ap.add_argument('--instances', type=int, default=8)
    ap.add_argument('--masks-per-instance', type=int, default=2)
    ap.add_argument('--mask-fraction', type=float, default=0.5)
    ap.add_argument('--guide-weight', type=float, default=1.0)
    main(ap.parse_args())
