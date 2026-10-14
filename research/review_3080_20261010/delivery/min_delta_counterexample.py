"""Fixed Euclidean engineering counterexample, not a research effect estimate.

The current customer's minimum insertion delta need not minimize the completed
route after another customer arrives. All choices are capacity feasible here.
"""
import json
import numpy as np


def counterexample():
    coords = np.asarray([[.4967018417953405, .46831253442657617],
        [.12769032289247773, .2575625049095238], [.0031811092933116125, .3810677478158191],
        [.5758730840369666, .42729877128295013], [.8351023473085695, .6164912514101898],
        [.2660839120112748, .8110221113180831]])
    dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    original = [1, 2, 3]
    chain = [0] + original + [0]
    options = sorted((float(dist[a, 4] + dist[4, b] - dist[a, b]), at)
                     for at, (a, b) in enumerate(zip(chain, chain[1:])))
    completed = []
    for delta, at in options:
        route = original.copy()
        route.insert(at, 4)
        chain = [0] + route + [0]
        _, second_at = min((float(dist[a, 5] + dist[5, b] - dist[a, b]), pos)
                          for pos, (a, b) in enumerate(zip(chain, chain[1:])))
        route.insert(second_at, 5)
        chain = [0] + route + [0]
        completed.append({'first_delta': delta, 'first_position': at, 'routes': [route],
                          'final_distance': float(sum(dist[a, b] for a, b in zip(chain, chain[1:])))})
    greedy = completed[0]
    best = min(completed, key=lambda row: row['final_distance'])
    assert best['first_delta'] > greedy['first_delta'] and best['final_distance'] < greedy['final_distance']
    return {'stage': 'constructed_synthetic_logic_counterexample', 'coordinates': coords.tolist(),
            'all_options': completed, 'greedy_immediate': greedy, 'better_completed': best,
            'no_learning_gain_claim': True, 'does_not_change_old_numeric_failure': True}


if __name__ == '__main__':
    print(json.dumps(counterexample(), indent=2))
