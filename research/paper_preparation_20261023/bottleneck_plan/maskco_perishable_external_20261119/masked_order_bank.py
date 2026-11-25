"""Build and price identical geometry-defined masked route orders for all rankers."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter, process_time
import json
import numpy as np
from directed_decoder_fit import HERE, read
from global_masked_split import chain, propose, split_order
from remaining_edges_study import development_model
from prepare_panel import independent_audit
from view_frame_probe import shuffle_probabilities

OUT = HERE / "results/order_bank01"
MASKS = (0, 6, 4, 10, 5, 11)


def edge_features(matrix):
    ranks = np.argsort(np.argsort(-matrix, axis=1, kind="stable"), axis=1, kind="stable")
    return -np.log1p(ranks)


def prepare_maps(nodes, distance, value):
    pair = np.full(distance.shape, -1.e5)
    singles = {i: value((i,)) for i in range(1, len(nodes))}
    for i in singles:
        for j in singles:
            if i != j:
                benefit = value((i, j)) - singles[i] - singles[j]
                if np.isfinite(benefit):
                    pair[i, j] = benefit
    load = nodes[:, 3] / 20000.
    maps = [-distance, edge_features(-distance), edge_features(pair),
            np.clip(pair, -1000., 1000.),
            -(load[:, None] + load[None, :]),
            -(np.maximum(0., nodes[:, 4, None] - nodes[None, :, 5]))]
    return pair, np.stack(maps)


def heat_maps(scores):
    ranks = edge_features(scores)
    std = np.maximum(scores.std(axis=1, keepdims=True), .001)
    standardized = np.clip((scores - scores.mean(axis=1, keepdims=True)) / std, -6., 6.)
    return np.stack((ranks, np.minimum(ranks, ranks.T), standardized))


def changed_features(old_edges, order, maps):
    edges = set(zip(order, order[1:]))
    added = sorted(edges - old_edges)
    removed = sorted(old_edges - edges)
    if not added or len(added) != len(removed):
        raise ValueError("Distinct customer permutation has no matched changes")
    def summarize(pairs):
        i, j = np.asarray(pairs).T
        a = maps[:, i, j]
        return np.stack((a.sum(axis=1), a.mean(axis=1), a.min(axis=1), a.max(axis=1)), axis=1)
    after, before = summarize(added), summarize(removed)
    return np.concatenate(((after - before).ravel(), after[:, 2], before[:, 2]))


def common_bank(order, mask, routes, distance, pair, masses, seed):
    seen = {order}
    orders = []
    streams = [propose(order, mask, routes, m, masses, seed + idx, 256)
               for idx, m in enumerate((-distance, pair))]
    for batch in zip(*streams):
        for seq in batch:
            if seq not in seen:
                seen.add(seq)
                orders.append(seq)
                if len(orders) == 48:
                    return orders
    raise ValueError("Fewer than 48 distinct orders")


def run_case(i):
    b = read(HERE / "headcapture01.json")["blocks"][i]
    case = b["case"]
    capture = HERE / "results/headcapture01/out"
    entry = read(HERE / "heldout_sources.json")["derived_instances"][case]
    states = read(HERE / "archived_states.json")[case]
    rec = read(capture / f"{i}.json")
    if rec["case"] != case or rec["max_logit_error"] > .002:
        raise ValueError("Archived neural representation differs")
    start, cpu = perf_counter(), process_time()
    nodes, d, model, value = development_model(case, entry)
    pair, phys = prepare_maps(nodes, d, value)
    rows = []
    requests, audits = 0, 0
    with np.load(capture / f"{i}.npz") as z:
        for k in MASKS:
            routes = tuple(map(tuple, states["initial" if k < 6 else "terminal"]))
            h, mask = z["embed"][k], z["mask"][k].astype(int).tolist()
            score = h @ h.T * rec["scale"]
            rng = np.random.default_rng([56303, i, k])
            shuffled = shuffle_probabilities(score, d, routes, mask, rng)
            maps = np.concatenate((phys, heat_maps(score), heat_maps(shuffled)))
            order = chain(routes, int(rng.integers(len(routes))), d)
            old_edges = set(zip(order, order[1:]))
            candidates = common_bank(order, mask, routes, d, pair, nodes[:, 3], 56309 + 101 * i + 3 * k)
            visible = set(order) - set(mask)
            anchors = tuple(x for x in order if x in visible)
            features, labels = [], []
            for seq in [order, *candidates]:
                if len(seq) != 100 or set(seq) != set(order) or tuple(x for x in seq if x in visible) != anchors:
                    raise ValueError("The common bank changed the visible customer order")
                u, fleet, calls = split_order(seq, nodes[:, 3], value)
                audited = independent_audit(model, [model.route(r) for r in fleet], True)
                if abs(u-audited) > 1.e-6:
                    raise ValueError("Full fleet price differs from independent audit")
                labels.append(audited)
                requests += calls
                audits += 1
                if seq != order:
                    features.append(np.concatenate([changed_features(old_edges, seq, part)
                                    for part in (maps[:6], maps[6:9], maps[9:])]).tolist())
            rows.append(dict(mask_index=k, features=features, U=labels, orders=candidates))
    doc = dict(case=case, rows=rows, segment_requests=requests, prices=value.cache_info().misses,
               audits=audits, cpu_s=process_time()-cpu, wall_s=perf_counter()-start)
    (OUT / f"{i}.json").write_text(json.dumps(doc, allow_nan=False, separators=(",", ":")), encoding="utf8")
    return i


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case", type=int)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.case is not None:
        print(run_case(args.case))
        return
    missing=[i for i in range(27) if not (OUT/f"{i}.json").exists()]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i in pool.map(run_case, missing):
            print(i, flush=True)


if __name__ == "__main__":
    main()
