"""Evaluate degree-constrained MASKCO shortlists on archived development fleets."""
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter, process_time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "maskco_perishable_external_20261119"
sys.path.insert(0, str(DEV))
from directed_decoder_fit import SOURCES, read
from prepare_panel import independent_audit
from remaining_edges_study import development_model
from view_frame_probe import shuffle_probabilities

METHODS = ("heat", "heat_balanced", "distance", "distance_balanced", "shuffled_balanced")
CANDIDATE = "heat_balanced"
EPS = 1e-6


def residual(routes, mask, count):
    hidden = np.zeros(count, bool)
    hidden[mask] = True
    outgoing = np.ones(count)
    incoming = np.ones(count)
    outgoing[0] = incoming[0] = len(routes)
    protected = set()
    missing = np.zeros((count, count), dtype=bool)
    for route in routes:
        for i, j in zip((0,) + route, route + (0,)):
            if not (hidden[i] or hidden[j]):
                protected.add((i, j))
                outgoing[i] -= 1
                incoming[j] -= 1
            else:
                missing[i, j] = True
    send, recv = np.flatnonzero(outgoing), np.flatnonzero(incoming)
    support = ((hidden[send, None] | hidden[recv][None, :]) &
               (send[:, None] != recv[None, :]))
    # An unused arc can carry flow iff it lies on an alternating cycle
    # relative to the original fleet, which is itself a feasible completion.
    reverse = missing[np.ix_(send, recv)].T
    graph = bmat([[None, csr_matrix(support)], [csr_matrix(reverse), None]],format="csr")
    _, component = connected_components(graph,directed=True,connection="strong")
    support &= component[:len(send),None] == component[None,len(send):]
    return send, recv, outgoing[send], incoming[recv], support, protected


def edge_matrices(scores, send, recv, outgoing, incoming, support):
    values = scores[np.ix_(send, recv)]
    affinity = np.zeros_like(values, dtype=float)
    for row, active in enumerate(support):
        v = values[row, active]
        asc = np.sort(v)
        rank = 1 + len(v) - np.searchsorted(asc, v, side="right")
        affinity[row, active] = 1 / rank
    matrix = affinity.copy()
    for iteration in range(2000):
        matrix *= (outgoing / matrix.sum(axis=1))[:, None]
        matrix *= incoming / matrix.sum(axis=0)
        error = float(np.abs(matrix.sum(axis=1) - outgoing).max())
        if error <= 1e-8:
            break
    else:
        raise ValueError(f"Unmet residual marginals: {error}")
    row_rank = np.zeros_like(scores, dtype=float)
    balanced = np.zeros_like(row_rank)
    row_rank[np.ix_(send, recv)] = affinity
    balanced[np.ix_(send, recv)] = matrix
    return row_rank, balanced, error


def relocation(fleet, index, u, v):
    si, ti = index[u], index[v]
    source, target = fleet[si], fleet[ti]
    rest = tuple(x for x in source if x != u)
    if si == ti:
        at = rest.index(v)
        return {si: rest[:at] + (u,) + rest[at:]}
    at = target.index(v)
    return {si: rest, ti: target[:at] + (u,) + target[at:]}


def check_fleet(model, fleet, expected, protected):
    if sorted(u for r in fleet for u in r) != list(range(1, 101)):
        raise ValueError("Incomplete or overlapping customer coverage")
    edges = {(i, j) for r in fleet for i, j in zip((0,) + r, r + (0,))}
    if not protected <= edges:
        raise ValueError("Changed visible skeleton edge")
    measured = independent_audit(model, [model.route(r) for r in fleet], True)
    if abs(measured - expected) > EPS:
        raise ValueError("Independent net benefit differs")
    return measured


def context(case, dataset, geometry, index, state, seed, k, mask, signal, distance, model, value, fleet):
    start = perf_counter()
    mask = mask.astype(int).tolist()
    parts = residual(fleet, mask, len(distance))
    send, recv, outgoing, incoming, support, protected = parts
    heat, balance, err_h = edge_matrices(signal, *parts[:5])
    spatial, dist_balance, err_d = edge_matrices(-distance, *parts[:5])
    shuffled = shuffle_probabilities(signal, distance, fleet, mask, np.random.default_rng([seed, index, k, 17]))
    _, shuffle_balance, err_s = edge_matrices(shuffled, *parts[:5])
    models = dict(zip(METHODS, (heat, balance, spatial, dist_balance, shuffle_balance)))
    chosen = np.random.default_rng([seed, index, k]).choice(mask, 20, replace=False).astype(int).tolist()
    route_index = {u: ri for ri, route in enumerate(fleet) for u in route}
    successor = {u: v for r in fleet for u, v in zip(r, r[1:] + (0,))}
    route_values = [value(r) for r in fleet]
    base = check_fleet(model, fleet, sum(route_values), protected)
    best = {a: (base, {}, None) for a in (*METHODS, "oracle")}
    candidates = {a: set() for a in METHODS}
    attempted = 0
    feasible = 0
    for u in chosen:
        targets = [int(v) for v in recv if heat[u,v] > 0 and v not in (0, u, successor[u])]
        for a, weights in models.items():
            for v in sorted(targets, key=lambda j: (-weights[u, j], j))[:3]:
                candidates[a].add((u, v))
        for v in targets:
            changed = relocation(fleet, route_index, u, v)
            score = base + sum((value(r) if r else 0.) - route_values[i] for i, r in changed.items())
            attempted += 1
            if not math.isfinite(score):
                continue
            feasible += 1
            for a in best:
                if (a == "oracle" or (u, v) in candidates[a]) and score > best[a][0] + EPS:
                    best[a] = (score, changed, (u, v))
    scores = {}
    for a, (score, changed, move) in best.items():
        updated = tuple(changed.get(i, r) for i, r in enumerate(fleet))
        updated = tuple(r for r in updated if r)
        audited = check_fleet(model, updated, score, protected)
        scores[a] = dict(gain_U=audited-base, move=move, routes=updated, U=audited)
    if any(len(keys) != len(chosen)*3 for keys in candidates.values()):
        raise ValueError("Inconsistent shortlist budget")
    if max(v["gain_U"] for v in scores.values()) > scores["oracle"]["gain_U"] + EPS:
        raise ValueError("Restricted arm exceeds finite-neighbourhood oracle")
    return dict(case=case, dataset=dataset, geometry=geometry, state=state, seed=seed, k=k,
                mask=mask, clients=chosen, base_U=base, permitted=attempted, feasible=feasible,
                proposals_per_arm=len(chosen)*3,
                replacement_fraction=1 - len(candidates["heat"] & candidates[CANDIDATE]) / len(candidates["heat"]),
                margin_error=max(err_h, err_d, err_s), arms=scores, wall_s=perf_counter()-start)


def geometry_means(rows, measure):
    cases = defaultdict(list)
    for row in rows:
        cases[row["geometry"], row["case"]].append(measure(row))
    geos = defaultdict(list)
    for (g, _), vals in cases.items():
        geos[g].append(mean(vals))
    return {g: mean(vals) for g, vals in geos.items()}


def report(rows):
    by_source = {}
    for s in SOURCES:
        source_rows = [r for r in rows if r["dataset"] == s]
        contrasts = {}
        for a in METHODS:
            if a == CANDIDATE:
                continue
            def delta(r):
                return r["arms"][CANDIDATE]["gain_U"] - r["arms"][a]["gain_U"]
            geo = geometry_means(source_rows, delta)
            seeds = sorted({r["seed"] for r in source_rows})
            contrasts[a] = dict(mean=mean(geo.values()), by_geometry=geo,
                                wins=sum(v > EPS for v in geo.values()), geometries=len(geo),
                                by_seed=[mean(geometry_means([r for r in source_rows if r["seed"] == i], delta).values()) for i in seeds])
        gains = {a:mean(geometry_means(source_rows,lambda r:r["arms"][a]["gain_U"]).values())
                 for a in (*METHODS,"oracle")}
        by_source[s] = dict(gain_U=gains, contrasts=contrasts,
                            changed=mean(geometry_means(source_rows, lambda r:r["replacement_fraction"]).values()))
    passed = True
    for v in by_source.values():
        contrast = v["contrasts"]["distance_balanced"]
        passed &= (v["contrasts"]["heat"]["mean"] > EPS and contrast["mean"] > EPS
                   and contrast["wins"]*2 > contrast["geometries"])
    for k in (0, 1):
        passed &= mean(v["contrasts"]["distance_balanced"]["by_seed"][k] for v in by_source.values()) > EPS
    passed &= mean(v["contrasts"]["shuffled_balanced"]["mean"] for v in by_source.values()) > EPS
    return by_source, bool(passed)


def phase(stage, indices, repeats, catalog, blocks, states):
    directory = HERE / "results" / stage
    directory.mkdir()
    start, cpu = perf_counter(), process_time()
    rows = []
    calls = 0
    for ix in indices:
        case = blocks[ix]["case"]
        receipt = read(DEV/f"results/headcapture01/out/{ix}.json")
        if receipt["case"] != case or receipt["max_logit_error"] > .002 or receipt["contexts"] != 12:
            raise ValueError("Archived forward differs")
        _, distance, model, value = development_model(case, catalog["derived_instances"][case])
        meta = catalog["cases"][case]
        sample = []
        with np.load(DEV/f"results/headcapture01/out/{ix}.npz") as z:
            for repeat, seed in repeats:
                for state, k in (("initial", repeat), ("terminal", repeat+6)):
                    fleet = tuple(map(tuple, states[case][state]))
                    h = z["embed"][k]
                    heat = (h @ h.T)*receipt["scale"]
                    sample.append(context(case, meta["dataset"], meta["geometry"], ix, state, seed, k,
                                          z["mask"][k], heat, distance, model, value, fleet))
        (directory/f"{case}.json").write_text(json.dumps(sample, allow_nan=False),encoding="utf8")
        rows.extend(sample)
        calls += value.cache_info().misses
    stats, passed = report(rows)
    cost = dict(contexts=len(rows),method_endpoints=len(rows)*len(METHODS),oracle_endpoints=len(rows),
                audit_calls=len(rows)*(len(METHODS)+2),eligible_moves=sum(r["permitted"] for r in rows),
                feasible_moves=sum(r["feasible"] for r in rows),unique_route_prices_within_case=calls,
                wall_s=perf_counter()-start,cpu_s=process_time()-cpu,max_margin_error=max(r["margin_error"] for r in rows))
    decision = ("RUN_DEVELOPMENT_REPLICATION" if stage=="degree01" else "ELIGIBLE_FOR_TIMED_DEVELOPMENT") if passed else "STOP_RESIDUAL_DEGREE"
    (directory/"analysis.json").write_text(json.dumps(dict(status="COMPLETE",decision=decision,passed=passed,
                                                          sources=stats,cost=cost),indent=2,allow_nan=False),encoding="utf8")
    with (directory/"paired.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f)
        writer.writerow(["case","dataset","geometry","state","seed","mask_index","base_U","shortlist_change",*METHODS,"oracle"])
        for r in rows:
            writer.writerow([r[x] for x in ("case","dataset","geometry","state","seed","k","base_U","replacement_fraction")]+
                            [r["arms"][a]["gain_U"] for a in (*METHODS,"oracle")])
    print(stage,decision,cost,flush=True)
    return passed


def fixture():
    routes = ((1,2,3),(4,5))
    mask = [1,3,5]
    parts = residual(routes,mask,6)
    assert parts[-1] == {(0,4)}
    scores = -np.abs(np.arange(6)[:,None]-np.arange(6)[None,:])
    _,p,e = edge_matrices(scores,*parts[:5])
    assert e <= 1e-8 and np.max(np.abs(p.sum(axis=0)[parts[1]]-parts[3])) <= 1e-8
    ri = {u:i for i,r in enumerate(routes) for u in r}
    assert relocation(routes,ri,1,3) == {0:(2,1,3)}
    assert relocation(routes,ri,3,5) == {0:(1,2),1:(4,3,5)}
    for u in mask:
        for v in parts[1]:
            if v and v != u:
                change = relocation(routes,ri,u,int(v))
                updated = [change.get(i,r) for i,r in enumerate(routes)]
                edges = {(i,j) for r in updated if r for i,j in zip((0,)+r,r+(0,))}
                assert parts[-1] <= edges
    print("PASS degree support, depot degree, matrix scaling, intra/interroute moves and visible edges")


if __name__ == "__main__":
    fixture()
    if sys.argv[1:] == ["--fixture"]:
        sys.exit(0)
    if sys.argv[1:]:
        raise SystemExit("Usage: residual_degree.py [--fixture]")
    catalog = read(DEV/"heldout_sources.json")
    blocks = read(DEV/"headcapture01.json")["blocks"]
    cases = {b["case"] for b in read(DEV/"fleetscreen01.json")["blocks"]}
    assert len(cases) == 11 and len(blocks)==27
    states = read(DEV/"archived_states.json")
    initial = [i for i,b in enumerate(blocks) if b["case"] in cases]
    rest = [i for i,b in enumerate(blocks) if b["case"] not in cases]
    if phase("degree01",initial,((0,69101),(1,69102)),catalog,blocks,states):
        phase("degree02",rest,((2,69111),(3,69112)),catalog,blocks,states)
