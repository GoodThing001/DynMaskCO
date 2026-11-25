"""Apply a single fleet-pricing prefix before the locked masked search."""
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import external_common as api
import fleet_timed_pool as pool
from fleet_pricing import MaskedMaster, propose


HERE = Path(__file__).resolve().parent
METHODS = ("FLEET_PRICE", "FLEET_PRICE_D", "PREFIX_DISTANCE", "ORDER_SPLIT", "DIST", "DIST_SLOT")


def prefix(task, provider, first, seed, clock, *, neural):
    start = time.perf_counter()
    before = task.route.cache_info()
    rng = np.random.default_rng(seed)
    mask = rng.choice(np.arange(1, 101), size=40, replace=False)
    parent = tuple(map(tuple, first))
    raw = None
    forward_s = 0.
    if neural:
        from base_inputs import BaseInferenceInputs
        from profit_neural import predict
        tick = time.perf_counter()
        builder = BaseInferenceInputs(task)
        raw = predict(provider.model, builder.build(parent, mask))
        forward_s = time.perf_counter() - tick
    nodes = np.asarray(task.ins["nodes"])
    distance = np.asarray(task.ins["distance"])

    def value(order):
        result = task.route(order)
        return result.net_benefit if result.feasible else -math.inf

    master = MaskedMaster(parent, mask.tolist(), value)
    common = master.common()
    prices = master.prices(common)
    base, base_U, base_gap = master.integer(common, parent)
    expanded, attempts = propose(master, common, prices, distance, raw, nodes[:, 3])
    proposed, proposed_U, gap = master.integer(expanded, base)
    after = task.route.cache_info()
    requests = after.hits + after.misses - before.hits - before.misses
    return proposed, dict(initial_U=clock.events[0]["score"]["net_benefit_U"],
                          common_U=base_U, proposed_U=proposed_U, proposed_routes=proposed,
                          accepted=False, common_gap_U=base_gap, mip_gap_U=gap,
                          mask=mask.tolist(), route_requests=requests,
                          unique_route_requests=after.misses-before.misses, forwards=int(neural),
                          forward_s=forward_s, elapsed_s=time.perf_counter()-start,
                          common_columns=len(common), columns=len(expanded), attempts=attempts)


def endpoint(arm, block, cfg):
    case, seed = block["case"], block["seed"]
    seconds = block.get("seconds", cfg["seconds"])
    if arm in ("DIST", "DIST_SLOT"):
        return api.endpoint(arm, case, seed, dict(cfg, seconds=seconds))
    if arm not in METHODS:
        raise ValueError(arm)
    inner = "ORDER_SPLIT_D" if arm == "FLEET_PRICE_D" else "ORDER_SPLIT"
    _, _, access, core, src, locked = api.install(inner, dict(candidate="ORDER_SPLIT", matched="ORDER_SPLIT_D"), False)
    from timed_search import single_cpu_guard
    affinity = single_cpu_guard()
    tick = time.perf_counter()
    lib = pool.library()
    policy = None
    if inner == "ORDER_SPLIT":
        policy, _ = pool.study.load("SLOT", access, core, src, locked, "c205")
    setup = time.perf_counter() - tick
    scope = access.Scope("fleet-pricing-development", frozenset([(0, "external_test", case)]),
                         frozenset(["initialize", "infer", "search"]))
    with access.restricted_scope(scope):
        callback = None
        if arm != "ORDER_SPLIT":
            from functools import partial
            callback = partial(prefix, neural=arm == "FLEET_PRICE")
        result = pool.solve(lib, core, locked, policy, case, "external_test", seed, inner, seconds, prefix=callback)
    result.update(method=arm, cpu=affinity, setup_s=setup)
    return result


def main(index):
    cfg = api.read(HERE/"phase.json")
    block = cfg["blocks"][index]
    if len(sys.argv) == 3:
        arm = sys.argv[2]
        if arm not in block["arms"]:
            raise ValueError(arm)
        row = endpoint(arm, block, cfg)
        with (HERE/"out"/f"{index}_{arm}.cell.json").open("x") as output:
            json.dump(row, output, allow_nan=False)
        return
    rows = {}
    for arm in block["arms"]:
        subprocess.run([sys.executable, "-B", str(HERE/"fleet_timed_worker.py"), str(index), arm], check=True)
        rows[arm] = api.read(HERE/"out"/f"{index}_{arm}.cell.json")
    if len({json.dumps(row["initial_routes"]) for row in rows.values()}) != 1:
        raise ValueError("Paired initial routes disagree")
    with (HERE/"out"/f"{index}.json").open("x") as output:
        json.dump(dict(case=block["case"], seed=block["seed"], arms=rows), output, allow_nan=False)


if __name__ == "__main__":
    main(int(sys.argv[1]))
