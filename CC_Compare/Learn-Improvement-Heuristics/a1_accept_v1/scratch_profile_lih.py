"""LIH 每步耗时剖面 + 线程数扫描（仅调试，不入库）。"""
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

from lih_provider import LIHProvider, _emdedding


def make_sp(coords, demands, capacity=50.0):
    n = len(coords)
    sp = SimpleNamespace(
        pool_customer_ids=tuple(range(1, n)),
        node_ids=tuple(range(n)),
        coords=tuple(tuple(float(x) for x in c) for c in coords),
        demands=tuple(float(d) for d in demands),
        tw_start=tuple(float(0.0) for _ in range(n)),
        tw_end=tuple(float(22.0) for _ in range(n)),
        service_time=tuple(float(0.05) for _ in range(n)),
        capacity=float(capacity),
        depot_tw_end=22.0,
    )
    sp.node_index = lambda c: sp.node_ids.index(int(c))
    return sp


def build_state(provider, sp, chunk):
    n = len(chunk)
    idx = {c: sp.node_index(c) for c in chunk}
    depot = [float(x) for x in sp.coords[sp.node_index(0)]]
    loc = np.zeros((1, 100, 2), dtype=np.float32)
    dem = np.zeros((1, 100), dtype=np.float32)
    dems = []
    for k, c in enumerate(chunk):
        loc[0, k] = [float(x) for x in sp.coords[idx[c]]]
        d = float(sp.demands[idx[c]]) / float(sp.capacity)
        dem[0, k] = d
        dems.append(d)
    for k in range(n, 100):
        loc[0, k] = depot
    split_depots = list(range(51, 101))
    fake = list(range(n + 1, 51))
    rec_tokens, di, seg = [], 0, 0.0
    for k in range(n):
        if seg > 0 and seg + dems[k] > 1.0 + 1e-6:
            rec_tokens.append(split_depots[di])
            di += 1
            seg = 0.0
        rec_tokens.append(k + 1)
        seg += dems[k]
    rec_tokens += split_depots[di:]
    rec_tokens += fake
    input_ = {'loc': torch.from_numpy(loc), 'demand': torch.from_numpy(dem)}
    rec = torch.tensor([rec_tokens], dtype=torch.long)
    return input_, rec


def main():
    ckpt = os.path.normpath(os.path.join(_HERE, '..', 'CVRP', 'CVRP50',
                                         'outputs', 'cvrp_50', 'run', 'epoch-199.pt'))
    rng = np.random.default_rng(0)
    n_pool = 50
    coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
    demands = np.concatenate([[0.0], rng.uniform(1, 3, n_pool)])
    sp = make_sp(coords, demands)
    chunk = list(range(1, n_pool + 1))
    provider = LIHProvider(ckpt, device='cpu', num_loc=100)
    provider._load()
    input_, rec = build_state(provider, sp, chunk)
    for nthreads in (1, 2, 4, 8):
        torch.set_num_threads(nthreads)
        pre_length, rec2, che_mask = provider._problem.get_costs(
            input_, rec, provider._dic, provider._cl)
        model = provider._model
        exchange = None
        rec = rec2
        t_emb = t_mod = t_cost = 0.0
        with torch.no_grad():
            for _ in range(30):
                t0 = time.perf_counter()
                info, pos = _emdedding(input_, rec)
                t_emb += time.perf_counter() - t0
                t0 = time.perf_counter()
                ll, now_length, rec, exchange, act, che_mask = model(
                    input_, rec, info, pos, exchange, che_mask,
                    provider._dic, provider._cl, action=None, test=False)
                t_mod += time.perf_counter() - t0
        # 单独测 get_costs（另跑一轮，用固定 rec）
        rec_f = torch.tensor([list(range(1, 101))], dtype=torch.long)
        for _ in range(3):
            t0 = time.perf_counter()
            provider._problem.get_costs(input_, rec_f, provider._dic,
                                        provider._cl, torch.tensor([[3, 9]]))
            t_cost += time.perf_counter() - t0
        print('threads=%d: emb=%.1fms mod=%.1fms cost=%.1fms/step'
              % (nthreads, t_emb / 30 * 1e3, t_mod / 30 * 1e3,
                 t_cost / 3 * 1e3), flush=True)


if __name__ == '__main__':
    main()
