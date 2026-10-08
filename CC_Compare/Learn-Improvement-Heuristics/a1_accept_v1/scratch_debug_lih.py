"""LIH provider 调试：定位 multinomial NaN 来源（仅调试用，不入库）。"""
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

from lih_provider import (LIHProvider, _emdedding, _install_shims)


def make_sp(coords, demands, capacity=50.0):
    n = len(coords)
    dist = [[np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
             for j in range(n)] for i in range(n)]
    sp = SimpleNamespace(
        pool_customer_ids=tuple(range(1, n)),
        node_ids=tuple(range(n)),
        coords=tuple(tuple(float(x) for x in c) for c in coords),
        demands=tuple(float(d) for d in demands),
        tw_start=tuple(float(0.0) for _ in range(n)),
        tw_end=tuple(float(22.0) for _ in range(n)),
        service_time=tuple(float(0.05) for _ in range(n)),
        dist_mat=tuple(tuple(float(x) for x in row) for row in dist),
        travel_mat=tuple(tuple(float(x) for x in row) for row in dist),
        capacity=float(capacity),
        depot_tw_end=22.0,
    )
    sp.node_index = lambda c: sp.node_ids.index(int(c))
    return sp


def main():
    ckpt = os.path.normpath(os.path.join(_HERE, '..', 'CVRP', 'CVRP50',
                                         'outputs', 'cvrp_50', 'run', 'epoch-199.pt'))
    provider = LIHProvider(ckpt, device='cpu', num_loc=100)
    provider._load()
    n_pool = 5
    rng = np.random.default_rng(0)
    coords = np.vstack([[0.5, 0.5], rng.uniform(0, 1, (n_pool, 2))])
    demands = np.concatenate([[0.0], rng.uniform(1, 3, n_pool)])
    sp = make_sp(coords, demands)
    chunk = [int(c) for c in sp.pool_customer_ids]
    n = len(chunk)
    idx = {c: sp.node_index(c) for c in chunk}
    depot = [float(x) for x in sp.coords[sp.node_index(0)]]
    loc = np.zeros((1, 100, 2), dtype=np.float32)
    loc[0, 0] = depot
    dem = np.zeros((1, 100), dtype=np.float32)
    for k, c in enumerate(chunk):
        loc[0, 1 + k] = [float(x) for x in sp.coords[idx[c]]]
        dem[0, 1 + k] = float(sp.demands[idx[c]]) / float(sp.capacity)
    for k in range(n, 100):
        loc[0, k] = depot
    input_ = {'loc': torch.from_numpy(loc), 'demand': torch.from_numpy(dem)}
    rec = torch.arange(1, 101, dtype=torch.long).view(1, 100)
    pre_length, rec, che_mask = provider._problem.get_costs(
        input_, rec, provider._dic, provider._cl)
    print('initial che_mask sum/len = %d/%d  pre_length=%s'
          % (int(che_mask.sum()), int(che_mask.numel()),
             pre_length.tolist()), flush=True)
    model = provider._model
    exchange = None
    with torch.no_grad():
        for step in range(100):
            info, pos = _emdedding(input_, rec)
            try:
                ll, now_length, rec, exchange, act, che_mask = model(
                    input_, rec, info, pos, exchange, che_mask,
                    provider._dic, provider._cl, action=None, test=False)
                if step % 10 == 0:
                    print('step %d: ll=%s len=%s che_mask sum=%d/%d'
                          % (step, ll.tolist(), now_length.tolist(),
                             int(che_mask.sum()), int(che_mask.numel())),
                          flush=True)
            except Exception as exc:  # noqa: BLE001
                print('step %d CRASH: %s' % (step, exc), flush=True)
                # 定位：one_attn 输出
                h_em = model.embedder.layers(model.init_embed(info) + pos)
                fusion = model.embedder.project_node(h_em) + \
                    model.embedder.project_graph(h_em.max(1)[0])[:, None, :]
                att, att_s = model.embedder.one_attn(fusion, exchange, che_mask)
                print('att finite=%s  att_s finite=%s  att_s nan=%s'
                      % (bool(torch.isfinite(att).all()),
                         bool(torch.isfinite(att_s).all()),
                         bool(torch.isnan(att_s).any())), flush=True)
                row = att_s[0, 0]
                print('att_s row: min=%s max=%s neg=%s nan=%s'
                      % (row.min().item(), row.max().item(),
                         (row < 0).sum().item(), torch.isnan(row).sum().item()),
                      flush=True)
                print('che_mask sum=%d/%d exchange=%s'
                      % (int(che_mask.sum()), int(che_mask.numel()),
                         exchange.tolist() if exchange is not None else None),
                      flush=True)
                print('rec head: %s' % (rec[0][:20].tolist()), flush=True)
                break


if __name__ == '__main__':
    main()
