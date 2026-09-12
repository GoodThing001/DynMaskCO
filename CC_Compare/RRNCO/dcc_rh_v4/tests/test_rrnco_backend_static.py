"""test_rrnco_backend_static.py — rrnco_backend 静态回归（Stage B.0 热修）。

纯 Python 部分本地必跑；torch/tensordict 部分在本地无 torch 时跳过，服务器 B2 实测。
覆盖：import 隔离、注入语义（ready_time 不加 anchor 服务）、抽样 seed 确定性、
_PoolStartNodes 校验/dtype/device、num_loc=100/normalize=True 冻结、N<25 抽样规则、
demand pre-reset n-1。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rrnco_backend as rb  # 顶层 import 不应触发 torch/tensordict/rl4co


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _backend(seed=0):
    f = tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False)
    f.write(b'x')
    f.close()
    return rb.RRNCOBackend(f.name, capacity=50.0, device='cpu', seed=seed)


def _sp(pool, anchor=0, ready_time=0.0, load=0.0,
        demands=None, service=None, depot_tw_end=100.0):
    from subproblem import build_subproblem
    from testutil import make_view
    coords = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1)]
    n = len(coords)
    demands = [0.0] * n if demands is None else demands
    service = [0.0] * n if service is None else service
    view = make_view(coords, demands, [0.0] * n, [100.0] * n, service,
                     [(1, anchor, ready_time, load, 'ready', ())],
                     [1], 50.0, depot_tw_end, pool, True)
    return build_subproblem(view, view.vehicles[0])


# ---------------------------------------------------------------------------
# 纯 Python（本地必跑）
# ---------------------------------------------------------------------------

def test_import_isolation():
    for mod in ('torch', 'tensordict', 'rl4co'):
        assert mod not in sys.modules, f'rrnco_backend 顶层 import 触发了 {mod}'
    print('  import 隔离：无 torch/tensordict/rl4co')


def test_compute_injection_no_extra_anchor_service():
    """ready_time 已是可出发时刻：current_time = ready_time*s，不再加 anchor 服务。"""
    sp = _sp(pool=[2, 3], anchor=1, ready_time=10.0, load=5.0,
             demands=[0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
             service=[0.0, 3.0, 0.0, 0.0, 0.0, 0.0], depot_tw_end=24.0)
    time_scaled, load_norm, visited = rb.compute_injection(sp, 50.0)
    s = rb.T_MAX / 24.0
    assert abs(time_scaled - 10.0 * s) < 1e-9, time_scaled
    assert abs(time_scaled - 13.0 * s) > 1e-9, '不应含 anchor service time'
    assert abs(load_norm - 5.0 / 50.0) < 1e-9, load_norm
    assert abs(load_norm - 7.0 / 50.0) > 1e-9, '不应含 anchor demand'
    assert visited == [sp.anchor_idx]
    print('  注入语义：ready_time 不加 anchor 服务 / load 不加 anchor 需求')


def test_compute_injection_depot_anchor():
    sp = _sp(pool=[1, 2, 3], anchor=0, ready_time=0.0, load=0.0)
    time_scaled, load_norm, visited = rb.compute_injection(sp, 50.0)
    assert time_scaled == 0.0
    assert load_norm == 0.0
    assert visited == [], visited
    print('  depot anchor：visited 为空、time/load=0')


def test_derive_sample_seed_deterministic():
    b = _backend(seed=42)
    sp_a = _sp(pool=[1, 2, 3])
    sp_b = _sp(pool=[1, 2, 3])   # 同 hash
    sp_c = _sp(pool=[1, 2])      # 不同 pool
    assert b._derive_sample_seed(sp_a) == b._derive_sample_seed(sp_b)
    assert b._derive_sample_seed(sp_a) != b._derive_sample_seed(sp_c)
    assert 0 <= b._derive_sample_seed(sp_a) < 2 ** 31
    print('  抽样 seed：(seed, subproblem hash) 派生、确定性')


def test_pool_local_ids_exclude_depot_and_anchor():
    b = _backend()
    sp = _sp(pool=[2, 3], anchor=1)   # node_ids=(0,1,2,3)
    assert b._pool_local_ids(sp) == [2, 3]
    print('  starts 只含 pool：排除 depot(0) 与 anchor(1)')


def test_pool_start_nodes_validation():
    pool = rb._PoolStartNodes()
    try:
        pool(None, 3)
        raise AssertionError('pool_local_ids=None 应抛 RuntimeError')
    except RuntimeError:
        pass
    pool.pool_local_ids = [1, 2, 3]
    try:
        pool(None, 2)
        raise AssertionError('num_starts 不匹配应抛 ValueError')
    except ValueError:
        pass
    assert pool.get_num_starts(None) == 3
    print('  _PoolStartNodes 校验：None/mismatch 拒绝、get_num_starts 正确')


def test_env_config_frozen():
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'rrnco_backend.py')
    src = open(src_path, encoding='utf-8').read()
    assert "'num_loc': 100" in src, 'num_loc 必须冻结为 100'
    assert 'normalize=True' in src, 'normalize 必须冻结为 True'
    assert 'visible_prob_sampling_v1' in src, '抽样策略必须冻结为 visible_prob_sampling_v1'
    print('  env 配置冻结：num_loc=100 / normalize=True / visible_prob_sampling_v1')


# ---------------------------------------------------------------------------
# torch 门控（本地无 torch 跳过，服务器 B2 实测）
# ---------------------------------------------------------------------------

def test_pool_start_nodes_select_dtype_device():
    torch = _torch()
    if torch is None:
        print('  skip (no torch)'); return
    from types import SimpleNamespace
    fake_node = SimpleNamespace(dtype=torch.int64)
    fake_td = SimpleNamespace(device='cpu')
    fake_td.__getitem__ = lambda self, k: fake_node if k == 'current_node' else (_ for _ in ()).throw(KeyError(k))
    pool = rb._PoolStartNodes()
    pool.pool_local_ids = [1, 2, 3]
    out = pool(fake_td, 3, None)
    assert out.dtype == torch.int64, out.dtype
    assert out.device.type == 'cpu', out.device
    assert out.tolist() == [1, 2, 3]
    print('  _select dtype/device 正确（torch.tensor，非 td.new_tensor）')


def test_visible_prob_sampling_rules():
    torch = _torch()
    if torch is None:
        print('  skip (no torch)'); return
    sample_size = 25
    for N in (2, 5, 24, 25, 50):
        torch.manual_seed(0)
        d = torch.rand(2, N, N, dtype=torch.float32)
        out = rb._visible_prob_sample_indices(d, N, sample_size, seed=123)
        assert out.shape == (2, N, sample_size), (N, out.shape)
        assert out.min().item() >= 0 and out.max().item() < N
        if N >= sample_size:  # replacement=False：每行无重复
            for b in range(2):
                for i in range(N):
                    assert len(set(out[b, i].tolist())) == sample_size, (N, b, i)
        print(f'    N={N} 抽样 shape/范围/无重复 通过')
    print('  visible_prob_sampling_v1 N=2/5/24/25/50 通过')


def test_visible_prob_sampling_determinism():
    torch = _torch()
    if torch is None:
        print('  skip (no torch)'); return
    torch.manual_seed(0)
    d = torch.rand(1, 5, 5, dtype=torch.float32)
    a = rb._visible_prob_sample_indices(d, 5, 25, seed=7)
    b = rb._visible_prob_sample_indices(d, 5, 25, seed=7)
    c = rb._visible_prob_sample_indices(d, 5, 25, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    print('  抽样确定性：同 seed 同结果、异 seed 异结果')


def test_subproblem_to_td_demand_shape():
    torch = _torch()
    if torch is None:
        print('  skip (no torch)'); return
    try:
        import tensordict  # noqa: F401
    except ImportError:
        print('  skip (no tensordict)'); return
    b = _backend()
    for anchor, pool in ((0, [1, 2, 3]), (1, [2, 3])):
        sp = _sp(pool=pool, anchor=anchor)
        td = b._subproblem_to_td(sp)
        nn = len(sp.node_ids)
        assert td['locs'].shape[-2] == nn
        assert td['demand_linehaul'].shape[-1] == nn - 1
        assert td['demand_backhaul'].shape[-1] == nn - 1
    print('  demand pre-reset = n-1（locs n）')


def main():
    test_import_isolation()
    test_compute_injection_no_extra_anchor_service()
    test_compute_injection_depot_anchor()
    test_derive_sample_seed_deterministic()
    test_pool_local_ids_exclude_depot_and_anchor()
    test_pool_start_nodes_validation()
    test_env_config_frozen()
    test_pool_start_nodes_select_dtype_device()
    test_visible_prob_sampling_rules()
    test_visible_prob_sampling_determinism()
    test_subproblem_to_td_demand_shape()
    print('PASS test_rrnco_backend_static')


if __name__ == '__main__':
    main()
