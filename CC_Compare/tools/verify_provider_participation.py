"""A-v1 对比方法「模型是否真实参与决策」通用对照脚本（2026-10-13）。

对任意 ordering provider 跑 4 组对照 + 2 项守恒，输出 JSON + markdown 片段：

  C1 真实权重      provider.order(sp)                    → o_real
  C2 随机权重替换   把模型参数替换为随机值（保持各自 std 量级）→ o_rand
  C3 EDD          按 tw_end 升序                        → o_edd（下界/规则对照）
  C4 shuffle      固定种子洗牌                          → o_shuf（随机对照）
  B6 未来扰动       只改**不可见**节点（非 pool）字段 → 排序与可见 payload hash 是否不变
  B3 确定性         **显式清缓存**后连续两次 order() → 是否逐位一致

设计要点（针对已知陷阱）：
- **缓存**：所有 provider 都有内容寻址缓存（`_cache`）；本脚本在每次"需要真解码"的调用前
  显式 `_cache.clear()`，否则两次调用相等只能证明缓存命中（MVMoE 已记录该陷阱）。
- **C2 不假装**：找不到 torch 模型时明确写 `"not_supported"`，不伪造"已做权重替换"。
- **只读**：不修改任何 provider / scripts 文件；provider 通过 `ordering_providers.build_ordering_provider`
  正常构造，与 40 天正式臂同一条加载路径。

用法（服务器，`cc_compare` env，CPU）：
    cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem
    /home/hzeng/envs/cc_compare/bin/python \
        ../CC_Compare/tools/verify_provider_participation.py \
        --provider pomo --ckpt ../CC_Compare/POMO/NEW_py_ver/CVRP/POMO/result/saved_CVRP100_model/checkpoint-30500.pt \
        --outdir ../CC_Compare/results/a1_compare_table/participation
"""
import argparse
import hashlib
import json
import os
import sys
import time

# --------------------------------------------------------------------------- #
# 负载纪律（必须在 import numpy/torch 之前设置）：
# F1 正式批次正在 1003 上跑（9 workers，每决策 10s 硬时限）——本脚本 MUST NOT 抢核。
# 默认单线程；未显式给 --threads 时不允许 provider 自己放开线程池。
# --------------------------------------------------------------------------- #
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
_EXT = os.path.join(_ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')
for _p in (os.path.join(_EXT, 'scripts'),
           os.path.join(_EXT, 'scripts', 'evaluation'),
           os.path.join(_EXT, 'scripts', 'simulation'),
           os.path.join(_EXT, 'scripts', 'coldchain')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --------------------------------------------------------------------------- #
# mock 子问题：字段与 CC_Compare/RRNCO/dcc_rh_v4/subproblem.py::SubProblem 逐字段一致
# --------------------------------------------------------------------------- #

SP_FIELDS = ('vehicle_id', 'anchor_node_id', 'anchor_idx', 'ready_time',
             'current_load', 'pool_customer_ids', 'node_ids', 'coords', 'demands',
             'tw_start', 'tw_end', 'service_time', 'dist_mat', 'travel_mat',
             'capacity', 'depot_tw_end')


def make_sp(node_ids_all, coords, demands, tw_end, pool, capacity=50.0,
            depot_tw_end=22.0, anchor=0, vehicle_id=1, ready_time=0.0,
            current_load=0.0):
    """构造与 SubProblem 同字段的 mock 子问题（任意真实客户 id）。

    node_ids_all/coords/demands/tw_end 是**完整**数组（含 pool 之外的"未揭示/不可见"节点），
    子问题只摘取 depot + anchor + pool 三部分 —— 因此 B6 可以改不可见节点来测隔离性。
    """
    ids = [0]
    if int(anchor) != 0 and int(anchor) in set(int(x) for x in node_ids_all):
        ids.append(int(anchor))
    for c in sorted(int(c) for c in pool):
        if c not in ids:
            ids.append(c)
    pos = {int(n): i for i, n in enumerate(node_ids_all)}

    def take(arr):
        return tuple(float(arr[pos[n]]) for n in ids)

    n = len(ids)
    dist = [[float(np.hypot(coords[pos[a]][0] - coords[pos[b]][0],
                            coords[pos[a]][1] - coords[pos[b]][1]))
             for b in ids] for a in ids]
    sp = SimpleNamespace(
        vehicle_id=int(vehicle_id),
        anchor_node_id=int(anchor),
        anchor_idx=ids.index(int(anchor)),
        ready_time=float(ready_time),
        current_load=float(current_load),
        pool_customer_ids=tuple(sorted(int(c) for c in pool)),
        node_ids=tuple(ids),
        coords=tuple(tuple(float(x) for x in coords[pos[c]]) for c in ids),
        demands=take(demands),
        tw_start=tuple(0.0 for _ in ids),
        tw_end=take(tw_end),
        service_time=tuple(0.0 for _ in ids),
        dist_mat=tuple(tuple(r) for r in dist),
        travel_mat=tuple(tuple(r) for r in dist),
        capacity=float(capacity),
        depot_tw_end=float(depot_tw_end),
    )
    sp.node_index = lambda c: sp.node_ids.index(int(c))

    def _payload():
        return {
            'vehicle_id': int(sp.vehicle_id),
            'anchor_node_id': int(sp.anchor_node_id),
            'ready_time': float(sp.ready_time),
            'current_load': float(sp.current_load),
            'nodes': [int(x) for x in sp.node_ids],
            'coords': [[float(x) for x in c] for c in sp.coords],
            'demands': [float(d) for d in sp.demands],
            'tw_start': [float(t) for t in sp.tw_start],
            'tw_end': [float(t) for t in sp.tw_end],
            'service_time': [float(s) for s in sp.service_time],
            'dist_mat': [[float(x) for x in row] for row in sp.dist_mat],
            'travel_mat': [[float(x) for x in row] for row in sp.travel_mat],
            'capacity': float(sp.capacity),
            'depot_tw_end': float(sp.depot_tw_end),
        }

    sp.canonical_payload = _payload
    sp.canonical_hash = lambda: hashlib.sha256(
        json.dumps(_payload(), sort_keys=True).encode('utf-8')).hexdigest()
    return sp


def build_instance(n_pool=40, n_hidden=60, seed=7):
    """一个固定实例：depot + pool 客户 + 一批**不可见**节点（用于 B6）。"""
    rng = np.random.default_rng(seed)
    pool = [100 + i * 2 for i in range(n_pool)]
    hidden = [5000 + i for i in range(n_hidden)]
    all_ids = [0] + pool + hidden
    n = len(all_ids)
    coords = np.vstack([[0.5, 0.5], rng.uniform(0.0, 1.0, (n - 1, 2))])
    demands = np.zeros(n)
    demands[1:1 + len(pool)] = rng.uniform(1.0, 3.0, len(pool))
    demands[1 + len(pool):] = rng.uniform(1.0, 3.0, n_hidden)
    tw_end = np.full(n, 22.0)
    tw_end[1:1 + len(pool)] = 4.0 + rng.uniform(0.0, 12.0, len(pool))
    tw_end[1 + len(pool):] = 4.0 + rng.uniform(0.0, 12.0, n_hidden)
    return dict(node_ids_all=all_ids, coords=coords, demands=demands,
                tw_end=tw_end, pool=pool, hidden=hidden)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #

def spearman(a, b):
    a = list(a)
    b = list(b)
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return None
    ra = {c: i for i, c in enumerate(a)}
    rb = {c: i for i, c in enumerate(b)}
    x = np.array([ra[c] for c in common], float)
    y = np.array([rb[c] for c in common], float)
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else None


def diff_positions(a, b):
    """同一集合的两个排序：位置不同的个数。"""
    n = min(len(a), len(b))
    return int(sum(1 for i in range(n) if a[i] != b[i]))


def first_k_overlap(a, b, k=12):
    sa, sb = set(list(a)[:k]), set(list(b)[:k])
    return int(len(sa & sb))


def edd_order(sp):
    idx = {int(c): i for i, c in enumerate(sp.node_ids)}
    return tuple(sorted((int(c) for c in sp.pool_customer_ids),
                        key=lambda c: (float(sp.tw_end[idx[c]]), c)))


def shuffle_order(pool, seed=0):
    rng = np.random.default_rng(seed)
    p = [int(c) for c in pool]
    rng.shuffle(p)
    return tuple(p)


def clear_cache(provider):
    """显式清空内容寻址缓存（返回被清键数；无缓存则 None）。"""
    n = None
    for name in ('_cache', 'cache', '_memo', '_order_cache', '_ordering_cache'):
        v = getattr(provider, name, None)
        if isinstance(v, dict):
            n = (n or 0) + len(v)
            v.clear()
    return n


def find_torch_models(provider, max_depth=2):
    """返回 (属性路径, torch.nn.Module) 列表。

    支持「wrapper 持 module」的写法（如 MAPT：`provider._policy.transformer`
    才是 nn.Module，`_policy` 本身只是普通对象）。
    """
    try:
        import torch
    except Exception:                                   # noqa: BLE001
        return []
    found = []
    seen = set()

    def visit(obj, path, depth):
        if len(found) > 8 or depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            found.append((path, obj))
            return
        if depth == max_depth:
            return
        try:
            items = list(vars(obj).items())
        except Exception:                               # noqa: BLE001
            return
        for name, v in items:
            if name.startswith('__') or isinstance(
                    v, (int, float, str, bool, bytes, type(None),
                        list, tuple, dict, set)):
                continue
            visit(v, ('%s.%s' % (path, name)) if path else name, depth + 1)

    for name in ('model', '_model', '_net', 'net', '_policy', 'policy',
                 '_network', '_transformer'):
        v = getattr(provider, name, None)
        if v is not None:
            visit(v, name, 0)
    if not found:
        for name, v in list(vars(provider).items()):
            if not name.startswith('__'):
                visit(v, name, 0)
    return found


def randomize_weights(mod, seed=0):
    """把模块参数替换为随机值（保持各自 std 量级）。返回替换的元素数。"""
    import torch
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    n = 0
    with torch.no_grad():
        for p in mod.parameters():
            if not p.is_floating_point():
                continue
            std = float(p.detach().float().std().item())
            if not np.isfinite(std) or std <= 0:
                std = 0.05
            noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
            p.copy_(noise.to(device=p.device, dtype=p.dtype) * std)
            n += int(p.numel())
    return n


def timed_order(provider, sp, clear=True):
    if clear:
        clear_cache(provider)
    t0 = time.perf_counter()
    try:
        order = tuple(int(x) for x in provider.order(sp))
        err = None
    except Exception as e:                      # noqa: BLE001
        order, err = None, '%s: %s' % (type(e).__name__, e)
    return order, time.perf_counter() - t0, err


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def probe_provider(provider, args):
    """对一个**已构造好**的 provider 跑四对照 + 两守恒，返回证据 dict。

    2026-10-13 从 `run()` 抽出（纯提取，行为不变），使构造方式不同的 provider
    （例如 RRNCO 原生面：`RRNCOPreferenceProvider(BackendConfig(...))`）能复用同一
    实测口径 —— 见 `verify_rrnco_c2.py`。
    """
    t0 = time.perf_counter()
    # 2026-10-14 修正：`COEnrichedMLProvider` **没有 `_load()`**（它在 `__init__` 里就完成加载），
    # 于是本 harness 一调用就 `AttributeError: no attribute '_load'` ⇒ 它是 13 个学习型方法里
    # **唯一没有 `participation/*.json`** 的那个，台账只能靠散文断言。这里改为「有 `_load` 才调」；
    # 对其余 provider（都有 `_load`）**行为逐字不变**，因此已冻结的 12 份证据不受影响。
    _load = getattr(provider, '_load', None)
    if callable(_load):
        _load()
    load_s = time.perf_counter() - t0
    # provider 可能在 _load 里自己放开线程池（AM 曾设 8）→ 统一压回
    try:
        import torch
        torch.set_num_threads(int(args.threads))
        torch.set_num_interop_threads(1)
    except Exception:                                   # noqa: BLE001
        pass
    t_start = time.perf_counter()

    inst = build_instance(n_pool=args.pool, n_hidden=args.hidden)
    sp = make_sp(inst['node_ids_all'], inst['coords'], inst['demands'],
                 inst['tw_end'], inst['pool'], capacity=args.capacity,
                 depot_tw_end=22.0)
    res = {
        'provider': args.provider,
        'ckpt': args.ckpt,
        'device': args.device,
        'n_pool': len(inst['pool']),
        'n_hidden': len(inst['hidden']),
        'seed': 7,
        'load_seconds': load_s,
        'subproblem_fields': list(SP_FIELDS),
        'subproblem_hash': sp.canonical_hash(),
    }

    # ---- C1 真实权重（清缓存后计时） ----
    o_real, t_real, err = timed_order(provider, sp)
    res['C1_real'] = {'order': list(o_real) if o_real else None, 'seconds': t_real,
                      'error': err, 'covered': bool(o_real) and
                      sorted(o_real) == sorted(sp.pool_customer_ids)}

    # ---- B3 确定性（清缓存两次） ----
    o_a, _, e1 = timed_order(provider, sp)
    o_b, _, e2 = timed_order(provider, sp)
    res['B3_determinism'] = {
        'cleared_between_calls': True,
        'same': (o_a is not None and o_a == o_b),
        'order_a': list(o_a) if o_a else None,
        'order_b': list(o_b) if o_b else None,
        'errors': [e1, e2],
    }

    # ---- B6 未来扰动（只改不可见节点字段） ----
    coords2 = inst['coords'].copy()
    dem2 = inst['demands'].copy()
    tw2 = inst['tw_end'].copy()
    hpos = [inst['node_ids_all'].index(h) for h in inst['hidden']]
    rng = np.random.default_rng(1234)
    coords2[hpos] = rng.uniform(0.0, 1.0, (len(hpos), 2))
    dem2[hpos] = rng.uniform(1.0, 3.0, len(hpos))
    tw2[hpos] = 6.0 + rng.uniform(0.0, 8.0, len(hpos))
    sp2 = make_sp(inst['node_ids_all'], coords2, dem2, tw2, inst['pool'],
                  capacity=args.capacity, depot_tw_end=22.0)
    o_p, _, e3 = timed_order(provider, sp2)
    res['B6_future_perturbation'] = {
        'perturbed': 'hidden(non-pool) coords/demands/tw_end only',
        'subproblem_hash_same': sp2.canonical_hash() == sp.canonical_hash(),
        'ordering_same': (o_p is not None and o_real is not None and o_p == o_real),
        'order_perturbed': list(o_p) if o_p else None,
        'error': e3,
    }

    # ---- C3 EDD / C4 shuffle ----
    o_edd = edd_order(sp)
    o_shuf = shuffle_order(sp.pool_customer_ids, seed=0)
    res['C3_edd_order'] = list(o_edd)
    res['C4_shuffle_order'] = list(o_shuf)

    # ---- C2 随机权重替换 ----
    models = find_torch_models(provider)
    if not models:
        res['C2_random_weights'] = {
            'status': 'NOT_SUPPORTED',
            'reason': '未在 provider 上找到 torch.nn.Module（可能是非 torch 实现）；'
                      '没有做权重替换，不得据此主张已做对照。',
        }
        o_rand = None
    else:
        names = [n for n, _ in models]
        n_rep = 0
        for _, m in models:
            n_rep += randomize_weights(m, seed=args.rand_seed)
        o_rand, t_rand, e4 = timed_order(provider, sp)
        res['C2_random_weights'] = {
            'status': 'DONE',
            'modules': names,
            'replaced_params': n_rep,
            'order': list(o_rand) if o_rand else None,
            'seconds': t_rand,
            'error': e4,
            'covered': bool(o_rand) and sorted(o_rand) == sorted(sp.pool_customer_ids),
        }

    # ---- 对照指标 ----
    if o_real:
        res['metrics'] = {
            'diff_positions_real_vs_edd': diff_positions(o_real, o_edd),
            'spearman_real_edd': spearman(o_real, o_edd),
            'diff_positions_real_vs_shuffle': diff_positions(o_real, o_shuf),
            'spearman_real_shuffle': spearman(o_real, o_shuf),
        }
        if o_rand:
            res['metrics'].update({
                'diff_positions_real_vs_rand': diff_positions(o_real, o_rand),
                'spearman_real_rand': spearman(o_real, o_rand),
                'first12_overlap_real_rand': first_k_overlap(o_real, o_rand, 12),
                'set_equal_real_rand': sorted(o_real) == sorted(o_rand),
                'order_changed_by_weights': bool(o_real != o_rand),
            })
    res['calls'] = {k: getattr(provider, k) for k in
                    ('n_calls', 'n_solves', 'n_edd_fallback') if hasattr(provider, k)}
    return res


def run(args):
    """按注册表 / 显式类路径构造 provider，再交给 `probe_provider`。"""
    if args.provider_dir:
        # 直接导入（用于尚未接入 ordering_providers 注册表的方法，例如 mapt/coenrich；
        # 注册表在 F1 正式批次期间禁止改动）。加载路径与注册表完全一致：
        # 把 provider 目录插入 sys.path → importlib.import_module → getattr(cls)。
        import importlib
        d = os.path.normpath(os.path.join(_ROOT, *args.provider_dir.split('/')))
        if not os.path.isdir(d):
            raise SystemExit('provider dir missing: %s' % d)
        if d not in sys.path:
            sys.path.insert(0, d)
        mod = importlib.import_module(args.provider_module)
        cls = getattr(mod, args.provider_class)
        provider = cls(args.ckpt, device=args.device)
    else:
        from ordering_providers import build_ordering_provider
        provider = build_ordering_provider(args.provider, args.ckpt,
                                           device=args.device)
    return probe_provider(provider, args)


def md_fragment(res):
    m = res.get('metrics') or {}
    c2 = res.get('C2_random_weights') or {}
    L = ['### %s' % res['provider'], '',
         '- checkpoint: `%s`' % res['ckpt'],
         '- n_pool=%d（不可见节点 %d），load %.2fs，单次 order()（清缓存）%.3fs'
         % (res['n_pool'], res['n_hidden'], res['load_seconds'],
            (res['C1_real'] or {}).get('seconds', float('nan'))),
         '- C1 覆盖全池: %s' % (res['C1_real'] or {}).get('covered'),
         '- **C2 权重替换**: %s%s' % (
             c2.get('status'),
             ('（%s，替换 %s 个参数）' % (c2.get('modules'), c2.get('replaced_params')))
             if c2.get('status') == 'DONE' else ''),
         '- **C3 与 EDD 不同位置**: %s / %s，Spearman=%.3f'
         % (m.get('diff_positions_real_vs_edd'), res['n_pool'],
            m.get('spearman_real_edd') or float('nan')),
         '- **C2 与随机权重不同位置**: %s，Spearman(o_real,o_rand)=%s，first12 重合=%s，'
         '**order_changed_by_weights=%s**' % (
             m.get('diff_positions_real_vs_rand'), m.get('spearman_real_rand'),
             m.get('first12_overlap_real_rand'), m.get('order_changed_by_weights')),
         '- B6 未来扰动: ordering_same=%s, subproblem_hash_same=%s'
         % (res['B6_future_perturbation']['ordering_same'],
            res['B6_future_perturbation']['subproblem_hash_same']),
         '- B3 确定性（清缓存两次）: %s' % res['B3_determinism']['same'],
         '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--provider-dir', default=None,
                    help='相对仓库根的 provider 目录（用于未接入注册表的方法）')
    ap.add_argument('--provider-module', default=None)
    ap.add_argument('--provider-class', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--pool', type=int, default=40)
    ap.add_argument('--hidden', type=int, default=60)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--rand-seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1,
                    help='强制线程数（默认 1；负载纪律，见文件头）')
    ap.add_argument('--max-seconds', type=int, default=180,
                    help='单方法墙钟上限；超时 SIGALRM 中止（默认 180s）')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    import signal

    def _on_alarm(signum, frame):                       # noqa: ARG001
        raise TimeoutError('wall-clock budget %ss exceeded' % args.max_seconds)

    # 墙钟熔断：SIGALRM **只有 POSIX 有**（Windows 的 `signal` 没有该属性）。
    # 2026-10-13 实测踩到：在 3080（Windows）上直接 `signal.signal(signal.SIGALRM, …)`
    # 抛 `AttributeError: module 'signal' has no attribute 'SIGALRM'`，探针**两次尝试都秒退**、
    # 一个产物都没写（`RRNCO_C2_DONE … rc=1 ok=False`）。故这里按平台能力启用，
    # 并在没有熔断时**明确打印**，不假装有保护。
    _armed = hasattr(signal, 'SIGALRM')
    if _armed:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(args.max_seconds))
    else:
        print('NOTE: 本平台无 SIGALRM（Windows）⇒ 跳过墙钟熔断；'
              '本脚本只做常数次 order() 调用，请用外部超时（本仓库用 ps1/pwsh 守候）兜底。')
    try:
        res = run(args)
    except TimeoutError as e:
        print('ABORTED: %s —— 未写产物（避免把半成品当证据）' % e)
        return 3
    finally:
        if _armed:
            signal.alarm(0)
    os.makedirs(args.outdir, exist_ok=True)
    jp = os.path.join(args.outdir, '%s.json' % args.provider)
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    mp = os.path.join(args.outdir, '%s.md' % args.provider)
    with open(mp, 'w', encoding='utf-8') as f:
        f.write(md_fragment(res))
    print(md_fragment(res))
    print('wrote', jp)


if __name__ == '__main__':
    main()
