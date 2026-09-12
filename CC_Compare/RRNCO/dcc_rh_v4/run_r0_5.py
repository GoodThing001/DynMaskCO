"""run_r0_5.py — R0.5 canonical runner（Gates B3-B7，产出 R0_5_RESULT.json）。

统一入口：加载真实 epoch_199.ckpt 后端，按固定顺序执行
  B3 确定性解码（同子问题两次 order 决策一致）
  B4 三快照（初始部分揭示 / 行程中间 anchor+load / 多车冻结前缀，真实 provider + 协调，零 fallback）
  B5 可变规模（pool 1/2/5/10/25/50 + 空 pool + anchor-only）
  B6 未来扰动（改未来坐标/需求/数量，可见子问题 ordering 不变）
  B7 模型贡献（真实 ckpt vs EDD/nearest/shuffle 至少一处改变决策）

运行（服务器，cc_compare env，tmux）：
    cd /home/hzeng/project/MASKCO-Main
    CUDA_VISIBLE_DEVICES=0 /home/hzeng/envs/cc_compare/bin/python -u \
        CC_Compare/RRNCO/dcc_rh_v4/run_r0_5.py \
        --checkpoint CC_Compare/RRNCO/checkpoints/rcvrptw/epoch_199.ckpt \
        --device cuda:0 --seed 0 \
        --output CC_Compare/RRNCO/dcc_rh_v4/results/r0_5_run_a.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

_DCC = os.path.dirname(os.path.abspath(__file__))
_RRNCO_ROOT = os.path.dirname(_DCC)
sys.path.insert(0, _DCC)
sys.path.insert(0, _RRNCO_ROOT)

import rrnco_backend as rb
from subproblem import build_subproblem
from preference import MockPreferenceProvider, validate_ordering
from coordinator import coordinate
from pickup_certificate import certify_suffix
from testutil import make_view

GATES = {}


def gate(name):
    def deco(fn):
        GATES[name] = fn
        return fn
    return deco


def _make_backend(ckpt, device, seed):
    return rb.RRNCOPreferenceProvider(
        rb.BackendConfig(checkpoint_path=ckpt, device=device, seed=seed))


def _ordering_map(backend, view):
    """对每个 replan 车计算真实 ordering。"""
    orderings = {}
    for v in view.vehicles:
        vid = int(v.vehicle_id)
        if vid in set(int(x) for x in view.replan_ids):
            sp = build_subproblem(view, v)
            orderings[vid] = tuple(int(c) for c in backend.order(sp))
    return orderings


def _certify_all(view, suffixes):
    vehicle_views = {int(v.vehicle_id): v for v in view.vehicles}
    for vid, suffix in suffixes.items():
        ok, reason = certify_suffix(view, vehicle_views[vid], suffix)
        if not ok:
            return False, f'certify {vid}: {reason}'
    return True, None


# ---------------------------------------------------------------------------
# B3 确定性
# ---------------------------------------------------------------------------

def _b3_subproblem():
    coords = [(0, 0), (2, 0), (4, 1), (6, 0), (8, 1)]
    n = len(coords)
    view = make_view(coords, [0, 2, 3, 4, 5], [0] * n, [24] * n, [0.5] * n,
                     [(1, 1, 10.0, 5.0, 'ready', ())], [1], 50.0, 24.0,
                     [2, 3, 4], True)
    return build_subproblem(view, view.vehicles[0])


@gate('B3_determinism')
def b3(backend, _):
    sp = _b3_subproblem()
    import torch
    with torch.inference_mode():
        backend.order(sp)
        a = backend.last_audit
        backend.order(sp)
        b = backend.last_audit
    checks = {
        'ordering': a.ordering == b.ordering,
        'selected_start': a.selected_start == b.selected_start,
        'raw_actions': a.raw_actions == b.raw_actions,
        'injected_state_hash': a.injected_state_hash == b.injected_state_hash,
        'tensor_hash': a.tensor_hash == b.tensor_hash,
        'sampling_seed': a.sampling_seed == b.sampling_seed,
    }
    return all(checks.values()), checks


# ---------------------------------------------------------------------------
# B4 三快照
# ---------------------------------------------------------------------------

@gate('B4_snapshots')
def b4(backend, _):
    import torch
    results = {}
    all_ok = True

    def run(name, coords, demands, tw_end, service, vehicle_specs, replan_ids,
            capacity, depot_tw_end, pool):
        nonlocal all_ok
        n = len(coords)
        view = make_view(coords, demands, [0] * n, tw_end, service, vehicle_specs,
                         replan_ids, capacity, depot_tw_end, pool, True)
        orderings = {}
        for v in view.vehicles:
            vid = int(v.vehicle_id)
            if vid in set(int(x) for x in replan_ids):
                sp = build_subproblem(view, v)
                with torch.inference_mode():
                    orderings[vid] = tuple(int(c) for c in backend.order(sp))
        # 完整覆盖 pool（每车）
        poolset = set(pool)
        for vid, ord_ in orderings.items():
            if set(ord_) != poolset:
                all_ok = False
                results[name] = f'ordering 不覆盖 pool: {vid} {ord_}'
                return
        r = coordinate(view, orderings)
        cert_ok, reason = _certify_all(view, r.suffixes)
        if r.fallback_reason is not None:
            all_ok = False
            results[name] = f'fallback: {r.fallback_reason}'
        elif not cert_ok:
            all_ok = False
            results[name] = f'certify fail: {reason}'
        else:
            results[name] = {'suffixes': {str(k): list(v) for k, v in r.suffixes.items()},
                             'deferred': list(r.deferred), 'fallback': None}

    run('snapshot1_initial_partial', [(0, 0), (1, 0), (2, 0), (3, 0), (40, 0), (41, 0)],
        [0, 3, 4, 5, 3, 4], [100] * 6, [0] * 6, [(1, 0, 0.0, 0.0, 'ready', ())],
        [1], 10.0, 100.0, [1, 2, 3])
    run('snapshot2_mid_trip', [(0, 0), (5, 0), (2, 0), (8, 0)],
        [0, 3, 4, 5], [100] * 4, [0] * 4, [(1, 2, 3.0, 5.0, 'ready', ())],
        [1], 10.0, 100.0, [1, 3])
    run('snapshot3_frozen_prefix', [(0, 0), (2, 0), (5, 0), (8, 0), (10, 0)],
        [0, 3, 4, 5, 6], [100] * 5, [0] * 5,
        [(1, 3, 5.0, 5.0, 'committed', (3,)), (2, 0, 0.0, 0.0, 'ready', ())],
        [2], 10.0, 100.0, [2, 4])
    return all_ok, results


# ---------------------------------------------------------------------------
# B5 可变规模
# ---------------------------------------------------------------------------

@gate('B5_variable_size')
def b5(backend, _):
    import torch
    ok = True
    details = {}
    for k in (1, 2, 5, 10, 25, 50):
        # depot(0) + anchor(1) + k 个 pool 客户；depot_tw_end=1000 保证返仓均可行
        n_total = k + 2
        coords = [(0.0, 0.0), (1.0, 0.0)] + [(2.0 + i, 0.0) for i in range(k)]
        demands = [0.0, 0.0] + [1.0] * k
        service = [0.0, 0.0] + [0.1] * k
        view = make_view(coords, demands, [0.0] * n_total, [1000.0] * n_total,
                         service, [(1, 1, 1.0, 0.0, 'ready', ())], [1],
                         50.0, 1000.0, list(range(2, k + 2)), True)
        sp = build_subproblem(view, view.vehicles[0])
        try:
            with torch.inference_mode():
                ordering = tuple(backend.order(sp))
            poolset = set(range(2, k + 2))
            good = (set(ordering) == poolset)
            details[str(k)] = {'ordering_len': len(ordering), 'complete': good,
                               'replacement': backend.last_audit.sampling_replacement}
            if not good:
                ok = False
        except Exception as exc:  # noqa: BLE001
            ok = False
            details[str(k)] = f'EXC {type(exc).__name__}: {exc}'

    # 空 pool / anchor-only
    for name, pool in (('empty_pool', []), ('anchor_only', [])):
        n_total = 3
        view = make_view([(0, 0), (1, 0), (2, 0)], [0, 0, 0], [0] * n_total,
                         [24] * n_total, [0] * n_total,
                         [(1, 1, 1.0, 0.0, 'ready', ())], [1],
                         50.0, 24.0, pool, True)
        sp = build_subproblem(view, view.vehicles[0])
        with torch.inference_mode():
            ordering = tuple(backend.order(sp))
        details[name] = {'ordering': list(ordering)}
        if ordering != ():
            ok = False
    return ok, details


# ---------------------------------------------------------------------------
# B6 未来扰动
# ---------------------------------------------------------------------------

@gate('B6_future_perturbation')
def b6(backend, _):
    import torch
    ok = True
    details = {}

    def make(future_coords, future_demands):
        # 可见：depot(0) + anchor(1) + pool(2,3)；未来：追加节点
        coords = [(0, 0), (1, 0), (2, 0), (3, 0)] + future_coords
        n = len(coords)
        demands = [0, 0, 2, 3] + future_demands
        return make_view(coords, demands, [0] * n, [24] * n, [0.1] * n,
                         [(1, 1, 1.0, 0.0, 'ready', ())], [1],
                         50.0, 24.0, [2, 3], True)

    view_base = make([(40, 40), (41, 41)], [3, 4])
    sp_base = build_subproblem(view_base, view_base.vehicles[0])
    with torch.inference_mode():
        backend.order(sp_base)
    base_audit = backend.last_audit

    variants = [
        ('coord_change', make([(30, 30), (41, 41)], [3, 4])),
        ('demand_change', make([(40, 40), (41, 41)], [9, 8])),
        ('count_change', make([(40, 40)], [3])),
    ]
    for label, view2 in variants:
        sp2 = build_subproblem(view2, view2.vehicles[0])
        with torch.inference_mode():
            backend.order(sp2)
        a2 = backend.last_audit
        same = (a2.subproblem_hash == base_audit.subproblem_hash
                and a2.tensor_hash == base_audit.tensor_hash
                and a2.injected_state_hash == base_audit.injected_state_hash
                and a2.ordering == base_audit.ordering)
        details[label] = {'subproblem_hash_same': a2.subproblem_hash == base_audit.subproblem_hash,
                          'ordering_same': a2.ordering == base_audit.ordering}
        if not same:
            ok = False
    return ok, details


# ---------------------------------------------------------------------------
# B7 模型贡献
# ---------------------------------------------------------------------------

@gate('B7_model_contribution')
def b7(backend, _):
    import torch
    coords = [(0, 0), (1, 0), (5, 0), (6, 0), (7, 0), (8, 0)]
    n = len(coords)
    view = make_view(coords, [0, 3, 2, 4, 5, 1], [0] * n, [24] * n, [0.2] * n,
                     [(1, 1, 1.0, 0.0, 'ready', ())], [1], 50.0, 24.0,
                     [2, 3, 4, 5], True)
    sp = build_subproblem(view, view.vehicles[0])
    with torch.inference_mode():
        real_ord = tuple(int(c) for c in backend.order(sp))

    controls = {}
    for mode in ('edd', 'nearest', 'shuffle'):
        controls[mode] = tuple(MockPreferenceProvider(mode, seed=7).order(sp))

    # 至少一个控制组与真实排序不同
    differs = {m: controls[m] != real_ord for m in controls}
    some_differs = any(differs.values())

    # 真实排序 vs EDD 在 coordinator 上是否改变决策
    def decide(orderings):
        return coordinate(view, {1: orderings})

    real_r = decide(real_ord)
    edd_r = decide(controls['edd'])
    decision_changed = (real_r.suffixes != edd_r.suffixes
                        or real_r.deferred != edd_r.deferred)

    details = {'real_ordering': list(real_ord),
               'controls': {m: list(o) for m, o in controls.items()},
               'differs': differs,
               'decision_changed_vs_edd': decision_changed}
    return some_differs and decision_changed, details


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_server_env():
    p = os.path.join(_DCC, 'SERVER_ENVIRONMENT.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', required=True)
    ap.add_argument('--gates', default='B3_determinism,B4_snapshots,B5_variable_size,'
                   'B6_future_perturbation,B7_model_contribution')
    args = ap.parse_args()

    # 前置：SERVER_ENVIRONMENT.json + checkpoint hash
    senv = _load_server_env()
    backend = _make_backend(args.checkpoint, args.device, args.seed)
    pre = {'server_env_loaded': senv is not None,
           'checkpoint_sha256': backend.ckpt_sha256}
    if senv is not None:
        pre['server_env_ckpt_sha256'] = senv.get('checkpoint', {}).get('sha256')
        pre['ckpt_hash_match'] = (senv.get('checkpoint', {}).get('sha256')
                                  == backend.ckpt_sha256)

    gate_list = args.gates.split(',')
    results = {'preflight': pre, 'gates': {}, 'verdict': 'PENDING'}
    all_pass = True
    for g in gate_list:
        if g not in GATES:
            results['gates'][g] = {'pass': False, 'detail': 'unknown gate'}
            all_pass = False
            continue
        t0 = time.time()
        print(f'[{g}] running...', flush=True)
        try:
            ok, detail = GATES[g](backend, None)
            results['gates'][g] = {'pass': ok, 'detail': detail,
                                   'runtime_s': time.time() - t0}
            print(f'[{g}] {"PASS" if ok else "FAIL"} ({time.time()-t0:.1f}s)',
                  flush=True)
            if not ok:
                all_pass = False
                break
        except Exception as exc:  # noqa: BLE001
            results['gates'][g] = {'pass': False,
                                   'detail': f'{type(exc).__name__}: {exc}',
                                   'runtime_s': time.time() - t0}
            print(f'[{g}] EXC {type(exc).__name__}: {exc}', flush=True)
            all_pass = False
            break

    results['verdict'] = 'PASS' if all_pass else 'FAIL'
    results['run_id'] = time.strftime('%Y%m%d-%H%M%S')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)
        f.write('\n')
    os.replace(tmp, args.output)
    print('R0_5_RESULT written:', args.output)
    print('verdict:', results['verdict'])
    for g, r in results['gates'].items():
        print(f'  {g}: {"PASS" if r["pass"] else "FAIL"}')


if __name__ == '__main__':
    main()
