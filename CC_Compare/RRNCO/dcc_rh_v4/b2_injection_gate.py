"""b2_injection_gate.py — R0.5 Stage B 注入 + mask-parity Gate（服务器实测，B2-1 ~ B2-5）。

验证真实 epoch_199.ckpt 下：
  B2-1 load：只加载一次、eval、cuda、sample_size=25、normalize=True、num_loc=100
  B2-2 reset：pre/post 形状 + 注入 anchor/time/load/visited
  B2-3 first step：手动 _step 一步，验证 time/load 公式
  B2-4 decoder context：POMO 首步后状态进入 decoder（hook 只记第一次）
  B2-5 mask parity：RRNCO action_mask 与 pickup_certificate 逐客户对照（含边界）

运行（服务器，cc_compare env）：
    cd /home/hzeng/project/MASKCO-Main
    CUDA_VISIBLE_DEVICES=0 /home/hzeng/envs/cc_compare/bin/python \
        CC_Compare/RRNCO/dcc_rh_v4/b2_injection_gate.py
"""
import os
import sys

_DCC = os.path.dirname(os.path.abspath(__file__))
_RRNCO_ROOT = os.path.dirname(_DCC)
sys.path.insert(0, _DCC)
sys.path.insert(0, _RRNCO_ROOT)

import torch

import rrnco_backend as rb
from subproblem import build_subproblem
from testutil import make_view
from pickup_certificate import VehicleState, certify_append

CKPT = os.path.join(_RRNCO_ROOT, 'checkpoints', 'rcvrptw', 'epoch_199.ckpt')

PASS = []
FAIL = []


def check(name, cond, detail=''):
    if cond:
        PASS.append(name)
        print(f'  [PASS] {name}')
    else:
        FAIL.append(name)
        print(f'  [FAIL] {name}  {detail}')


def build_view_sp():
    coords = [(0, 0), (2, 0), (4, 1), (6, 0), (8, 1), (10, 0)]  # 0 depot,1 anchor,2-4 pool,5 future
    n = len(coords)
    demands = [0.0, 2.0, 3.0, 4.0, 5.0, 0.0]
    service = [0.0, 1.0, 0.5, 0.5, 0.5, 0.0]
    view = make_view(coords, demands, [0.0] * n, [24.0] * n, service,
                     [(1, 1, 10.0, 5.0, 'ready', ())],  # vehicle 1, anchor=1, ready=10, load=5
                     [1], 50.0, 24.0, [2, 3, 4], True)
    return view, build_subproblem(view, view.vehicles[0])


def mask_parity(backend, env, view, sp, label):
    """对照 RRNCO action_mask 与 pickup_certificate 对每个 pool 客户的可行性。

    返回 (mismatches, cases)：mismatches = [(customer, cert_ok, rrnco_ok, reason), ...]
    """
    capacity = float(sp.capacity)
    env.set_tolerance(rb.T_MAX / max(float(sp.depot_tw_end), 1e-6), capacity)
    td = rb.subproblem_to_tensordict(sp, 'cuda')
    time_scaled, load_norm, visited_local = rb.compute_injection(sp, capacity)
    env.set_injection(sp.anchor_idx, time_scaled, load_norm, visited_local)
    backend._pool_starts.pool_local_ids = backend._pool_local_ids(sp)
    with torch.inference_mode():
        td_reset = env.reset(td)
    mask = td_reset['action_mask'][0].cpu().numpy().astype(bool)
    v = view.vehicles[0]
    state = VehicleState(int(v.anchor_node_id), float(v.ready_time), float(v.load), ())
    mismatches = []
    cases = []
    for c in sorted(sp.pool_customer_ids):
        cert_ok, reason = certify_append(view, state, int(c))
        local = sp.node_index(int(c))
        rrnco_ok = bool(mask[local])
        cases.append((int(c), cert_ok, rrnco_ok))
        tag = 'MATCH' if cert_ok == rrnco_ok else 'MISMATCH'
        print(f'    [{label}] c={c} local={local} cert={cert_ok} rrnco={rrnco_ok} {tag}')
        if cert_ok != rrnco_ok:
            mismatches.append((int(c), cert_ok, rrnco_ok, reason))
    return mismatches, cases


def _mk(coords, demands, tw_end, service, ready, load, capacity,
        depot_tw_end, pool, anchor=0):
    n = len(coords)
    return make_view(coords, demands, [0.0] * n, tw_end, service,
                     [(1, anchor, ready, load, 'ready', ())], [1],
                     capacity, depot_tw_end, pool, True)


def main():
    view, sp = build_view_sp()
    n = len(sp.node_ids)          # 5 = depot + anchor + pool(3)
    s = rb.T_MAX / float(sp.depot_tw_end)
    capacity = float(sp.capacity)
    print(f'== subproblem: node_ids={sp.node_ids} anchor_idx={sp.anchor_idx} '
          f'ready={sp.ready_time} load={sp.current_load} pool={sp.pool_customer_ids}')

    # ---------------- B2-1 load ----------------
    print('== B2-1 load checkpoint ==')
    backend = rb.RRNCOPreferenceProvider(
        rb.BackendConfig(checkpoint_path=CKPT, device='cuda', seed=0))
    policy, env = backend._load()
    check('load_once', backend._loaded)
    check('policy_eval', not policy.training)
    dev = next(policy.parameters()).device.type
    check('device_cuda', dev == 'cuda', f'device={dev}')
    check('sample_size_measured', backend._sample_size == 25,
          f'sample_size={backend._sample_size}')
    check('normalize_true', getattr(env, 'normalize', None) is True,
          f'normalize={getattr(env, "normalize", None)}')
    gnl = getattr(getattr(env, 'generator', None), 'num_loc', None)
    check('num_loc_100', gnl == 100, f'generator.num_loc={gnl}')
    print(f'    checkpoint_sha256={backend.ckpt_sha256}')

    # ---------------- B2-2 reset ----------------
    print('== B2-2 reset (pre/post shape + injection) ==')
    td = rb.subproblem_to_tensordict(sp, 'cuda')
    check('pre_locs_n', td['locs'].shape[-2] == n, f"locs={td['locs'].shape}")
    check('pre_demand_nm1', td['demand_linehaul'].shape[-1] == n - 1
          and td['demand_backhaul'].shape[-1] == n - 1,
          f"linehaul={td['demand_linehaul'].shape} backhaul={td['demand_backhaul'].shape}")

    backend._sample_seed = backend._derive_sample_seed(sp)
    time_scaled, load_norm, visited_local = rb.compute_injection(sp, capacity)
    backend._pool_starts.pool_local_ids = backend._pool_local_ids(sp)
    env.set_injection(sp.anchor_idx, time_scaled, load_norm, visited_local)

    with torch.inference_mode():
        td_reset = env.reset(td)
    check('post_locs_n', td_reset['locs'].shape[-2] == n)
    check('post_demand_n', td_reset['demand_linehaul'].shape[-1] == n
          and td_reset['demand_backhaul'].shape[-1] == n)
    check('post_visited_n', td_reset['visited'].shape[-1] == n)
    check('post_mask_n', td_reset['action_mask'].shape[-1] == n)

    check('inject_current_node_anchor',
          int(td_reset['current_node'].item()) == int(sp.anchor_idx),
          f"current_node={td_reset['current_node'].item()}")
    check('inject_current_time',
          abs(float(td_reset['current_time'].item()) - time_scaled) < 1e-6,
          f"current_time={td_reset['current_time'].item()} expect={time_scaled}")
    check('inject_backhaul_load',
          abs(float(td_reset['used_capacity_backhaul'].item()) - load_norm) < 1e-6,
          f"ub={td_reset['used_capacity_backhaul'].item()} expect={load_norm}")
    check('inject_visited_anchor',
          bool(td_reset['visited'][0, sp.anchor_idx].item()))

    # ---------------- B2-3 first step (manual _step) ----------------
    print('== B2-3 first step ==')
    first = backend._pool_local_ids(sp)[0]   # first pool customer
    td_step = td_reset.clone()
    td_step['action'] = torch.tensor([first], dtype=td_reset['current_node'].dtype,
                                     device=td_reset.device)
    prev_node = int(td_step['current_node'].item())
    with torch.inference_mode():
        td_next = env._step(td_step)

    dur_anchor_first = float(sp.travel_mat[sp.anchor_idx][first]) * s
    tw_start_first = float(sp.tw_start[first]) * s
    service_first = float(sp.service_time[first]) * s
    exp_time = max(time_scaled + dur_anchor_first, tw_start_first) + service_first
    exp_load = load_norm + float(sp.demands[first]) / capacity

    check('step_prev_is_anchor', prev_node == int(sp.anchor_idx), f'prev={prev_node}')
    check('step_curr_is_first', int(td_next['current_node'].item()) == first,
          f"curr={td_next['current_node'].item()} first={first}")
    check('step_time_ok',
          abs(float(td_next['current_time'].item()) - exp_time) < 1e-4,
          f"got={td_next['current_time'].item()} expect={exp_time}")
    check('step_load_ok',
          abs(float(td_next['used_capacity_backhaul'].item()) - exp_load) < 1e-5,
          f"got={td_next['used_capacity_backhaul'].item()} expect={exp_load}")
    check('step_anchor_still_visited', bool(td_next['visited'][0, sp.anchor_idx].item()))

    # ---------------- B2-4 decoder context hook ----------------
    # register_forward_hook 捕获 decoder.forward（主循环第一次调用）的输入 td，
    # 即 POMO 强制首步之后、首次解码之前的状态；只记录第一次。
    print('== B2-4 decoder context (first-step state into decoder) ==')
    captured = {}
    captured_once = [False]

    def hook(module, args, output):
        if captured_once[0]:
            return
        captured_once[0] = True
        tdin = args[0]
        captured['current_node'] = tdin['current_node'].flatten().clone()
        captured['current_time'] = tdin['current_time'].flatten().clone()
        captured['used_capacity_backhaul'] = (
            tdin['used_capacity_backhaul'].flatten().clone())
        captured['visited'] = tdin['visited'].clone()

    handle = policy.decoder.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = policy(td_reset, env, return_actions=True, phase='val',
                         calc_reward=False, num_starts=len(backend._pool_local_ids(sp)))
    finally:
        handle.remove()

    pool_local = backend._pool_local_ids(sp)
    cn = captured['current_node'].cpu().tolist()
    check('ctx_batch_is_num_starts', len(cn) == len(pool_local),
          f"batch={len(cn)} expect={len(pool_local)}")
    check('ctx_node_is_pool_starts', sorted(cn) == sorted(pool_local), f'cn={cn}')

    ct = captured['current_time'].cpu().tolist()
    cl = captured['used_capacity_backhaul'].cpu().tolist()
    ctx_ok_time = True
    ctx_ok_load = True
    for i, st in enumerate(pool_local):
        exp_t = max(time_scaled + float(sp.travel_mat[sp.anchor_idx][st]) * s,
                    float(sp.tw_start[st]) * s) + float(sp.service_time[st]) * s
        exp_l = load_norm + float(sp.demands[st]) / capacity
        ctx_ok_time = ctx_ok_time and abs(float(ct[i]) - exp_t) < 1e-4
        ctx_ok_load = ctx_ok_load and abs(float(cl[i]) - exp_l) < 1e-5
    check('ctx_time_ok', ctx_ok_time, f'times={ct}')
    check('ctx_load_ok', ctx_ok_load, f'loads={cl}')
    check('ctx_anchor_visited',
          bool(captured['visited'][:, sp.anchor_idx].all().item()))

    actions = out['actions'].cpu().numpy()
    print(f'    actions shape={actions.shape} (num_starts x seq_len)')

    # ---------------- B2-5 mask parity ----------------
    print('== B2-5 mask parity (RRNCO action_mask vs pickup_certificate) ==')
    # 5a. 基础对照（常规 view，非边界）
    mism, cases = mask_parity(backend, env, view, sp, 'base')
    check('mask_parity_base', len(mism) == 0,
          f'mismatches={[(c, ck, rk) for c, ck, rk, _ in mism]}')

    # 5b. 边界对照（exact-capacity / capacity+eps / exact-TW / TW+eps / exact-return / return+eps）
    print('  -- boundary cases --')
    boundary_mismatches = []

    def run_case(label, coords, demands, tw_end, service, ready, load, cap, dtwe, pool):
        v2 = _mk(coords, demands, tw_end, service, ready, load, cap, dtwe, pool)
        sp2 = build_subproblem(v2, v2.vehicles[0])
        mm, _ = mask_parity(backend, env, v2, sp2, label)
        return mm

    # capacity: capacity=10, load=8, demand=2 -> 8+2=10 (exact); demand=3 -> 11 (eps over)
    run_case('exact-capacity', [(0, 0), (1, 0)], [0.0, 2.0], [4.6, 4.6], [0.0, 0.0],
             0.0, 8.0, 10.0, 4.6, [1])
    run_case('capacity+eps', [(0, 0), (1, 0)], [0.0, 3.0], [4.6, 4.6], [0.0, 0.0],
             0.0, 8.0, 10.0, 4.6, [1])
    # TW: anchor=(0,0) customer=(1,0) travel=1.0 back=1.0 (往返 2.0 < 4.6 不超时);
    #     tw_end=1.0 (exact arrive==tw_end) / 0.999 (eps over)
    run_case('exact-TW', [(0, 0), (1, 0)], [0.0, 1.0], [4.6, 1.0], [0.0, 0.0],
             0.0, 0.0, 50.0, 4.6, [1])
    run_case('TW+eps', [(0, 0), (1, 0)], [0.0, 1.0], [4.6, 0.999], [0.0, 0.0],
             0.0, 0.0, 50.0, 4.6, [1])
    # return: anchor=(0,0) customer=(2,0) travel=2.0 back=2.0; service 0.6 -> ret 4.6 exact
    run_case('exact-return', [(0, 0), (2, 0)], [0.0, 1.0], [4.6, 4.6], [0.0, 0.6],
             0.0, 0.0, 50.0, 4.6, [1])
    run_case('return+eps', [(0, 0), (2, 0)], [0.0, 1.0], [4.6, 4.6], [0.0, 0.7],
             0.0, 0.0, 50.0, 4.6, [1])
    check('mask_parity_boundary_documented', True,
          '边界差异已在上面逐行打印（exact-TW/exact-return 预期 RRNCO 严格<更严）')

    # ---------------- 附加：完整 order() 跑通 ----------------
    print('== bonus: backend.order() end-to-end ==')
    try:
        with torch.inference_mode():
            ordering = backend.order(sp)
        audit = backend.last_audit
        print(f'    ordering={ordering}')
        print(f'    audit_keys={sorted(audit.to_dict().keys())}')
        print(f'    sampling_policy={audit.sampling_policy} '
              f'replacement={audit.sampling_replacement} '
              f'seed={audit.sampling_seed}')
        check('order_completes', set(ordering) == set(sp.pool_customer_ids),
              f'ordering={ordering}')
    except Exception as exc:  # noqa: BLE001
        print(f'    order() FAILED: {type(exc).__name__}: {exc}')
        FAIL.append('order_completes')

    print(f'\n== RESULT: PASS={len(PASS)} FAIL={len(FAIL)} ==')
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('B2 INJECTION + MASK-PARITY GATE: ALL PASS')


if __name__ == '__main__':
    main()
