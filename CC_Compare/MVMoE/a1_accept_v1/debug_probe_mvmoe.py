"""MVMoE 臂 fallback 诊断探针（只读驱动，不改任何 driver/replanner 文件）。

在真实 driver 导入链 + 真实 SubProblem 下重放 provider.order，打印异常 traceback。
用法（服务器，C-VRP_Cold-chainVehicleRoutingProblem 目录）：
    /home/hzeng/envs/cc_compare/bin/python ../CC_Compare/MVMoE/a1_accept_v1/debug_probe_mvmoe.py
"""
import os
import sys
import traceback

for p in ('scripts/evaluation', 'scripts/coldchain', 'scripts/simulation',
          'scripts'):
    if p not in sys.path:
        sys.path.insert(0, p)

import ordering_providers  # noqa: E402

_orig = ordering_providers.build_ordering_provider


def wrapped(name, ckpt, device='cuda', **kw):
    p = _orig(name, ckpt, device, **kw)
    if hasattr(p, 'order'):
        _o = p.order

        def logged(sp):
            try:
                r = _o(sp)
                print('ORDER_OK pool=%d first=%s' % (
                    len(sp.pool_customer_ids), list(r)[:3] if r else '()'),
                    flush=True)
                return r
            except Exception as e:  # noqa: BLE001
                print('ORDER_EXC pool=%d nodes=%s %s: %s' % (
                    len(sp.pool_customer_ids), list(sp.node_ids),
                    type(e).__name__, e), flush=True)
                traceback.print_exc()
                raise
        p.order = logged
    return p


ordering_providers.build_ordering_provider = wrapped

import run_a1_external_accept  # noqa: E402

run_a1_external_accept.main([
    '--solver', 'ordering', '--provider', 'mvmoe',
    '--provider-ckpt',
    '../CC_Compare/MVMoE/pretrained/pomo_vrptw_n100/epoch-5000.pt',
    '--provider-device', 'cpu',
    '--baseline-from', 'results/a1_step2_gate_c1_20260926/gate.json',
    '--dev-instances', '1', '--workers', '1',
    '--penalty', 'p_c=0',
    '--out', 'results/a1_ext_extra/mvmoe_dbg1',
])
