"""复现 cada 臂「全 fallback」：跑真实 day 0 一个决策，打印 provider 异常 traceback。

用法（服务器，C-VRP_Cold-chainVehicleRoutingProblem 目录，cc_compare env）：
    /home/hzeng/envs/cc_compare/bin/python ../../CC_Compare/CaDA/a1_accept_v1/repro_one.py
只读复现 + 诊断；不写任何结果文件。CPU。
"""
import json
import os
import sys
import traceback

_SCRIPTS = os.path.join(os.getcwd(), 'scripts')
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'evaluation'),
           os.path.join(_SCRIPTS, 'coldchain'), os.path.join(_SCRIPTS, 'simulation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_exp_reserve import generate_dataset, BOOKING_HORIZON, SPEED_KMH, KM_PER_UNIT
from scenario_saa import make_c0_contract_v2
from coldchain_evaluator_a1 import add_v2_initial_quality
from strict_online_env import StrictOnlineEnv
from rrnco_accept_replanner import OrderingAcceptReplanner
from ordering_providers import build_ordering_provider

CKPT = os.path.normpath(os.path.join(os.getcwd(), '..', 'CC_Compare', 'CaDA',
                                     '100', 'result', '2024-1121-1355',
                                     'checkpoint-300.pt'))


def main():
    contract = make_c0_contract_v2()
    gate_ds = add_v2_initial_quality(generate_dataset(40, 200, 20260926), contract)
    baseline = json.load(open(os.path.join(
        os.getcwd(), 'results', 'a1_step2_gate_c1_20260926', 'gate.json'),
        encoding='utf-8'))
    B = float(baseline['budget']['B'])
    cs = float(baseline['budget']['cooling_share'])
    print('building provider...', flush=True)
    provider = build_ordering_provider('cada', CKPT, 'cpu')
    provider._load()
    print('provider loaded', flush=True)
    rp = OrderingAcceptReplanner(budget=B, capacity=50.0,
                                 booking_horizon=BOOKING_HORIZON,
                                 contract=contract, cooling_share=cs,
                                 provider=provider, time_limit=10.0)

    orig = rp._adapter.preference_provider.order
    n_err = [0]

    def wrapped(sp):
        try:
            return orig(sp)
        except Exception:
            n_err[0] += 1
            if n_err[0] <= 3:
                print('--- provider exception #%d ---' % n_err[0], flush=True)
                traceback.print_exc()
                print('sp: vehicle=%s anchor_idx=%s node_ids=%s pool=%s' % (
                    sp.vehicle_id, sp.anchor_idx, sp.node_ids[:5],
                    sp.pool_customer_ids[:5]), flush=True)
            raise

    rp._adapter.preference_provider.order = wrapped

    env = StrictOnlineEnv(gate_ds, capacity=50.0, num_vehicles=15,
                          tw_speed=SPEED_KMH / KM_PER_UNIT, replanner=rp,
                          coldchain_contract=contract,
                          booking_horizon=BOOKING_HORIZON)
    print('running day 0...', flush=True)
    traces, _ = env.run(0)
    print('accepted=%d rejected=%d' % (len(rp._accepted), len(rp._rejected)))
    print('reject_reasons=%s' % dict(rp._reject_reasons))
    print('n_solves=%d n_fail=%d timeouts=%d' % (rp.n_solves, rp.n_fail,
                                                 rp.timeouts))
    print('provider_exceptions_seen=%d' % n_err[0])


if __name__ == '__main__':
    main()
