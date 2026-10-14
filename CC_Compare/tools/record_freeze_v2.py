"""记录 A-v1 正式批次 v2 的源码冻结基线（启动前快照 + 事后对账）。

列出与批次身份相关的全部文件（共享计算文件 + 两个驱动 + 求解器适配 +
ordering 入口 + 11 个 provider + rrnco 适配/后端），打印 sha256。

用法（在 C-VRP_Cold-chainVehicleRoutingProblem 下）：
    python ../CC_Compare/tools/record_freeze_v2.py [--json out.json]
"""
import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

FILES = [
    # 共享计算文件（配对必需，source_stable 判定对象）
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/scenario_saa.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_exp_energy_c0.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/coldchain_evaluator_a1.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_exp_reserve.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/strict_online_env.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_state.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_contract.py',
    # 驱动与身份
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_a1_step2_gate.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_a1_external_accept.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/solver_accept_replanner.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_identity.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/run_a1_downstream_ablation.py',
    # ordering 入口 + 11 个 provider
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/ordering_providers.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/rrnco_accept_replanner.py',
    'CC_Compare/AttentionModel/a1_accept_v1/am_provider.py',
    'CC_Compare/DeepACO/a1_accept_v1/deepaco_provider.py',
    'CC_Compare/Omni-VRP/a1_accept_v1/omnivrp_provider.py',
    'CC_Compare/Learn-Improvement-Heuristics/a1_accept_v1/lih_provider.py',
    'CC_Compare/SGBS/a1_accept_v1/sgbs_provider.py',
    'CC_Compare/Sym-NCO/a1_accept_v1/symnco_provider.py',
    'CC_Compare/POMO/a1_accept_v1/pomo_provider.py',
    'CC_Compare/RouteFinder/a1_accept_v1/routefinder_provider.py',
    'CC_Compare/CaDA/a1_accept_v1/cada_provider.py',
    'CC_Compare/MVMoE/a1_accept_v1/mvmoe_provider.py',
    # rrnco 适配/后端（driving dcc_rh_v4）
    'CC_Compare/RRNCO/dcc_rh_v4/rrnco_guided_adapter.py',
    'CC_Compare/RRNCO/dcc_rh_v4/rrnco_backend.py',
    'CC_Compare/RRNCO/dcc_rh_v4/coordinator.py',
    'CC_Compare/RRNCO/dcc_rh_v4/identity.py',
    'CC_Compare/RRNCO/dcc_rh_v4/FROZEN_CONFIG.json',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    out = {}
    for rel in FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            out[rel] = None
            continue
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 20), b''):
                h.update(blk)
        out[rel] = h.hexdigest()
    txt = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            f.write(txt)
        print('wrote', args.json, '(%d files)' % len(out))
    else:
        print(txt)


if __name__ == '__main__':
    main()
