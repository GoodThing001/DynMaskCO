"""N1 诊断（表 2）：固定状态学习器 greedy 重建 vs teacher(regret-2) 重建。

用现有 teacher 记录（每条 P_partial/mask/P_target = regret-2 的产物）作固定状态，从同一
P_partial + mask 出发，比较学习器 greedy 重建与 regret-2 重建：
  - 完整修复成功率（mask 客户全部插回）；
  - 与 regret-2 目标计划的一致性（plan_hash / **客户集合**）；
  - 重建耗时。

证据边界（重要）：
  - `_customer_set` 只保留客户集合，**丢掉车辆归属、访问顺序与重复次数**；「集合一致」不
    代表分车/顺序一致，「无额外改动」也不代表没有改变分车或顺序。同一个 mask 的完整修复
    本来就应该覆盖相同客户集合。
  - 未计算 `J_vis`，因此**不能判定学习器与 regret-2 的完整方案谁更好**。
  - 耗时**未分离 JIT 编译与稳态**（首次调用含编译、均摊到 100 条记录），且学习器计时
    **不含 encoder 编码**（encode 在计时前完成）。完整 J_vis 对账需带状态的二次导出。
"""
import json
import os
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'training',
          'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

from strict_online_env import StrictOnlineEnv
from action_contract import VehiclePlan, plan_hash
from dynmaskco_cc_graph import build_visible_index
from train_dynmaskco_cc import _build_encoder_raw
from dynmaskco_cc_repair_replanner import load_repair_model, RepairScorer, greedy_reconstruct
from repair_teacher import reconstruct_with_steps


def _dict_to_plans(d):
    return {int(vid): VehiclePlan(int(vid), int(v['anchor_node']), float(v['anchor_time']),
                                  float(v['anchor_load']), tuple(int(x) for x in v['suffix']))
            for vid, v in d.items()}


def _customer_set(plans):
    """只保留客户集合（丢掉车辆归属/顺序/重复）。仅用于「集合一致」层面，不代表计划一致。"""
    return frozenset(int(x) for p in plans.values() for x in p.suffix)


def main():
    ROOT = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem'
    encoder_ckpt = f'{ROOT}/ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt'
    data_path = f'{ROOT}/data/m0_scale/dcc_50_r1_edod05_train_teacher.npz'
    records_path = f'{ROOT}/results/m0_scale/repair_teacher/records.jsonl'

    records = [json.loads(l) for l in open(records_path)]
    data_npz = dict(np.load(data_path))
    tw_max = float(data_npz['tw_end'][:, 0].max())
    num_nodes = data_npz['coords'].shape[1]
    env = StrictOnlineEnv(data_npz, 50.0, 1.0, 25, replanner=None, coldchain_contract=None)

    model = load_repair_model(f'{ROOT}/results/m0_scale/repair_s42/model.ckpt', encoder_ckpt)
    scorer = RepairScorer(model)

    def _encode(record):
        inst = int(record['inst']); clock = float(record['clock'])
        visible = np.zeros(num_nodes, bool); visible[0] = True
        visible |= (data_npz['reveal_time'][inst] <= clock + 1e-6)
        vis_ids = [int(i) for i in range(1, num_nodes) if visible[i]]
        node_to_local, local_to_node, Nv = build_visible_index(vis_ids)
        raw_j, vis_j = _build_encoder_raw(data_npz, inst, clock, visible, 50.0, tw_max)
        H = scorer.encode(raw_j, visible_mask=vis_j)
        Hv = np.asarray(H[0])[np.asarray(local_to_node, dtype=np.int32)]
        return Hv, node_to_local, Nv

    n_ok = 0
    n_agree_plan = 0
    n_agree_own = 0
    n_improve_own = 0
    n_worse_own = 0
    t_learn = 0.0
    t_r = 0.0
    for r in records[:100]:
        P = _dict_to_plans(r['P_partial'])
        mutable_ids = set(int(x) for x in r['mutable_ids'])
        mask = [int(x) for x in r['mask']]
        Hv, node_to_local, Nv = _encode(r)
        t0 = time.time()
        M, _ = greedy_reconstruct(scorer, Hv, P, mask, mutable_ids, env, int(r['inst']),
                                  node_to_local, Nv)
        t_learn += time.time() - t0
        t0 = time.time()
        R, _ = reconstruct_with_steps(env, int(r['inst']), P, mask, mutable_ids)
        t_r += time.time() - t0
        if M is None:
            continue
        n_ok += 1
        T = _dict_to_plans(r['P_target'])
        if plan_hash(M) == plan_hash(T):
            n_agree_plan += 1
        own_M = _customer_set(M); own_T = _customer_set(T)
        if own_M == own_T:
            n_agree_own += 1
        # 相对 regret-2 目标的「额外改动的客户数」（仅客户集合；M 比 T 多/少改的客户）
        base_own = _customer_set(P)
        changed_M = own_M ^ base_own
        changed_T = own_T ^ base_own
        extra_M = len(changed_M - changed_T)
        if extra_M == 0:
            n_improve_own += 1  # 未额外改动（仅客户集合层面）= 复现 regret-2 的修改集合
        else:
            n_worse_own += 1
    print(f'  fixed-state records evaluated = {n_ok}/100')
    print(f'  plan_hash agree with regret-2 = {n_agree_plan}/{n_ok}')
    print(f'  customer-SET agree with regret-2 = {n_agree_own}/{n_ok}  '
          f'(set only, no vehicle/order)')
    print(f'  no-extra customer-SET change vs regret-2 = {n_improve_own}/{n_ok}  '
          f'(set only)')
    print(f'  avg learn time = {t_learn/max(n_ok,1)*1000:.1f}ms  avg regret-2 time = '
          f'{t_r/max(n_ok,1)*1000:.1f}ms  (not compile/steady separated; learn excludes encoder)')


if __name__ == '__main__':
    main()
