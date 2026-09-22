"""N1 诊断（表 1）：归档 + 训练/部署一致性 + 标签回放 + 跳过计数。

只读现有 checkpoint（不重训），用现有 teacher 记录做最小正确性诊断：
  1. 归档：checkpoint / 关键源码 / 配置 的 SHA-256；
  2. 训练式 padding（补零到训练 `Nv_max`） vs 在线实际 Nv 的动作排序是否一致；
  3. 标签回放：对每条 teacher 记录，steps 从 P_partial 重放能否得到 P_target 的**客户集合**；
  4. 跳过计数：训练中「目标步骤不在枚举 / 无合法插入」而被跳过的记录/步骤数。

证据边界（重要）：
  - 标签回放只比较**客户集合**（`own_P == own_T`），丢掉了车辆归属/顺序/重复；尚未证明
    逐车有序路线完整一致。
  - padding 翻转范围 = s42、前 20 条训练记录的**第一步选择**；补齐长度取 checkpoint 存出的
    `Nv_max`（与总节点数 `num_nodes` 一并打印核对），不是整个部署分布的翻转率。
"""
import json
import os
import sys
import hashlib
import pickle

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'training',
          'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

import jax
import jax.numpy as jnp

from strict_online_env import StrictOnlineEnv
from action_contract import (VehiclePlan, FleetAction, ActionSlot, enumerate_actions_from_plans,
                             apply_action, plan_hash)
from dynmaskco_cc_graph import build_visible_index, build_plan_adjacency, resolve_action_endpoints
from train_fleet_head import load_base_model
from train_dynmaskco_cc import _build_encoder_raw
from dynmaskco_cc_repair_replanner import load_repair_model, RepairScorer


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()[:16]


def _dict_to_plans(d):
    return {int(vid): VehiclePlan(int(vid), int(v['anchor_node']), float(v['anchor_time']),
                                  float(v['anchor_load']), tuple(int(x) for x in v['suffix']))
            for vid, v in d.items()}


def main():
    ROOT = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem'
    PY = '/home/hzeng/miniconda3/envs/MASKCO_env/bin/python'
    encoder_ckpt = f'{ROOT}/ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt'
    data_path = f'{ROOT}/data/m0_scale/dcc_50_r1_edod05_train_teacher.npz'
    records_path = f'{ROOT}/results/m0_scale/repair_teacher/records.jsonl'

    records = [json.loads(l) for l in open(records_path)]
    data_npz = dict(np.load(data_path))
    tw_max = float(data_npz['tw_end'][:, 0].max())

    # --- 1. archive ---
    print('=== 1. archive ===')
    for sd in (42, 43, 44):
        ck = f'{ROOT}/results/m0_scale/repair_s{sd}/model.ckpt'
        print(f'  repair_s{sd} model.ckpt sha256 = {_sha(ck)}')
    print(f'  encoder ckpt sha256 = {_sha(encoder_ckpt)}')

    # --- 2. padding consistency ---
    num_nodes = data_npz['coords'].shape[1]
    ckpt_path = f'{ROOT}/results/m0_scale/repair_s42/model.ckpt'
    with open(ckpt_path, 'rb') as f:
        _ckpt_meta = pickle.load(f)
    Nv_max_train = int(_ckpt_meta.get('Nv_max', num_nodes))
    print(f'=== 2. padding consistency (actual Nv vs train Nv_max={Nv_max_train}, '
          f'num_nodes={num_nodes}) ===')
    model = load_repair_model(ckpt_path, encoder_ckpt)
    scorer = RepairScorer(model)
    env = StrictOnlineEnv(data_npz, 50.0, 1.0, 25, replanner=None, coldchain_contract=None)

    def _score_first_step(record, pad_to):
        inst = int(record['inst']); clock = float(record['clock'])
        P = _dict_to_plans(record['P_partial'])
        mutable_ids = set(int(x) for x in record['mutable_ids'])
        mask = [int(x) for x in record['mask']]
        visible = np.zeros(num_nodes, bool); visible[0] = True
        visible |= (data_npz['reveal_time'][inst] <= clock + 1e-6)
        vis_ids = [int(i) for i in range(1, num_nodes) if visible[i]]
        node_to_local, local_to_node, Nv = build_visible_index(vis_ids)
        A = build_plan_adjacency(P, node_to_local, Nv)
        # encoder Hv (actual Nv)
        raw_j, vis_j = _build_encoder_raw(data_npz, inst, clock, visible, 50.0, tw_max)
        H = scorer.encode(raw_j, visible_mask=vis_j)
        Hv_actual = np.asarray(H[0])[np.asarray(local_to_node, dtype=np.int32)]
        # enumerate first-step options (remaining = all mask customers)
        ends, vals, exps = [], [], []
        for c in mask:
            cands, _ = enumerate_actions_from_plans(env, inst, P, int(c),
                                                    allowed_vehicle_ids=mutable_ids)
            for cand in cands:
                if not cand.feasible:
                    continue
                a = cand.action
                e, v = resolve_action_endpoints(a.customer, a.slot, a.predecessor, a.successor,
                                                P, node_to_local)
                ends.append(e); vals.append(v)
                exps.append([float(cand.incremental_distance),
                             0.0 if cand.tw_slack is None else float(cand.tw_slack),
                             0.0 if cand.cap_slack is None else float(cand.cap_slack),
                             0.0 if cand.return_slack is None else float(cand.return_slack)])
        if not ends:
            return None
        M = len(ends)
        if pad_to is not None:
            Npad = int(pad_to)
            Hv = np.zeros((1, Npad, 256), np.float32)
            Hv[:, :Nv, :] = Hv_actual[None]
            A_in = np.zeros((1, Npad, Npad), np.float32)
            A_in[:, :Nv, :Nv] = A[None]
        else:
            Hv = Hv_actual[None]; A_in = A[None]
        endpoints = jnp.array(np.stack(ends)[None].astype(np.int32))
        valid = jnp.array(np.stack(vals)[None].astype(bool))
        explicit = jnp.array(np.stack(exps)[None].astype(np.float32))
        ts = jnp.array([0.5], jnp.float32)
        Z = scorer.decode(Hv, ts, jnp.array(A_in.astype(np.float32)))
        sc = np.asarray(scorer.score_with_Z(Hv, Z, endpoints, valid, explicit))[0]
        return sc, int(np.argmax(sc)), M

    flip = 0; total = 0
    for r in records[:20]:
        a = _score_first_step(r, pad_to=None)
        b = _score_first_step(r, pad_to=Nv_max_train)
        if a is None or b is None:
            continue
        total += 1
        if a[1] != b[1]:
            flip += 1
            print(f"  inst{r['inst']}/evt{r['event']}: argmax differs (Nv={a[2]} opts) "
                  f"no-pad={a[1]} pad={b[1]}")
    print(f'  argmax flip: {flip}/{total}  (s42, first {len(records[:20])} records, first-step)')

    # --- 3. label replay ---
    print('=== 3. label replay (steps -> P_target CUSTOMER SET only) ===')
    replay_ok = 0; replay_bad = 0
    for r in records:
        P = _dict_to_plans(r['P_partial'])
        mutable_ids = set(int(x) for x in r['mutable_ids'])
        for st in r['steps']:
            act = FleetAction(customer=int(st['customer']),
                              slot=ActionSlot(st['slot_kind'], int(st['slot_anchor'])),
                              position=int(st['position']), predecessor=int(st['predecessor']),
                              successor=int(st['successor']), incumbent=False)
            try:
                P = apply_action(P, act, allowed_vehicle_ids=mutable_ids)
            except (KeyError, ValueError):
                replay_bad += 1
                break
        # 只比较客户集合（不含分车/顺序/重复）；逐车有序路线一致未证明
        tgt = _dict_to_plans(r['P_target'])
        own_P = {int(x) for p in P.values() for x in p.suffix}
        own_T = {int(x) for p in tgt.values() for x in p.suffix}
        if own_P == own_T:
            replay_ok += 1
        else:
            replay_bad += 1
    print(f'  replay ok/bad = {replay_ok}/{replay_bad} (of {len(records)}; customer set only)')

    # --- 4. skip counting (re-run training tensor construction, count skips) ---
    print('=== 4. skip counting (target step not found / no options) ===')
    from train_dynmaskco_cc_repair import _reconstruct_record_tensors
    base_model = load_base_model(encoder_ckpt)
    skipped = 0; steps_total = 0
    for r in records:
        try:
            Hv, A_partial, ts, per_step = _reconstruct_record_tensors(
                r, data_npz, base_model, 50.0, tw_max, env)
        except (KeyError, ValueError, IndexError):
            skipped += 1
            continue
        steps_total += len(per_step)
        if len(per_step) < len(r['steps']):
            skipped += (len(r['steps']) - len(per_step))
    print(f'  records with errors/truncation = {skipped}, total replayed steps = {steps_total}')


if __name__ == '__main__':
    main()
