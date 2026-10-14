"""N1 训练：MaskCO 条件联合修复器（冻结 encoder + 可训练 decoder/插入头）。

训练目标 = 条件插入步骤监督：给定 P_partial（移除 mask 后）的可见状态与邻接条件，对剩余
mask 客户的合法插入打分，用 cross-entropy 对齐 teacher 的下一步插入（customer+slot+position）。
不是旧 g_B 回归。

用法（本地/服务器）：
    python scripts/training/train_dynmaskco_cc_repair.py \
        --records results/m0_scale/repair_teacher/records.jsonl \
        --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
        --ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
        --num-steps 10000 --batch-size 16 --lr 1e-4 --seed 42 --out results/m0_scale/repair_s42
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'expert'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(_CVRPTW, '..', 'MASKCO_code', 'models'))

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from dynmaskco_cc import MaskCODecoder, ScoringMLP, gather_endpoints
from dynmaskco_cc_graph import (build_visible_index, build_plan_adjacency, compute_timestep,
                                resolve_action_endpoints)
from train_fleet_head import load_base_model
from train_dynmaskco_cc import _build_encoder_raw
from cvrptw_utils import coord_normalize_visible
from action_contract import ActionSlot, FleetAction, VehiclePlan, enumerate_actions_from_plans
from strict_online_env import StrictOnlineEnv

ENCODER_EMBED_DIM = 256
EXPLICIT_DIM = 4   # incremental_distance / tw_slack / cap_slack / return_slack


def dict_to_plans(d):
    return {int(vid): VehiclePlan(int(vid), int(v['anchor_node']), float(v['anchor_time']),
                                  float(v['anchor_load']), tuple(int(x) for x in v['suffix']))
            for vid, v in d.items()}


def _reconstruct_record_tensors(record, data_npz, base_model, capacity, tw_max, env):
    """一条 teacher 记录 → (Hv, A_partial, ts, per_step)。

    per_step[k] = (endpoints[M_k,4], valid[M_k,4], explicit[M_k,4], target_idx)。
    M_k = 第 k 步剩余客户的合法插入数；target_idx = teacher 插入在 options 里的下标。
    """
    inst = int(record['inst'])
    clock = float(record['clock'])
    num_nodes = data_npz['coords'].shape[1]
    reveal = data_npz['reveal_time'][inst]
    visible = np.zeros(num_nodes, bool)
    visible[0] = True
    visible |= (reveal <= clock + 1e-6)
    visible_ids = [int(i) for i in range(1, num_nodes) if visible[i]]
    node_to_local, local_to_node, Nv = build_visible_index(visible_ids)

    partial = dict_to_plans(record['P_partial'])
    A_partial = build_plan_adjacency(partial, node_to_local, Nv)
    ts = 0.5  # 固定中性 timestep；条件主要来自 A_partial（部分计划邻接）

    raw_j, vis_j = _build_encoder_raw(data_npz, inst, clock, visible, capacity, tw_max)
    H = base_model.encode(raw_j, visible_mask=vis_j)
    Hv = np.asarray(H[0])[np.asarray(local_to_node, dtype=np.int32)]

    mutable_ids = set(int(x) for x in record['mutable_ids'])
    P = dict_to_plans(record['P_partial'])
    remaining = list(record['mask'])
    per_step = []
    for target in record['steps']:
        ends, vals, exps = [], [], []
        t_idx = None
        for i, c in enumerate(remaining):
            cands, _ = enumerate_actions_from_plans(env, inst, P, int(c),
                                                    allowed_vehicle_ids=mutable_ids)
            for cand in cands:
                if not cand.feasible:
                    continue
                a = cand.action
                e, v = resolve_action_endpoints(a.customer, a.slot, a.predecessor, a.successor,
                                                P, node_to_local)
                ends.append(e)
                vals.append(v)
                cert = {'incr': cand.incremental_distance, 'tw': cand.tw_slack,
                        'cap': cand.cap_slack, 'ret': cand.return_slack}
                exps.append([0.0 if cert['incr'] is None else float(cert['incr']),
                             0.0 if cert['tw'] is None else float(cert['tw']),
                             0.0 if cert['cap'] is None else float(cert['cap']),
                             0.0 if cert['ret'] is None else float(cert['ret'])])
                if (int(a.customer) == int(target['customer'])
                        and a.slot.kind == target['slot_kind']
                        and int(a.slot.anchor) == int(target['slot_anchor'])
                        and int(a.position) == int(target['position'])):
                    t_idx = len(ends) - 1
        if t_idx is None or not ends:
            # teacher 插入不在本次枚举（罕见：slot 匹配失败）→ 跳过该步
            break
        per_step.append((np.stack(ends).astype(np.int32), np.stack(vals).astype(bool),
                         np.stack(exps).astype(np.float32), int(t_idx)))
        # 应用 teacher 插入，推进部分计划
        act = FleetAction(customer=int(target['customer']),
                          slot=ActionSlot(target['slot_kind'], int(target['slot_anchor'])),
                          position=int(target['position']), predecessor=int(target['predecessor']),
                          successor=int(target['successor']), incumbent=False)
        from action_contract import apply_action
        P = apply_action(P, act, allowed_vehicle_ids=mutable_ids)
        remaining.remove(int(target['customer']))
    return Hv, A_partial, ts, per_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num-steps', type=int, default=10000)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records = [json.loads(l) for l in open(args.records)]
    data_npz = dict(np.load(args.data))
    tw_max = float(data_npz['tw_end'][:, 0].max())
    base_model = load_base_model(args.ckpt)
    env = StrictOnlineEnv(data_npz, args.capacity, 1.0, 25, replanner=None,
                          coldchain_contract=None)

    print("building repair tensors...", flush=True)
    samples = []   # (Hv, A_partial, ts, per_step) per record
    n_steps_total = 0
    for r in records:
        try:
            Hv, A_partial, ts, per_step = _reconstruct_record_tensors(
                r, data_npz, base_model, args.capacity, tw_max, env)
        except (KeyError, ValueError, IndexError):
            continue
        if per_step:
            samples.append((Hv, A_partial, ts, per_step))
            n_steps_total += len(per_step)
    print(f"  n_records={len(samples)} n_steps={n_steps_total}", flush=True)
    if not samples:
        raise RuntimeError("无可用训练样本")

    head_in = 4 * ENCODER_EMBED_DIM * 2 + 4 + EXPLICIT_DIM
    decoder = MaskCODecoder(d_enc=ENCODER_EMBED_DIM, d_dec=ENCODER_EMBED_DIM,
                            num_layers=6, num_heads=8, rngs=args.seed)
    head = ScoringMLP(head_in, hidden=(128, 64), rngs=args.seed)
    d_gd, d_params = nnx.split(decoder)
    h_gd, h_params = nnx.split(head)
    tx = optax.adamw(args.lr, weight_decay=1e-2)
    params = {'decoder': d_params, 'head': h_params}
    opt_state = tx.init(params)

    # 全局 padding（固定 batch 内最大节点/选项维，JIT 形状恒定）
    Nv_max = max(s[0].shape[0] for s in samples)
    M_max = max(max(p[0].shape[0] for p in s[3]) for s in samples)

    def _pad_nodes(x):
        top = Nv_max - x.shape[0]
        return np.pad(x, ((0, top), (0, 0)), constant_values=0)

    def _pad_square(x):
        top = Nv_max - x.shape[0]
        return np.pad(x, ((0, top), (0, top)), constant_values=0)

    def _sample_tensors(sample, k):
        _Hv, _A_partial, _ts, per_step = sample
        ends, vals, exps, tgt = per_step[k]
        M = ends.shape[0]
        top = M_max - M
        ends_p = np.pad(ends, ((0, top), (0, 0)), constant_values=0)
        vals_p = np.pad(vals, ((0, top), (0, 0)), constant_values=False)
        exps_p = np.pad(exps, ((0, top), (0, 0)), constant_values=0)
        opt_mask = np.zeros(M_max, bool); opt_mask[:M] = True
        return (ends_p, vals_p, exps_p, opt_mask, tgt)

    def loss_fn(params, Hv, A_in, ts, endpoints, valid, explicit, opt_mask, tgt):
        dec = nnx.merge(d_gd, params['decoder'])
        hd = nnx.merge(h_gd, params['head'])
        Z = dec(Hv, ts, A_in)
        struct = gather_endpoints(Hv, Z, endpoints, valid)
        x = jnp.concatenate([struct, explicit], axis=-1)
        scores = hd(x)                              # [B, M_max]
        logits = jnp.where(opt_mask, scores, -1e9)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -log_probs[jnp.arange(log_probs.shape[0]), tgt].mean()

    @jax.jit
    def train_step(params, opt_state, Hv, A_in, ts, endpoints, valid, explicit, opt_mask, tgt):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, Hv, A_in, ts, endpoints, valid, explicit, opt_mask, tgt)
        updates, new_opt = tx.update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), new_opt

    rng = np.random.default_rng(args.seed)
    loss = float('nan')
    for step in range(args.num_steps):
        batch = [samples[int(rng.integers(len(samples)))]
                 for _ in range(min(args.batch_size, len(samples)))]
        Hv = jnp.array(np.stack([_pad_nodes(s[0]) for s in batch])).astype(jnp.float32)
        A_in = jnp.array(np.stack([_pad_square(s[1]) for s in batch])).astype(jnp.float32)
        ts = jnp.array([s[2] for s in batch], jnp.float32)
        ends, vals, exps, masks, tgts = [], [], [], [], []
        for s in batch:
            k = int(rng.integers(len(s[3])))
            e, v, x, m, t = _sample_tensors(s, k)
            ends.append(e); vals.append(v); exps.append(x); masks.append(m); tgts.append(t)
        ends = jnp.array(np.stack(ends)).astype(jnp.int32)
        vals = jnp.array(np.stack(vals)).astype(bool)
        exps = jnp.array(np.stack(exps)).astype(jnp.float32)
        masks = jnp.array(np.stack(masks)).astype(bool)
        tgts = jnp.array(np.stack(tgts)).astype(jnp.int32)
        loss, params, opt_state = train_step(params, opt_state, Hv, A_in, ts, ends, vals, exps,
                                             masks, tgts)
        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{args.num_steps} loss={float(loss):.4f}", flush=True)

    with open(os.path.join(args.out, 'model.ckpt'), 'wb') as f:
        pickle.dump({'decoder': params['decoder'], 'head': params['head'],
                     'explicit_dim': EXPLICIT_DIM, 'd_model': ENCODER_EMBED_DIM,
                     'Nv_max': Nv_max, 'M_max': M_max}, f)
    json.dump({'ckpt': args.ckpt, 'num_steps': args.num_steps, 'lr': args.lr,
               'seed': args.seed, 'n_records': len(samples), 'n_steps': n_steps_total,
               'final_loss': float(loss), 'explicit_dim': EXPLICIT_DIM,
               'Nv_max': Nv_max, 'M_max': M_max},
              open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
    print(f"saved: {args.out}/model.ckpt (loss={float(loss):.4f})")


if __name__ == '__main__':
    main()
