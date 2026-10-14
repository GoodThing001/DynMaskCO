"""DynMaskCO-CC M1 训练入口（joint：utility loss + reconstruction loss）。

从 M0 teacher 数据集构造训练张量：对每个 context 恢复 fleet 状态 → incumbent P0 → 邻接 A0 →
枚举候选计划邻接 → event mask M / A_in → 冻结 encoder H/Hv → masked decoder Z → 候选端点
gather + 显式特征 → 联合损失训练 decoder + utility head（encoder 冻结）。

用法（服务器 GPU）：
    python scripts/training/train_dynmaskco_cc.py \
        --teacher-dir results/m0dev/train --data data/m0dev/dcc_50_r1_edod05_train_teacher.npz \
        --ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
        --objective-profile results/o0cc/scale_v2/objective_profile.json \
        --num-steps 200 --batch-size 8 --seed 42 --out results/m0dev/m1
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))          # scripts/training
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

from coldchain_teacher_dataset import load_teacher_dataset
from coldchain_visible_features import (extract_context_features, extract_action_features,
                                        ACTION_FEAT_DIM)
from coldchain_utility_head import feature_only_context, FEATURE_CONTEXT_DIM
from dynmaskco_cc_graph import (build_visible_index, build_plan_adjacency, build_event_mask,
                                compute_timestep, resolve_action_endpoints)
from dynmaskco_cc import MaskCODecoder, ScoringMLP, DynMaskCOModel
from train_fleet_head import load_base_model
from cvrptw_utils import coord_normalize_visible
from action_contract import ActionSlot, FleetAction, apply_action
from dynmaskco_cc_context import compute_incumbent_plans
from jf1h_repair import make_continuation

ENCODER_EMBED_DIM = 256
EXPLICIT_DIM = FEATURE_CONTEXT_DIM + 2 * ACTION_FEAT_DIM + 4 + 3   # 25 + 16 + 4 + 3 = 48


def _build_encoder_raw(data_npz, inst_idx, clock, visible, capacity, tw_max):
    coords = data_npz['coords'][inst_idx].astype(np.float32)
    demands = data_npz['demands'][inst_idx].astype(np.float32)
    tw_start = data_npz['tw_start'][inst_idx].astype(np.float32)
    tw_end = data_npz['tw_end'][inst_idx].astype(np.float32)
    temp_class = data_npz['temp_class'][inst_idx].astype(np.float32)
    reveal_time = data_npz['reveal_time'][inst_idx].astype(np.float32)
    raw = np.concatenate([
        coords, (demands / capacity)[..., None], (tw_start / tw_max)[..., None],
        (tw_end / tw_max)[..., None], (temp_class / 2.0)[..., None],
        (reveal_time / tw_max)[..., None]], axis=-1).astype(np.float32)
    vis = visible[..., None]
    raw[..., 2:] = raw[..., 2:] * vis
    raw[..., :2] = raw[..., :2] * vis + (1.0 - vis) * 0.5
    raw_j = jnp.array(raw[None])
    raw_j = raw_j.at[..., :2].set(coord_normalize_visible(raw_j[..., :2], jnp.array(visible[None])))
    return raw_j, jnp.array(visible[None])


def build_training_tensors(ds, data_npz, base_model, capacity, tw_max, deindex=False,
                           keep_all=False):
    """从 teacher 数据集构造训练张量。返回 dict of lists（每 context 一份）。

    deindex=True 时屏蔽显式特征（ctx_summary 车队编号 + 动作编号）里的编号通道；encoder H
    与 masked decoder Z 基于几何表征（节点身份是位置性的，非显式编号），本身无编号通道。

    keep_all=True（评价用）时不再跳过「无重构变化」的 context（M 置零、A_in=A0、ts=1.0），
    使输出与 ds.contexts 逐一对齐，便于离线 margin 扫描与三组同口径比较。
    """
    contexts, cands_by = ds.contexts, ds.candidates_by_context
    out = {'Hv': [], 'A_in': [], 'timestep': [], 'local_to_node': [],
           'endpoints': [], 'valid': [], 'explicit': [], 'labels': [], 'supervision': [],
           'M': [], 'A0': [], 'A_teacher': [], 'instance_ids': [], 'n_ctx': 0,
           'skipped_no_struct': 0}

    for ctx in contexts:
        ctx_id = ctx['context_id']
        inst_idx = int(ctx['inst_idx'])
        snapshot = ctx['snapshot']
        visible = np.asarray(snapshot['visible_mask'], bool)
        visible_ids = [int(i) for i in range(len(visible)) if visible[i] and i != 0]
        node_to_local, local_to_node, Nv = build_visible_index(visible_ids)

        # post-baseline incumbent：跑 baseline planner（与 teacher 的 _incumbent_plans 一致）
        env = _make_env(data_npz, capacity, base_model, tw_max)
        P0, _vehicles, _replan_ids = compute_incumbent_plans(env, snapshot, make_continuation())
        A0 = build_plan_adjacency(P0, node_to_local, Nv)

        cands = cands_by[ctx_id]
        # 候选计划邻接（feasible 且非 DEFER 的普通动作）
        cand_adjs = []
        for c in cands:
            if c.get('is_pseudo') or not c.get('feasible', False):
                continue
            act = _fleet_action(c['action'])
            try:
                Pa = apply_action(P0, act)
            except (KeyError, ValueError):
                continue
            cand_adjs.append(build_plan_adjacency(Pa, node_to_local, Nv))

        if not cand_adjs:
            if not keep_all:
                out['skipped_no_struct'] += 1
                continue
            M = np.zeros((Nv, Nv), bool)
            A_in = np.asarray(A0).copy()
        else:
            M, A_in = build_event_mask(A0, cand_adjs)
            if int(M.astype(int).sum()) == 0 and not keep_all:
                out['skipped_no_struct'] += 1
                continue

        ts = compute_timestep(A0, A_in)
        # teacher 目标邻接：teacher 选中的普通动作应用后的计划；KEEP/DEFER 则为 A0。
        teacher_adj = A0
        for c in cands:
            if c.get('selected_by_teacher') and not c.get('is_pseudo'):
                try:
                    teacher_adj = build_plan_adjacency(
                        apply_action(P0, _fleet_action(c['action'])), node_to_local, Nv)
                except (KeyError, ValueError):
                    pass
                break
        raw_j, vis_j = _build_encoder_raw(data_npz, inst_idx, float(ctx['clock']), visible,
                                          capacity, tw_max)
        H = base_model.encode(raw_j, visible_mask=vis_j)
        Hv = np.asarray(H[0])[np.asarray(local_to_node, dtype=np.int32)]  # [Nv, D]

        # 显式 context 摘要（feature_only）
        feats = extract_context_features(data_npz, inst_idx, snapshot, deindex=deindex)
        ctx_summary = np.asarray(feature_only_context(
            jnp.asarray(feats['order_feats']), jnp.asarray(feats['node_visible']),
            jnp.asarray(feats['fleet_feats']), jnp.asarray(feats['vehicle_valid'])))

        endpoints, valid, explicit, labels, sup = [], [], [], [], []
        for c in cands:
            av, avd = extract_action_features(c, deindex=deindex)
            cert = c.get('certificate') or {}
            incr = cert.get('incremental_distance')
            tws = cert.get('tw_slack')
            caps = cert.get('cap_slack')
            rets = cert.get('return_slack')
            cert_vals = np.array([
                0.0 if incr is None else float(incr),
                0.0 if tws is None else float(tws),
                0.0 if caps is None else float(caps),
                0.0 if rets is None else float(rets)], np.float32)
            ptype = c.get('pseudo')  # None / 'KEEP' / 'DEFER'
            type_oh = np.array([ptype is None, ptype == 'KEEP', ptype == 'DEFER'], np.float32)
            exp = np.concatenate([ctx_summary, av, avd.astype(np.float32), cert_vals, type_oh])
            if c.get('is_pseudo') and c.get('pseudo') == 'DEFER':
                e, v = resolve_action_endpoints(int(c['action']['customer']), None, None, None,
                                                P0, node_to_local)
            elif c.get('is_pseudo') and c.get('pseudo') == 'KEEP':
                act = _fleet_action(c['action'])
                e, v = resolve_action_endpoints(act.customer, act.slot, act.predecessor,
                                                act.successor, P0, node_to_local)
            else:
                act = _fleet_action(c['action'])
                e, v = resolve_action_endpoints(act.customer, act.slot, act.predecessor,
                                                act.successor, P0, node_to_local)
            endpoints.append(e); valid.append(v); explicit.append(exp)
            sup.append(bool(ds.supervision_mask([c])[0]))
            d = c.get('delta_vs_keep')
            labels.append(0.0 if d is None else float(d))

        out['Hv'].append(Hv); out['A_in'].append(A_in); out['timestep'].append(ts)
        out['local_to_node'].append(local_to_node)
        out['endpoints'].append(np.stack(endpoints))
        out['valid'].append(np.stack(valid))
        out['explicit'].append(np.stack(explicit))
        out['labels'].append(np.array(labels, np.float32))
        out['supervision'].append(np.array(sup, bool))
        out['M'].append(M); out['A0'].append(A0); out['A_teacher'].append(teacher_adj)
        out['instance_ids'].append(inst_idx)
        out['n_ctx'] += 1
    return out


def _make_env(data_npz, capacity, base_model, tw_max):
    # 只需 dist_mat 用于 build_vehicle_plans/certify；构造最小 env 对象
    from strict_online_env import StrictOnlineEnv
    return StrictOnlineEnv(data_npz, capacity, 1.0, 25, replanner=None, coldchain_contract=None)


def build_plans_from_snapshot(env, inst_idx, snapshot):
    """从 teacher snapshot 字段直接重建 incumbent FleetPlan（绕过 state_hash 校验）。

    快照经 JSON 往返后 state_hash 不复现（序列化差异），但 vehicle 字段仍在；据此重建
    anchor_node/anchor_time/anchor_load/suffix，语义与 build_vehicle_plans 一致。
    """
    from action_contract import VehiclePlan
    K = int(snapshot['num_vehicles'])
    status = list(snapshot['vehicle_status'])
    node = [int(x) for x in snapshot['vehicle_node']]
    ready = [float(x) for x in snapshot['vehicle_ready']]
    load = [float(x) for x in snapshot['vehicle_load']]
    cn = [int(x) for x in snapshot['committed_next']]
    cf = [float(x) for x in snapshot['committed_finish']]
    suffix = snapshot['mutable_suffix']
    slen = [int(x) for x in snapshot['mutable_suffix_len']]
    plans = {}
    for k in range(K):
        if status[k] == 'closed':
            continue
        if status[k] == 'committed' and cn[k] not in (None, 0, -1):
            anchor_node = cn[k]
            anchor_time = cf[k]
            anchor_load = load[k] + float(env.demands[inst_idx, anchor_node])
        else:
            anchor_node = node[k]
            anchor_time = ready[k]
            anchor_load = load[k]
        s = tuple(int(x) for x in suffix[k][:slen[k]] if int(x) != 0)
        plans[k] = VehiclePlan(k, anchor_node, anchor_time, anchor_load, s)
    return plans


def _fleet_action(a):
    return FleetAction(customer=int(a['customer']), slot=ActionSlot(a['slot_kind'], int(a['slot_anchor'])),
                       position=int(a['position']), predecessor=int(a['predecessor']),
                       successor=int(a['successor']), incumbent=bool(a.get('incumbent', False)))


def huber(err, delta=1.0):
    a = jnp.abs(err)
    return jnp.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True, help='冻结 encoder')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num-steps', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--tw-max', type=float, default=None)
    ap.add_argument('--deindex', action='store_true',
                    help='去编号消融：屏蔽显式特征（ctx 车队编号 + 动作编号）里的编号通道')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = load_teacher_dataset(args.teacher_dir, data_path=args.data)
    data_npz = dict(np.load(args.data))
    tw_max = args.tw_max if args.tw_max is not None else float(data_npz['tw_end'][:, 0].max())
    base_model = load_base_model(args.ckpt)

    print("building training tensors...", flush=True)
    t0 = time.time()
    tensors = build_training_tensors(ds, data_npz, base_model, args.capacity, tw_max,
                                     deindex=args.deindex)
    print(f"  n_ctx={tensors['n_ctx']} skipped_no_struct={tensors['skipped_no_struct']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if tensors['n_ctx'] == 0:
        raise RuntimeError("无可用训练 context（所有 context 结构分支无重构变化）")

    # 尺度 s（TRAIN 监督标签）
    all_labels = np.concatenate(tensors['labels'])
    all_sup = np.concatenate(tensors['supervision'])
    s = max(float(np.std(all_labels[all_sup])), 1e-6)

    decoder = MaskCODecoder(d_enc=ENCODER_EMBED_DIM, d_dec=ENCODER_EMBED_DIM,
                            num_layers=6, num_heads=8, rngs=args.seed)
    head = ScoringMLP(4 * ENCODER_EMBED_DIM * 2 + 4 + EXPLICIT_DIM, hidden=(128, 64),
                      rngs=args.seed)
    d_graphdef, d_params = nnx.split(decoder)
    h_graphdef, h_params = nnx.split(head)
    tx = optax.adamw(args.lr, weight_decay=1e-2)
    opt_state = tx.init({'decoder': d_params, 'head': h_params})

    def loss_fn(params, Hv, A_in, ts, endpoints, valid, explicit, M, A_teacher, y, sup, s):
        dec = nnx.merge(d_graphdef, params['decoder'])
        hd = nnx.merge(h_graphdef, params['head'])
        Z = dec(Hv, ts, A_in)
        struct = _gather(Hv, Z, endpoints, valid)
        x = jnp.concatenate([struct, explicit], axis=-1)
        score = hd(x)                       # [B, M]
        y_n = -y / s
        err = score - y_n
        util = jnp.where(sup, huber(err), 0.0)
        util = util.sum() / jnp.maximum(sup.sum(), 1.0)
        logits = _decoder_logits(dec, Hv, ts, A_in)
        recon = _recon_loss(logits, M, A_teacher)
        return util + 1.0 * recon, (util, recon)

    @jax.jit
    def train_step(params, opt_state, Hv, A_in, ts, endpoints, valid, explicit, M, A_teacher, y,
                   sup, s):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, Hv, A_in, ts, endpoints, valid, explicit, M, A_teacher, y, sup, s)
        updates, new_opt = tx.update(grads, opt_state, params)
        return loss, aux, optax.apply_updates(params, updates), new_opt

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(tensors['n_ctx'])
    params = {'decoder': d_params, 'head': h_params}
    # 全局 padding：固定 batch 内节点/候选最大维，使 JIT 形状恒定（编译/autotune 一次）。
    Nv_max = max(tensors['Hv'][i].shape[0] for i in range(tensors['n_ctx']))
    M_max = max(tensors['endpoints'][i].shape[0] for i in range(tensors['n_ctx']))
    def _pad_nodes(key, square):
        out = []
        for i in batch:
            x = np.asarray(tensors[key][i])
            top = Nv_max - x.shape[0]
            pw = ((0, top), (0, top)) if square else ((0, top), (0, 0))
            out.append(np.pad(x, pw, constant_values=0))
        return jnp.asarray(np.stack(out)).astype(jnp.float32)
    def pad(a, fill):
        arr = [np.asarray(tensors[a][i]) for i in batch]
        out = []
        for x in arr:
            top = M_max - x.shape[0]
            pw = ((0, top), (0, 0)) if x.ndim == 2 else ((0, top),)
            out.append(np.pad(x, pw, constant_values=fill))
        return jnp.asarray(np.stack(out))
    for step in range(args.num_steps):
        batch = rng.choice(idxs, size=min(args.batch_size, tensors['n_ctx']), replace=True)
        Hv = _pad_nodes('Hv', square=False)
        A_in = _pad_nodes('A_in', square=True)
        ts = jnp.array([tensors['timestep'][i] for i in batch], jnp.float32)
        endpoints = pad('endpoints', 0).astype(jnp.int32)
        valid = pad('valid', 0).astype(bool)
        explicit = pad('explicit', 0).astype(jnp.float32)
        y = pad('labels', 0).astype(jnp.float32)
        sup = pad('supervision', 0).astype(bool)
        M = _pad_nodes('M', square=True)
        A_teacher = _pad_nodes('A_teacher', square=True)
        loss, aux, params, opt_state = train_step(
            params, opt_state, Hv, A_in, ts, endpoints, valid, explicit, M, A_teacher, y, sup, s)
        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{args.num_steps} loss={float(loss):.4f} "
                  f"util={float(aux[0]):.4f} recon={float(aux[1]):.4f}", flush=True)

    with open(os.path.join(args.out, 'm1.ckpt'), 'wb') as f:
        pickle.dump({'decoder': params['decoder'], 'head': params['head'],
                     's': s, 'explicit_dim': EXPLICIT_DIM, 'd_model': ENCODER_EMBED_DIM}, f)
    json.dump({'ckpt': args.ckpt, 's': s, 'n_ctx': tensors['n_ctx'],
               'num_steps': args.num_steps, 'tw_max': tw_max, 'deindex': args.deindex,
               'final_loss': float(loss), 'explicit_dim': EXPLICIT_DIM},
              open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
    print(f"saved: {args.out}/m1.ckpt (s={s:.4f})")


def _gather(Hv, Z, endpoints, valid):
    from dynmaskco_cc import gather_endpoints
    return gather_endpoints(Hv, Z, endpoints, valid)


def _decoder_logits(dec, Hv, ts, A_in):
    """对称边 logits：Z @ Z^T（与 MaskCO feature2logit 一致，无 final_norm 简化）。"""
    Z = dec(Hv, ts, A_in)
    return jnp.matmul(Z, Z.transpose(0, 2, 1))


def _recon_loss(logits, M, A_teacher):
    """M 上三角区域的对称边 BCE，正负边分别归一化。"""
    tri = jnp.triu(jnp.ones_like(M), k=1)
    region = (M * tri) > 0.5
    target = (A_teacher * tri) > 0.5
    bce = -jax.nn.log_sigmoid(logits) * target - jax.nn.log_sigmoid(-logits) * (1.0 - target)
    pos = region & target
    neg = region & (~target)
    pos_loss = (bce * pos).sum() / jnp.maximum(pos.sum(), 1.0)
    neg_loss = (bce * neg).sum() / jnp.maximum(neg.sum(), 1.0)
    return 0.5 * pos_loss + 0.5 * neg_loss


if __name__ == '__main__':
    main()
