"""
JF2 Phase 3 Step 14-16 — 冻结 MaskCO encoder，训练 FleetAssignmentHead（M0 seed42）。

读 collect_assignment_states.py 采的 state（features）+ OR-joint teacher_vehicle，用**冻结**的
Causal MaskCO encoder 算 H，训练 FleetAssignmentHead（CE loss 只在 sound-feasible vehicles 上）。

M0 关键指标（主控文档 Step 16）：不是 overall top-1，而是「teacher != JF1-H 的状态上模型能不能
学对」（disagreement subset accuracy）——teacher==JF1-H 的 easy state 不产生优化价值。

用法（服务器）：
    python scripts/training/train_fleet_head.py \
      --ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
      --states results/jf2/data/r1_edod05/states_train_0.npz \
      --teacher results/jf2/data/r1_edod05/teacher_vehicle.npz \
      --data data/baseline/50_node/train/dcc_50_r1_edod05_train.npz \
      --num_steps 2000 --batch_size 64 --lr 1e-3 --seed 42 \
      --out results/jf2/m0_seed42
"""
import sys, os, argparse, time
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
_MASKCO = os.path.dirname(_CVRPTW)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from fleet_features import Fv, Fp
from fleet_assignment_head import FleetAssignmentHead


def load_base_model(ckpt_path):
    from training import load_ckpt
    from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    if params is None:
        raise RuntimeError(f"load_ckpt 返回空 params: {ckpt_path}")
    return nnx.merge(nnx.graphdef(model), params)


def build_H_batch(base_model, inst_data, states, batch_idx, tw_max, capacity):
    """批量构造 7D 特征 → 冻结 encoder → H [B, N, D]。"""
    from cvrptw_utils import coord_normalize_visible
    inst_ids = states['instance_id'][batch_idx]
    clock = states['clock'][batch_idx]
    N = inst_data['coords'].shape[1]

    coords = inst_data['coords'][inst_ids].astype(np.float32)
    demands = inst_data['demands'][inst_ids].astype(np.float32)
    tw_start = inst_data['tw_start'][inst_ids].astype(np.float32)
    tw_end = inst_data['tw_end'][inst_ids].astype(np.float32)
    temp_class = inst_data['temp_class'][inst_ids].astype(np.float32)
    reveal_time = inst_data['reveal_time'][inst_ids].astype(np.float32)

    raw = np.concatenate([
        coords, (demands / capacity)[..., None], (tw_start / tw_max)[..., None],
        (tw_end / tw_max)[..., None], (temp_class / 2.0)[..., None],
        (reveal_time / tw_max)[..., None],
    ], axis=-1).astype(np.float32)

    visible_mask = (reveal_time <= clock[:, None])
    visible_mask[:, 0] = True
    vis = visible_mask[..., None]
    raw[..., 2:] = raw[..., 2:] * vis
    raw[..., :2] = raw[..., :2] * vis + (1.0 - vis) * 0.5

    raw_j = jnp.array(raw)
    raw_j = raw_j.at[..., :2].set(coord_normalize_visible(raw_j[..., :2], jnp.array(visible_mask)))
    H = base_model.encode(raw_j, visible_mask=jnp.array(visible_mask))
    return np.array(H)


def compute_canonical_teacher(anchor_ids, veh_feat, teacher):
    """把 teacher 映射到其等价类里最小 vehicle id（消除 idle@depot 对称噪声）。

    等价类 = (anchor_node, round(time*1e3), round(load*1e3))。idle@depot 车全部等价 → 映射到
    最小 id 的 idle 车，使 CE label 确定性（head 只需预测「某个 idle 车」，而非 OR 恰好挑的那辆）。
    真实分歧（ready@不同节点）各自成类，保留信号。
    """
    S, K = anchor_ids.shape
    N = teacher.shape[1]
    out = np.full_like(teacher, -1)
    for s in range(S):
        class_min = {}
        for k in range(K):
            key = (int(anchor_ids[s, k]),
                   int(round(float(veh_feat[s, k, 0]) * 1e3)),
                   int(round(float(veh_feat[s, k, 1]) * 1e3)))
            class_min[key] = min(class_min.get(key, k), k)
        for j in range(1, N):
            kt = int(teacher[s, j])
            if kt == -1:
                continue
            key = (int(anchor_ids[s, kt]),
                   int(round(float(veh_feat[s, kt, 0]) * 1e3)),
                   int(round(float(veh_feat[s, kt, 1]) * 1e3)))
            out[s, j] = class_min[key]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--states', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--num_steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. 数据
    states = dict(np.load(args.states))
    teacher = np.load(args.teacher)['teacher_vehicle']
    # 对称性正则化：把 teacher 映射到等价类最小 vehicle id（消除 idle@depot 噪声）
    teacher = compute_canonical_teacher(states['anchor_ids'], states['veh_feat'], teacher)
    inst_data = dict(np.load(args.data))
    tw_max = float(inst_data['tw_end'].max())
    N = inst_data['coords'].shape[1]
    S = states['instance_id'].shape[0]

    # 2. 冻结 base + head
    base_model = load_base_model(args.ckpt)
    head = FleetAssignmentHead(embed_dim=256, veh_dim=Fv, pair_dim=Fp, hidden_dim=256,
                               rngs=nnx.Rngs(args.seed))
    graphdef, params = nnx.split(head)
    tx = optax.adamw(args.lr, weight_decay=1e-2)
    opt_state = tx.init(params)

    # 3. 预计算 H（冻结 encoder）
    H_path = os.path.join(args.out, 'H.npy')
    if os.path.exists(H_path):
        H = np.load(H_path)
    else:
        H = np.zeros((S, N, 256), dtype=np.float32)
        t0 = time.time()
        for lo in range(0, S, args.batch_size):
            hi = min(lo + args.batch_size, S)
            H[lo:hi] = build_H_batch(base_model, inst_data, states, np.arange(lo, hi),
                                     tw_max, args.capacity)
            if hi % 1024 == 0 or hi == S:
                print(f"  encode H {hi}/{S} | {time.time()-t0:.0f}s", flush=True)
        np.save(H_path, H)

    # 4. 训练循环（functional nnx）
    anchor_ids = states['anchor_ids'].astype(np.int32)
    veh_feat = states['veh_feat'].astype(np.float32)
    pair_feat = states['pair_feat'].astype(np.float32)
    base_score = states['base_score'].astype(np.float32)
    cand_mask = states['candidate_mask'].astype(bool)

    valid_states = np.array([s for s in range(S) if (teacher[s] != -1).any()])
    print(f"  valid states (>=1 teacher label): {len(valid_states)}/{S}", flush=True)

    def loss_fn(params_, H_b, a_b, v_b, p_b, b_b, c_b, t_b):
        h = nnx.merge(graphdef, params_)
        _score, residual = h(H_b, a_b, v_b, p_b, b_b, c_b)
        score = jnp.where(c_b, b_b + args.alpha * residual, -1e9)
        logits = score.transpose(0, 2, 1)                    # [B,N,K]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, t_b)
        mask = (t_b != -1)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1)

    @jax.jit
    def train_step(params_, opt_state_, H_b, a_b, v_b, p_b, b_b, c_b, t_b):
        loss, grads = jax.value_and_grad(loss_fn)(params_, H_b, a_b, v_b, p_b, b_b, c_b, t_b)
        updates, new_opt = tx.update(grads, opt_state_, params_)
        return loss, optax.apply_updates(params_, updates), new_opt

    rng = np.random.default_rng(args.seed)
    for step in range(args.num_steps):
        idx = rng.choice(valid_states, size=args.batch_size, replace=True)
        loss, params, opt_state = train_step(
            params, opt_state, jnp.array(H[idx]), jnp.array(anchor_ids[idx]),
            jnp.array(veh_feat[idx]), jnp.array(pair_feat[idx]),
            jnp.array(base_score[idx]), jnp.array(cand_mask[idx]), jnp.array(teacher[idx]))
        if (step + 1) % 200 == 0:
            print(f"  step {step+1}/{args.num_steps} | loss={float(loss):.4f}", flush=True)

    # 5. 报告（Step 16：overall top-1 + disagreement top-1，分批避免 OOM）
    def predict_batched(pred_params, idx, chunk=256):
        preds = []
        for lo in range(0, len(idx), chunk):
            hi = min(lo + chunk, len(idx))
            b = idx[lo:hi]
            h = nnx.merge(graphdef, pred_params)
            _s, residual = h(jnp.array(H[b]), jnp.array(anchor_ids[b]),
                             jnp.array(veh_feat[b]), jnp.array(pair_feat[b]),
                             jnp.array(base_score[b]), jnp.array(cand_mask[b]))
            score = jnp.where(jnp.array(cand_mask[b]),
                              jnp.array(base_score[b]) + args.alpha * residual, -1e9)
            preds.append(np.array(jnp.argmax(score, axis=1)))
        return np.concatenate(preds, axis=0)

    jf1h = jnp.argmax(jnp.where(cand_mask[valid_states], base_score[valid_states], -1e9), axis=1)
    pred = predict_batched(params, valid_states)
    t_valid = teacher[valid_states]
    labeled = (t_valid != -1)
    top1 = float(((pred == t_valid) & labeled).sum() / labeled.sum())
    disagr = labeled & (t_valid != jf1h)
    disagr_top1 = float(((pred == t_valid) & disagr).sum() / disagr.sum()) if disagr.sum() > 0 else float('nan')
    print(f"\n=== JF2-M0 seed42 report ===")
    print(f"  states={len(valid_states)}  disagreement_states={int(disagr.sum())}")
    print(f"  overall top-1 (labeled) = {top1:.3%}")
    print(f"  disagreement top-1     = {disagr_top1:.3%}   <-- 关键指标")
    print(f"  alpha = {args.alpha}")

    # 保存 head params（Flax 0.10.4 无 nnx.save，用 pickle 存 nnx.State + 配置）
    import pickle
    with open(os.path.join(args.out, 'fleet_head.ckpt'), 'wb') as f:
        pickle.dump({'params': params, 'alpha': args.alpha, 'canonicalize': True,
                     'embed_dim': 256, 'veh_dim': Fv, 'pair_dim': Fp, 'hidden_dim': 256}, f)


if __name__ == '__main__':
    main()
