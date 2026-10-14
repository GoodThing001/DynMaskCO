"""
HFR-M0 Step 3 — 训练 GroupingHead + RouteResidualAdapter（frozen base MaskCO）。

读 Step 1 的 OR-joint teacher（teacher_routes + teacher_group_id，同一次 OR solve），构建 G/A
targets，冻结 Causal MaskCO encoder + 原 route head，只训 GroupingHead + RouteResidualAdapter。

关键（导师 §17）：L_route 必须作用于 A_joint（A_joint = A_base + ΔA + β·log_sigmoid(G)），
让 route supervision 反向塑造 G（∂L_A/∂G ≠ 0），实现真正的 G→A joint conditioning。

Loss（§16）：L = L_G + λ_r·L_A。

实现说明：
  - H        = frozen encoder 输出（预计算）。
  - A_base   = frozen route head 的 edge logits（预计算，adjmat=None + timestep=0，与推理
               hfr_replanner 完全一致，避免 train/inference mismatch；plain-loop 避免在 jit
               闭包里捕获 base nnx.Module）。
  - 训练只动 GroupingHead + RouteResidualAdapter（functional nnx，graphdef 捕获 + params 参数化）。

Sanity Gate（§24）在 held-out dev（按 instance 切）上报告：
  S1  G 有真实信号（AUPRC / positive F1 / balanced accuracy）
  S2  L_route(A_joint) < L_route(A_base)
  S3  L_route(G_real) < L_route(G_shuffle)

用法（服务器）：
    python scripts/training/train_hfr_m0.py \
      --ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
      --states results/jf2/data/r1_edod05/states_train_0.npz \
      --teacher_dir results/jf2/data/r1_edod05 \
      --data data/baseline/50_node/train/dcc_50_r1_edod05_train.npz \
      --num_steps 2000 --batch_size 64 --lr 1e-3 --seed 42 \
      --out results/hfr/m0_seed42
"""
import sys, os, argparse, time, json
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_BOOTSTRAP = os.path.dirname(_BASE)
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))
sys.path.insert(0, _BASE)

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from hierarchical_fleet_route import HierarchicalFleetRouteModel
from train_fleet_head import load_base_model, build_H_batch


def _atomic_save(path, arr):
    """原子写：先写 .tmp 再 os.replace，避免 SSH drop 留下 header 完整但数据截断的坏缓存。

    注意：np.save 对非 .npy 结尾的字符串路径会自动补 .npy（写成 .tmp.npy），故这里传文件对象
    绕过该行为；with 块负责 flush+close，再原子 rename。
    """
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, arr)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Loss（§13 / §17）
# --------------------------------------------------------------------------- #
def balanced_group_loss(logits, target, pair_mask):
    """balanced BCE：pos / neg 各占 0.5，避免 negative pair 淹没（§13）。"""
    pos = pair_mask & (target > 0.5)
    neg = pair_mask & (target < 0.5)
    log_p = jax.nn.log_sigmoid(logits)
    log_1mp = jax.nn.log_sigmoid(-logits)
    pos_loss = -jnp.sum(jnp.where(pos, log_p, 0.0)) / jnp.maximum(pos.sum(), 1.0)
    neg_loss = -jnp.sum(jnp.where(neg, log_1mp, 0.0)) / jnp.maximum(neg.sum(), 1.0)
    return 0.5 * pos_loss + 0.5 * neg_loss


def route_ce(logits, A_target):
    """route CE：softmax over columns（节点），对 A_target 的 1 边求和，按边数归一。"""
    lp = jax.nn.log_softmax(logits, axis=-1)
    n_edges = jnp.maximum(A_target.sum(), 1.0)
    return -jnp.sum(lp * A_target) / n_edges


# --------------------------------------------------------------------------- #
# Target 构造（§19 / §21）：G* 与 A* 来自同一次 OR-joint solution
# --------------------------------------------------------------------------- #
def build_targets(teacher_group_id, teacher_routes):
    """G_target = 1[same route]，A_target = 1[customer-customer 相邻]（对称）。

    只监督 mutable 客户（group_id != -1）。depot（index 0）group_id 恒 -1 自动排除。
    """
    S, N = teacher_group_id.shape
    gid = teacher_group_id
    labeled = gid != -1                       # [S, N]
    G_target = (gid[:, :, None] == gid[:, None, :]) & labeled[:, :, None] & labeled[:, None, :]
    G_target = G_target.astype(np.float32)
    eye = np.eye(N, dtype=bool)[None]
    pair_mask = labeled[:, :, None] & labeled[:, None, :] & ~eye   # [S, N, N]

    S2, K, L = teacher_routes.shape
    assert S2 == S, f"teacher_routes 与 teacher_group_id 的 state 数不一致: {S2} vs {S}"
    A_target = np.zeros((S, N, N), dtype=np.float32)
    for s in range(S):
        for k in range(K):
            custs = [int(c) for c in teacher_routes[s, k] if c != -1]
            for p in range(len(custs) - 1):
                i, j = custs[p], custs[p + 1]
                A_target[s, i, j] = 1.0
                A_target[s, j, i] = 1.0

    bad = int(((A_target > 0) & (G_target == 0)).sum())
    print(f"  targets: G[{G_target.shape}] A[{A_target.shape}]  A⇒G violation={bad}"
          f"  labeled_customers={int(labeled.sum())}")
    assert bad == 0, "QC-6 FAIL: A⇒G 必须为 0"
    return G_target, A_target, pair_mask


# --------------------------------------------------------------------------- #
# 预计算 A_base（frozen route head 的 edge logits，adjmat=None + timestep=0，与推理一致）
# --------------------------------------------------------------------------- #
def build_A_base(base_model, H, chunk=64):
    S, N, _ = H.shape
    A_base = np.zeros((S, N, N), dtype=np.float32)
    t0 = time.time()
    for lo in range(0, S, chunk):
        hi = min(lo + chunk, S)
        out = base_model.decode(jnp.array(H[lo:hi]),
                                jnp.zeros((hi - lo,), dtype=jnp.float32), None)
        A_base[lo:hi] = np.array(out)
        if hi % 2048 == 0 or hi == S:
            print(f"  decode A_base {hi}/{S} | {time.time()-t0:.0f}s", flush=True)
    return A_base


# --------------------------------------------------------------------------- #
# G 指标（§24 S1）：AUPRC / AUROC / positive F1 / balanced accuracy
# --------------------------------------------------------------------------- #
def group_metrics(logits, labels):
    """logits/labels 为扁平化后的 pair（只含 pair_mask 内的）。

    sigmoid 单调，ranking 直接用 logit；pred = logit > 0（等价 sigmoid > 0.5），
    避免极端 logit 的 exp overflow。
    """
    labels = labels.astype(bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return dict(auprc=float('nan'), auroc=float('nan'),
                    f1=float('nan'), bal_acc=float('nan'))
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    order_asc = np.argsort(logits)              # ascending：rank 1 = 最小 logit
    lab_asc = labels[order_asc].astype(np.int32)
    ranks = np.arange(1, len(lab_asc) + 1, dtype=np.float64)
    auroc = float((ranks[lab_asc == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    lab_desc = lab_asc[::-1]                    # descending：高 logit 在前
    tp = np.cumsum(lab_desc)
    fp = np.cumsum(1 - lab_desc)
    prec = tp / np.maximum(tp + fp, 1e-12)
    auprc = float((prec * lab_desc).sum() / n_pos)

    pred = logits > 0.0
    tp_ = int((pred & labels).sum()); fp_ = int((pred & ~labels).sum())
    tn_ = int((~pred & ~labels).sum()); fn_ = int((~pred & labels).sum())
    prec_ = tp_ / max(tp_ + fp_, 1); rec_ = tp_ / max(tp_ + fn_, 1)
    f1 = 2 * prec_ * rec_ / max(prec_ + rec_, 1e-12)
    bal_acc = 0.5 * (tp_ / max(tp_ + fn_, 1) + tn_ / max(tn_ + fp_, 1))
    return dict(auprc=auprc, auroc=auroc, f1=float(f1), bal_acc=float(bal_acc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='frozen base MaskCO checkpoint')
    parser.add_argument('--states', required=True, help='states_<split>_<shard>.npz')
    parser.add_argument('--teacher_dir', required=True,
                        help='含 teacher_routes.npz + teacher_group_id.npz 的目录')
    parser.add_argument('--data', required=True, help='DCC 数据（coords/demands/tw/reveal）')
    parser.add_argument('--out', required=True)
    parser.add_argument('--num_steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--lambda_r', type=float, default=1.0, help='L_A 权重（§16）')
    parser.add_argument('--beta', type=float, default=1.0, help='log_sigmoid(G) 偏置权重')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--dev_frac', type=float, default=0.2, help='按 instance 切 dev 比例（§22）')
    parser.add_argument('--max_states', type=int, default=None, help='smoke：截断 state 数')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. 数据 + teacher
    states = dict(np.load(args.states))
    inst_data = dict(np.load(args.data))
    teacher_routes = np.load(os.path.join(args.teacher_dir, 'teacher_routes.npz'))['teacher_routes']
    teacher_group_id = np.load(os.path.join(args.teacher_dir, 'teacher_group_id.npz'))['teacher_group_id']

    tw_max = float(inst_data['tw_end'].max())
    N = inst_data['coords'].shape[1]
    S_orig = states['instance_id'].shape[0]
    S = S_orig
    if args.max_states is not None:
        S = min(S, args.max_states)
        teacher_routes = teacher_routes[:S]
        teacher_group_id = teacher_group_id[:S]
        for k in list(states.keys()):
            if isinstance(states[k], np.ndarray) and states[k].shape[0] == S_orig:
                states[k] = states[k][:S]

    G_target, A_target, pair_mask = build_targets(teacher_group_id, teacher_routes)

    # 2. 冻结 base + 预计算 H / A_base（缓存加载带 shape 校验，坏缓存自动重建）
    base_model = load_base_model(args.ckpt)
    H_path = os.path.join(args.out, 'H.npy')
    H = None
    if os.path.exists(H_path):
        try:
            _H = np.load(H_path)
            if _H.shape == (S, N, 256) and _H.dtype == np.float32:
                H = _H
            else:
                print(f"  H.npy shape 不符 {_H.shape} != {(S, N, 256)} — 重新生成", flush=True)
        except (ValueError, OSError) as e:
            print(f"  H.npy 损坏（{e}）— 重新生成", flush=True)
    if H is None:
        H = np.zeros((S, N, 256), dtype=np.float32)
        t0 = time.time()
        for lo in range(0, S, args.batch_size):
            hi = min(lo + args.batch_size, S)
            H[lo:hi] = build_H_batch(base_model, inst_data, states, np.arange(lo, hi),
                                     tw_max, args.capacity)
            if hi % 1024 == 0 or hi == S:
                print(f"  encode H {hi}/{S} | {time.time()-t0:.0f}s", flush=True)
        _atomic_save(H_path, H)
    d_model = H.shape[-1]

    A_path = os.path.join(args.out, 'A_base.npy')
    A_base = None
    if os.path.exists(A_path):
        try:
            _A = np.load(A_path)
            if _A.shape == (S, N, N) and _A.dtype == np.float32:
                A_base = _A
            else:
                print(f"  A_base.npy shape 不符 {_A.shape} != {(S, N, N)} — 重新生成", flush=True)
        except (ValueError, OSError) as e:
            print(f"  A_base.npy 损坏（{e}）— 重新生成", flush=True)
    if A_base is None:
        A_base = build_A_base(base_model, H)
        _atomic_save(A_path, A_base)

    # 3. trainable heads（复用 wrapper，保证训练/推理架构一致）
    model = HierarchicalFleetRouteModel(base_model, beta=args.beta, d_model=d_model,
                                        rngs=nnx.Rngs(args.seed))
    g_graphdef, g_params = nnx.split(model.group_head)
    ra_graphdef, ra_params = nnx.split(model.route_adapter)
    params = {'group': g_params, 'adapter': ra_params}
    tx = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = tx.init(params)

    # 4. instance-level split（§22）：同 episode 的相邻 recourse state 不跨 train/dev
    rng = np.random.default_rng(args.seed)
    uniq = rng.permutation(np.unique(states['instance_id'][:S]))
    n_dev_inst = max(1, int(len(uniq) * args.dev_frac))
    dev_inst = set(uniq[:n_dev_inst].tolist())
    dev_idx = np.array([i for i in range(S) if int(states['instance_id'][i]) in dev_inst])
    train_idx = np.array([i for i in range(S) if int(states['instance_id'][i]) not in dev_inst])
    print(f"  split: train={len(train_idx)} dev={len(dev_idx)} states "
          f"(dev instances={n_dev_inst}/{len(uniq)})", flush=True)

    beta = args.beta
    lambda_r = args.lambda_r

    # 5. 训练（functional nnx：只差分组头 + adapter）
    def loss_fn(params_, H_b, A_b, G_t_b, A_t_b, pair_b):
        g = nnx.merge(g_graphdef, params_['group'])
        ra = nnx.merge(ra_graphdef, params_['adapter'])
        G_logits = g(H_b)
        delta_A = ra(H_b, G_logits)
        A_joint = A_b + delta_A + beta * jax.nn.log_sigmoid(G_logits)
        L_G = balanced_group_loss(G_logits, G_t_b, pair_b)
        L_A = route_ce(A_joint, A_t_b)
        return L_G + lambda_r * L_A, (L_G, L_A)

    @jax.jit
    def train_step(params_, opt_state_, H_b, A_b, G_t_b, A_t_b, pair_b):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params_, H_b, A_b, G_t_b, A_t_b, pair_b)
        updates, new_opt = tx.update(grads, opt_state_, params_)
        return loss, aux, optax.apply_updates(params_, updates), new_opt

    def _arr(a):
        return jnp.asarray(a)

    t0 = time.time()
    for step in range(args.num_steps):
        idx = rng.choice(train_idx, size=args.batch_size, replace=True)
        loss, aux, params, opt_state = train_step(
            params, opt_state, _arr(H[idx]), _arr(A_base[idx]),
            _arr(G_target[idx]), _arr(A_target[idx]), _arr(pair_mask[idx]))
        if (step + 1) % 200 == 0:
            print(f"  step {step+1}/{args.num_steps} | loss={float(loss):.4f} "
                  f"L_G={float(aux[0]):.4f} L_A={float(aux[1]):.4f} | {time.time()-t0:.0f}s",
                  flush=True)

    # 6. 报告（dev 上 S1/S2/S3，分 chunk 避免 OOM）
    @jax.jit
    def eval_step(params_, H_b, A_b, G_t_b, A_t_b, pair_b, perm):
        g = nnx.merge(g_graphdef, params_['group'])
        ra = nnx.merge(ra_graphdef, params_['adapter'])
        G_logits = g(H_b)
        delta = ra(H_b, G_logits)
        A_joint = A_b + delta + beta * jax.nn.log_sigmoid(G_logits)
        B = H_b.shape[0]
        G_sh = G_logits[jnp.arange(B)[:, None], perm]   # [B,N,N]，行置换破坏 (i,j) 配对
        delta_sh = ra(H_b, G_sh)
        A_sh = A_b + delta_sh + beta * jax.nn.log_sigmoid(G_sh)
        return (G_logits, balanced_group_loss(G_logits, G_t_b, pair_b),
                route_ce(A_b, A_t_b), route_ce(A_joint, A_t_b), route_ce(A_sh, A_t_b))

    G_list = []
    l_g = l_a_base = l_a_joint = l_a_shuf = 0.0
    n_chunk = 0
    for lo in range(0, len(dev_idx), 256):
        hi = min(lo + 256, len(dev_idx))
        b = dev_idx[lo:hi]
        perm = np.stack([rng.permutation(N) for _ in range(hi - lo)]).astype(np.int32)
        Glog, lg, lab, laj, lash = eval_step(
            params, _arr(H[b]), _arr(A_base[b]), _arr(G_target[b]),
            _arr(A_target[b]), _arr(pair_mask[b]), jnp.array(perm))
        G_list.append(np.array(Glog))
        l_g += float(lg); l_a_base += float(lab); l_a_joint += float(laj); l_a_shuf += float(lash)
        n_chunk += 1
    l_g /= n_chunk; l_a_base /= n_chunk; l_a_joint /= n_chunk; l_a_shuf /= n_chunk

    # S1：G 指标（扁平化全部 dev pairs）
    G_all = np.concatenate(G_list, axis=0)
    lab = (G_target[dev_idx] > 0.5)
    pm = pair_mask[dev_idx]
    m_ = group_metrics(G_all[pm], lab[pm])          # 直接用 logit，避免 sigmoid overflow

    print(f"\n=== HFR-M0 report (dev {len(dev_idx)} states) ===")
    print(f"  S1 G:  AUPRC={m_['auprc']:.4f}  AUROC={m_['auroc']:.4f}  "
          f"F1={m_['f1']:.4f}  bal_acc={m_['bal_acc']:.4f}")
    print(f"  L_G            = {l_g:.4f}")
    print(f"  S2 route: L_A(base)={l_a_base:.4f}  L_A(joint)={l_a_joint:.4f}  "
          f"Δ={l_a_joint - l_a_base:+.4f}  {'PASS' if l_a_joint < l_a_base else 'FAIL'}")
    print(f"  S3 shuffle: L_A(real)={l_a_joint:.4f}  L_A(shuffle)={l_a_shuf:.4f}  "
          f"Δ={l_a_shuf - l_a_joint:+.4f}  {'PASS' if l_a_joint < l_a_shuf else 'FAIL'}")

    # 7. 保存（§43：wrapper 不 merge 回原 ckpt）
    import pickle
    with open(os.path.join(args.out, 'group_head.ckpt'), 'wb') as f:
        pickle.dump({'params': params['group'], 'd_model': d_model, 'hidden': 256}, f)
    with open(os.path.join(args.out, 'route_adapter.ckpt'), 'wb') as f:
        pickle.dump({'params': params['adapter'], 'd_model': d_model, 'hidden': 128}, f)
    config = dict(ckpt=args.ckpt, states=args.states, teacher_dir=args.teacher_dir,
                  data=args.data, seed=args.seed, beta=beta, lambda_r=lambda_r,
                  num_steps=args.num_steps, batch_size=args.batch_size, lr=args.lr,
                  dev_frac=args.dev_frac, d_model=d_model, num_nodes=N)
    manifest = dict(config=config, base_maskco_ref=args.ckpt, seed=args.seed,
                    n_states=S, n_train=len(train_idx), n_dev=len(dev_idx),
                    dev_instances=sorted(int(i) for i in dev_inst),
                    sanity=dict(S1=m_, L_G=round(l_g, 4),
                                S2=dict(base=round(l_a_base, 4), joint=round(l_a_joint, 4),
                                        pass_=bool(l_a_joint < l_a_base)),
                                S3=dict(real=round(l_a_joint, 4), shuffle=round(l_a_shuf, 4),
                                        pass_=bool(l_a_joint < l_a_shuf))))
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  saved: {args.out}/group_head.ckpt + route_adapter.ckpt + config.json + manifest.json")


if __name__ == '__main__':
    main()
