"""修正版诊断（v2）：修复 4 处计算偏差，不重训 12 模型。

修正点：
  1. 实例等权聚合（net/regret 先实例内平均再实例等权），与正式评价器一致；
  2. 阈值分解用 1/(N·C_i) 加权，校验 G(0)=保留+过滤、G(tau*)=保留；
  3. 有货比例分母用 legal & ~pseudo（不含 padding）；
  4. f=y 参照的 s 从 checkpoint 读取，且不称「下限」。
"""
import sys, os, pickle, json, numpy as np, statistics, collections
R = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem'
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'expert', 'training'):
    sys.path.insert(0, os.path.join(R, 'scripts', p))
sys.path.insert(0, os.path.join(R, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(R, '..', 'MASKCO_code', 'models'))
import jax.numpy as jnp
from flax import nnx
from coldchain_teacher_dataset import load_teacher_dataset
from train_m1_compact import build_explicit_and_labels
from train_state_2x2 import build_x_state
from dynmaskco_cc_compact import ScoringMLP
from run_margin_scan import TAU_GRID, _context_decision

EPS = 1e-6
CELLS = ['F_H', 'F_R', 'S_H', 'S_R']
SEEDS = [42, 43, 44]


def _f(x):
    return 'nan' if x is None else f'{x:.4f}'


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float('nan')


# ---------- 实例等权聚合（与正式评价器一致） ----------
def decisions(scores, t, s):
    return [_context_decision(t, scores, i, s) for i in range(scores.shape[0])]


def inst_equal_net(decisions, inst, subset, tau):
    per = {}
    for i, (gh, gain, status) in enumerate(decisions):
        u = int(inst[i])
        if subset is not None and u not in subset:
            continue
        p = per.setdefault(u, [0.0, 0, 0, 0])  # [gain_sum, n_ctx, n_sf, n_lm]
        p[1] += 1
        if status == 'keep' or gh is None or gh <= tau:
            pass
        elif status == 'service_fail':
            p[2] += 1
        elif status == 'label_missing':
            p[3] += 1
        else:
            p[0] += float(gain)
    vals = [p[0] / p[1] for p in per.values() if p[1] > 0]
    n_sf = sum(p[2] for p in per.values())
    n_lm = sum(p[3] for p in per.values())
    return (statistics.mean(vals) if vals else 0.0), n_sf, n_lm


def select_tau(decisions, inst, subset, grid):
    best_tau, best_g = None, None
    for tau in grid:
        g, n_sf, n_lm = inst_equal_net(decisions, inst, subset, tau)
        if n_sf > 0 or n_lm > 0:
            continue
        if best_g is None or g > best_g + 1e-12:
            best_tau, best_g = tau, g
    return best_tau


# ---------- 一次性构建 splits ----------
SPLITS = {}
for name, td, data in [
        ('train', f'{R}/results/m0_scale/train', f'{R}/data/m0_scale/dcc_50_r1_edod05_train_teacher.npz'),
        ('cal', f'{R}/results/m0_scale/cal', f'{R}/data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz'),
        ('dev', f'{R}/results/m0_scale/dev_check', f'{R}/data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz')]:
    ds = load_teacher_dataset(td, data_path=data)
    npz = dict(np.load(data))
    tw_max = float(npz['tw_end'][:, 0].max())
    explicit, t = build_explicit_and_labels(ds, npz, 50.0, True)
    xs = build_x_state(ds, npz, 50.0, tw_max)
    # has_cargo 分组：context 是否存在至少一个 legal & ~pseudo & has_cargo 候选
    legal = np.asarray(t['legal'])
    pseudo = np.asarray(t['is_pseudo'])
    hc = xs[..., 12] > 0.5
    has_hc_ctx = np.array([bool((hc[i] & legal[i] & ~pseudo[i]).any())
                           for i in range(xs.shape[0])])
    SPLITS[name] = dict(ds=ds, npz=npz, tw_max=tw_max, explicit=explicit, t=t, xs=xs,
                        has_hc_ctx=has_hc_ctx)
    print(f"built {name}: contexts={explicit.shape[0]} M={explicit.shape[1]}", flush=True)


def score_cell(cell, seed, split):
    d = os.path.join(R, f'results/m0_scale/state2x2_full_v1/{cell}_s{seed}')
    ck = pickle.load(open(f'{d}/model.ckpt', 'rb'))
    sp = SPLITS[split]
    X = sp['explicit']
    if ck['use_state']:
        X = np.concatenate([sp['explicit'], sp['xs']], -1)
    gd, _ = nnx.split(ScoringMLP(ck['in_dim'], (128, 64), rngs=0))
    sc = np.asarray(nnx.merge(gd, ck['params'])(jnp.asarray(X))).astype(np.float32)
    return sc, ck['s']


# ---------- 表1: TRAIN 实例等权选择结果 ----------
print('\n===== 表1(v2): TRAIN 选择结果 (tau=0, 实例等权) =====')
tr_t = SPLITS['train']['t']
tr_inst = np.asarray(tr_t['instance_ids'])
tr_sup = np.asarray(tr_t['supervision'])
tr_delta = np.asarray(tr_t['delta'], float)
tr_pseudo = np.asarray(tr_t['is_pseudo'])
# g* per context
gstar = np.zeros(len(tr_inst))
for i in range(len(tr_inst)):
    m = tr_sup[i]
    if m.any():
        gstar[i] = max(0.0, float(np.max(-tr_delta[i][m])))
opp = gstar > EPS

print(f"{'cell':5s} {'seed':4s} {'net0':>7s} {'regret':>7s} {'hit':>6s} {'wrong':>6s} {'acc':>4s} {'sf/lm':>6s}")
for cell in CELLS:
    for seed in SEEDS:
        sc, s = score_cell(cell, seed, 'train')
        dec = decisions(sc, tr_t, s)
        # net0 + regret 实例等权
        net0, n_sf, n_lm = inst_equal_net(dec, tr_inst, None, 0.0)
        # regret 实例等权：per-context reg = g* - gain(τ=0)
        reg_per = {}
        n_hit = 0
        n_wrong = 0
        n_acc = 0
        for i, (gh, gain, status) in enumerate(dec):
            u = int(tr_inst[i])
            p = reg_per.setdefault(u, [0.0, 0])
            p[1] += 1
            if status == 'keep' or gh is None or gh <= 0.0:
                p[0] += float(gstar[i])
            elif gain is not None:
                p[0] += float(gstar[i] - gain)
                n_acc += 1
                if opp[i] and gain > EPS:
                    n_hit += 1
                if (not opp[i]) and gain < -EPS:
                    n_wrong += 1
        regret = statistics.mean([p[0] / p[1] for p in reg_per.values() if p[1] > 0])
        n_opp = int(opp.sum())
        hit_rate = n_hit / n_opp if n_opp else float('nan')
        wrong_rate = n_wrong / (len(tr_inst) - n_opp) if (len(tr_inst) - n_opp) else float('nan')
        print(f"{cell:5s} {seed:4d} {net0:+7.4f} {regret:7.4f} {hit_rate:6.3f} {wrong_rate:6.3f} "
              f"{n_acc:4d} {n_sf}/{n_lm}")

# f=y 参照（s 从第一个 checkpoint 读）
s_ref = pickle.load(open(f'{R}/results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt', 'rb'))['s']
y_ref = np.where(tr_sup, -tr_delta / s_ref, 0.0).astype(np.float32)
dec_ref = decisions(y_ref, tr_t, s_ref)
net0_ref, _, _ = inst_equal_net(dec_ref, tr_inst, None, 0.0)
reg_per = {}
for i, (gh, gain, status) in enumerate(dec_ref):
    u = int(tr_inst[i])
    p = reg_per.setdefault(u, [0.0, 0])
    p[1] += 1
    p[0] += float(gstar[i] - (gain if gain is not None else 0.0))
regret_ref = statistics.mean([p[0] / p[1] for p in reg_per.values() if p[1] > 0])
from train_state_2x2 import compute_loss
_, lmag_ref, lrank_ref = compute_loss(jnp.asarray(y_ref), jnp.asarray(y_ref),
                                      tr_sup, tr_pseudo, tr_delta, True, s_ref)
print(f"{'f=y':5s} {'ref':4s} {net0_ref:+7.4f} {regret_ref:7.4f} 1.000 0.000 — — "
      f"(s={s_ref:.4f}, L_mag={float(lmag_ref):.4f}, L_rank={float(lrank_ref):.4f}; "
      f"rank loss 是「幅度正确时仍非零」的参照，不是下限)")

# ---------- 表2: DEV 加权分解 + 恒等式校验 ----------
print('\n===== 表2(v2): DEV tau* 加权分解 (1/(N·C_i)) =====')
dev_t = SPLITS['dev']['t']
dev_inst = np.asarray(dev_t['instance_ids'])
dev_sup = np.asarray(dev_t['supervision'])
dev_delta = np.asarray(dev_t['delta'], float)
N_dev = len(set(int(x) for x in dev_inst))
C_i = collections.Counter(int(x) for x in dev_inst)

print(f"{'cell':5s} {'seed':4s} {'tau*':>5s} {'G(0)':>7s} {'保留贡献':>8s} {'过滤贡献':>8s} {'G(tau*)':>7s} {'核对':>5s}")
for cell in CELLS:
    for seed in SEEDS:
        tj = json.load(open(f'{R}/results/m0_scale/state2x2_full_v1/eval/{cell}_s{seed}/threeway.json'))
        tau = tj['cal_selected_tau']
        sc, s = score_cell(cell, seed, 'dev')
        dec = decisions(sc, dev_t, s)
        # G(0), G(tau*), kept, filtered 贡献（加权 1/(N·C_i)）
        kept = filtered = 0.0
        for i, (gh, gain, status) in enumerate(dec):
            u = int(dev_inst[i])
            w = 1.0 / (N_dev * C_i[u])
            if status == 'keep' or gh is None or gh <= 0.0:
                continue
            if gain is None:
                continue  # sf/lm 阻断
            if gh > tau:
                kept += float(gain) * w
            else:
                filtered += float(gain) * w
        g0 = kept + filtered
        gt = kept
        # 校验：G(0) 应与正式 evaluator 的 net@0 一致（实例等权）
        ok = 'OK' if abs(g0 - tj['dev']['rows']['0.0']['net_gain_instance_mean']) < 1e-4 else 'DRIFT'
        print(f"{cell:5s} {seed:4d} {str(tau):>5s} {g0:+7.4f} {kept:+8.4f} {filtered:+8.4f} {gt:+7.4f} {ok:>5s}")

# ---------- 表3: 有货分层 + 固定分组比较 ----------
print('\n===== 表3(v2): 有货 分层 + 固定分组比较 =====')
for name in ('train', 'cal', 'dev'):
    sp = SPLITS[name]
    t = sp['t']
    xs = sp['xs']
    legal = np.asarray(t['legal'])
    pseudo = np.asarray(t['is_pseudo'])
    sup = np.asarray(t['supervision'])
    delta = np.asarray(t['delta'], float)
    hc = xs[..., 12] > 0.5
    reg = legal & ~pseudo  # 合法普通候选（不含 padding、不含伪动作）
    cand = int(reg.sum())
    cand_hc = int((hc & reg).sum())
    ctx_hc = int(sp['has_hc_ctx'].sum())
    # 机会级：g* 最优候选是否涉及有货（平局取第一个最优）
    opp_hc = 0
    opp_tot = 0
    for i in range(xs.shape[0]):
        m = sup[i]
        if not m.any():
            continue
        best = int(np.flatnonzero(m)[np.argmax(-delta[i][m])])
        if -delta[i][best] > EPS:
            opp_tot += 1
            if hc[i][best]:
                opp_hc += 1
    print(f"  {name}: 合法普通候选有货={cand_hc}/{cand}={cand_hc/cand:.3f} | "
          f"context含合法有货={ctx_hc}/{xs.shape[0]}={ctx_hc/xs.shape[0]:.3f} | "
          f"机会涉及有货={opp_hc}/{opp_tot}={opp_hc/opp_tot if opp_tot else float('nan'):.3f}")

# 固定分组比较（TRAIN）：有货 context vs 无货 context，四格实例等权 net
print('\n  固定分组 TRAIN 实例等权 net（有货 context / 无货 context）：')
has_hc = SPLITS['train']['has_hc_ctx']
for cell in CELLS:
    vals_hc = []
    vals_no = []
    for seed in SEEDS:
        sc, s = score_cell(cell, seed, 'train')
        dec = decisions(sc, tr_t, s)
        sub_hc = set(int(u) for u in tr_inst[has_hc])
        sub_no = set(int(u) for u in tr_inst[~has_hc])
        nh, _, _ = inst_equal_net(dec, tr_inst, sub_hc, 0.0)
        nn, _, _ = inst_equal_net(dec, tr_inst, sub_no, 0.0)
        vals_hc.append(nh)
        vals_no.append(nn)
    print(f"  {cell}: 有货context net={mean(vals_hc):+.4f} ({', '.join(f'{v:+.4f}' for v in vals_hc)}) | "
          f"无货context net={mean(vals_no):+.4f} ({', '.join(f'{v:+.4f}' for v in vals_no)})")

# ---------- 校验：修订 select_tau 在完整 CAL 上 == threeway.json ----------
print('\n===== 校验: 修订 select_tau(full CAL) vs threeway.json ====')
cal_t = SPLITS['cal']['t']
cal_inst = np.asarray(cal_t['instance_ids'])
mismatch = 0
for cell in CELLS:
    for seed in SEEDS:
        tj = json.load(open(f'{R}/results/m0_scale/state2x2_full_v1/eval/{cell}_s{seed}/threeway.json'))
        sc, s = score_cell(cell, seed, 'cal')
        dec = decisions(sc, cal_t, s)
        st = select_tau(dec, cal_inst, None, TAU_GRID)
        if str(st) != str(tj['cal_selected_tau']):
            mismatch += 1
            print(f"  MISMATCH {cell}_s{seed}: revised={st} threeway={tj['cal_selected_tau']}")
print(f"  select_tau 一致: {12 - mismatch}/12")
