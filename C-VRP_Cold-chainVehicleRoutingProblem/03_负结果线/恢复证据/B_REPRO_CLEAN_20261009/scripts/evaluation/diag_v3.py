"""诊断 v3：修正有货/无货 context 分层 + 三路收益分解（选哪个 vs 接不接受）。

不重训，只用现有 12 个 checkpoint 的预测。
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
from run_margin_scan import _context_decision

EPS = 1e-6
CELLS = ['F_H', 'F_R', 'S_H', 'S_R']
SEEDS = [42, 43, 44]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float('nan')


# ---------- 构建 splits ----------
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
    legal = np.asarray(t['legal'])
    pseudo = np.asarray(t['is_pseudo'])
    hc = xs[..., 12] > 0.5
    has_hc_ctx = np.array([bool((hc[i] & legal[i] & ~pseudo[i]).any())
                           for i in range(xs.shape[0])])
    SPLITS[name] = dict(ds=ds, npz=npz, tw_max=tw_max, explicit=explicit, t=t, xs=xs,
                        has_hc_ctx=has_hc_ctx)
    print(f"built {name}: contexts={explicit.shape[0]}", flush=True)


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


# 每 context 分析：g_full(完整hindsight) / g_fixed(固定首选+事后完美接受) / g_actual(实际阈值)
def context_analysis(scores, t, s, tau):
    legal = np.asarray(t['legal'])
    sup = np.asarray(t['supervision'])
    pseudo = np.asarray(t['is_pseudo'])
    delta = np.asarray(t['delta'], float)
    inst = np.asarray(t['instance_ids'])
    C = scores.shape[0]
    gstar = np.zeros(C)
    for i in range(C):
        m = sup[i]
        if m.any():
            gstar[i] = max(0.0, float(np.max(-delta[i][m])))
    out = []
    for i in range(C):
        lm = legal[i]
        pm = pseudo[i]
        mod = np.flatnonzero(lm & ~pm)
        if mod.size == 0:
            out.append(dict(g_full=gstar[i], g_fixed=0.0, g_actual=0.0, inst=int(inst[i]),
                            hc=False, sf=0, lm2=0))
            continue
        w = mod[np.argmax(scores[i][mod])]
        kp = np.flatnonzero(lm & pm)
        ks = scores[i][kp[0]] if kp.size else scores[i][lm][0]
        gh = s * (scores[i][w] - ks)
        accept = gh > tau
        # g_full
        g_full = gstar[i]
        # g_fixed: 固定首选 w，事后完美接受（真实收益>0 才接受）
        if sup[i][w]:
            g_fixed = max(0.0, float(-delta[i][w]))
        else:
            g_fixed = 0.0
        # g_actual
        if accept and sup[i][w]:
            g_actual = float(-delta[i][w])
        else:
            g_actual = 0.0
        sf = 1 if (accept and not sup[i][w] and not t['service_ok'][i][w]) else 0
        lm2 = 1 if (accept and not sup[i][w] and t['service_ok'][i][w]) else 0
        out.append(dict(g_full=g_full, g_fixed=g_fixed, g_actual=g_actual, inst=int(inst[i]),
                        hc=bool(SPLITS.get('_hc', [False])[0]), sf=sf, lm2=lm2))
    return out


def inst_equal_of(rows, key):
    per = {}
    for r in rows:
        p = per.setdefault(r['inst'], [0.0, 0])
        p[0] += float(r[key])
        p[1] += 1
    vals = [p[0] / p[1] for p in per.values() if p[1] > 0]
    return statistics.mean(vals) if vals else 0.0


# ===== 1. 修正分层：有货/无货 context 对总收益的贡献（1/(N·C_i) 权重，τ=0）=====
print('\n===== v3-1: 有货/无货 context 贡献 (TRAIN/CAL/DEV, tau=0, 加权 1/(N*C_i)) =====')
for split in ('train', 'cal', 'dev'):
    sp = SPLITS[split]
    t = sp['t']
    inst = np.asarray(t['instance_ids'])
    N = len(set(int(x) for x in inst))
    C_i = collections.Counter(int(x) for x in inst)
    has_hc = sp['has_hc_ctx']
    print(f'  [{split}] N={N}')
    for cell in CELLS:
        vals_hc = []
        vals_no = []
        for seed in SEEDS:
            sc, s = score_cell(cell, seed, split)
            dec = [_context_decision(t, sc, i, s) for i in range(sc.shape[0])]
            cont_hc = 0.0
            cont_no = 0.0
            for i, (gh, gain, status) in enumerate(dec):
                if status == 'keep' or gh is None or gh <= 0.0:
                    continue
                if gain is None:
                    continue
                w = 1.0 / (N * C_i[int(inst[i])])
                if has_hc[i]:
                    cont_hc += float(gain) * w
                else:
                    cont_no += float(gain) * w
            vals_hc.append(cont_hc)
            vals_no.append(cont_no)
        print(f'    {cell}: 有货贡献={mean(vals_hc):+.4f}  无货贡献={mean(vals_no):+.4f}  '
              f'(和={mean([h+n for h,n in zip(vals_hc,vals_no)]):+.4f})')

# ===== 1b. 有货子集内的 S−F 配对差异（同一固定 context 子集）=====
print('\n===== v3-1b: 有货 context 子集内 S−F 配对差异 (per-context, 加权) =====')
for split in ('train', 'dev'):
    sp = SPLITS[split]
    t = sp['t']
    inst = np.asarray(t['instance_ids'])
    N = len(set(int(x) for x in inst))
    C_i = collections.Counter(int(x) for x in inst)
    has_hc = sp['has_hc_ctx']
    # 对 S 组与 F 组，逐 context 算 gain（τ=0）
    def ctx_gain_map(cell):
        g = {}
        for seed in SEEDS:
            sc, s = score_cell(cell, seed, split)
            for i in range(sc.shape[0]):
                gh, gain, status = _context_decision(t, sc, i, s)
                gv = 0.0
                if status != 'keep' and gh is not None and gh > 0.0 and gain is not None:
                    gv = float(gain)
                g[(seed, i)] = gv
        return g
    gF_H = ctx_gain_map('F_H'); gS_H = ctx_gain_map('S_H')
    gF_R = ctx_gain_map('F_R'); gS_R = ctx_gain_map('S_R')
    for label, gS, gF in [('S_H-F_H (Huber)', gS_H, gF_H), ('S_R-F_R (rank)', gS_R, gF_R)]:
        diffs = []
        for (seed, i) in gS:
            if not has_hc[i]:
                continue
            w = 1.0 / (N * C_i[int(inst[i])])
            diffs.append((gS[(seed, i)] - gF[(seed, i)]) * w)
        print(f'  [{split}] {label}: 有货子集加权配对差={statistics.mean(diffs):+.4f} '
              f'(逐seed: ' + ', '.join(
                  f'{statistics.mean([(gS[(s,i)]-gF[(s,i)])*w for (s2,i) in gS if s2==s and has_hc[i]]):+.4f}'
                  for s in SEEDS) + ')')

# ===== 2. 三路收益分解（DEV，τ=*）=====
print('\n===== v3-2: 三路收益分解 (DEV, tau*=CAL阈值, 实例等权) =====')
dev_sp = SPLITS['dev']
dev_t = dev_sp['t']
dev_inst = np.asarray(dev_t['instance_ids'])
dev_sup = np.asarray(dev_t['supervision'])
dev_delta = np.asarray(dev_t['delta'], float)
dev_legal = np.asarray(dev_t['legal'])
dev_pseudo = np.asarray(dev_t['is_pseudo'])
dev_svo = np.asarray(dev_t['service_ok'])
# g* per context
gstar = np.zeros(len(dev_inst))
for i in range(len(dev_inst)):
    m = dev_sup[i]
    if m.any():
        gstar[i] = max(0.0, float(np.max(-dev_delta[i][m])))

print(f"{'cell':5s} {'seed':4s} {'full_ref':>9s} {'fixed_ref':>9s} {'actual':>8s} {'pick_loss':>9s} {'accept_loss':>10s}")
for cell in CELLS:
    for seed in SEEDS:
        tj = json.load(open(f'{R}/results/m0_scale/state2x2_full_v1/eval/{cell}_s{seed}/threeway.json'))
        tau = tj['cal_selected_tau']
        sc, s = score_cell(cell, seed, 'dev')
        per = {}
        for i in range(sc.shape[0]):
            u = int(dev_inst[i])
            p = per.setdefault(u, [0.0, 0.0, 0.0, 0])
            p[3] += 1
            lm = dev_legal[i]; pm = dev_pseudo[i]
            mod = np.flatnonzero(lm & ~pm)
            if mod.size == 0:
                continue
            w_idx = mod[np.argmax(sc[i][mod])]
            kp = np.flatnonzero(lm & pm)
            ks = sc[i][kp[0]] if kp.size else sc[i][lm][0]
            gh = s * (sc[i][w_idx] - ks)
            accept = gh > tau
            # full
            p[0] += gstar[i]
            # fixed
            p[1] += (max(0.0, float(-dev_delta[i][w_idx])) if dev_sup[i][w_idx] else 0.0)
            # actual
            p[2] += (float(-dev_delta[i][w_idx]) if (accept and dev_sup[i][w_idx]) else 0.0)
        full = statistics.mean([p[0]/p[3] for p in per.values()])
        fixed = statistics.mean([p[1]/p[3] for p in per.values()])
        actual = statistics.mean([p[2]/p[3] for p in per.values()])
        print(f"{cell:5s} {seed:4d} {full:9.4f} {fixed:9.4f} {actual:8.4f} "
              f"{full-fixed:+9.4f} {fixed-actual:+10.4f}")

# ===== 3. 独立 G(tau*) 对账 =====
print('\n===== v3-3: 独立 G(tau*) 对账 =====')
for cell in CELLS:
    for seed in SEEDS:
        tj = json.load(open(f'{R}/results/m0_scale/state2x2_full_v1/eval/{cell}_s{seed}/threeway.json'))
        tau = tj['cal_selected_tau']
        sc, s = score_cell(cell, seed, 'dev')
        dec = [_context_decision(dev_t, sc, i, s) for i in range(sc.shape[0])]
        N = len(set(int(x) for x in dev_inst))
        C_i = collections.Counter(int(x) for x in dev_inst)
        kept = 0.0
        for i, (gh, gain, status) in enumerate(dec):
            if status == 'keep' or gh is None or gh <= 0.0:
                continue
            if gain is None:
                continue
            if gh > tau:
                kept += float(gain) / (N * C_i[int(dev_inst[i])])
        formal_gt = tj['dev']['rows'][str(tau)]['net_gain_instance_mean']
        if abs(kept - formal_gt) > 1e-4:
            print(f"  MISMATCH {cell}_s{seed}: computed G(tau*)={kept:+.4f} formal={formal_gt:+.4f}")
print("  G(tau*) 独立对账完成（无输出即全部一致）")
