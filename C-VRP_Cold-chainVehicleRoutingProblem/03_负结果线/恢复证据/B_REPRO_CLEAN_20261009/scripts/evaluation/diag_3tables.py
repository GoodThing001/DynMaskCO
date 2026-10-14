import sys, os, pickle, json, numpy as np, statistics
R = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem'
for p in ('models','data','simulation','evaluation','baselines','coldchain','expert','training'):
    sys.path.insert(0, os.path.join(R, 'scripts', p))
sys.path.insert(0, os.path.join(R, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(R, '..', 'MASKCO_code', 'models'))
import jax.numpy as jnp
from flax import nnx
from coldchain_teacher_dataset import load_teacher_dataset
from train_m1_compact import build_explicit_and_labels
from train_state_2x2 import build_x_state, compute_loss
from dynmaskco_cc_compact import ScoringMLP

EPS = 1e-6
CELLS = ['F_H', 'F_R', 'S_H', 'S_R']
SEEDS = [42, 43, 44]


def _f(x):
    return 'nan' if x is None else f'{x:.4f}'


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float('nan')


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
    SPLITS[name] = dict(ds=ds, npz=npz, tw_max=tw_max, explicit=explicit, t=t, xs=xs)
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


def outcomes(scores, t, s):
    legal = np.asarray(t['legal'])
    sup = np.asarray(t['supervision'])
    pseudo = np.asarray(t['is_pseudo'])
    delta = np.asarray(t['delta'], float)
    svo = np.asarray(t['service_ok'])
    C, M = scores.shape
    gs = np.zeros(C)
    for i in range(C):
        m = sup[i]
        if m.any():
            gs[i] = max(0.0, float(np.max(-delta[i][m])))
    opp = gs > EPS
    per = []
    for i in range(C):
        lm = legal[i]
        pm = pseudo[i]
        mod = np.flatnonzero(lm & ~pm)
        if mod.size == 0:
            per.append(dict(gain=0., accept=False, reg=gs[i], opp=opp[i], hit=False,
                            wrong=False, sf=False, lm2=False, g_hat=-1.0, w=-1))
            continue
        w = mod[np.argmax(scores[i][mod])]
        kp = np.flatnonzero(lm & pm)
        ks = scores[i][kp[0]] if kp.size else scores[i][lm][0]
        gh = s * (scores[i][w] - ks)
        accept = gh > 0
        if accept and sup[i][w]:
            gain = -delta[i][w]
            reg = gs[i] - gain
            hit = opp[i] and gain > EPS
            wrong = (not opp[i]) and gain < -EPS
            sf = lm2 = False
        elif accept:
            gain = 0.
            reg = gs[i]
            hit = wrong = False
            sf = not svo[i][w]
            lm2 = svo[i][w]
        else:
            gain = 0.
            reg = gs[i]
            hit = wrong = sf = lm2 = False
        per.append(dict(gain=gain, accept=accept, reg=reg, opp=opp[i], hit=hit, wrong=wrong,
                        sf=sf, lm2=lm2, g_hat=gh, w=int(w)))
    return per, gs, opp


def agg(per):
    C = len(per)
    net = statistics.mean([p['gain'] for p in per])
    reg = statistics.mean([p['reg'] for p in per])
    n_acc = sum(p['accept'] for p in per)
    ben = sum(p['accept'] and p['gain'] > EPS for p in per)
    neu = sum(p['accept'] and abs(p['gain']) <= EPS for p in per)
    harm = sum(p['accept'] and p['gain'] < -EPS for p in per)
    pos = sum(p['gain'] for p in per if p['gain'] > EPS)
    neg = sum(p['gain'] for p in per if p['gain'] < -EPS)
    n_opp = int(sum(p['opp'] for p in per))
    hit = sum(p['hit'] for p in per)
    n_nopp = C - n_opp
    wrong = sum(p['wrong'] for p in per)
    sf = sum(p['sf'] for p in per)
    lm2 = sum(p['lm2'] for p in per)
    return dict(net=net, reg=reg, n_acc=n_acc, ben=ben, neu=neu, harm=harm, pos=pos, neg=neg,
                hit_rate=(hit / n_opp if n_opp else None),
                wrong_rate=(wrong / n_nopp if n_nopp else None),
                n_opp=n_opp, n_nopp=n_nopp, sf=sf, lm2=lm2)


# ===== 表1: TRAIN 选择结果 =====
print('\n===== 表1: TRAIN 选择结果 (tau=0) =====')
hdr = f"{'cell':5s} {'seed':4s} {'net0':>7s} {'regret':>7s} {'acc':>4s} {'b/n/h':>9s} {'pos':>6s} {'neg':>6s} {'hit':>6s} {'wrong':>6s} {'sf/lm':>6s}"
print(hdr)
for cell in CELLS:
    for seed in SEEDS:
        sc, s = score_cell(cell, seed, 'train')
        per, gs, opp = outcomes(sc, SPLITS['train']['t'], s)
        a = agg(per)
        bnh = f"{a['ben']}/{a['neu']}/{a['harm']}"
        print(f"{cell:5s} {seed:4d} {a['net']:+7.4f} {a['reg']:7.4f} {a['n_acc']:4d} "
              f"{bnh:>9s} {a['pos']:+6.3f} {a['neg']:+6.3f} {_f(a['hit_rate']):>6s} "
              f"{_f(a['wrong_rate']):>6s} {a['sf']}/{a['lm2']}")

# f=y 参照
t = SPLITS['train']['t']
y = np.where(np.asarray(t['supervision']), -np.asarray(t['delta']) / 0.0392, 0.0)
per_ref, gs_ref, opp_ref = outcomes(y.astype(np.float32), t, 0.0392)
a_ref = agg(per_ref)
print(f"{'f=y':5s} {'ref':4s} {a_ref['net']:+7.4f} {a_ref['reg']:7.4f} {a_ref['n_acc']:4d} "
      f"{a_ref['ben']}/{a_ref['neu']}/{a_ref['harm']:>3d} {a_ref['pos']:+6.3f} {a_ref['neg']:+6.3f} "
      f"{_f(a_ref['hit_rate']):>6s} {_f(a_ref['wrong_rate']):>6s} {a_ref['sf']}/{a_ref['lm2']}")
_, lmag_ref, lrank_ref = compute_loss(jnp.asarray(y), jnp.asarray(y),
                                      np.asarray(t['supervision']), np.asarray(t['is_pseudo']),
                                      np.asarray(t['delta']), True, 0.0392)
print(f"  f=y 参照: L_mag={float(lmag_ref):.4f} L_rank={float(lrank_ref):.4f} (幅度完美时 rank loss 的下限)")

# ===== 表2: DEV tau* 分解 =====
print('\n===== 表2: DEV tau* 分解 (被保留 vs 被过滤) =====')
for cell in CELLS:
    for seed in SEEDS:
        tj = json.load(open(f'{R}/results/m0_scale/state2x2_full_v1/eval/{cell}_s{seed}/threeway.json'))
        tau = tj['cal_selected_tau']
        sc, s = score_cell(cell, seed, 'dev')
        t = SPLITS['dev']['t']
        legal = np.asarray(t['legal'])
        sup = np.asarray(t['supervision'])
        pseudo = np.asarray(t['is_pseudo'])
        delta = np.asarray(t['delta'], float)
        kept = {'n': 0, 'pos': 0., 'neg': 0.}
        filt = {'n': 0, 'pos': 0., 'neg': 0.}
        for i in range(sc.shape[0]):
            lm = legal[i]
            pm = pseudo[i]
            mod = np.flatnonzero(lm & ~pm)
            if mod.size == 0:
                continue
            w = mod[np.argmax(sc[i][mod])]
            kp = np.flatnonzero(lm & pm)
            ks = sc[i][kp[0]] if kp.size else sc[i][lm][0]
            gh = s * (sc[i][w] - ks)
            if gh <= 0:
                continue
            gain = -delta[i][w] if sup[i][w] else 0.
            target = kept if gh > tau else filt
            target['n'] += 1
            if gain > EPS:
                target['pos'] += gain
            elif gain < -EPS:
                target['neg'] += gain
        print(f"  {cell}_s{seed} tau*={tau}: 保留 n={kept['n']} pos={kept['pos']:+.3f} neg={kept['neg']:+.3f}"
              f" | 过滤 n={filt['n']} pos={filt['pos']:+.3f} neg={filt['neg']:+.3f}")

# ===== 表3: 有货分层 =====
print('\n===== 表3: 有货 分层 =====')
for name in ('train', 'cal', 'dev'):
    sp = SPLITS[name]
    t = sp['t']
    xs = sp['xs']
    pseudo = np.asarray(t['is_pseudo'])
    sup = np.asarray(t['supervision'])
    delta = np.asarray(t['delta'], float)
    legal = np.asarray(t['legal'])
    has_cargo = xs[..., 12] > 0.5
    nonp = ~pseudo
    cand = int(nonp.sum())
    cand_hc = int((has_cargo & nonp).sum())
    ctx_hc = 0
    for i in range(xs.shape[0]):
        if (has_cargo[i] & nonp[i] & legal[i]).any():
            ctx_hc += 1
    opp_hc = 0
    opp_tot = 0
    for i in range(xs.shape[0]):
        m = sup[i]
        if not m.any():
            continue
        best = int(np.flatnonzero(m)[np.argmax(-delta[i][m])])
        if -delta[i][best] > EPS:
            opp_tot += 1
            if has_cargo[i][best]:
                opp_hc += 1
    print(f"  {name}: 候选有货={cand_hc}/{cand}={cand_hc/cand:.3f} | "
          f"context含合法有货={ctx_hc}/{xs.shape[0]}={ctx_hc/xs.shape[0]:.3f} | "
          f"机会中涉及有货={opp_hc}/{opp_tot}={opp_hc/opp_tot if opp_tot else float('nan'):.3f}")
