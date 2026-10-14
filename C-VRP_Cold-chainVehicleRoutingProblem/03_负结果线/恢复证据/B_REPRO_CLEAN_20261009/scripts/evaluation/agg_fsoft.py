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
from dynmaskco_cc_compact import ScoringMLP

CELLS = ['F_H', 'F_R', 'F_soft']
SEEDS = [42, 43, 44]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float('nan')


ds = load_teacher_dataset(f'{R}/results/m0_scale/dev_check', data_path=f'{R}/data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz')
npz = dict(np.load(f'{R}/data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'))
explicit, t = build_explicit_and_labels(ds, npz, 50.0, True)
legal = np.asarray(t['legal'])
sup = np.asarray(t['supervision'])
pseudo = np.asarray(t['is_pseudo'])
delta = np.asarray(t['delta'], float)
inst = np.asarray(t['instance_ids'])
N = len(set(int(x) for x in inst))


def score(cell, seed):
    ck = pickle.load(open(f'{R}/results/m0_scale/state2x2_full_v1/{cell}_s{seed}/model.ckpt', 'rb'))
    gd, _ = nnx.split(ScoringMLP(ck['in_dim'], (128, 64), rngs=0))
    return np.asarray(nnx.merge(gd, ck['params'])(jnp.asarray(explicit))).astype(np.float32), ck['s']


# g* per context
gstar = np.zeros(len(inst))
for i in range(len(inst)):
    m = sup[i]
    if m.any():
        gstar[i] = max(0.0, float(np.max(-delta[i][m])))


def threeway(cell, seed):
    sc, s = score(cell, seed)
    tj = json.load(open(f'{R}/results/m0_scale/state2x2_full_v1/eval/{cell}_s{seed}/threeway.json'))
    tau = tj['cal_selected_tau']
    per = {}
    for i in range(sc.shape[0]):
        u = int(inst[i])
        p = per.setdefault(u, [0.0, 0.0, 0.0, 0, 0.0])
        p[3] += 1
        lm = legal[i]; pm = pseudo[i]
        mod = np.flatnonzero(lm & ~pm)
        if mod.size == 0:
            continue
        w = mod[np.argmax(sc[i][mod])]
        kp = np.flatnonzero(lm & pm)
        ks = sc[i][kp[0]] if kp.size else sc[i][lm][0]
        gh = s * (sc[i][w] - ks)
        accept = gh > tau
        p[0] += gstar[i]                                                    # full
        p[1] += (max(0.0, float(-delta[i][w])) if sup[i][w] else 0.0)        # fixed
        p[2] += (float(-delta[i][w]) if (accept and sup[i][w]) else 0.0)     # actual
        if sup[i][w]:
            p[4] += abs(gh - float(-delta[i][w]))                            # |s·m − (−δ)| margin err
    full = statistics.mean([p[0]/p[3] for p in per.values()])
    fixed = statistics.mean([p[1]/p[3] for p in per.values()])
    actual = statistics.mean([p[2]/p[3] for p in per.values()])
    marg = statistics.mean([p[4]/p[3] for p in per.values()])
    return dict(tau=tau, net0=tj['dev']['rows']['0.0']['net_gain_instance_mean'],
                net_t=tj['dev']['rows'][str(tau)]['net_gain_instance_mean'],
                regret=tj['dev']['rows']['0.0']['regret_instance_mean'],
                spm=tj['dev']['spearman_mae']['rank_corr_spearman'],
                mae=tj['dev']['spearman_mae']['utility_mae'],
                full=full, fixed=fixed, actual=actual,
                pick=full-fixed, accept=fixed-actual, marg=marg)

rows = {c: [threeway(c, s) for s in SEEDS] for c in CELLS}
print(f"{'cell':7s} {'net0':>8s} {'net_t*':>8s} {'fixed_ref':>9s} {'pick':>7s} {'accept':>7s} {'marg_err':>8s} {'spm':>6s} {'mae':>6s}")
for c in CELLS:
    rs = rows[c]
    print(f"{c:7s} {mean([r['net0'] for r in rs]):+8.4f} {mean([r['net_t'] for r in rs]):+8.4f} "
          f"{mean([r['fixed'] for r in rs]):9.4f} {mean([r['pick'] for r in rs]):7.4f} "
          f"{mean([r['accept'] for r in rs]):7.4f} {mean([r['marg'] for r in rs]):8.4f} "
          f"{mean([r['spm'] for r in rs]):6.3f} {mean([r['mae'] for r in rs]):6.3f}")

print('\n逐 seed fixed_ref / margin_err:')
for c in CELLS:
    print(f"  {c}: fixed_ref=" + ', '.join(f"{r['fixed']:.4f}" for r in rows[c]) +
          f" | marg_err=" + ', '.join(f"{r['marg']:.4f}" for r in rows[c]))
print('\n配对（同 seed）:')
print(f"{'seed':5s} {'fixed(F_R-F_H)':>14s} {'fixed(F_soft-F_H)':>16s} {'marg(F_soft-F_R)':>16s} {'net_t*(F_soft-F_R)':>17s}")
for si, s in enumerate(SEEDS):
    fH = rows['F_H'][si]
    fR = rows['F_R'][si]
    fS = rows['F_soft'][si]
    print(f"{s:5d} {fR['fixed']-fH['fixed']:+14.4f} {fS['fixed']-fH['fixed']:+16.4f} "
          f"{fS['marg']-fR['marg']:+16.4f} {fS['net_t']-fR['net_t']:+17.4f}")
