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

CELLS = ['F_H', 'F_R', 'S_H', 'S_R']
SEEDS = [42, 43, 44]
EPS = 1e-6

ds = load_teacher_dataset(f'{R}/results/m0_scale/cal', data_path=f'{R}/data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz')
npz = dict(np.load(f'{R}/data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz'))
tw_max = float(npz['tw_end'][:, 0].max())
explicit, t = build_explicit_and_labels(ds, npz, 50.0, True)
xs = build_x_state(ds, npz, 50.0, tw_max)
inst = np.asarray(t['instance_ids'])

print('CAL leave-one-instance (16 instances, select tau on 15, eval on 1):')
print(f"{'cell':5s} {'seed':4s} {'tau*频率':>22s} {'留出net均值':>10s} {'留出accept均值':>12s}")
for cell in CELLS:
    for seed in SEEDS:
        ck = pickle.load(open(f'{R}/results/m0_scale/state2x2_full_v1/{cell}_s{seed}/model.ckpt', 'rb'))
        X = explicit
        if ck['use_state']:
            X = np.concatenate([explicit, xs], -1)
        gd, _ = nnx.split(ScoringMLP(ck['in_dim'], (128, 64), rngs=0))
        sc = np.asarray(nnx.merge(gd, ck['params'])(jnp.asarray(X))).astype(np.float32)
        s = ck['s']
        # 每 context 决策
        C = sc.shape[0]
        dec = [_context_decision(t, sc, i, s) for i in range(C)]
        uniq = sorted(set(int(x) for x in inst))
        # 每实例每 tau 的 gain
        # 先按实例聚合 gain（τ 依赖）
        def inst_gain(insts_subset, tau):
            gains = []
            for u in insts_subset:
                gi = [0.0]
                for i in range(C):
                    if int(inst[i]) != u:
                        continue
                    gh, gain, status = dec[i]
                    if status == 'keep' or gh is None or gh <= tau:
                        gi.append(0.0)
                    elif gain is not None:
                        gi.append(float(gain))
                    # service_fail/label_missing: 不计
                gains.append(statistics.mean(gi))
            return statistics.mean(gains) if gains else 0.0
        held_gains = []
        held_acc = []
        tau_freq = collections.Counter()
        for u in uniq:
            others = [x for x in uniq if x != u]
            best_tau = None
            best_g = None
            for tau in TAU_GRID:
                g = inst_gain(others, tau)
                if best_g is None or g > best_g + 1e-12:
                    best_g, best_tau = g, tau
            tau_freq[str(best_tau)] += 1
            # 留出实例 u 在 best_tau 的 gain + accept
            gi = []
            acc = 0
            for i in range(C):
                if int(inst[i]) != u:
                    continue
                gh, gain, status = dec[i]
                if status == 'keep' or gh is None or gh <= best_tau:
                    gi.append(0.0)
                else:
                    acc += 1
                    gi.append(float(gain) if gain is not None else 0.0)
            held_gains.append(statistics.mean(gi))
            held_acc.append(acc)
        tf = ', '.join(f'{k}:{v}' for k, v in tau_freq.most_common(4))
        print(f"{cell:5s} {seed:4d} {tf:>22s} {statistics.mean(held_gains):+10.4f} {statistics.mean(held_acc):12.2f}")
