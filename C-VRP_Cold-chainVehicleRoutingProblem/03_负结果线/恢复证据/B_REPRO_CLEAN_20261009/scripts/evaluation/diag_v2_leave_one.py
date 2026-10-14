"""修正版 CAL 留一（v2）：去掉 spurious 零 context，复用正式 select_tau 规则。"""
import sys, os, statistics, collections, numpy as np
R = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem'
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain', 'expert', 'training'):
    sys.path.insert(0, os.path.join(R, 'scripts', p))
sys.path.insert(0, os.path.join(R, '..', 'MASKCO_code'))
sys.path.insert(0, os.path.join(R, '..', 'MASKCO_code', 'models'))
from diag_v2 import select_tau, decisions, score_cell, SPLITS, inst_equal_net
from run_margin_scan import TAU_GRID

CELLS = ['F_H', 'F_R', 'S_H', 'S_R']
SEEDS = [42, 43, 44]

cal_t = SPLITS['cal']['t']
cal_inst = np.asarray(cal_t['instance_ids'])
uniq = sorted(set(int(x) for x in cal_inst))

print('CAL leave-one (v2): 15 选 tau / 1 评，复用 select_tau（实例等权 + 排除失败）')
print(f"{'cell':5s} {'seed':4s} {'tau*频率':>24s} {'留出net均值':>10s} {'留出accept均值':>12s}")
for cell in CELLS:
    for seed in SEEDS:
        sc, s = score_cell(cell, seed, 'cal')
        dec = decisions(sc, cal_t, s)
        tau_freq = collections.Counter()
        held_gains = []
        held_acc = []
        for u in uniq:
            others = set(x for x in uniq if x != u)
            st = select_tau(dec, cal_inst, others, TAU_GRID)
            tau_freq[str(st)] += 1
            g, _, _ = inst_equal_net(dec, cal_inst, {u}, st)
            acc = 0
            for i, (gh, gain, status) in enumerate(dec):
                if int(cal_inst[i]) != u:
                    continue
                if status != 'keep' and gh is not None and gh > st:
                    acc += 1
            held_gains.append(g)
            held_acc.append(acc)
        tf = ', '.join(f'{k}:{v}' for k, v in tau_freq.most_common(4))
        print(f"{cell:5s} {seed:4d} {tf:>24s} {statistics.mean(held_gains):+10.4f} "
              f"{statistics.mean(held_acc):12.2f}")
