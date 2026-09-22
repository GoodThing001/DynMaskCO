"""MaxIterations 扫描：只用于确定距离收敛 / 耗时 / feasible / deferred /
warning / 确定性——不按 J_CC 选值（PyVRP-RH-D 优化的是距离）。"""
import json, os, sys, time
_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401
import numpy as np
from strict_online_runner import run_instance, load_objective_profile, dataset_hash
from pyvrp_adapter import PyVRPRHDAdapter

DATA_DIR = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
            r'\data\baseline\50_node\dev_proto')
PROFILE = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
           r'\results\o0cc\scale_v2\objective_profile.json')
OUT = os.path.join(_DCC_VRP, 'results', 'iter_sweep.json')
SWEEP = (100, 300, 1000, 3000)
CASES = [('R1', 0.5, 0), ('C1', 0.2, 0), ('RC1', 0.8, 0)]

results = []
for typ, ed, inst in CASES:
    fname = f'dcc_50_{typ.lower()}_edod{str(ed).replace(".", "")}_dev_proto.npz'
    ds = dict(np.load(os.path.join(DATA_DIR, fname)))
    dh = dataset_hash(ds)
    profile = load_objective_profile(PROFILE)
    row = {'cell': f'{typ}_edod{ed}', 'inst': inst, 'iterations': []}
    for it in SWEEP:
        t0 = time.time()
        rec = run_instance(ds, 50, 25,
                           adapter_factory=lambda: PyVRPRHDAdapter(
                               max_iterations=it, seed=0),
                           inst_idx=inst, objective='coldchain',
                           profile=profile, seed=0, data_sha256=dh,
                           instance_seed=inst)
        row['iterations'].append({
            'max_iterations': it,
            'distance_cost': float(rec['outcome']['distance_cost']),
            'complete': bool(rec['outcome']['complete']),
            'n_solver_calls': rec['stats']['n_solver_calls'],
            'n_unplanned_customer_events':
                rec['stats']['n_unplanned_customer_events'],
            'n_penalty_warnings': rec['stats']['n_penalty_warnings'],
            'runtime_s': round(time.time() - t0, 2),
            'decision_hash': rec['decision_hash'],
        })
        print(f"  [{row['cell']} inst{inst}] it={it:5d} dist="
              f"{rec['outcome']['distance_cost']:.3f} "
              f"complete={rec['outcome']['complete']} "
              f"deferred_ev={rec['stats']['n_unplanned_customer_events']} "
              f"warn={rec['stats']['n_penalty_warnings']} "
              f"t={time.time()-t0:.1f}s", flush=True)
    results.append(row)
    # 确定性：同 iterations 重跑一次对 decision_hash
    rec2 = run_instance(ds, 50, 25,
                        adapter_factory=lambda: PyVRPRHDAdapter(
                            max_iterations=SWEEP[-1], seed=0),
                        inst_idx=inst, objective='coldchain',
                        profile=profile, seed=0, data_sha256=dh,
                        instance_seed=inst)
    det = rec2['decision_hash'] == row['iterations'][-1]['decision_hash']
    assert det, f'[{row["cell"]}] 3000 迭代确定性失败'
    print(f"  [{row['cell']}] determinism(3000): True", flush=True)

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\nsaved {OUT}')
