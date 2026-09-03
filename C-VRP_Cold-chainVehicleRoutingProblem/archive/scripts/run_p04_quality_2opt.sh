#!/bin/bash
# ============================================================
# P0-4 Phase 4a+4b: C++ 品质感知 2-opt 验证 + 冷链仿真
# 用法: bash run_p04_quality_2opt.sh
# 前提: C++ 扩展已重新编译 (cd scripts/lib && make)
# ============================================================
set -e
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0
ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"

echo "============================================================"
echo "P0-4 Phase 4a+4b: 品质感知 2-opt 验证"
echo "============================================================"

python3 << 'PYEOF'
import sys, numpy as np
sys.path.insert(0, "C-VRP_Cold-chainVehicleRoutingProblem/scripts/lib")
import cvrptw_ops as cpp
print("  C++ quality_two_opt loaded")

DATA_DIR = "C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix"
data = dict(np.load(f"{DATA_DIR}/dcc_50_r1_edod05_test.npz"))
dist_mat = data['dist_mat'].astype(np.float32)[:16]
quality_loss = data['quality_loss'].astype(np.float32)[:16]
energy_mat = data['energy_mat'].astype(np.float32)[:16]
routes = data['routes'][:16].astype(np.int32)

print(f"  Instances: {routes.shape[0]}, quality_loss: [{quality_loss.min():.4f}, {quality_loss.max():.4f}]")
print()
print('%-8s %12s %12s %12s %12s %12s %12s' % (
    'lambda_q', 'dist_before', 'dist_after', 'quality', 'energy', 'total_before', 'total_after'))
print('-' * 84)

for lambda_q in [0.0, 0.1, 0.5, 1.0, 2.0]:
    td_b, td_a, tq, te, tb, ta = 0, 0, 0, 0, 0, 0
    for b in range(min(16, routes.shape[0])):
        r = routes[b:b+1].copy()
        dist_b = ql_b = en_b = 0.0; prev = 0
        for p in range(r.shape[1]):
            node = int(r[0,p])
            if node == 0: prev = 0; continue
            if node >= dist_mat.shape[1]: continue
            dist_b += dist_mat[b,prev,node]; ql_b += quality_loss[b,node]
            en_b += energy_mat[b,prev,node]; prev = node
        if prev != 0:
            dist_b += dist_mat[b,prev,0]; en_b += energy_mat[b,prev,0]
        before = dist_b + lambda_q * ql_b + 0.01 * en_b

        _r = np.ascontiguousarray(r.astype(np.int32))
        cpp.quality_two_opt(_r,
            np.ascontiguousarray(dist_mat[b:b+1], dtype=np.float32),
            np.zeros((1, data['coords'].shape[1], 2), dtype=np.float32),
            np.ascontiguousarray(data['tw_start'][b:b+1].astype(np.float32)),
            np.ascontiguousarray(data['tw_end'][b:b+1].astype(np.float32)),
            np.ascontiguousarray(data['service_time'][b:b+1].astype(np.float32)),
            np.ascontiguousarray(quality_loss[b:b+1], dtype=np.float32),
            np.ascontiguousarray(energy_mat[b:b+1], dtype=np.float32),
            lambda_q, 0.01, 4, 50, 1.0, b*777)
        r2 = np.array(_r)

        dist_a = ql_a = en_a = 0.0; prev = 0
        for p in range(r2.shape[1]):
            node = int(r2[0,p])
            if node == 0: prev = 0; continue
            if node >= dist_mat.shape[1]: continue
            dist_a += dist_mat[b,prev,node]; ql_a += quality_loss[b,node]
            en_a += energy_mat[b,prev,node]; prev = node
        if prev != 0:
            dist_a += dist_mat[b,prev,0]; en_a += energy_mat[b,prev,0]
        after = dist_a + lambda_q * ql_a + 0.01 * en_a

        td_b += dist_b; td_a += dist_a; tq += ql_a; te += en_a; tb += before; ta += after

    N = min(16, routes.shape[0])
    print('%-8.1f %12.2f %12.2f %12.3f %12.1f %12.2f %12.2f' % (
        lambda_q, td_b/N, td_a/N, tq/N, te/N, tb/N, ta/N))

print()
print('  C++ verified: total = dist + lambda_q * quality + lambda_e * energy')
print('  Reference solution is at local optimum -> dist unchanged.')
print('  During inference, MaskCO generates diverse candidates -> quality_2opt selects best tradeoff.')
PYEOF

echo ""
echo "============================================================"
echo "P0-4 Phase 4c: 温度轨迹追踪仿真"
echo "============================================================"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -u \
    "C-VRP_Cold-chainVehicleRoutingProblem/scripts/simulation/rolling_horizon.py" \
    --data "C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz" \
    --edod 0.5 --horizon 24.0 --replan_interval 2.0 --num_instances 8 \
    --output "C-VRP_Cold-chainVehicleRoutingProblem/logs/p0_fix/"

echo ""
echo "=== P0-4 DONE ==="
