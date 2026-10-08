#!/bin/bash
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
/home/hzeng/miniconda3/envs/MASKCO_env/bin/python - 2>/dev/null <<'EOF'
import sys, os
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'scripts/evaluation'); sys.path.insert(0, 'scripts/models')
sys.path.insert(0, 'scripts/coldchain'); sys.path.insert(0, 'scripts/simulation')
from dataclasses import replace
from mpre import load_cvrp_model
cvrp, cfg, step = load_cvrp_model('../MASKCO_code/ckpts/cvrp100.ckpt')
rc = replace(cfg, rngs=42).construct_model()
print('random_cvrp construct OK, type=', type(rc).__name__, 'dtype=', getattr(rc, 'dtype', None))
# 与 pretrained 参数量一致（同架构）
import flax
a = flax.core.FrozenDict({})  # noqa: F841
n_pt = sum(x.size for x in __import__('jax').tree.leaves(
    __import__('flax').nnx.state(cvrp).to_pure_dict()))
n_rc = sum(x.size for x in __import__('jax').tree.leaves(
    __import__('flax').nnx.state(rc).to_pure_dict()))
print('param counts: pretrained=%d random_cvrp=%d equal=%s' % (n_pt, n_rc, n_pt == n_rc))
EOF
