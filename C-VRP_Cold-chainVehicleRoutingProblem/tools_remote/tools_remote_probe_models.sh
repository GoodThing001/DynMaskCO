#!/bin/bash
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
/home/hzeng/miniconda3/envs/MASKCO_env/bin/python - 2>/dev/null <<'EOF'
import sys, os
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/evaluation')
sys.path.insert(0, 'scripts/models'); sys.path.insert(0, 'scripts/coldchain')
sys.path.insert(0, 'scripts/simulation')
import numpy as np
from flax import nnx
from mpre import load_cvrp_model
from maskco_scenario import MaskCOScenarioModel, load_model

cvrp, cfg, _ = load_cvrp_model('../MASKCO_code/ckpts/cvrp100.ckpt')

def trainable_map(m):
    fs = nnx.state(m, nnx.Param).flat_state()
    return {str(p): np.asarray(v.value) for p, v in zip(fs.paths, fs.leaves)
            if not any(str(x).startswith('_cvrp') for x in p)}

init = MaskCOScenarioModel(dim=128, arm='pretrained', cvrp_model=cvrp, rngs=nnx.Rngs(42))
t_init = trainable_map(init)
for name in ('maskco_train160_clock_pretrained_s42', 'maskco_train160_partial_pretrained_s42'):
    m = MaskCOScenarioModel(dim=128, arm='pretrained', cvrp_model=cvrp, rngs=nnx.Rngs(42))
    m = load_model(m, 'results/a1_step3/%s/model.bin' % name)
    t = trainable_map(m)
    diffs = {k: float(np.abs(t[k] - t_init[k]).max()) for k in t if np.abs(t[k] - t_init[k]).max() > 0}
    print('%s: n_trainable=%d, n_changed_vs_init=%d, max_change=%.3e' % (
        name, len(t), len(diffs), max(diffs.values()) if diffs else 0.0))
    if diffs:
        print('  sample changed:', list(diffs.items())[:3])
EOF
