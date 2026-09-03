#!/bin/bash
# ============================================================
# Phase 3d: Cold-Chain Thermal Physics Evaluation
# 用法: bash run_coldchain_thermal_eval.sh
# ============================================================
set -e

ROOT="/home/hzeng/project/MASKCO-Main"
cd "$ROOT"
source /home/hzeng/envs/MASKCO_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "Phase 3d: Endogenous Cold-Chain Thermal Physics"
echo "============================================================"
echo ""

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python3 -c "
import sys, os, numpy as np
sys.path.insert(0, 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding')
sys.path.insert(0, 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models')
sys.path.insert(0, 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/lib')
sys.path.insert(0, 'models')

import jax, jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModelConfig
from modules.functional import coord_normalize
from resource_beam import ResourceBeamSearcher
from thermal_state import compute_route_thermal_metrics, ThermalState

print('Loading model...')
params, _, _, model_config, _, _ = load_ckpt(
    'C-VRP_Cold-chainVehicleRoutingProblem/ckpts/p0_fix/phase3c/step50000.ckpt')
if model_config is None:
    model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
    model_config.encoder_input_dim = 7
model = model_config.construct_model()
if params is not None: model = nnx.merge(nnx.graphdef(model), params)
_MODEL_IN = int(model.init_proj.kernel.shape[0])

@jax.jit
def encode_fn(raw_features):
    raw_features = raw_features.at[..., :2].set(coord_normalize(raw_features[..., :2]))
    return model.encode(raw_features[..., :_MODEL_IN])

@jax.jit
def logit_fn(features, timestep):
    adjmat = jnp.zeros((features.shape[0], 51, 51), dtype=jnp.int8)
    return model.decode(features, timestep, adjmat.astype(jnp.float32))

print('Loading data...')
data = dict(np.load('C-VRP_Cold-chainVehicleRoutingProblem/data/p0_fix/dcc_50_r1_edod05_test.npz'))
N = min(32, data['coords'].shape[0])
cap, tw_max = 50, float(data['tw_end'].max())

f_list = [
    data['coords'][:N], (data['demands'][:N]/cap)[...,None],
    (data['tw_start'][:N]/tw_max)[...,None], (data['tw_end'][:N]/tw_max)[...,None],
    (data['temp_class'][:N].astype(np.float32)/2.0)[...,None],
]
if 'reveal_time' in data: f_list.append((data['reveal_time'][:N].astype(np.float32)/tw_max)[...,None])
raw_features = np.concatenate(f_list, axis=-1).astype(np.float32)
feats = np.array(encode_fn(jnp.array(raw_features)))
logits_all = np.array(logit_fn(jnp.array(feats), jnp.full(N, 0.3, dtype=jnp.float32)))

print(f'Evaluating {N} instances with K=16 beam + thermal tracking...')
print()

# Headers
print(f'{\"Inst\":<6} {\"Dist\":<8} {\"Feas\":<8} {\"Energy\":<10} {\"QLoss\":<10} {\"TTI\":<10} {\"Viol\":<8} {\"Doors\":<8} {\"AvgT°\":<8}')
print('-' * 78)

results = []
for idx in range(N):
    logits_b = logits_all[idx].copy()
    logits_b[:, 0] = -1e9; np.fill_diagonal(logits_b, -1e9)

    searcher = ResourceBeamSearcher(
        data['coords'][idx], data['tw_start'][idx], data['tw_end'][idx],
        data['service_time'][idx], data['demands'][idx], cap, 1.0, K=16, tw_margin=0.05)
    route, score = searcher.generate(logits_b, max_steps=200)

    if route and len(route) > 2:
        m = compute_route_thermal_metrics(
            route, data['coords'][idx], data['tw_start'][idx], data['tw_end'][idx],
            data['service_time'][idx], data['demands'][idx], data['temp_class'][idx])
        results.append(m)
        print(f'{idx:<6} {m[\"total_distance\"]:<8.2f} {\"✓\":<8} {m[\"total_energy_kwh\"]:<10.3f} '
              f'{m[\"quality_loss\"]:<10.4f} {m[\"tti\"]:<10.1f} '
              f'{m[\"thermal_violations\"]:<8} {m[\"num_door_opens\"]:<8} {m[\"avg_cabin_temp\"]:<8.1f}')

print()
print('=== Summary (n={}) ==='.format(len(results)))
if results:
    dists = [r['total_distance'] for r in results]
    energies = [r['total_energy_kwh'] for r in results]
    qls = [r['quality_loss'] for r in results]
    ttis = [r['tti'] for r in results]
    viols = [r['thermal_violations'] for r in results]
    doors = [r['num_door_opens'] for r in results]
    temps = [r['avg_cabin_temp'] for r in results]

    print(f'  Distance:          {np.mean(dists):.2f} ± {np.std(dists):.2f}')
    print(f'  Energy (kWh):      {np.mean(energies):.3f} ± {np.std(energies):.3f}')
    print(f'  Quality Loss:      {np.mean(qls):.4f} ± {np.std(qls):.4f}')
    print(f'  TTI (℃·h):         {np.mean(ttis):.1f} ± {np.std(ttis):.1f}')
    print(f'  Thermal Violations:{np.sum(viols)} total ({np.mean(viols):.1f}/inst)')
    print(f'  Door Opens:         {np.sum(doors)} total ({np.mean(doors):.1f}/inst)')
    print(f'  Avg Cabin Temp:    {np.mean(temps):.1f} °C')

print()
print('=== Thermal Model Parameters ===')
print(f'  T_ambient = {25.0}°C, T_target = [25, 4, -18]°C')
print(f'  UA/C = 0.5 /h, P_cool = [0, 150, 400] W')
print(f'  Q_door = 500 J, COP = 2.5')
print(f'  k_0 = [0.01, 0.002, 0.0002] /h')
"

echo ""
echo "=== DONE ==="
