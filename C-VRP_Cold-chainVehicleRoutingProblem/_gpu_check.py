import sys
import jax
import jax.numpy as jnp
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'scripts/training')
sys.path.insert(0, 'scripts/models')
sys.path.insert(0, 'scripts/simulation')
sys.path.insert(0, '../MASKCO_code')
sys.path.insert(0, '../MASKCO_code/models')
print('devices', jax.devices())
from train_fleet_head import load_base_model
from dynmaskco_cc import MaskCODecoder
m = load_base_model('ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt')
print('encoder loaded')
raw = jnp.zeros((1, 51, 7), jnp.float32)
vis = jnp.ones((1, 51), jnp.float32)
H = m.encode(raw, visible_mask=vis)
print('H shape', H.shape, H.dtype)
d = MaskCODecoder(d_enc=256, d_dec=256, num_layers=2, num_heads=8, rngs=0)
Z = d(H[:, :10], jnp.array([0.5]), jnp.zeros((1, 10, 10)))
print('Z shape', Z.shape)
print('GPU OK')
