"""Capture actual frozen-CVRP input for special tokens, without a real checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
from flax import nnx

SCRIPTS = Path(__file__).resolve().parents[1]
for rel in ('models', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(SCRIPTS / rel))
from maskco_scenario import PretrainedEncoder, MASK_TOKEN, PAD_TOKEN


def probe():
    captured = {}

    class CaptureEncoder:
        def encode(self, x, attn_options=None):
            captured['x'] = np.asarray(x)
            captured['bias'] = np.asarray(attn_options['bias'])
            return jnp.zeros(x.shape[:-1] + (512,))

    encoder = PretrainedEncoder(4, CaptureEncoder(), nnx.Rngs(0))
    encoder(jnp.asarray([[1, 19, MASK_TOKEN, PAD_TOKEN]], dtype=jnp.int32))
    return {'probe': 'real PretrainedEncoder.__call__ with capturing CVRP stub',
            'mask_normalized_demand': float(captured['x'][0, 2, 2]),
            'pad_normalized_demand': float(captured['x'][0, 3, 2]),
            'mask_is_attention_blocked': bool(np.all(captured['bias'][0, 2] < -1e8)),
            'pad_is_attention_blocked': bool(np.all(captured['bias'][0, 3] < -1e8)),
            'matches_doc_zero_raw_demand': bool(np.all(captured['x'][0, 2:, 2] == 0)),
            'causes_online_loss_proven': False,
            'action': 'keep existing model/version; register any representation variant before retraining'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    report = probe()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
