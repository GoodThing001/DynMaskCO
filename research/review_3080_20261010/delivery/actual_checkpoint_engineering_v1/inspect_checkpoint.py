"""Read a trusted, SHA-pinned original checkpoint without optimizer restoration."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import types

EXPECTED = '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'


def load_trusted(checkpoint, upstream):
    sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if sha != EXPECTED:
        raise ValueError('Actual original checkpoint SHA mismatch')
    sys.path.insert(0, str(upstream))
    if 'tensorboardX' not in sys.modules:
        tb = types.ModuleType('tensorboardX')
        class UnusedWriter:
            def __init__(self, *args, **kwargs):
                raise RuntimeError('Unused logging is not part of this engineering entry')
        tb.SummaryWriter = UnusedWriter
        sys.modules['tensorboardX'] = tb
    # Trusted original hash only: do not deserialize arbitrary caller files.
    with Path(checkpoint).open('rb') as stream:
        saved = pickle.load(stream)
    return saved, sha


def inspect(saved, sha):
    import numpy as np
    params = saved['params']
    entries = [{'path': list(path), 'shape': list(np.asarray(v.value).shape),
                'dtype': str(np.asarray(v.value).dtype), 'elements': int(np.asarray(v.value).size)}
               for path, v in params.flat_state()]
    return {'checkpoint_sha256': sha, 'checkpoint_step': int(saved['step']),
            'model_config': vars(saved['model_config']),
            'parameter_leaves': len(entries), 'parameter_elements': sum(x['elements'] for x in entries),
            'parameter_shape_inventory': entries,
            'optimizer_state_present': saved.get('opt_state') is not None,
            'read_only_inspection_not_training': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    saved, sha = load_trusted(args.checkpoint, args.upstream)
    result = inspect(saved, sha)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('checkpoint_step', 'model_config', 'parameter_leaves', 'parameter_elements')}))
