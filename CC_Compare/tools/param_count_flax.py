"""Count parameters in a **Flax** checkpoint (used for the main model `Ours` row).

`param_count.py` handles torch checkpoints; the MaskCO main model is a Flax/JAX checkpoint
(305 MB), which torch cannot unpickle (and it needs `flax`). This traverses the restored
msgpack tree and sums array sizes — no model class needed.

Usage (use an env that has flax, e.g. the project's MASKCO_env):
    python CC_Compare/tools/param_count_flax.py <ckpt> [<ckpt2> ...]
"""
import sys

import numpy as np


def walk(obj, path=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(walk(v, '%s/%s' % (path, k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(walk(v, '%s/%d' % (path, i)))
    else:
        size = getattr(obj, 'size', None)
        if size is None:
            try:
                size = int(np.asarray(obj).size)
            except Exception:  # noqa: BLE001
                return {}
        out[path or 'root'] = int(size)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    try:
        import flax.serialization as fs
    except Exception as exc:  # noqa: BLE001
        print('FLAX_UNAVAILABLE %s: %s' % (type(exc).__name__, exc))
        return 1
    for path in sys.argv[1:]:
        print('=== %s' % path)
        try:
            obj = fs.msgpack_restore(open(path, 'rb').read())
        except Exception as exc:  # noqa: BLE001
            print('  LOAD_FAIL %s: %s' % (type(exc).__name__, exc))
            continue
        d = walk(obj)
        top = {}
        for k, v in d.items():
            head = k.split('/')[1] if k.count('/') > 1 else k
            top[head] = top.get(head, 0) + v
        for k in sorted(top, key=lambda x: -top[x])[:8]:
            print('  %-28s %d' % (k, top[k]))
        tot = sum(d.values())
        print('  TOTAL_PARAMS %d (%.3f M)' % (tot, tot / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main())
