"""Probe a checkpoint's container format and, if it is a plain array tree, count its parameters.

Tries (in order): a single msgpack object, a msgpack **stream** of objects, pickle.
Prints the first bytes so the format is identifiable even if all loaders fail.

Usage: python CC_Compare/tools/probe_ckpt_format.py <ckpt>
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


def count_nnx(state):
    """Sum array sizes in a Flax **NNX** state (it is not a plain dict of arrays)."""
    tot = 0
    n = 0
    flat = None
    try:
        flat = state.flat_state()
    except Exception:  # noqa: BLE001
        flat = None
    if flat is not None:
        try:
            items = list(flat.items())
        except AttributeError:
            try:
                items = [(k, flat[k]) for k in flat]
            except Exception:  # noqa: BLE001
                items = []
        for _path, var in items:
            v = getattr(var, 'value', var)
            try:
                tot += int(np.asarray(v).size)
                n += 1
            except Exception:  # noqa: BLE001
                pass
    return n, tot


def main():
    path = sys.argv[1]
    raw = open(path, 'rb').read()
    print('size=%d head=%r' % (len(raw), raw[:40]))
    obj = None
    try:
        import msgpack
        u = msgpack.Unpacker(raw=False)
        u.feed(raw)
        objs = list(u)
        print('msgpack stream objects: %d' % len(objs))
        if len(objs) == 1:
            obj = objs[0]
        elif objs:
            obj = objs
    except Exception as exc:  # noqa: BLE001
        print('msgpack failed: %s: %s' % (type(exc).__name__, exc))
    if obj is None:
        try:
            import pickle
            obj = pickle.loads(raw)
            print('pickle OK: %s' % type(obj))
        except Exception as exc:  # noqa: BLE001
            print('pickle failed: %s: %s' % (type(exc).__name__, exc))
            return 1
    if isinstance(obj, dict):
        print('top keys: %s' % list(obj)[:10])
    d = walk(obj)
    if d:
        print('array leaves=%d  total=%d (%.3f M)' % (len(d), sum(d.values()), sum(d.values()) / 1e6))
        for k in sorted(d, key=lambda x: -d[x])[:5]:
            print('  %-40s %d' % (k, d[k]))
    else:
        print('no array leaves found')
    # Flax NNX：params 是 State 对象，需按 flat_state 统计
    if isinstance(obj, dict) and 'params' in obj:
        for key in ('params', 'opt_state'):
            if key not in obj:
                continue
            n, tot = count_nnx(obj[key])
            print('NNX[%s]: %d arrays  total=%d (%.3f M)' % (key, n, tot, tot / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main())
