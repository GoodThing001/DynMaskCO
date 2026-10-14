"""Count parameters in a torch checkpoint WITHOUT needing the method's model code.

Used to fill the AEGIS-CD-style 参数量(M) column, and to independently cross-check the
counts the participation harness reported (`participation/*.json` → `C2_random_weights.replaced_params`).

Usage: python CC_Compare/tools/param_count.py <ckpt1> [ckpt2 ...] [--bridge]

`--bridge` applies the old→new torchrl class-name aliases (same in-process trick the
providers use, e.g. `CompositeSpec` → `Composite`); needed for checkpoints pickled with an
old torchrl. Note the checkpoint's own package must also be importable, e.g.
    PYTHONPATH=<repo>/CC_Compare/RRNCO python param_count.py <ckpt> --bridge
"""
import sys

import torch


def apply_torchrl_bridge():
    try:
        import torchrl.data.tensor_specs as ts
    except Exception as exc:  # noqa: BLE001
        print('  bridge: torchrl unavailable (%s)' % exc)
        return
    aliases = {
        'CompositeSpec': 'Composite', 'BoundedTensorSpec': 'Bounded',
        'UnboundedTensorSpec': 'Unbounded',
        'UnboundedContinuousTensorSpec': 'Unbounded',
        'BoundedContinuousTensorSpec': 'Bounded',
        'DiscreteTensorSpec': 'Discrete',
        'UnboundedDiscreteTensorSpec': 'UnboundedDiscrete',
        'OneHotDiscreteTensorSpec': 'OneHot',
        'BinaryDiscreteTensorSpec': 'Binary',
        'MultiDiscreteTensorSpec': 'MultiDiscrete',
        'MultiOneHotDiscreteTensorSpec': 'MultiOneHot',
    }
    applied = []
    for old, new in aliases.items():
        if not hasattr(ts, old) and hasattr(ts, new):
            setattr(ts, old, getattr(ts, new))
            applied.append('%s->%s' % (old, new))
    print('  bridge applied: %d aliases' % len(applied))


def sum_tensors(d):
    n = 0
    k = 0
    for _key, v in d.items():
        if torch.is_tensor(v):
            n += int(v.numel())
            k += 1
    return k, n


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    bridge = '--bridge' in sys.argv
    if not args:
        print(__doc__)
        return 2
    if bridge:
        apply_torchrl_bridge()
    for path in args:
        print('=== %s' % path)
        try:
            obj = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print('  LOAD_FAIL %s: %s' % (type(exc).__name__, exc))
            continue
        if not isinstance(obj, dict):
            print('  not a dict: %s' % type(obj))
            continue
        sd = obj.get('state_dict', obj)
        if not isinstance(sd, dict):
            print('  state_dict not a dict: %s' % type(sd))
            continue
        k, n = sum_tensors(sd)
        print('  top-level keys: %s' % list(obj)[:6])
        print('  tensors=%d  params=%d  (%.3f M)' % (k, n, n / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main())
