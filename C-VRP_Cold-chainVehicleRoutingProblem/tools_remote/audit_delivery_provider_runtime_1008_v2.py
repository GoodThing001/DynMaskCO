"""Read-only parameter-name and compilation-reserve diagnostic, no dev scores."""
from pathlib import Path
from copy import deepcopy
from collections.abc import Mapping
import hashlib
import json
import os
import sys
import time
import traceback


def arrays_by_name(value, prefix=()):
    import numpy as np
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            result.update(arrays_by_name(child, prefix + (str(key),)))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for i, child in enumerate(value):
            result.update(arrays_by_name(child, prefix + (str(i),)))
        return result
    return {'/'.join(prefix): np.asarray(value)}


def main():
    project = Path(__file__).resolve().parents[1]
    out = project / 'results/delivery_provider_audit_1008_v2'
    out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {54})
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                      PYTHONDONTWRITEBYTECODE='1')
    root = project / 'results/delivery_pilot_1008_v1/src'
    sys.path.insert(0, str(root / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/delivery'))
    try:
        from delivery_maskco import MaskCORouteProvider
        from flax import nnx
        import numpy as np
        checkpoint = project.parent / 'MASKCO_code/ckpts/cvrp100.ckpt'
        checkpoint_start = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        p = MaskCORouteProvider(checkpoint, root / 'MASKCO_code')
        from training import load_ckpt
        _, _, _, cfg, _, _ = load_ckpt(str(checkpoint))
        cfg = deepcopy(cfg)
        cfg.dtype = 'float32'
        fresh = cfg.construct_model()
        loaded = arrays_by_name(nnx.state(p.model, nnx.Param).to_pure_dict())
        initial = arrays_by_name(nnx.state(fresh, nnx.Param).to_pure_dict())
        missing = sorted(set(loaded) ^ set(initial))
        shape_errors = [k for k in loaded.keys() & initial.keys()
                        if loaded[k].shape != initial[k].shape]
        assert not missing and not shape_errors, (missing, shape_errors)
        changed = [k for k in sorted(loaded) if not np.array_equal(loaded[k], initial[k])]
        p.prewarm(51, 50.)
        including_compile_max = p.max_decode_s
        data = {'coords': np.stack([np.arange(51), np.zeros(51)], axis=1),
                'demands': np.r_[0., np.ones(50)]}
        h = p.features(data, 50.)
        a = np.zeros((51, 51), np.float32)
        steady = []
        for _ in range(3):
            t = time.perf_counter()
            p.logits(h, a)
            steady.append(time.perf_counter() - t)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_start
        report = {'checkpoint_sha256': checkpoint_start,
                  'checkpoint_step_metadata': p.identity['checkpoint_step'],
                  'parameter_arrays': len(loaded), 'same_named_parameter_shapes': True,
                  'parameter_arrays_different_from_original_config_initialization': len(changed),
                  'changed_parameter_names': changed,
                  'checkpoint_is_not_original_config_initialization': bool(changed),
                  'prewarm_decode_max_including_compilation_s': including_compile_max,
                  'postcompile_fixed_synthetic_decode_wall_s': steady,
                  'old_decode_reserve_s': 2 * including_compile_max,
                  'old_reserve_alone_blocks_10s_loop': 2 * including_compile_max >= 10.,
                  'source_root': str(root), 'read_only_original_source_and_weights': True,
                  'development_scores_not_used': True, 'online_gain_established': False}
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        (out / 'run.exit').write_text('0')
        print(json.dumps({k: v for k, v in report.items() if k != 'changed_parameter_names'}), flush=True)
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps(
            {'error': str(exc), 'traceback': traceback.format_exc()}, indent=2))
        (out / 'run.exit').write_text('1')
        raise


if __name__ == '__main__':
    main()
