"""Independent saved-only audit of the fixed-first-layer, sliced-final-layer head.

No request, backbone, C0 evaluator, action consumer, or old QA is run here.
The original timed audit and its 28 failed checks remain unchanged.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/delivery/split_final_timed_execution_root_v1"
ARCHIVE = BASE / "downloaded"
REG = BASE / "SAVED_SPLIT_AUDIT_REGISTRATION.json"
OUT = BASE / "INDEPENDENT_SAVED_SPLIT_AUDIT.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError("Saved audit already exists")
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert reg["script_sha256"] == sha(__file__)
    assert reg["download_manifest_sha256"] == sha(BASE / "DOWNLOAD_MANIFEST.json")
    assert reg["original_audit_sha256"] == sha(ARCHIVE / "INDEPENDENT_TIMED_REQUEST_AUDIT.json")
    assert reg["expected_records"] == 50
    original = json.loads((ARCHIVE / "INDEPENDENT_TIMED_REQUEST_AUDIT.json").read_text(encoding="utf-8"))
    assert original["failed"] == 28
    rng = nnx.Rngs(20261012)
    first = nnx.Linear(3075, 64, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rng)
    last = nnx.Linear(64, 1, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rng)
    params = {f"('{name}', '{leaf}')": sha_value(value)
              for name, layer in (("input", first), ("output", last))
              for leaf, value in (("kernel", layer.kernel.value), ("bias", layer.bias.value))}
    checks = []
    if params != reg["head_param_sha256"]:
        checks.append({"name": "head_parameter_identity", "passed": False})
    else:
        checks.append({"name": "head_parameter_identity", "passed": True})
    max_abs = 0.0
    for arm in reg["arms"]:
        directory = ARCHIVE / f"{arm}_fault0_worker/request"
        for step in range(25):
            path = directory / f"{step:02d}.npz"
            with np.load(path, allow_pickle=False) as z:
                h = jnp.asarray(z["latent"][0], jnp.float32)
                actions = z["actions"]
                n = len(actions)
                parts = {}
                for key in ("actions", "route_weights", "prefix_weights", "suffix_weights", "slot_position_length"):
                    value = z[key]
                    parts[key] = np.concatenate([value, np.repeat(value[:1], 750 - n, axis=0)], axis=0)
                aa = jnp.asarray(parts["actions"], jnp.int32)
                x = jnp.concatenate(
                    [h[aa[:, 0]], h[aa[:, 3]], h[aa[:, 4]]]
                    + [jnp.asarray(parts[key], jnp.float32) @ h for key in ("route_weights", "prefix_weights", "suffix_weights")]
                    + [jnp.asarray(parts["slot_position_length"], jnp.float32)], axis=-1)
                hidden = jnp.tanh(first(x))
                expected = np.asarray(last(hidden[:n])[:, 0])
                actual = z["scores"]
                equal = np.array_equal(expected, actual)
                error = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
                max_abs = max(max_abs, error)
                checks.append({"name": f"{arm}:{step:02d}", "passed": bool(equal), "n": n,
                               "expected_sha256": sha_value(expected), "actual_sha256": sha_value(actual),
                               "max_abs_error": error})
    result = {"checks_count": len(checks), "failed": sum(not x["passed"] for x in checks),
              "max_abs_error": max_abs, "checks": checks,
              "original_audit_failed": original["failed"], "original_audit_unchanged_sha256": sha(ARCHIVE / "INDEPENDENT_TIMED_REQUEST_AUDIT.json"),
              "saved_only": True, "request_rerun": False, "model_backbone_rerun": False,
              "full_request_qualified": False, "scientific_family3_started": False}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("checks_count", "failed", "max_abs_error")}))
    if result["failed"]:
        raise SystemExit(1)


def sha_value(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


if __name__ == "__main__":
    main()
