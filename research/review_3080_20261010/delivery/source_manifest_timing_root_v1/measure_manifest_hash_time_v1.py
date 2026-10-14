"""Single actual 3080 timing of the old saved source-start file set.

This measures a subset of the eventual close gate. It does not load the model,
enumerate new runtime modules, run a request, or move verification outside one.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import psutil

BASE = Path(__file__).resolve().parent
REG = BASE / "REGISTRATION.json"
OUT = BASE / "RESULT.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError("Single registered measurement already exists")
    reg = json.loads(REG.read_text(encoding="utf-8"))
    if sha(__file__) != reg["script_sha256"] or sha(reg["source_start_path"]) != reg["source_start_sha256"]:
        raise RuntimeError("Registered source differs")
    me = psutil.Process()
    me.cpu_affinity([15])
    available = psutil.virtual_memory().available / 2**30
    if available < 8:
        raise RuntimeError("8GiB runtime barrier")
    start = json.loads(Path(reg["source_start_path"]).read_text(encoding="utf-8"))
    files = start["files"]
    if len(files) != reg["expected_file_count"]:
        raise RuntimeError("Saved file count differs")
    t0 = time.perf_counter()
    mismatches = []
    nbytes = 0
    for name, expected in sorted(files.items()):
        path = Path(name)
        if not path.is_file():
            mismatches.append({"path": name, "reason": "missing"})
            continue
        nbytes += path.stat().st_size
        actual = sha(path)
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    t1 = time.perf_counter()
    out = {"registered_source_start_sha256": reg["source_start_sha256"],
           "file_count": len(files), "bytes_hashed": nbytes, "mismatch_count": len(mismatches),
           "mismatches": mismatches, "hash_elapsed_seconds": t1 - t0,
           "cpu_affinity": me.cpu_affinity(), "available_ram_gib_before": available,
           "actual_remote_runtime": True, "single_measurement": True,
           "no_model_or_request_or_C0": True,
           "source_close_gate_duration": None,
           "full_8s_request_qualified": False, "scientific_family3_started": False}
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps({"file_count": len(files), "bytes_hashed": nbytes,
                      "mismatch_count": len(mismatches), "hash_elapsed_seconds": t1 - t0}))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
