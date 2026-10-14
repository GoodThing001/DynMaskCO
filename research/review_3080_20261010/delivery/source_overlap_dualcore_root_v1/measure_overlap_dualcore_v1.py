"""One bounded dual-core overlap stress test of the old source hash list.

No model or delivery request is run. A synthetic CPU demand is deliberately
not interpreted as the real scoring workload.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

BASE = Path(__file__).resolve().parent
REG = BASE / "REGISTRATION.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(name, obj):
    with (BASE / name).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def hash_child(reg):
    psutil.Process().cpu_affinity(reg["hash_cpu_affinity"])
    files = json.loads(Path(reg["source_start_path"]).read_text(encoding="utf-8"))["files"]
    if len(files) != reg["expected_file_count"]:
        raise RuntimeError("source file count changed")
    started = time.perf_counter()
    mismatches = []
    nbytes = 0
    for name, expected in sorted(files.items()):
        path = Path(name)
        if not path.is_file():
            mismatches.append([name, "missing"])
            continue
        nbytes += path.stat().st_size
        got = sha(path)
        if got != expected:
            mismatches.append([name, "hash_mismatch"])
    save("HASH_CHILD_RESULT.json", {"elapsed_seconds": time.perf_counter() - started,
                                    "file_count": len(files), "bytes_hashed": nbytes,
                                    "mismatch_count": len(mismatches), "mismatches": mismatches})


def cpu_child(reg):
    psutil.Process().cpu_affinity(reg["synthetic_cpu_affinity"])
    started = time.perf_counter()
    c0 = time.thread_time()
    h = b"maskco-r16-fixed-cpu-demand"
    loops = 0
    while time.thread_time() - c0 < reg["synthetic_cpu_seconds"]:
        h = hashlib.sha256(h).digest()
        loops += 1
    save("CPU_CHILD_RESULT.json", {"wall_seconds": time.perf_counter() - started,
                                   "cpu_seconds": time.thread_time() - c0,
                                   "loops": loops, "final_digest": h.hex()})


def main():
    if (BASE / "RESULT.json").exists():
        raise RuntimeError("single registered attempt already exists")
    reg = json.loads(REG.read_text(encoding="utf-8"))
    if sha(__file__) != reg["script_sha256"] or sha(reg["source_start_path"]) != reg["source_start_sha256"]:
        raise RuntimeError("registered source identity differs")
    me = psutil.Process()
    me.cpu_affinity(reg["parent_cpu_affinity"])
    available = psutil.virtual_memory().available / 2**30
    if available < reg["ram_gate_gib"]:
        raise RuntimeError("registered 8GiB RAM barrier")
    t0 = time.perf_counter()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    children = {}
    for mode in ("hash", "cpu"):
        children[mode] = subprocess.Popen([sys.executable, "-B", str(__file__), "--" + mode],
                                          cwd=str(BASE), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          creationflags=flags)
    deadline = t0 + reg["active_deadline_seconds"]
    while time.perf_counter() < deadline and any(p.poll() is None for p in children.values()):
        time.sleep(0.02)
    at_deadline = {mode: p.poll() is not None for mode, p in children.items()}
    for p in children.values():
        if p.poll() is None:
            p.kill()
    receipts = {}
    for mode, p in children.items():
        try:
            out, err = p.communicate(timeout=reg["reap_deadline_seconds"])
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate(timeout=1)
        receipts[mode] = {"pid": p.pid, "exit_code": p.returncode,
                          "completed_by_8s": at_deadline[mode], "stdout": out.decode(errors="replace")[-1000:],
                          "stderr": err.decode(errors="replace")[-2000:], "reaped": p.poll() is not None}
    elapsed = time.perf_counter() - t0
    hash_path = BASE / "HASH_CHILD_RESULT.json"
    cpu_path = BASE / "CPU_CHILD_RESULT.json"
    out = {"registration_sha256": sha(REG), "available_ram_gib_before": available,
           "cpu_affinity_parent": me.cpu_affinity(), "parent_elapsed_seconds": elapsed,
           "within_8_plus_2": elapsed <= reg["active_deadline_seconds"] + reg["reap_deadline_seconds"],
           "receipts": receipts,
           "hash_result": json.loads(hash_path.read_text(encoding="utf-8")) if hash_path.exists() else None,
           "cpu_result": json.loads(cpu_path.read_text(encoding="utf-8")) if cpu_path.exists() else None,
           "synthetic_stress_only": True, "actual_model_or_request": False,
           "full_source_or_parameter_close_gate_qualified": False,
           "scientific_family3_started": False}
    save("RESULT.json", out)
    print(json.dumps({"hash_completed_by_8s": receipts["hash"]["completed_by_8s"],
                      "cpu_completed_by_8s": receipts["cpu"]["completed_by_8s"],
                      "parent_elapsed_seconds": elapsed,
                      "hash_elapsed_seconds": out["hash_result"]["elapsed_seconds"] if out["hash_result"] else None}))
    if not all(x["reaped"] for x in receipts.values()) or not out["within_8_plus_2"]:
        raise SystemExit(2)
    if out["hash_result"] and out["hash_result"]["mismatch_count"]:
        raise SystemExit(3)


if __name__ == "__main__":
    reg = json.loads(REG.read_text(encoding="utf-8"))
    if sys.argv[1:] == ["--hash"]:
        hash_child(reg)
    elif sys.argv[1:] == ["--cpu"]:
        cpu_child(reg)
    else:
        main()
