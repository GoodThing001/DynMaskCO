"""One registered four-shard source-byte hash stress beside fixed synthetic CPU work."""
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
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def save(name, obj):
    with (BASE / name).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def hash_child(reg, shard):
    psutil.Process().cpu_affinity([reg["hash_cpus"][shard]])
    files = json.loads(Path(reg["source_start_path"]).read_text(encoding="utf-8"))["files"]
    if len(files) != reg["expected_file_count"]:
        raise RuntimeError("registered file count changed")
    started = time.perf_counter()
    mismatches = []
    selected = sorted(files.items())[shard::len(reg["hash_cpus"])]
    nbytes = 0
    for name, expected in selected:
        path = Path(name)
        if not path.is_file():
            mismatches.append([name, "missing"])
            continue
        nbytes += path.stat().st_size
        if sha(path) != expected:
            mismatches.append([name, "hash_mismatch"])
    save(f"SHARD_{shard}.json", {"shard": shard, "elapsed_seconds": time.perf_counter() - started,
                                  "file_count": len(selected), "bytes_hashed": nbytes,
                                  "mismatch_count": len(mismatches), "mismatches": mismatches})


def cpu_child(reg):
    psutil.Process().cpu_affinity([reg["synthetic_cpu"]])
    start = time.perf_counter()
    begin = time.thread_time()
    digest = b"maskco-r18-fixed-cpu-demand"
    loops = 0
    while time.thread_time() - begin < reg["synthetic_cpu_seconds"]:
        digest = hashlib.sha256(digest).digest()
        loops += 1
    save("CPU_CHILD_RESULT.json", {"wall_seconds": time.perf_counter() - start,
                                   "cpu_seconds": time.thread_time() - begin,
                                   "loops": loops, "final_digest": digest.hex()})


def main():
    if (BASE / "RESULT.json").exists():
        raise RuntimeError("single registered attempt already exists")
    reg = json.loads(REG.read_text(encoding="utf-8"))
    if sha(__file__) != reg["script_sha256"] or sha(reg["source_start_path"]) != reg["source_start_sha256"]:
        raise RuntimeError("registered source/script differs")
    me = psutil.Process()
    me.cpu_affinity([reg["parent_cpu"]])
    available = psutil.virtual_memory().available / 2**30
    if available < reg["ram_gate_gib"]:
        raise RuntimeError("registered RAM gate")
    t0 = time.perf_counter()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    children = {}
    for shard in range(len(reg["hash_cpus"])):
        children[f"hash_{shard}"] = subprocess.Popen([sys.executable, "-B", str(__file__), "--hash", str(shard)],
                                                      cwd=str(BASE), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                      creationflags=flags)
    children["cpu"] = subprocess.Popen([sys.executable, "-B", str(__file__), "--cpu"], cwd=str(BASE),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    deadline = t0 + reg["active_deadline_seconds"]
    while time.perf_counter() < deadline and any(p.poll() is None for p in children.values()):
        time.sleep(0.02)
    by_deadline = {label: p.poll() is not None for label, p in children.items()}
    for p in children.values():
        if p.poll() is None:
            p.kill()
    receipts = {}
    for label, p in children.items():
        try:
            out, err = p.communicate(timeout=reg["reap_deadline_seconds"])
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate(timeout=1)
        receipts[label] = {"pid": p.pid, "exit_code": p.returncode, "completed_by_8s": by_deadline[label],
                           "stdout": out.decode(errors="replace")[-1000:],
                           "stderr": err.decode(errors="replace")[-2000:], "reaped": p.poll() is not None}
    shard_paths = [BASE / f"SHARD_{i}.json" for i in range(len(reg["hash_cpus"]))]
    shards = [json.loads(p.read_text(encoding="utf-8")) if p.exists() else None for p in shard_paths]
    cpu_path = BASE / "CPU_CHILD_RESULT.json"
    elapsed = time.perf_counter() - t0
    out = {"registration_sha256": sha(REG), "ram_gib_before": available,
           "parent_elapsed_seconds": elapsed,
           "within_8_plus_2": elapsed <= reg["active_deadline_seconds"] + reg["reap_deadline_seconds"],
           "receipts": receipts, "shards": shards,
           "all_source_files_verified": all(s is not None for s in shards)
                                        and sum(s["file_count"] for s in shards) == reg["expected_file_count"]
                                        and sum(s["mismatch_count"] for s in shards) == 0,
           "cpu_result": json.loads(cpu_path.read_text(encoding="utf-8")) if cpu_path.exists() else None,
           "synthetic_stress_only": True, "actual_model_or_request": False,
           "full_source_or_parameter_close_gate_qualified": False, "scientific_family3_started": False}
    save("RESULT.json", out)
    print(json.dumps({"all_source_files_verified": out["all_source_files_verified"],
                      "hash_children_done_by_8s": all(by_deadline[f"hash_{i}"] for i in range(len(shards))),
                      "cpu_done_by_8s": by_deadline["cpu"], "elapsed": elapsed}))
    if not all(x["reaped"] for x in receipts.values()) or not out["within_8_plus_2"]:
        raise SystemExit(2)
    if any(s is not None and s["mismatch_count"] for s in shards):
        raise SystemExit(3)


if __name__ == "__main__":
    reg = json.loads(REG.read_text(encoding="utf-8"))
    if len(sys.argv) == 3 and sys.argv[1] == "--hash":
        hash_child(reg, int(sys.argv[2]))
    elif sys.argv[1:] == ["--cpu"]:
        cpu_child(reg)
    else:
        main()
