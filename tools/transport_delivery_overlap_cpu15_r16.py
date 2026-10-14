"""One-shot exclusive 3080 PY/JSON stress and bounded read-only collection."""
import argparse
import hashlib
import json
import posixpath
from datetime import datetime, timezone
from pathlib import Path

import paramiko
from _srv import _cfg
from _srv_exec_v2 import build, run_capture

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/review_3080_20261010/delivery/source_overlap_cpu15_root_v1"
CONFIG = str(ROOT / ".vscode/sftp.3080.json")
WIN = r"C:\Users\zh\project\MASKCO-Main"
PY = r"D:\anaconda\envs\MASKCO_env\python.exe"
REL = "research/review_3080_20261010/source_overlap_cpu15_root_v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    with (BASE / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def connect():
    h, u, p = _cfg(CONFIG)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(h, port=22, username=u, password=p, timeout=30)
    return c


def remote():
    base = json.loads(Path(CONFIG).read_text(encoding="utf-8"))["remotePath"].rstrip("/")
    return posixpath.join(base, REL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "collect"])
    mode = parser.parse_args().mode
    reg_bytes = (BASE / "REGISTRATION.json").read_bytes()
    reg = json.loads(reg_bytes)
    script = (BASE / "measure_overlap_cpu15_v1.py").read_bytes()
    if sha(script) != reg["script_sha256"] or reg["expected_file_count"] != 8903:
        raise RuntimeError("Registered script/count differs")
    target = remote()
    if mode == "run":
        if (BASE / "RUN_ATTEMPT.json").exists():
            raise RuntimeError("Single attempt already recorded")
        code = """import json,os,psutil
me={os.getpid(),os.getppid()};owned=[]
for p in psutil.process_iter(['pid','name','cmdline','create_time']):
 try:
  a=p.info['cmdline'] or []
  if p.pid not in me and 'python' in (p.info['name'] or '').lower() and any('review_3080_20261010' in s and s.endswith('.py') for s in a):owned.append(p.info)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
print(json.dumps(dict(ram_gib=psutil.virtual_memory().available/2**30,cpu15=psutil.cpu_percent(interval=2,percpu=True)[15],allowed=psutil.Process().cpu_affinity(),owned=owned)))"""
        pre = run_capture(build(CONFIG, WIN, PY, "-B", ["-c", code], "windows"), 30, CONFIG)
        save("RESOURCE_PREFLIGHT.json", pre)
        if pre["exit_code"] != 0:
            raise RuntimeError("Resource query failed")
        obs = json.loads(pre["stdout"])
        if obs["ram_gib"] < 8 or obs["cpu15"] > 20 or 15 not in obs["allowed"] or obs["owned"]:
            raise RuntimeError("Resource/owned-tree barrier")
        save("RUN_ATTEMPT.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "registration_sha256": sha(reg_bytes), "remote": REL})
        c = connect()
        try:
            s = c.open_sftp()
            s.get_channel().settimeout(30)
            s.mkdir(target)
            for name, contents in (("REGISTRATION.json", reg_bytes), ("measure_overlap_cpu15_v1.py", script)):
                with s.open(target + "/" + name, "wx") as f:
                    f.write(contents)
                with s.open(target + "/" + name, "rb") as f:
                    if sha(f.read()) != sha(contents):
                        raise RuntimeError("Upload hash differs: " + name)
            s.close()
        finally:
            c.close()
        save("UPLOAD_ACCEPTANCE.json", {"registration_sha256": sha(reg_bytes), "script_sha256": sha(script), "verified": True})
        run = run_capture(build(CONFIG, WIN, PY, "-B", [REL + "/measure_overlap_cpu15_v1.py"], "windows"), 60, CONFIG)
        save("RUN_RESPONSE.json", run)
        print(json.dumps({"exit_code": run["exit_code"], "timed_out": run.get("timed_out"), "stdout": run["stdout"][-500:]}))
    else:
        if (BASE / "COLLECT_ATTEMPT.json").exists():
            raise RuntimeError("Collection already attempted")
        save("COLLECT_ATTEMPT.json", {"created_utc": datetime.now(timezone.utc).isoformat()})
        c = connect()
        try:
            s = c.open_sftp()
            s.get_channel().settimeout(30)
            for name in ("RESULT.json", "HASH_CHILD_RESULT.json", "CPU_CHILD_RESULT.json"):
                try:
                    with s.open(target + "/" + name, "rb") as f:
                        n = f.stat().st_size
                        if n > 1000000:
                            raise RuntimeError("Bounded result size exceeded")
                        data = list(f.readv([(0, n)]))[0]
                except FileNotFoundError:
                    save(name + ".missing.json", {"remote_missing": True, "name": name})
                    continue
                if len(data) != n:
                    raise RuntimeError("Short bounded read")
                (BASE / name).write_bytes(data)
                save(name + ".receipt.json", {"bytes": n, "sha256": sha(data), "readonly": True})
            s.close()
        finally:
            c.close()
        result = json.loads((BASE / "RESULT.json").read_text(encoding="utf-8"))
        print(json.dumps({"hash_completed_by_8s": result["receipts"]["hash"]["completed_by_8s"],
                          "cpu_completed_by_8s": result["receipts"]["cpu"]["completed_by_8s"],
                          "parent_elapsed_seconds": result["parent_elapsed_seconds"]}))


if __name__ == "__main__":
    main()
