"""Single read-only 3080 observation; does not launch a research workload."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from _srv_exec_v2 import build, run_capture  # noqa: E402

CONFIG = str(ROOT / ".vscode" / "sftp.3080.json")
WIN = r"C:\Users\zh\project\MASKCO-Main"
PY = r"D:\anaconda\envs\MASKCO_env\python.exe"

CODE = """import json,os,psutil,time
me={os.getpid(),os.getppid()}
rows=[]
for p in psutil.process_iter(['pid','name','cmdline','username','create_time']):
 try:
  i=p.info; a=i['cmdline'] or [];s=' '.join(a).lower()
  if p.pid in me or not ('review_3080_20261010' in s or 'cc_compare' in s):continue
  rows.append(dict(pid=p.pid,owner=i['username'],created=i['create_time'],name=i['name'],research_owned='review_3080_20261010' in s,comparison='cc_compare' in s,rss=p.memory_info().rss,cpu_percent=p.cpu_percent(interval=None)))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
print(json.dumps(dict(observed_epoch=time.time(),available_ram_gib=psutil.virtual_memory().available/2**30,cpu15_busy_percent=psutil.cpu_percent(interval=1,percpu=True)[15],allowed_cpu=psutil.Process().cpu_affinity(),matching_processes=rows)))"""


def main():
    target = HERE / "SERVER_RESOURCE_OBSERVATION.json"
    if target.exists():
        raise FileExistsError("read-only observation already saved")
    command = build(CONFIG, WIN, PY, "-B", ["-c", CODE], "windows")
    result = run_capture(command, timeout=45, config=CONFIG)
    observed = json.loads(result["stdout"]) if result["exit_code"] == 0 else None
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "readonly": True,
        "research_launched": False,
        "remote_exit_code": result["exit_code"],
        "observation": observed,
        "stderr": result["stderr"][:1000],
    }
    with target.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(receipt, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return int(result["exit_code"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
