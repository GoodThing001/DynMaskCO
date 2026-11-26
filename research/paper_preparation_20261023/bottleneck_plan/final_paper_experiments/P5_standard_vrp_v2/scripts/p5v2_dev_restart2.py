import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
PY = "/home/hzeng/envs/MASKCO_env/bin/python"
RUN = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01"
def step(cmd, label):
    try:
        out = transport.remote_exec(client, cmd)
        print("[%s] %s" % (label, (out or "").strip()[:300]))
    except Exception as exc:
        print("[%s] FAILED %s" % (label, str(exc)[:200]))
try:
    step("cd " + RUN + " && tail -4 runner.log 2>/dev/null; echo '---'; ls -la . | head -8", "before")
    step("cd " + RUN + " && rm -rf out status.json runner.log && echo cleaned", "clean")
    step("cd " + RUN + " && setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & echo launched", "launch")
    time.sleep(10)
    step("ps -eo pid,etime,args | grep unified_runner | grep -v grep | head -3", "ps")
    step("cd " + RUN + " && head -c 200 status.json 2>/dev/null; tail -3 runner.log 2>/dev/null", "status")
finally:
    client.close()
