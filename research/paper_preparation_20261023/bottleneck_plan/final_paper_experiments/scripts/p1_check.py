# -*- coding: utf-8 -*-
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"
TARGET = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p1/p1"

PROBE = r'''
import json, subprocess, time
from pathlib import Path
p = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p1/p1")
try:
    st = json.loads((p/"status.json").read_text())
    print("status:", {k: st[k] for k in ("complete","cells","total_cells","failed","end")})
    print("active:", json.dumps(st["active"], indent=0))
except FileNotFoundError:
    print("no status.json")
print("tmux:", subprocess.run(["tmux","ls"],capture_output=True,text=True).stdout.splitlines())
print("workers:", subprocess.run(["bash","-c","ps aux | grep unified_worker | grep -v grep | wc -l"],capture_output=True,text=True).stdout.strip())
'''

def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        out = transport.remote_exec(client, PYTHON + " -B -c " + shlex.quote(PROBE))
        print(out)
        out2 = transport.remote_exec(client, "tail -c 2000 " + TARGET + "/controller.log")
        print("--- controller.log tail ---")
        print(out2[-2000:])
    finally:
        client.close()

if __name__ == "__main__":
    main()
