import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
try:
    cmd = ("cd " + P5V2 + "/dev01 && echo \"[now] $(date -u +%H:%M:%S)\"; "
           "echo \"[json] $(ls out/*.json 2>/dev/null | wc -l) logs $(ls out/*.log 2>/dev/null | wc -l)\"; "
           "echo '[ps]'; ps -eo pid,etime,args | grep -E 'unified_(runner|worker)' | grep -v grep | head -4; "
           "echo '[runnerlog]'; tail -5 runner.log 2>/dev/null; "
           "echo '[status]'; head -c 300 status.json 2>/dev/null")
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
