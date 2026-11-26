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
    cmd = ("cd " + P5V2 + "/dev01 && echo \"[json] $(ls out/*.json 2>/dev/null | wc -l)\"; "
           "echo \"[failed] $(grep -o 'failed.: .[^]]*' status.json | head -c 300)\"; "
           "ps -eo etime,args | grep 'dev01/unified_worker' | grep -v grep | head -4")
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
