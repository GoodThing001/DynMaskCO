import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
RUN = "/tmp/p5v2_dev/dev01"
try:
    cmd = ("echo \"[now] $(date -u +%H:%M:%S)\"; cd " + RUN + " && "
           "echo \"[json] $(ls out/*.json 2>/dev/null | wc -l)\"; "
           "echo \"[failed] $(grep -o 'failed.: .[][]' status.json | head -c 200)\"; "
           "ps -eo etime,args | grep 'dev01/unified_worker' | grep -v grep | wc -l")
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
