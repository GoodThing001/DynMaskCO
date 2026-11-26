import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
D = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01"
try:
    print(transport.remote_exec(client, "cd " + D + " && for p in $(pgrep -f 'diag01/unified_worker[.]py'); do kill $p; done; for p in $(pgrep -f 'unified_runner[.]py'); do c=$(cat /proc/$p/cwd 2>/dev/null); done; sleep 1; echo done"))
    print(transport.remote_exec(client, "ps -eo pid,args | grep unified_runner | grep -v grep | wc -l"))
finally:
    client.close()
