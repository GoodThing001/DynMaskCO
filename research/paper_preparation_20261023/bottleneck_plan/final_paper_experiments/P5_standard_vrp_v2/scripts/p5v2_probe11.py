import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    cmd = "; ".join([
        "cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01",
        "echo \"[now] $(date -u +%H:%M:%S)\"",
        "echo \"[start] $(grep -o 'start.: .[^\\\"]*' status.json | head -1)\"",
        "echo \"[json] $(ls out/*.json 2>/dev/null | wc -l)\"",
        "echo '[procs]'; ps -eo etime,pcpu,args | grep 'dev01/unified_worker' | grep -v grep | head -2",
        "echo \"[ext2 json] $(ls /tmp/p5v2_formal/ext_pyvrp2/out/*.json 2>/dev/null | wc -l)\"",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
