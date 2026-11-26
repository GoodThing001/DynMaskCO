import sys, time
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
        "echo \"[json] $(ls out/*.json 2>/dev/null | wc -l)\"",
        "echo \"[workers] $(ps aux | grep unified_worker | grep -v grep | wc -l)\"",
        "echo '[latest json]'; ls -t out/*.json 2>/dev/null | head -3",
        "echo '[now]'; date -u +%H:%M:%S",
        "echo '[logtails]'; for f in $(ls -t out/*.log 2>/dev/null | head -2); do echo \"== $f\"; tail -3 $f; done",
        "echo '[disk]'; df -h /home | tail -1",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
