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
        "echo '[procs]'; ps -eo pid,etime,pcpu,args | grep 'dev01/unified_worker' | grep -v grep | head -3",
        "echo '[json]'; ls out/*.json 2>/dev/null | wc -l",
        "echo '[logs]'; ls -lt out/*.log 2>/dev/null | head -4",
        "echo '[date]'; date -u +%H:%M:%S",
        "for f in $(ls -t out/*.log 2>/dev/null | head -2); do echo \"== $f\"; grep -v 'jax\\|cuInit\\|CUDA\\|plugin' $f | tail -5; done",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
