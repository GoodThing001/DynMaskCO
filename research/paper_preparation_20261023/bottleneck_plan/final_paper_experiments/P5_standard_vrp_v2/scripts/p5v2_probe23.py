import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
RUN = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01"
try:
    cmd = ("echo \"[now] $(date -u +%H:%M:%S)\"; "
           "ps -eo pid,etime,pcpu,args | grep 'dev01/unified_worker' | grep -v grep | head -4; "
           "cd " + RUN + " && echo \"[json] $(ls out/*.json | wc -l)\"; "
           "echo '[latest]'; ls -t out/*.json | head -2; "
           "echo '[logtail]'; tail -3 $(ls -t out/*.log | head -1) 2>/dev/null")
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
