import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    cmd = ("cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01 && "
           "ls out/*.json | sed 's/.*\\///' | cut -d_ -f1 | sort -n | tail -1 && "
           "grep -c 'V1_CRASH' out/*.json 2>/dev/null | grep -v ':0' | head -5 ; "
           "grep -o '\"failed\": \\[[^]]*' status.json | head -c 200")
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
