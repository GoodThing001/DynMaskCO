import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    print(transport.remote_exec(client, "ps -eo pid,etime,args | grep unified_runner | grep -v grep | head -6; echo '---'; cd /tmp/p5v2_dev/dev01 && echo dev=$(ls out/*.json 2>/dev/null | wc -l); cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01 && echo diag=$(ls out/*.json 2>/dev/null | wc -l) err=$(ls out/*.error.txt 2>/dev/null | wc -l)"))
finally:
    client.close()
