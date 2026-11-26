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
    print(transport.remote_exec(client, "cd " + D + " && tail -14 out/0_DIAG.error.txt"))
finally:
    client.close()
