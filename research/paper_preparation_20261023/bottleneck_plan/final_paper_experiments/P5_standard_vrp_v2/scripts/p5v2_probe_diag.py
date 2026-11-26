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
    cmd = ("cd " + D + " && ls out/ 2>/dev/null | head -5; echo '---'; "
           "cat out/0_DIAG.error.txt 2>/dev/null | tail -14; echo '---log---'; "
           "grep -v 'jax\\|cuInit\\|CUDA\\|plugin' out/0_DIAG.log 2>/dev/null | tail -12")
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
