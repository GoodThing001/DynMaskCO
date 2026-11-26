import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    for label, path in (("cvrp", "/tmp/p5v2_formal/cvrp/out/0_SLOT.log"),
                        ("diag", "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01/out/0_DIAG.log")):
        cmd = ("grep -v 'jax\\|cuInit\\|CUDA\\|plugin' " + path + " | tail -8")
        print("==", label)
        print(transport.remote_exec(client, cmd))
finally:
    client.close()
