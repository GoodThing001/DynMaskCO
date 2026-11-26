import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    cmd = ("cd /tmp/p5v2_formal/cvrp && grep -v 'jax\\|cuInit\\|CUDA\\|plugin' out/6_SLOT.log | "
           "grep -n 'Error\\|error\\|line [0-9]*, in\\|Timeout' | tail -8")
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
