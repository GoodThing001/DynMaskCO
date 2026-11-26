import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
PY = "/home/hzeng/envs/MASKCO_env/bin/python"
D = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01"
try:
    print(transport.remote_exec(client, "cd " + D + " && rm -rf out status.json runner.log && (setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null &) ; echo launched-diag"))
finally:
    client.close()
