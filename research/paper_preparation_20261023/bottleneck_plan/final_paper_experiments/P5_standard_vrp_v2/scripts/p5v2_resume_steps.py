import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
try:
    steps = [
        "mkdir -p " + P5V2 + "/dev01_stash",
        "cp -n " + P5V2 + "/dev01/out/*.json " + P5V2 + "/dev01_stash/ 2>/dev/null; ls " + P5V2 + "/dev01_stash | wc -l",
        "rm -rf " + P5V2 + "/dev01/out " + P5V2 + "/dev01/status.json " + P5V2 + "/dev01/runner.log; ls " + P5V2 + "/dev01 | tr '\\n' ' '",
    ]
    for i, cmd in enumerate(steps):
        print("step", i, "->", transport.remote_exec(client, cmd))
finally:
    client.close()
