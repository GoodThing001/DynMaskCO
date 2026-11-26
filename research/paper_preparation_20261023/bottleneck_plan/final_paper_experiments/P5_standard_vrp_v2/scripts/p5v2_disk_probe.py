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
        "cd /home/hzeng/project/MASKCO-Main",
        "echo '[p5 tree sizes]'; du -sm _maskco_final_paper_p5 _maskco_final_paper_p5v2 2>/dev/null",
        "echo '[home top]'; du -sm /home/hzeng/* 2>/dev/null | sort -rn | head -8",
        "echo '[project top]'; du -sm /home/hzeng/project/MASKCO-Main/* 2>/dev/null | sort -rn | head -10",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
