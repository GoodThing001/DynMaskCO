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
        "cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem",
        "echo '[top]'; du -sm * 2>/dev/null | sort -rn | head -8",
        "echo '[results]'; du -sm results/* 2>/dev/null | sort -rn | head -10",
        "echo '[archive]'; du -sm archive/* 2>/dev/null | sort -rn | head -6",
        "echo '[checkpoints]'; du -sm checkpoints/* 2>/dev/null | sort -rn | head -5",
        "echo '[models]'; du -sm models/* 2>/dev/null | sort -rn | head -5",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
