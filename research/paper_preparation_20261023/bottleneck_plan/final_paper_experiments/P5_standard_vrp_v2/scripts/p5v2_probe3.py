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
        "cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5",
        "echo '[solomon inst]'; ls solomon/instances | head -5",
        "echo \"[n] $(ls solomon/instances | wc -l)\"",
        "echo '[bks sample]'; head -6 solomon/bks/C101.sol",
        "echo '[dev01]'; ls /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01/out 2>/dev/null | wc -l",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
