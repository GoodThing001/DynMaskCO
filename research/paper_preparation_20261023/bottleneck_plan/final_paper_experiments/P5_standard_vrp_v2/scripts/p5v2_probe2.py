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
        "echo \"[cvrp bks] $(ls cvrplib/bks/*.sol 2>/dev/null | wc -l)\"",
        "echo \"[cvrp inst] $(ls cvrplib/instances/*.vrp 2>/dev/null | wc -l)\"",
        "echo \"[solomon bks] $(ls solomon/bks 2>/dev/null | wc -l)\"",
        "echo \"[solomon inst] $(ls solomon/instances/*.txt 2>/dev/null | wc -l)\"",
        "echo '[solomon bks sample]'; ls solomon/bks 2>/dev/null | head -6",
        "echo '[manifest]'; head -c 700 cvrplib/source_manifest.json",
        "echo; echo '[solomon manifest]'; head -c 700 solomon/source_manifest.json",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
