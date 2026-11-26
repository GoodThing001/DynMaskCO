import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    with client.open_sftp() as s:
        s.get("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/stage01/unified_runner.py",
              "_vendor_unified_runner.py")
    print("ok", Path("_vendor_unified_runner.py").stat().st_size)
finally:
    client.close()
