import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
dest = V2 / "external_results/pyvrp_vrptw"
(dest / "out").unlink() if (dest / "out").is_file() else None
import shutil
if (dest / "out").exists():
    shutil.rmtree(dest / "out")
counts = {}
try:
    with client.open_sftp() as s:
        for tag, sub in (("run1", "ext_pyvrp"), ("run2", "ext_pyvrp2")):
            base = "/tmp/p5v2_formal/" + sub
            out = dest / tag
            out.mkdir(parents=True, exist_ok=True)
            names = s.listdir(base + "/out")
            n = 0
            for name in names:
                if name.endswith(".json") and not name.startswith("status"):
                    s.get(base + "/out/" + name, str(out / name))
                    n += 1
            for extra in ("phase.json", "status.json"):
                try:
                    s.get(base + "/" + extra, str(dest / ("%s_%s" % (tag, extra))))
                except OSError:
                    pass
            counts[tag] = n
finally:
    client.close()
print("fetched", counts)
