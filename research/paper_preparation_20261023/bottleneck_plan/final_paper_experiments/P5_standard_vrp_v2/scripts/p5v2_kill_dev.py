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
        "for p in $(pgrep -f 'dev01/unified_worker[.]py'); do kill $p 2>/dev/null; done",
        "for p in $(pgrep -f 'dev01/unified_runner[.]py'); do kill $p 2>/dev/null; done",
        "sleep 2",
        "echo \"[json] $(ls /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01/out/*.json | wc -l)\"",
        "echo \"[workers] $(pgrep -fc 'dev01/unified_worker[.]py' || true)\"",
        "echo \"[runners] $(pgrep -fc 'dev01/unified_runner[.]py' || true)\"",
        "echo \"[ext json] $(ls /tmp/p5v2_formal/ext_pyvrp/out/*.json 2>/dev/null | wc -l)\"",
        "echo \"[ext procs] $(pgrep -fc 'ext_pyvrp' || true)\"",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
