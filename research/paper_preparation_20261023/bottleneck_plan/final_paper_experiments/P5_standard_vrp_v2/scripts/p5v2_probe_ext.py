import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    cmd = ("echo \"[ext] $(ls /tmp/p5v2_formal/ext_pyvrp/out/*.json 2>/dev/null | wc -l)/168\"; "
           "echo \"[ext failed] $(python3 -c \"import json;d=json.load(open('/tmp/p5v2_formal/ext_pyvrp/status.json'));print(len(d.get('failed') or []),d.get('cells'))\" 2>/dev/null)\"; "
           "echo \"[procs] $(pgrep -fc 'ext_pyvrp' || true)\"")
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
