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
        "cd /tmp/p5v2_formal/ext_pyvrp",
        "echo '[failed list]'; python3 -c \"import json;d=json.load(open('status.json'));print([(f['block'],f['case'] if 'case' in f else '') for f in (d.get('failed') or [])][:10]);print((d.get('failed') or [{}])[0].get('error','')[:400])\" 2>/dev/null",
        "echo '[one log]'; ls out/*.log 2>/dev/null | head -2",
        "echo '[tail]'; for f in $(ls -t out/*.log 2>/dev/null | head -2); do echo \"== $f\"; grep -v 'jax\\|cuInit\\|CUDA' $f | tail -8; done",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
