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
        "echo '[ext procs]'; ps -eo pid,etime,args | grep 'ext_pyvrp' | grep -v grep | head -3",
        "echo \"[ext json] $(ls /tmp/p5v2_formal/ext_pyvrp/out/*.json 2>/dev/null | wc -l)\"",
        "echo '[ext status end]'; python3 -c \"import json;d=json.load(open('/tmp/p5v2_formal/ext_pyvrp/status.json'));print('end',d.get('end'),'cells',d.get('cells'),'failed',len(d.get('failed') or []))\" 2>/dev/null",
        "echo \"[dev json] $(ls /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01/out/*.json 2>/dev/null | wc -l)\"",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
