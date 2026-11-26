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
        "echo \"[dev] $(ls /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01/out/*.json 2>/dev/null | wc -l)\"",
        "echo \"[ext] $(ls /tmp/p5v2_formal/ext_pyvrp/out/*.json 2>/dev/null | wc -l)\"",
        "echo '[ext failed]'; python3 -c \"import json;d=json.load(open('/tmp/p5v2_formal/ext_pyvrp/status.json'));print(d.get('failed'),d.get('cells'),d.get('total_cells'))\" 2>/dev/null",
        "echo '[dev failed]'; python3 -c \"import json;d=json.load(open('/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01/status.json'));print(d.get('failed'),d.get('cells'),d.get('total_cells'),d.get('end'))\" 2>/dev/null",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
