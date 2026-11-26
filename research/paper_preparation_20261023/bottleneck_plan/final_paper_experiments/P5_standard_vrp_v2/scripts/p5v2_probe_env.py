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
        "echo \"[dev cells] $(ls /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01/out/*.json 2>/dev/null | wc -l)\"",
        "echo '[pyvrp]'; /home/hzeng/envs/MASKCO_env/bin/python -c \"import pyvrp;print(pyvrp.__version__)\" 2>&1 | tail -1",
        "echo '[pyvrp cc]'; /home/hzeng/envs/cc_compare/bin/python -c \"import pyvrp;print(pyvrp.__version__)\" 2>&1 | tail -1",
        "echo '[ortools]'; /home/hzeng/envs/MASKCO_env/bin/python -c \"import ortools;print(ortools.__version__)\" 2>&1 | tail -1",
        "echo '[free]'; df -h / | tail -1",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
