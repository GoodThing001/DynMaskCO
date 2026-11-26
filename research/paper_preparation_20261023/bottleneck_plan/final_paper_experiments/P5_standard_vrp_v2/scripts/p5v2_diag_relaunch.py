import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
D = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"
try:
    with client.open_sftp() as s:
        s.put(str(HERE / "p5v2_fixed_candidates.py"), D + "/p5v2_fixed_candidates.py")
    transport.remote_exec(client, "cd " + D + " && for p in $(pgrep -f 'unified_runner[.]py'); do kill $p; done; sleep 1; rm -rf out status.json runner.log; echo cleaned")
    transport.remote_exec(client, "cd " + D + " && setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & echo launched")
    time.sleep(90)
    print(transport.remote_exec(client, "cd " + D + " && echo \"[json] $(ls out/*.json 2>/dev/null | wc -l)\"; grep -o 'failed.: .[^]]*' status.json | head -c 200; ls out/*.error.txt 2>/dev/null | head -2"))
finally:
    client.close()
