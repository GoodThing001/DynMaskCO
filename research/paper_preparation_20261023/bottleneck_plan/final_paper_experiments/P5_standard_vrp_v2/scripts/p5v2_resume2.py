import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
try:
    for p in ("for p in $(pgrep -f 'unified_runner[.]py'); do kill $p 2>/dev/null; done",
              "for p in $(pgrep -f 'dev01/unified_worker[.]py'); do kill $p 2>/dev/null; done"):
        try:
            transport.remote_exec(client, p)
        except Exception as exc:
            print("kill step:", exc)
    with client.open_sftp() as s:
        s.put(str(HERE / "p5v2_worker.py"), P5V2 + "/dev01/unified_worker.py")
        s.put(str(HERE / "p5v2_worker.py"), P5V2 + "/dev01/future_dispatch.py")
        for f in sorted((HERE.parent / "standard_vrp").rglob("*.py")):
            rel = f.relative_to(HERE.parent).as_posix()
            try:
                s.put(str(f), P5V2 + "/" + rel)
            except OSError as exc:
                print("put", rel, exc)
    cmd = ("cd " + P5V2 + "/dev01 && rm -f out/*.log status.json runner.log && "
           "setsid nohup /home/hzeng/envs/MASKCO_env/bin/python -B unified_runner.py > runner.log 2>&1 "
           "< /dev/null & sleep 6; ps -eo pid,etime,args | grep 'unified_runner' | grep -v grep | head -2")
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
