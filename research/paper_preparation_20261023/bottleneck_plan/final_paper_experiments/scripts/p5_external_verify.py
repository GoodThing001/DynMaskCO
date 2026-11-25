# -*- coding: utf-8 -*-
"""重部署 external01 并直接运行两格 PyVRP 冒烟（taskset 空闲核）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
WORKER = HERE.parent / "P5_standard_vrp/p5_external_worker.py"

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external01
mkdir -p out
rm -f out/0_PyVRP.json out/1_PyVRP.json out/0_PyVRP.log out/1_PyVRP.log
JAX_PLATFORMS=cpu taskset -c 14 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 0 PyVRP > out/0_PyVRP.log 2>&1
echo CVRP_exit=$?
JAX_PLATFORMS=cpu taskset -c 14 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 1 PyVRP > out/1_PyVRP.log 2>&1
echo VRPTW_exit=$?
python3 - <<'EOF'
import json
for f in ("out/0_PyVRP.json", "out/1_PyVRP.json"):
    try:
        r = json.load(open(f))
        print(f, r["case"], r["objective"], "K=%s" % r["vehicles"], "D=%s" % r["distance"],
              "online=%.2f" % r["online_s"], "initU=%s" % r["initial_U"])
    except Exception as e:
        print(f, "MISSING", e)
EOF
echo ===LOG===
tail -3 out/0_PyVRP.log
tail -3 out/1_PyVRP.log
"""


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            def makedirs(path):
                try:
                    s.mkdir(path)
                except OSError:
                    pass
            base = P5 + "/external01"
            makedirs(base)
            s.put(str(WORKER), base + "/unified_worker.py")
            wb = WORKER.read_bytes()
            with s.open(base + "/future_dispatch.py", "w") as f:
                f.write(wb)
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
