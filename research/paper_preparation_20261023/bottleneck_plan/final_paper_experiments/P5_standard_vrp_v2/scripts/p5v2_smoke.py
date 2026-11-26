# -*- coding: utf-8 -*-
"""V2 冒烟：远端运行 dev 块 0 的三个版本（STD_FIX/STD_DIST/STD_OLD），后台 nohup。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"

SCRIPT = r"""#!/bin/bash
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/dev01
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
mkdir -p out smoke_logs
rm -f out/0_STD_FIX.json out/0_STD_DIST.json out/0_STD_OLD.json
for v in STD_FIX STD_DIST STD_OLD; do
  taskset -c 5 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 0 $v > smoke_logs/0_$v.log 2>&1
  echo "EXIT_$v=$?"
done
echo SMOKE_DONE > smoke_done
"""


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            with s.open(P5V2 + "/dev01/run_smoke.sh", "w") as f:
                f.write(SCRIPT.encode("utf-8"))
        cmd = ("cd " + P5V2 + "/dev01 && rm -f smoke_done && chmod +x run_smoke.sh && "
               "setsid nohup bash run_smoke.sh > smoke_runner.log 2>&1 < /dev/null & sleep 3; "
               "ps aux | grep run_smoke | grep -v grep | wc -l")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main()
