# -*- coding: utf-8 -*-
"""远端预检：在 stage01 工作区直接运行 ALNS/VNS 两格，验证新代码路径。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/stage01
mkdir -p out
rm -f out/0_ALNS.json out/0_VNS.json out/0_ALNS.log out/0_VNS.log
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= taskset -c 3 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 0 ALNS > out/0_ALNS.log 2>&1
echo ALNS_exit=$?
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= taskset -c 3 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 0 VNS > out/0_VNS.log 2>&1
echo VNS_exit=$?
python3 - <<'EOF'
import json
for m in ("ALNS", "VNS"):
    r = json.load(open("out/0_" + m + ".json"))
    print(m, r["case"], r["seed"], "D=%.2f" % r["distance"], "K=%d" % r["vehicles"],
          "online=%.2f" % r["online_s"], "initU=%.2f" % r["initial_U"], "epochs=%d" % r["epochs"])
EOF
echo ===LOGTAIL===
tail -2 out/0_ALNS.log
tail -2 out/0_VNS.log
"""


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
