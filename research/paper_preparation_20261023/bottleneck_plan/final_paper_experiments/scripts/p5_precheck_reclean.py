# -*- coding: utf-8 -*-
"""清理并重跑预检快格（MVMoE/CaDA/RouteFinder；RRNCO 已验证）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
tmux kill-session -t p5_precheck 2>/dev/null || true
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external_precheck
rm -f out/1_CaDA.json out/1_RouteFinder.json out/1_MVMoE.json out/1_*.log precheck_done
JAX_PLATFORMS=cpu taskset -c 30 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 1 MVMoE > out/1_MVMoE.log 2>&1
echo EXIT_MVMoE=$?
JAX_PLATFORMS=cpu taskset -c 30 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 1 CaDA > out/1_CaDA.log 2>&1
echo EXIT_CaDA=$?
JAX_PLATFORMS=cpu taskset -c 30 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 1 RouteFinder > out/1_RouteFinder.log 2>&1
echo EXIT_RouteFinder=$?
python3 - <<'EOF'
import glob, json
for f in sorted(glob.glob("out/1_*.json")):
    r = json.load(open(f))
    print(r["method"], r["status"], "K=%d" % r["vehicles"], "D=%.2f" % r["distance"],
          "online=%.1f" % r["online_s"])
EOF
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
