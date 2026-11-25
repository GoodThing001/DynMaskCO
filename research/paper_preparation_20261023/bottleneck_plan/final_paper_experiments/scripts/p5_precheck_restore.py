# -*- coding: utf-8 -*-
"""恢复双块预检 phase（block0 CVRP 7法 / block1 C101 4法）并重跑 block1 快格。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
MASKCO = "/home/hzeng/envs/MASKCO_env/bin/python"
CC = "/home/hzeng/envs/cc_compare/bin/python"

PHASE = dict(
    name="p5_external_precheck", seconds=60, training_only=False,
    methods=["AttentionModel", "POMO", "DeepACO", "CaR", "MVMoE", "CaDA", "RouteFinder", "RRNCO"],
    interpreters={**{m: MASKCO for m in
                     ("AttentionModel", "POMO", "DeepACO", "CaR", "MVMoE", "CaDA", "RouteFinder")},
                  "RRNCO": CC},
    blocks=[
        dict(case="X-n101-k25", seed=68001, objective="cvrp",
             methods=["AttentionModel", "POMO", "DeepACO", "CaR", "MVMoE", "CaDA", "RouteFinder"]),
        dict(case="C101", seed=69001, objective="vrptw",
             methods=["MVMoE", "CaDA", "RouteFinder", "RRNCO"]),
    ],
    max_cores=1, schedule_note="外部作者模型调度预检（CVRP 7 法 + VRPTW 4 法）")

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external_precheck
rm -f out/1_MVMoE.json out/1_CaDA.json out/1_RouteFinder.json out/1_*.log
for m in MVMoE CaDA RouteFinder; do
  JAX_PLATFORMS=cpu taskset -c 30 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 1 $m > out/1_$m.log 2>&1
  echo EXIT_$m=$?
done
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
        with client.open_sftp() as s:
            with s.open(P5 + "/external_precheck/phase.json", "w") as f:
                f.write(json.dumps(PHASE, ensure_ascii=False, indent=2).encode("utf-8"))
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
