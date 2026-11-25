# -*- coding: utf-8 -*-
"""PyVRP 向量化矩阵修复后的单格验证（X-n1001-k43 与 X-n502-k39）。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external_precheck
rm -f out/0_PyVRP.json out/0_PyVRP.log out/1_PyVRP.json out/1_PyVRP.log out/2_*.log
python3 - <<'EOF'
import json
phase = {
  "name": "p5_pyvrp_precheck", "seconds": 60, "training_only": False,
  "methods": ["PyVRP"],
  "interpreters": {"PyVRP": "/home/hzeng/envs/MASKCO_env/bin/python"},
  "blocks": [
    {"case": "X-n1001-k43", "seed": 68001, "objective": "cvrp", "methods": ["PyVRP"]},
    {"case": "X-n502-k39", "seed": 68001, "objective": "cvrp", "methods": ["PyVRP"]}
  ],
  "max_cores": 1, "schedule_note": "PyVRP 大实例预算回归"
}
open("phase.json", "w").write(json.dumps(phase, ensure_ascii=False, indent=2))
EOF
for ix in 0 1; do
  JAX_PLATFORMS=cpu taskset -c 30 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py $ix PyVRP > out/2_$ix.log 2>&1
  echo EXIT_$ix=$?
done
python3 - <<'EOF'
import glob, json
for f in sorted(glob.glob("out/2_*.json")):
    r = json.load(open(f))
    print(r["case"], r["status"], "K=%d" % r["vehicles"], "D=%.2f" % r["distance"],
          "online=%.2f" % r["online_s"])
EOF
"""


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            wb = (HERE.parent / "P5_standard_vrp/p5_external_worker.py").read_bytes()
            s.put(str(HERE.parent / "P5_standard_vrp/p5_external_worker.py"),
                  P5 + "/external_precheck/unified_worker.py")
            with s.open(P5 + "/external_precheck/future_dispatch.py", "w") as f:
                f.write(wb)
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
