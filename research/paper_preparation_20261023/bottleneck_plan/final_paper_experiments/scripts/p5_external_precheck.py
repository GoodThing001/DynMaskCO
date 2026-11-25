# -*- coding: utf-8 -*-
"""external_precheck：p5_external_worker 作者调度预检（AM/POMO/DeepACO/CaR 四格）。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"
METHODS = ["AttentionModel", "POMO", "DeepACO", "CaR"]

PHASE = dict(
    name="p5_external_precheck", seconds=60, training_only=False, methods=METHODS,
    interpreters={m: PY for m in METHODS},
    blocks=[dict(case="X-n101-k25", seed=68001, objective="cvrp", methods=METHODS)],
    max_cores=1, schedule_note="外部作者模型调度预检（单块四法）")

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/external_precheck
mkdir -p out
rm -f out/*.json out/*.log
for m in AttentionModel POMO DeepACO CaR; do
  JAX_PLATFORMS=cpu taskset -c 30 /home/hzeng/envs/MASKCO_env/bin/python -B unified_worker.py 0 $m > out/0_$m.log 2>&1
  echo EXIT_$m=$?
done
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob("out/*.json")):
    r = json.load(open(f))
    print(r["method"], r["status"], "D=%.1f" % r["distance"], "K=%d" % r["vehicles"],
          "online=%.2f" % r["online_s"], "initU=%s" % r["initial_U"])
EOF
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
            base = P5 + "/external_precheck"
            makedirs(base)
            wb = (HERE.parent / "P5_standard_vrp/p5_external_worker.py").read_bytes()
            s.put(str(HERE.parent / "P5_standard_vrp/p5_external_worker.py"), base + "/unified_worker.py")
            with s.open(base + "/future_dispatch.py", "w") as f:
                f.write(wb)
            with s.open(base + "/phase.json", "w") as f:
                f.write(json.dumps(PHASE, ensure_ascii=False, indent=2).encode("utf-8"))
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
