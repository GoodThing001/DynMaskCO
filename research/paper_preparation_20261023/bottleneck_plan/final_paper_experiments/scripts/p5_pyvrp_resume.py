# -*- coding: utf-8 -*-
"""PyVRP 批次续跑：仅运行未完成块（385..467），写入 external_pyvrp2 工作区。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"
START = 385


def main():
    phase = json.loads((HERE.parent / "P5_standard_vrp/phase_external_pyvrp.json").read_text(encoding="utf-8"))
    blocks = phase["blocks"][START:]
    resume = dict(phase, name="p5_external_pyvrp_rest", blocks=blocks,
                  schedule_note="PyVRP 续跑：块 %d..%d（缺失 %d 块）" % (START, START + len(blocks) - 1, len(blocks)))
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            def makedirs(path):
                try:
                    s.mkdir(path)
                except OSError:
                    pass
            base = P5 + "/external_pyvrp2"
            makedirs(base)
            wb = (HERE.parent / "P5_standard_vrp/p5_external_worker.py").read_bytes()
            s.put(str(HERE.parent / "P5_standard_vrp/p5_external_worker.py"), base + "/unified_worker.py")
            with s.open(base + "/future_dispatch.py", "w") as f:
                f.write(wb)
            with s.open(base + "/phase.json", "w") as f:
                f.write(json.dumps(resume, ensure_ascii=False, indent=2).encode("utf-8"))
        cmd = ("cp " + P5 + "/stage01/unified_runner.py " + P5 + "/external_pyvrp2/ && cd " + P5
               + "/external_pyvrp2 && rm -rf out status.json runner.log && tmux new-session -d -s p5_pyvrp_rest"
               " 'cd " + P5 + "/external_pyvrp2 && " + PY + " -B unified_runner.py > runner.log 2>&1' && sleep 4 && tmux ls")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main()
