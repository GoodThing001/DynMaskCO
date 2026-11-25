# -*- coding: utf-8 -*-
"""重跑 PyVRP 全量批次（收据曾因工作区清理丢失；确定性固定种子可复现）。

工作区 external_pyvrp3，max_cores=6（避开 stage03 的 0-7 与作者批次的 9-19）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
PY = "/home/hzeng/envs/MASKCO_env/bin/python"


def main():
    phase = json.loads((HERE.parent / "P5_standard_vrp/phase_external_pyvrp.json").read_text(encoding="utf-8"))
    phase = dict(phase, name="p5_external_pyvrp3", max_cores=6,
                 schedule_note="PyVRP 全量重跑（468 格，6 核；确定性种子）")
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            def makedirs(path):
                try:
                    s.mkdir(path)
                except OSError:
                    pass
            base = P5 + "/external_pyvrp3"
            makedirs(base)
            wb = (HERE.parent / "P5_standard_vrp/p5_external_worker.py").read_bytes()
            s.put(str(HERE.parent / "P5_standard_vrp/p5_external_worker.py"), base + "/unified_worker.py")
            with s.open(base + "/future_dispatch.py", "w") as f:
                f.write(wb)
            with s.open(base + "/phase.json", "w") as f:
                f.write(json.dumps(phase, ensure_ascii=False, indent=2).encode("utf-8"))
        cmd = ("cp " + P5 + "/stage01/unified_runner.py " + P5 + "/external_pyvrp3/ && "
               "sed -i 's/timeout=1500/timeout=300/' " + P5 + "/external_pyvrp3/unified_runner.py && "
               "cd " + P5 + "/external_pyvrp3 && rm -rf out status.json runner.log && "
               "tmux new-session -d -s p5_pyvrp3 'cd " + P5 + "/external_pyvrp3 && " + PY
               + " -B unified_runner.py > runner.log 2>&1' && sleep 4 && tmux ls")
        print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    main()
