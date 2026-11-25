# -*- coding: utf-8 -*-
"""P5-A 正式阶段部署：建 stage01/02/03 工作区、同步 worker/runner/phase、启动 stage01。

- standard_vrp 共享路径（_maskco_final_paper_p5/standard_vrp），先由 p5_ship_standard 同步；
- unified_worker.py 与 future_dispatch.py 均为最新 p5_worker.py 副本（远端 check 从 future_dispatch 导入）；
- unified_runner.py 从 smoke01 复制（与 P4 同款，env 已含 JAX_PLATFORMS=cpu）；
- 仅启动 stage01（阶段顺序执行，避免核心争用），stage02/03 待 stage01 完成后单独启动。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
WORKER = HERE.parent / "P5_standard_vrp/p5_worker.py"


def main(launch="stage01"):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            import posixpath
            def makedirs(path):
                try:
                    s.mkdir(path)
                except OSError:
                    pass
            worker_bytes = WORKER.read_bytes()
            for stage in ("stage01", "stage02", "stage03"):
                base = P5 + "/" + stage
                makedirs(base)
                s.put(str(WORKER), base + "/unified_worker.py")
                with s.open(base + "/future_dispatch.py", "w") as f:
                    f.write(worker_bytes)
                s.put(str(HERE.parent / ("P5_standard_vrp/phase_" + stage + ".json")),
                      base + "/phase.json")
                print("OK", stage, "worker/check/phase")
        # 复制 runner（smoke01 验证过的版本）
        cmd = ("cp /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/smoke01/unified_runner.py "
               + P5 + "/stage01/ && cp " + P5 + "/stage01/unified_runner.py "
               + P5 + "/stage02/ && cp " + P5 + "/stage01/unified_runner.py " + P5 + "/stage03/")
        transport.remote_exec(client, cmd)
        print("OK runner copied")
        if launch:
            cmd = ("cd " + P5 + "/" + launch + " && rm -rf out status.json runner.log && "
                   "tmux new-session -d -s p5a_" + launch + " 'cd " + P5 + "/" + launch
                   + " && /home/hzeng/envs/MASKCO_env/bin/python -B unified_runner.py > runner.log 2>&1'"
                   " && sleep 4 && tmux ls")
            print(transport.remote_exec(client, cmd))
    finally:
        client.close()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "stage01"
    main(None if arg in ("nolaunch", "none") else arg)
