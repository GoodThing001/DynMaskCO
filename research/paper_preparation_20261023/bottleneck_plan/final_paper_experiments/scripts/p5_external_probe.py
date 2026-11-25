# -*- coding: utf-8 -*-
"""远端探测：PyVRP 可用性 + 作者模型仓库推理入口。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
/home/hzeng/envs/MASKCO_env/bin/python -B - <<'EOF'
try:
    import pyvrp
    print("PYVRP", pyvrp.__version__ if hasattr(pyvrp, "__version__") else "ok")
except Exception as e:
    print("PYVRP_MISSING", type(e).__name__, str(e)[:120])
try:
    import torch
    print("TORCH", torch.__version__)
except Exception as e:
    print("TORCH_MISSING", type(e).__name__, str(e)[:120])
EOF
echo ===UPSTREAM===
ls -d /home/hzeng/project/MASKCO-Main/upstream_extra_20261021/*/ 2>/dev/null | head -20
echo ===RECENT5===
ls /home/hzeng/project/MASKCO-Main/_maskco_recent5_20261120/assets/ 2>/dev/null | head -30
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
