"""DEV-GATE 薄入口：固定 `role=dev_gate`，保持旧命令兼容。

本文件不再承载驱动逻辑——正式门控核心在 `run_formal_gate.py`，共享契约在
`formal_gate_contract.py`。此处只做两件事：
  1. 把旧 CLI（`--data-dir --profile --workers --out`）翻译为通用驱动调用；
  2. 再导出 `COMPUTE_FILES` / `CONTROL_FILES` / `ANALYSIS_FILES` / `_version` /
     `_frozen_compute`，供 `test_control_chain.py`、`test_dev_gate_control.py` 与
     旧 `freeze_source_archive.py` 兼容使用。

用法（不变）：
    python scripts/evaluation/run_dev_gate.py \
        --data-dir data/baseline/50_node/dev_gate \
        --profile results/o0cc/scale_v2/objective_profile.json \
        --workers 8 --out results/o0cc/dev_gate
"""
import argparse
import os

# 再导出：兼容 test_control_chain / test_dev_gate_control 对 run_dev_gate.X 的引用。
from formal_gate_contract import (  # noqa: F401
    COMPUTE_FILES, CONTROL_FILES, ANALYSIS_FILES, _version, _frozen_compute,
)

from run_formal_gate import run as _run_formal_gate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--coldchain-contract', default=None,
                    help='calibrated contract JSON（新 DEV 必填；缺省回退 pilot）')
    args = ap.parse_args(argv)
    manifest = os.path.join(args.data_dir, 'DEV_MANIFEST.json')
    _run_formal_gate(role='dev_gate', manifest=manifest, profile=args.profile,
                     workers=args.workers, out=args.out,
                     coldchain_contract=args.coldchain_contract)


if __name__ == '__main__':
    main()
