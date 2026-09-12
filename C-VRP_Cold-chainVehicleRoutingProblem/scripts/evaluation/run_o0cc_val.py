"""O0-CC VAL 入口：只接受 sealed archive（自包含），不接任意文件拼装。

正式 VAL 的唯一入口。它：
  1. 核验 archive bundle（文件集合 + 逐文件 hash + bundle 重算 + 本地批准 hash）；
  2. 核验 active compute/control/analysis 源码 hash 与封存 seal 完全一致（防止工作区源码
     在封存后变化仍被用来处理冻结输入）；
  3. 从 archive `inputs/` 解析 manifest / profile / 规范化 split registry；
  4. 调通用门控核心执行。

不再接受 `--manifest --profile --split-registry` 显式拼装；通用 `run_formal_gate.py` 的
`--role val` 也被拒绝，避免绕过本入口。

用法：
    python scripts/evaluation/run_o0cc_val.py \
        --archive <sealed_val_archive> --expected-bundle-sha256 <64-hex> \
        --workers 8 --out <fresh_output_dir>
"""
import argparse
import os

from run_formal_gate import run as _run_formal_gate
import formal_gate_contract as contract


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True, help='sealed VAL archive 路径')
    ap.add_argument('--expected-bundle-sha256', required=True,
                    help='本地批准的 archive bundle sha256（64 位 hex）')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True, help='全新输出目录')
    args = ap.parse_args(argv)

    archive = os.path.abspath(args.archive)
    seal = contract.verify_archive(archive, args.expected_bundle_sha256)
    contract.verify_active_code(seal)

    manifest = os.path.join(archive, 'inputs', 'manifest.json')
    profile = os.path.join(archive, 'inputs', 'objective_profile.json')
    split_registry = os.path.join(archive, 'inputs', 'split_registry.json')
    contract_file = os.path.join(archive, 'inputs', 'contract.json')
    statistical_plan = os.path.join(archive, 'inputs', 'statistical_analysis_plan.json')
    for p, name in ((manifest, 'manifest.json'), (profile, 'objective_profile.json'),
                    (split_registry, 'split_registry.json'), (contract_file, 'contract.json'),
                    (statistical_plan, 'statistical_analysis_plan.json')):
        if not os.path.exists(p):
            raise SystemExit(f'archive 缺 inputs/{name}')

    print(f'VAL archive 校验通过（bundle={seal.get("bundle_sha256", "")[:16]}）；'
          f'active code hash 与封存一致')
    _run_formal_gate(role='val', manifest=manifest, profile=profile,
                     workers=args.workers, out=args.out, split_registry=split_registry,
                     coldchain_contract=contract_file, statistical_plan=statistical_plan)


if __name__ == '__main__':
    main()
