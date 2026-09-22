"""生成严格的 DEV/VAL 独立 split 与 manifest（统一 schema）。

与 `generate_dev_split.py` 同 generator/参数（R1/C1/RC1 × EDoD 0.2/0.5/0.8，默认
temp_dist，capacity=50，problem_size=50），只换 seed；额外写入统一 schema
`o0cc-split-manifest-v1`，登记 generator/contract/profile identity，并对每个
`--disjoint-manifest` 做 instance_seeds 与 scene_instance_ids 的无重叠硬校验。

VAL 的 root_seed 不应现在确定；应在 CAL-PHYS 与新 DEV 路线确认完成后一次性登记并冻结。
本脚本是框架：只有显式传入 `--seed`（root_seed）并冻结身份后才会产出正式 VAL manifest。

用法（服务器）：
    python scripts/data/generate_o0cc_split.py \
        --split-role val --seed 123456 --num-instances 128 \
        --profile <frozen_profile.json> \
        --disjoint-manifest <train_manifest> \
        --disjoint-manifest <dev_cal_manifest> \
        --out data/baseline/50_node/val
"""
import argparse
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))          # scripts/data
_SCRIPTS = os.path.dirname(_BASE)                           # scripts
_EVAL = os.path.join(_SCRIPTS, 'evaluation')
for p in (_SCRIPTS, _EVAL):
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_coldchain_data import generate_dataset
import formal_gate_contract as contract

TYPES = ('R1', 'C1', 'RC1')
EDODS = (0.2, 0.5, 0.8)
SCHEMA = 'o0cc-split-manifest-v1'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-role', required=True, choices=['dev_proto', 'dev_gate', 'val'])
    ap.add_argument('--seed', type=int, required=True, help='root_seed（一次性登记并冻结）')
    ap.add_argument('--num-instances', type=int, default=128)
    ap.add_argument('--out', required=True)
    ap.add_argument('--profile', default=None,
                    help='frozen objective_profile.json；用于登记 profile/contract identity')
    ap.add_argument('--disjoint-manifest', action='append', default=None,
                    help='已登记 split 的 manifest（独立性校验，可重复）')
    args = ap.parse_args(argv)

    if args.num_instances != contract.INSTANCES_PER_CELL:
        raise SystemExit(f'正式门控要求每 cell {contract.INSTANCES_PER_CELL} 实例，'
                         f'得到 {args.num_instances}')

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    inst_seeds = rng.integers(0, 2**31, size=args.num_instances, dtype=np.int32)
    cells = []
    for t in TYPES:
        for e in EDODS:
            ed = f'{int(e*10):02d}'
            fname = f'dcc_50_{t.lower()}_edod{ed}_{args.split_role}.npz'
            out_path = os.path.join(args.out, fname)
            print(f'generating {fname} (type={t}, edod={e}, n={args.num_instances}, '
                  f'seed={args.seed})')
            dataset = generate_dataset(args.num_instances, 50, t, 50, seed=args.seed, edod=e)
            np.savez_compressed(out_path, **dataset)
            cells.append({
                'type': t, 'edod': e, 'file': fname,
                'sha256': contract.sha256_file(out_path),
                'num_instances': args.num_instances, 'seed': args.seed,
                'scene_instance_ids': [
                    f'{args.split_role}__dcc_50_{t.lower()}_edod{ed}__{i}'
                    for i in range(args.num_instances)],
                'instance_seeds': [int(s) for s in inst_seeds],
            })

    manifest = {
        'schema': SCHEMA,
        'split_role': args.split_role,
        'root_seed': args.seed,
        'problem_size': 50,
        'capacity': 50,
        'instances_per_cell': args.num_instances,
        'cells': cells,
        'generator_identity': {
            'generator': 'generate_coldchain_data.py',
            'sha256': contract.sha256_file(
                os.path.join(_SCRIPTS, 'data', 'generate_coldchain_data.py')),
        },
        'contract_identity': {},
        'profile_identity': {},
        'disjoint_from': [],
    }
    if args.profile:
        profile = contract.load_json(args.profile)
        manifest['profile_identity'] = {
            'name': profile.get('name'),
            'profile_hash': profile.get('profile_hash'),
        }
        manifest['contract_identity'] = {
            'contract_hash': (profile.get('provenance') or {}).get('contract_hash'),
        }
    if args.disjoint_manifests:
        report = contract.check_disjoint(manifest, args.disjoint_manifests)
        bad = [e for e in report if e['seed_overlap'] or e['scene_id_overlap']]
        if bad:
            raise SystemExit('生成的 split 与已登记 split 重叠（独立性违反）：'
                             f'\n{json.dumps(bad, indent=2, ensure_ascii=False)}')
        manifest['disjoint_from'] = report

    out_manifest = os.path.join(args.out, f'{args.split_role.upper()}_MANIFEST.json')
    with open(out_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'saved {out_manifest} (split_role={args.split_role}, root_seed={args.seed}, '
          f'{len(cells)} cells × {args.num_instances})')


if __name__ == '__main__':
    main()
