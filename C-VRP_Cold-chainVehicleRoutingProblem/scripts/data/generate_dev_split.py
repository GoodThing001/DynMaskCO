"""生成 DEV-PROTO / DEV-GATE 数据（与 DEV-CAL 同参数、新 seed），并写 manifest。

DEV-CAL(seed=777) 只用于 scale；DEV-PROTO 用于协议调试，DEV-GATE 用于一次性 lookahead 判定。
两者与 dev_cal 用相同 generator/参数（R1/C1/RC1 × EDoD 0.2/0.5/0.8，默认 temp_dist=[0.4,0.35,0.25]，
capacity=50，problem_size=50），只换 seed。

用法（服务器）：
    python scripts/data/generate_dev_split.py --seed 7781 --num_instances 32 --out data/baseline/50_node/dev_proto
    python scripts/data/generate_dev_split.py --seed 7782 --num_instances 128 --out data/baseline/50_node/dev_gate
"""
import argparse, os, sys, hashlib, json
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from generate_coldchain_data import generate_dataset

TYPES = ('R1', 'C1', 'RC1')
EDODS = (0.2, 0.5, 0.8)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--num_instances', type=int, required=True)
    ap.add_argument('--split-role', required=True,
                    help='dev_proto / dev_gate（写入 manifest.split_role + scene_instance_id）')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # 内部 instance seed（与 generate_dataset 一致：rng.integers(0, 2**31, size=n)）
    rng = np.random.default_rng(args.seed)
    inst_seeds = rng.integers(0, 2**31, size=args.num_instances, dtype=np.int32)
    cells = []
    for t in TYPES:
        for e in EDODS:
            ed = f'{int(e*10):02d}'
            fname = f'dcc_50_{t.lower()}_edod{ed}_{args.split_role}.npz'
            out_path = os.path.join(args.out, fname)
            print(f"generating {fname} (type={t}, edod={e}, n={args.num_instances}, seed={args.seed})")
            dataset = generate_dataset(args.num_instances, 50, t, 50, seed=args.seed, edod=e)
            np.savez_compressed(out_path, **dataset)
            cells.append({
                'type': t, 'edod': e, 'file': fname,
                'sha256': _sha256_file(out_path),
                'num_instances': args.num_instances, 'seed': args.seed,
                'scene_instance_ids': [
                    f"{args.split_role}__dcc_50_{t.lower()}_edod{ed}__{i}"
                    for i in range(args.num_instances)],
                'instance_seeds': [int(s) for s in inst_seeds],
            })

    manifest = {
        'split_role': args.split_role,
        'seed': args.seed,
        'problem_size': 50,
        'num_instances_per_cell': args.num_instances,
        'capacity': 50,
        'cells': cells,
        'generator': 'generate_coldchain_data.py',
        'generator_sha256': _sha256_file(os.path.join(_SCRIPTS, 'data', 'generate_coldchain_data.py')),
    }
    with open(os.path.join(args.out, 'DEV_MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"saved {args.out}/DEV_MANIFEST.json (split_role={args.split_role}, {len(cells)} cells)")


if __name__ == '__main__':
    main()
