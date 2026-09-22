"""生成 M0 研发用 fresh 独立实例集（50-node R1 EDoD=0.5）。

修复 seed 身份：逐实例显式 seed 驱动 `generate_coldchain_instance`（不再依赖 `generate_dataset`
内部隐式抽 seed），manifest 登记的 `instance_seeds` 就是真正驱动该实例生成的 seed。

三集合明确分离（root_seed 独立，与 DEV-CAL(777)/DEV-PROTO(7781)/DEV-GATE(7782)/VAL/TEST
不同源）：
  - TRAIN-64  (root_seed 9701)：参与梯度更新；
  - CAL-16    (root_seed 9702)：选择接受 margin；
  - DEV-CHECK-16 (root_seed 9703)：margin 固定后评价，标签不回灌训练。

对实例内容做去重检查（coords 的 bytes 指纹），保留旧数据不覆盖。

用法（服务器）：
    python scripts/data/generate_m0_teacher_split.py \
        --train-instances 64 --cal-instances 16 --dev-check-instances 16 --out data/m0_scale
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))          # scripts/data
_SCRIPTS = os.path.dirname(_BASE)                           # scripts
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from generate_coldchain_data import generate_coldchain_instance

TRAIN_SEED = 9701
CAL_SEED = 9702
DEV_CHECK_SEED = 9703
SCHEMA = 'm0-teacher-split-v2'

_FLOAT_KEYS = ('coord', 'tw_', 'service', 'reveal', 'dist_mat', 'quality', 'energy', 'cost')


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _stack(insts):
    keys = ['coords', 'demands', 'tw_start', 'tw_end', 'service_time', 'temp_class',
            'reveal_time', 'visible_mask', 'quality_loss', 'energy_mat',
            'legacy_delivery_reference_quality', 'legacy_edge_energy_proxy', 'initial_quality',
            'routes', 'opt_costs', 'dist_mat']
    out = {}
    for k in keys:
        arrs = []
        for inst in insts:
            arr = inst[k]
            if k == 'routes':
                arr = np.pad(arr, (0, max(0, 200 - len(arr))), constant_values=0)[:200]
            arrs.append(arr)
        dtype = np.float32 if any(kk in k for kk in _FLOAT_KEYS) else np.int32
        out[k] = np.stack(arrs, axis=0).astype(dtype)
    return out


def _gen_cell(n_instances, root_seed, split_role):
    rng = np.random.default_rng(root_seed)
    inst_seeds = [int(s) for s in rng.integers(0, 2 ** 31, size=n_instances, dtype=np.int32)]
    insts = [generate_coldchain_instance(50, 'R1', 50, seed=s, edod=0.5)
             for s in inst_seeds]
    dataset = _stack(insts)
    scene_ids = [f'm0_{split_role}__dcc_50_r1_edod05__{i}' for i in range(n_instances)]
    return dataset, inst_seeds, scene_ids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-instances', type=int, default=64)
    ap.add_argument('--cal-instances', type=int, default=16)
    ap.add_argument('--dev-check-instances', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    if args.train_instances <= 0 or args.cal_instances <= 0 or args.dev_check_instances <= 0:
        raise SystemExit('--train/--cal/--dev-check instances 必须为正')

    os.makedirs(args.out, exist_ok=True)
    roles = [('train_teacher', args.train_instances, TRAIN_SEED),
             ('cal_teacher', args.cal_instances, CAL_SEED),
             ('dev_check_teacher', args.dev_check_instances, DEV_CHECK_SEED)]
    cells = []
    seen_coords = {}
    for role, n, seed in roles:
        fname = f'dcc_50_r1_edod05_{role}.npz'
        out_path = os.path.join(args.out, fname)
        dataset, inst_seeds, scene_ids = _gen_cell(n, seed, role)
        # 去重检查：coords bytes 指纹不跨集合重复
        for i in range(n):
            fp = hashlib.sha256(dataset['coords'][i].tobytes()).hexdigest()[:16]
            if fp in seen_coords:
                raise SystemExit(f'{role}[{i}] coords 与 {seen_coords[fp]} 重复（seed 重叠）')
            seen_coords[fp] = f'{role}[{i}]'
        np.savez_compressed(out_path, **dataset)
        cells.append({
            'type': 'R1', 'edod': 0.5, 'file': fname, 'sha256': _sha256_file(out_path),
            'num_instances': n, 'seed': seed, 'instance_seeds': inst_seeds,
            'scene_instance_ids': scene_ids,
        })
        print(f'  {fname}: n={n} seed={seed} sha256={cells[-1]["sha256"][:16]}', flush=True)

    manifest = {
        'schema': SCHEMA,
        'purpose': 'M0 扩规模开发（TRAIN-64 / CAL-16 / DEV-CHECK-16，非论文样本量）',
        'problem_size': 50, 'capacity': 50, 'edod': 0.5, 'type': 'R1',
        'train_seed': TRAIN_SEED, 'cal_seed': CAL_SEED, 'dev_check_seed': DEV_CHECK_SEED,
        'generator': 'generate_coldchain_data.py',
        'generator_sha256': _sha256_file(os.path.join(_BASE, 'generate_coldchain_data.py')),
        'cells': cells,
        'disjoint_from': ['DEV-CAL(777)', 'DEV-PROTO(7781)', 'DEV-GATE(7782)', 'VAL128', 'TEST'],
    }
    mpath = os.path.join(args.out, 'M0_TEACHER_MANIFEST.json')
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'saved {mpath}')


if __name__ == '__main__':
    main()
