"""生成独立 DEV-CAL 数据（决策：新建独立 dev_cal，不从 train 截取，不用 VAL）。

9 个 cell：R1/C1/RC1 × EDoD 0.2/0.5/0.8，每 cell 128 个独立实例，seed=777（与
train/val/test 默认 seed 不同）。保存生成参数、代码版本（生成器 sha256）、实例 ID 与
文件 sha256 到 DEV_CAL_MANIFEST.json。

DEV-CAL 只用于 scale / 权重邻域 / 参数诊断，不进训练、O0 Gate 或 TEST。
"""
import argparse, hashlib, json, os, subprocess, sys

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT

_CVRPTW = str(EXTENSION_ROOT)
CELLS = [(t, e) for t in ('R1', 'C1', 'RC1') for e in ('0.2', '0.5', '0.8')]
SEED = 777
GENERATOR = os.path.join(_CVRPTW, 'scripts', 'data', 'generate_coldchain_data.py')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='生成 DEV-CAL 数据')
    parser.add_argument('--out_dir', default='data/baseline/50_node/dev_cal')
    args = parser.parse_args()

    out_dir = os.path.join(_CVRPTW, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    generator_sha = _sha256(GENERATOR)
    manifest = {
        'seed': SEED,
        'problem_size': 50,
        'num_instances_per_cell': 128,
        'capacity': 50,
        'cells': [],
        'generator': 'generate_coldchain_data.py',
        'generator_sha256': generator_sha,
    }

    for t, edod in CELLS:
        name = f"dcc_50_{t.lower()}_edod{edod.replace('.', '')}_devcal.npz"
        out_path = os.path.join(out_dir, name)
        cmd = [sys.executable, GENERATOR,
               '--problem_size', '50', '--num_instances', '128',
               '--type', t, '--capacity', '50', '--edod', edod,
               '--seed', str(SEED), '--output', out_path]
        subprocess.run(cmd, check=True, cwd=_CVRPTW)
        file_sha = _sha256(out_path)
        manifest['cells'].append({
            'type': t, 'edod': float(edod), 'file': name, 'sha256': file_sha,
            'num_instances': 128, 'seed': SEED,
        })
        print(f"  {name}: sha256={file_sha[:16]}", flush=True)

    with open(os.path.join(out_dir, 'DEV_CAL_MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"saved: {out_dir}/DEV_CAL_MANIFEST.json")


if __name__ == '__main__':
    main()
