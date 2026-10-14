"""记录训练身份：checkpoint hash + 数据 hash + 关键源码 hash + summary/state_manifest。

写入 training_identity.json，供后续冻结与复现。
"""
import argparse
import hashlib
import json
import os

SRC_FILES = [
    'scripts/training/train_mpre_reinforce.py',
    'scripts/simulation/dynmaskco_cc_context.py',
    'scripts/simulation/cc_lns_replanner.py',
    'scripts/simulation/mtrained_replanner.py',
    'scripts/simulation/mpre_policy.py',
    'scripts/models/mpre.py',
    'scripts/models/mpre_trained.py',
    'scripts/simulation/action_contract.py',
    'scripts/simulation/visible_state.py',
    'scripts/simulation/strict_online_env.py',
    'scripts/data/repair_state.py',
    'scripts/coldchain/coldchain_contract.py',
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    ident = {
        'model_ckpt_sha256': sha256(os.path.join(args.out_dir, 'model.ckpt')),
        'data_sha256': sha256(os.path.join(args.root, args.data)),
        'src_sha256': {p: sha256(os.path.join(args.root, p)) for p in SRC_FILES},
    }
    for name in ('summary.json', 'state_manifest.json'):
        p = os.path.join(args.out_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                ident[name.replace('.json', '')] = json.load(f)
    out = os.path.join(args.out_dir, 'training_identity.json')
    with open(out, 'w') as f:
        json.dump(ident, f, indent=2)
    print(f"saved: {out}")
    print(json.dumps({'model_ckpt_sha256': ident['model_ckpt_sha256'],
                      'data_sha256': ident['data_sha256']}, indent=2))


if __name__ == '__main__':
    main()
