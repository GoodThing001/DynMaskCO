"""RRNCO-Ordering-RH-D R1 冻结：一键生成器（tools/gen_freeze.py）。

按顺序重算并写出完整身份链（纯 stdlib，可重跑复验）：
  SOURCE_MANIFEST.json → FROZEN_CONFIG.json（含 manifest sha）→ FREEZE_SEAL.json
  （含 manifest/config/evidence/checkpoint sha）→ verify_freeze 对账。

用法（仓库根）：
    python CC_Compare/RRNCO/dcc_rh_v4/tools/gen_freeze.py [--revision N]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

_DCC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_SHA = 'b1ff3191ee2f28c19e4807748b3c1b83522e1e96c6fb7bf9238442cf859464b3'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd, cwd):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
        return out.returncode, out.stdout.strip(), out.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return -1, '', repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--revision', type=int, default=1)
    args = ap.parse_args()

    # 1) SOURCE_MANIFEST
    rc, so, se = _run([sys.executable, os.path.join(_DCC, 'tools', 'gen_source_manifest.py')],
                      os.path.dirname(_DCC))
    if rc != 0:
        print('gen_source_manifest failed:', so, se, file=sys.stderr)
        sys.exit(1)
    manifest_hash = sha256_file(os.path.join(_DCC, 'SOURCE_MANIFEST.json'))
    ev_hash = sha256_file(os.path.join(_DCC, 'results', 'R0_5_EVIDENCE_MANIFEST.json'))

    # 2) FROZEN_CONFIG
    cfg_path = os.path.join(_DCC, 'FROZEN_CONFIG.json')
    cfg = json.load(open(cfg_path, encoding='utf-8')) if os.path.exists(cfg_path) else {}
    cfg.update({
        'schema_version': 'rrnco-dcc-rh-v4-frozen-config-v1',
        'revision': args.revision,
        'method': 'RRNCO-Ordering-RH-D',
        'scope': '旧 strict-online DCC-VRP 协议（SubProblem/coordinator/公共 Bridge 管线）; '
                 'A-v1 accept 协议适配不在此冻结内',
        'source_manifest_sha256': manifest_hash,
        'backend_config': {
            'checkpoint_path': 'CC_Compare/RRNCO/checkpoints/rcvrptw/epoch_199.ckpt',
            'checkpoint_sha256': CKPT_SHA,
            'device': 'cuda', 'seed': 0, 't_max': 4.6, 'num_loc': 100,
            'normalize': True, 'sampling_policy': 'visible_prob_sampling_v1',
            'distance_sample_size': 25, 'native_backhaul_mapping': True,
        },
        'coordinator_rules': [
            'open-new-vehicle 优先（已启动车续用，空载 depot 闲置车才新开）',
            '归一化排名 → 增量距离 → vehicle_id → customer_id',
            '排序非法 → EDD fallback（带原因）',
            'provider 每车只调一次；SubProblem 无 view 引用；禁止默认 EDD',
        ],
        'certificate': {'tolerance': 1e-6, 'pickup_to_depot': True},
        'evidence': {'evidence_manifest_sha256': ev_hash,
                     'run_a_id': '802c3a4e7a1a471bb366d0101a7ff801',
                     'run_b_id': '0551f33ebe724eaabf18802c10750e12'},
        'decision_ref': 'R1_FREEZE_PLAN.md (2026-09-26, 用户授权)',
    })
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')
    cfg_hash = sha256_file(cfg_path)

    # 3) FREEZE_SEAL
    seal = {
        'schema_version': 'rrnco-dcc-rh-v4-seal-v1',
        'revision': args.revision,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'RRNCO-Ordering-RH-D',
        'source_manifest_sha256': manifest_hash,
        'frozen_config_sha256': cfg_hash,
        'evidence_manifest_sha256': ev_hash,
        'checkpoint_sha256': CKPT_SHA,
        'decision_ref': 'R1_FREEZE_PLAN.md (2026-09-26, 用户授权; R0.5 verify ALL PASS)',
    }
    seal_path = os.path.join(_DCC, 'FREEZE_SEAL.json')
    with open(seal_path, 'w', encoding='utf-8') as f:
        json.dump(seal, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')

    # 4) verify
    rc, so, se = _run([sys.executable, os.path.join(_DCC, 'verify_freeze.py'),
                       '--seal', seal_path], os.path.dirname(_DCC))
    print(so)
    if se:
        print(se, file=sys.stderr)
    sys.exit(rc)


if __name__ == '__main__':
    main()
