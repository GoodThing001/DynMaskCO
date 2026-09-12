"""verify_r0_5.py — R0.5 只读核验器（不写任何结果，失败非零退出）。

验证一个 results 目录内的完整证据：
  - SERVER_ENVIRONMENT.json 存在 + checkpoint sha256
  - B2_RESULT.json verdict PASS 且 29 项全过
  - r0_5_run_a.json / r0_5_run_b.json verdict PASS + 全部 gate PASS
  - 双跑 run_id 不同；除 runtime/run_id 外决策证据完全一致
  - （可选）证据 manifest 逐文件 hash 对账

用法：
    python verify_r0_5.py --results CC_Compare/RRNCO/dcc_rh_v4/results
"""
import argparse
import hashlib
import json
import os
import sys

REQUIRED_GATES = ['B3_determinism', 'B4_snapshots', 'B4_public_chain',
                  'B5_variable_size', 'B6_future_perturbation',
                  'B7_model_contribution']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    args = ap.parse_args()
    R = args.results
    errors = []

    def require(cond, msg):
        if cond:
            print(f'  [PASS] {msg}')
        else:
            errors.append(msg)
            print(f'  [FAIL] {msg}')

    # ---- SERVER_ENVIRONMENT.json ----
    checkpoints = []
    senv_p = os.path.join(R, 'SERVER_ENVIRONMENT.json')
    if os.path.exists(senv_p):
        senv = json.load(open(senv_p))
        ckpt = senv.get('checkpoint', {}).get('sha256')
        require(bool(ckpt), 'SERVER_ENVIRONMENT.json checkpoint sha256 非空')
        require('dependencies' in senv, 'SERVER_ENVIRONMENT.json 含依赖版本')
        if ckpt:
            checkpoints.append(ckpt)
    else:
        require(False, 'SERVER_ENVIRONMENT.json 存在')

    # ---- B2_RESULT.json ----
    b2_p = os.path.join(R, 'B2_RESULT.json')
    if os.path.exists(b2_p):
        b2 = json.load(open(b2_p))
        require(b2.get('verdict') == 'PASS', f"B2 verdict PASS (={b2.get('verdict')})")
        require(b2.get('n_fail') == 0, f"B2 n_fail==0 (={b2.get('n_fail')})")
        require(b2.get('n_pass') == 29, f"B2 n_pass==29 (={b2.get('n_pass')})")
        boundary = b2.get('mask_parity_boundary', [])
        all_match = all(b.get('match') for b in boundary)
        require(len(boundary) == 9 and all_match,
                f'B2 mask parity 边界 ==9 且全 MATCH (n={len(boundary)})')
        if b2.get('checkpoint_sha256'):
            checkpoints.append(b2['checkpoint_sha256'])
    else:
        require(False, 'B2_RESULT.json 存在')

    # ---- run_a / run_b ----
    a_p = os.path.join(R, 'r0_5_run_a.json')
    b_p = os.path.join(R, 'r0_5_run_b.json')
    if os.path.exists(a_p) and os.path.exists(b_p):
        a = json.load(open(a_p))
        b = json.load(open(b_p))
        require(a.get('verdict') == 'PASS', f"run_a verdict PASS (={a.get('verdict')})")
        require(b.get('verdict') == 'PASS', f"run_b verdict PASS (={b.get('verdict')})")
        for x in (a, b):
            ck = x.get('preflight', {}).get('checkpoint_sha256')
            if ck:
                checkpoints.append(ck)
        ga = sorted(a.get('gates', {}).keys())
        gb = sorted(b.get('gates', {}).keys())
        require(ga == sorted(REQUIRED_GATES), f'run_a gate 集合精确 ==6 项 (={ga})')
        require(gb == sorted(REQUIRED_GATES), f'run_b gate 集合精确 ==6 项 (={gb})')
        for g in REQUIRED_GATES:
            for label, x in (('a', a), ('b', b)):
                require(x.get('gates', {}).get(g, {}).get('pass') is True,
                        f'run_{label} {g} PASS')
        require(a.get('run_id') != b.get('run_id'),
                f"run_id 不同 (a={a.get('run_id')} b={b.get('run_id')})")

        def strip(x):
            d = json.loads(json.dumps(x))
            for g in d.get('gates', {}).values():
                g.pop('runtime_s', None)
            d.pop('run_id', None)
            return d
        require(strip(a) == strip(b), 'run_a/run_b 决策证据一致（排除 runtime/run_id）')
    else:
        require(False, 'r0_5_run_a.json 与 r0_5_run_b.json 都存在')

    # checkpoint SHA 必须彼此相同
    require(len(set(checkpoints)) == 1,
            f'B0/B2/runA/runB checkpoint SHA 一致 (n_unique={len(set(checkpoints))})')

    # ---- manifest 必需 ----
    m_p = os.path.join(R, 'R0_5_EVIDENCE_MANIFEST.json')
    if os.path.exists(m_p):
        m = json.load(open(m_p))
        files = m.get('files', {})
        require(bool(files), 'manifest 含 files 清单')
        ok = True
        for name, exp in files.items():
            fp = os.path.join(R, name)
            if not os.path.exists(fp):
                ok = False
                print(f'    manifest 缺文件: {name}')
                continue
            got = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
            if got != exp:
                ok = False
                print(f'    manifest hash 不符: {name}')
        require(ok, 'manifest 逐文件 hash 对账一致')
    else:
        require(False, 'R0_5_EVIDENCE_MANIFEST.json 必需')

    if errors:
        print(f'\nverify_r0_5: {len(errors)} 项失败')
        for e in errors:
            print('  -', e)
        sys.exit(1)
    print('\nverify_r0_5: ALL PASS')


if __name__ == '__main__':
    main()
