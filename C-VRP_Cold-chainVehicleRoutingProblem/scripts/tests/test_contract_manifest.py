"""ColdChainContract 序列化/反序列化 loader 与严格 schema 测试（纯 stdlib，不跑 oracle）。

覆盖 Step 14a 接口骨架：
  - `default_pilot_contract().to_manifest()` → JSON → `load_coldchain_contract` 往返 hash 一致；
  - `validate_contract_manifest` 拒绝：缺 section / 未知字段 / 未知顶层字段 / 坏 hash /
    错 schema；
  - `write_pilot_contract` + `load_coldchain_contract` 文件往返一致。

用法：python scripts/tests/test_contract_manifest.py
"""
import json
import os
import sys
import tempfile

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
for p in (os.path.join(_BASE, 'coldchain'), _BASE):
    if p not in sys.path:
        sys.path.insert(0, p)

from coldchain_contract import (
    ColdChainContract, SCHEMA_VERSION, default_pilot_contract,
    load_coldchain_contract, validate_contract_manifest, write_pilot_contract,
)

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _expect_fail(fn, *args):
    try:
        fn(*args)
        return False, '未拒绝'
    except ValueError:
        return True, None


def test_roundtrip_hash_consistent():
    c = default_pilot_contract()
    m = c.to_manifest()
    ok = m['contract_hash'] == c.contract_hash and len(c.contract_hash) == 64
    # JSON 往返（tuples -> lists）后 hash 必须不变
    data = json.loads(json.dumps(m))
    c2 = ColdChainContract.from_manifest(data)
    ok = ok and c2.contract_hash == c.contract_hash
    record('contract_roundtrip_hash_consistent', ok, f"hash={c.contract_hash[:12]}")
    return ok


def test_file_roundtrip():
    tmp = tempfile.mkdtemp(prefix='contract_mf_')
    try:
        p = os.path.join(tmp, 'pilot_contract.json')
        write_pilot_contract(p)
        c = load_coldchain_contract(p)
        ok = c.contract_hash == default_pilot_contract().contract_hash
        record('contract_file_roundtrip', ok, f"hash={c.contract_hash[:12]}")
        return ok
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_validate_rejects():
    m = default_pilot_contract().to_manifest()
    ok = True
    # 缺 section
    d = json.loads(json.dumps(m)); del d['thermal']
    ok &= _expect_fail(validate_contract_manifest, d)[0]
    # 未知顶层字段
    d = json.loads(json.dumps(m)); d['foo'] = 1
    ok &= _expect_fail(validate_contract_manifest, d)[0]
    # 未知 section 字段
    d = json.loads(json.dumps(m)); d['units']['extra'] = 1
    ok &= _expect_fail(validate_contract_manifest, d)[0]
    # 错 schema
    d = json.loads(json.dumps(m)); d['schema_version'] = 'bad'
    ok &= _expect_fail(validate_contract_manifest, d)[0]
    # 缺失 contract_hash → 拒绝（P1：必填，不是可选字段）
    d = json.loads(json.dumps(m)); del d['contract_hash']
    ok &= _expect_fail(validate_contract_manifest, d)[0]
    # 坏 contract_hash（非 64-hex）
    d = json.loads(json.dumps(m)); d['contract_hash'] = 'zz'
    ok &= _expect_fail(validate_contract_manifest, d)[0]
    # hash 与内容不符 → from_manifest 拒绝
    d = json.loads(json.dumps(m)); d['contract_hash'] = '0' * 64
    ok &= _expect_fail(ColdChainContract.from_manifest, d)[0]
    record('contract_validate_rejects', ok)
    return ok


def test_explicit_pilot_equals_implicit():
    """显式 pilot JSON 与隐式 default_pilot_contract 语义完全一致（Step 14a parity 基础）。"""
    tmp = tempfile.mkdtemp(prefix='contract_parity_')
    try:
        p = os.path.join(tmp, 'pilot.json')
        write_pilot_contract(p)
        loaded = load_coldchain_contract(p)
        implicit = default_pilot_contract()
        ok = loaded.contract_hash == implicit.contract_hash
        from coldchain_contract import apply_objective_profile, ObjectiveProfile
        prof = ObjectiveProfile('p', 18.0, 2.9, 1193.0, 1.0, 1.0, 'devmean')
        ok = ok and apply_objective_profile(loaded, prof).contract_hash \
             == apply_objective_profile(implicit, prof).contract_hash
        record('explicit_pilot_equals_implicit', ok, f"hash={loaded.contract_hash[:12]}")
        return ok
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ok = [test_roundtrip_hash_consistent(), test_file_roundtrip(), test_validate_rejects(),
          test_explicit_pilot_equals_implicit()]
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
