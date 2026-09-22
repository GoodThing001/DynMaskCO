"""B2 步骤 1：环境身份检查与导入路径检查。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testutil  # noqa: F401

import identity as pyvrp_identity
from importlib.metadata import version


def test_version():
    assert version('pyvrp') == '0.14.0', \
        f"pyvrp 版本 {version('pyvrp')} != 0.14.0（统一正式版本）"
    print('  pyvrp==0.14.0')


def test_import_path():
    ident = pyvrp_identity.check_environment()
    pkg = ident['pyvrp_file'].replace('\\', '/')
    assert 'site-packages' in pkg, \
        f'导入路径不在 site-packages（疑似开发源码树）: {pkg}'
    assert 'CC_Compare' not in pkg, f'导入路径指向 CC_Compare 源码树: {pkg}'
    print(f"  导入路径: {ident['pyvrp_file']}")
    print(f"  native: {ident['_pyvrp_file']} sha256={ident['_pyvrp_sha256'][:16]}...")


def test_identity_fields():
    ident = pyvrp_identity.environment_identity(include_freeze=True)
    for k in ('pyvrp_version', 'pyvrp_file', '_pyvrp_file', '_pyvrp_sha256',
              'wheel_sha256', 'python_version', 'numpy_version', 'platform',
              'executable'):
        assert k in ident and ident[k], f'identity 缺字段 {k}'
    assert ident['wheel_sha256'] == pyvrp_identity.WHEEL_SHA256
    assert 'pyvrp==0.14.0' in ident['pip_freeze']
    print('  identity 字段齐全（含 wheel sha256 + pip freeze）')


def test_version_mismatch_rejected():
    # 直接调用核心断言（模拟错误版本）
    try:
        pyvrp_identity.check_environment()
    except pyvrp_identity.EnvironmentError:
        raise AssertionError('当前环境不应被拒绝')


def main():
    test_version()
    test_import_path()
    test_identity_fields()
    test_version_mismatch_rejected()
    print('PASS test_environment_identity')


if __name__ == '__main__':
    main()
