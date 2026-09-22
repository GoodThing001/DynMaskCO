"""PyVRP 环境身份检查与落盘（B2 步骤 1）。

统一正式版本 pyvrp==0.14.0（两端一致）：
  - adapter 启动强制断言版本 == 0.14.0；
  - 强制确认导入来自独立环境的 site-packages，而不是 CC_Compare/PyVRP 源码树；
  - 落盘：版本 / pyvrp.__file__ / _pyvrp 二进制扩展 hash / Python / NumPy /
    wheel sha256 / pip freeze / 平台。

参考资产：CC_Compare/PyVRP 上游源码快照（commit 4b2aa285，1.0.0a0）标记为
reference_only，不作为运行依赖。
"""
import hashlib
import os
import platform
import subprocess
import sys

REQUIRED_PYVRP_VERSION = '0.14.0'
WHEEL_SHA256 = 'b3ce60d4714be39dbd382c40296fb754d649b1bfe5b462c9748ff50786d8c12f'  # cp312 win_amd64


class EnvironmentError(RuntimeError):
    pass


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_environment():
    """启动强制检查。返回环境身份 dict。"""
    from importlib.metadata import version

    installed = version('pyvrp')
    if installed != REQUIRED_PYVRP_VERSION:
        raise EnvironmentError(
            f'pyvrp 版本不匹配: 安装={installed} 要求={REQUIRED_PYVRP_VERSION}'
            f'（统一正式版本，禁用 1.0.0a0 main 源码）')

    import pyvrp
    pkg_file = os.path.normpath(pyvrp.__file__)
    if 'site-packages' not in pkg_file.replace('\\', '/'):
        raise EnvironmentError(
            f'pyvrp 导入路径不在 site-packages（疑似导入开发源码树）: {pkg_file}')

    # _pyvrp 原生扩展 hash（二进制身份）
    import pyvrp._pyvrp as native
    native_file = getattr(native, '__file__', None)
    if native_file is None:
        raise EnvironmentError('无法定位 pyvrp._pyvrp 原生扩展')
    native_file = os.path.normpath(native_file)
    native_hash = _sha256_file(native_file)

    import numpy
    identity = {
        'pyvrp_version': installed,
        'pyvrp_file': pkg_file,
        '_pyvrp_file': native_file,
        '_pyvrp_sha256': native_hash,
        'wheel_sha256': WHEEL_SHA256,
        'python_version': sys.version,
        'numpy_version': numpy.__version__,
        'platform': platform.platform(),
        'executable': sys.executable,
    }
    return identity


def environment_identity(include_freeze=False):
    """完整身份（含 pip freeze），用于 manifest 落盘。"""
    identity = check_environment()
    if include_freeze:
        try:
            out = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                 capture_output=True, text=True, timeout=60)
            identity['pip_freeze'] = out.stdout.strip().splitlines()
        except Exception as exc:  # noqa: BLE001
            identity['pip_freeze_error'] = repr(exc)
    return identity


if __name__ == '__main__':
    import json
    ident = environment_identity(include_freeze=True)
    print(json.dumps(ident, indent=2, ensure_ascii=False))
