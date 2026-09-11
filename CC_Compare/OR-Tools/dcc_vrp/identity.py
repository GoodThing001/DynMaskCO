"""OR-Tools 环境身份（OR0 / OR4.2）。

统一正式版本 ortools==9.11.4210（两端一致）；独立 cc_ortools env
（py3.12，不装进 MASKCO_env / cc_pyvrp）。adapter 启动强制版本断言 +
site-packages 导入路径检查。

身份字段（OR4.2 P0-2 修正）：
  - wrapper_sha256           = pywrapcp.py（Python 包装层）文件 hash；
  - native_extension_sha256  = _pywrapcp.pyd/.so（真实原生扩展）文件 hash
    —— 之前误把 pywrapcp.__file__（wrapper）当 native 记录，已修正；
  - wheel_sha256             = 从磁盘 assets/ 下的 wheel 实算，并与
    WHEEL_SHA256 预期值比对（不再只返回硬编码常量）；
  - Windows/Linux 是不同环境身份（wheel/.pyd vs wheel/.so），native hash
    不要求跨平台相同——跨平台 parity 用两份环境身份 + 相同 decision/output 对账。
"""
import hashlib
import os
import platform
import subprocess
import sys

REQUIRED_ORTOOLS_VERSION = '9.11.4210'
WHEEL_SHA256 = 'bc1b6e4cc0a121ef888481a99194765e6df72d4d3da81f928543171a2bac8cbb'  # cp312 win_amd64
_WHEEL_NAME = 'ortools-9.11.4210-cp312-cp312-win_amd64.whl'


class EnvironmentError(RuntimeError):
    pass


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_environment():
    from importlib.metadata import version
    installed = version('ortools')
    if installed != REQUIRED_ORTOOLS_VERSION:
        raise EnvironmentError(
            f'ortools 版本不匹配: 安装={installed} 要求={REQUIRED_ORTOOLS_VERSION}')

    import ortools.constraint_solver.pywrapcp as wrapper
    import ortools.constraint_solver._pywrapcp as native
    wrapper_file = os.path.normpath(wrapper.__file__)
    native_file = os.path.normpath(native.__file__)
    for label, f in (('wrapper', wrapper_file), ('native', native_file)):
        if 'site-packages' not in f.replace('\\', '/'):
            raise EnvironmentError(
                f'ortools {label} 导入路径不在 site-packages: {f}')
    if os.path.splitext(wrapper_file)[1] != '.py':
        raise EnvironmentError(f'wrapper 路径异常: {wrapper_file}')
    wrapper_sha256 = _sha256_file(wrapper_file)
    native_extension_sha256 = _sha256_file(native_file)

    # wheel 从磁盘实算并与预期比对
    wheel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'assets', _WHEEL_NAME)
    if not os.path.exists(wheel_path):
        raise EnvironmentError(f'冻结 wheel 不存在: {wheel_path}')
    actual_wheel = _sha256_file(wheel_path)
    if actual_wheel != WHEEL_SHA256:
        raise EnvironmentError(
            f'wheel 磁盘 hash 与预期不一致: 磁盘={actual_wheel} 预期={WHEEL_SHA256}')

    import numpy
    return {
        'ortools_version': installed,
        'wrapper_file': wrapper_file,
        'wrapper_sha256': wrapper_sha256,
        'native_file': native_file,
        'native_extension_sha256': native_extension_sha256,
        'wheel_sha256': actual_wheel,
        'python_version': sys.version,
        'numpy_version': numpy.__version__,
        'platform': platform.platform(),
        'executable': sys.executable,
    }


def environment_identity(include_freeze=False):
    identity = check_environment()
    if include_freeze:
        try:
            out = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                 capture_output=True, text=True, timeout=60)
            identity['pip_freeze'] = out.stdout.strip().splitlines()
        except Exception as exc:  # noqa: BLE001
            identity['pip_freeze_error'] = repr(exc)
    return identity
