"""CC_Compare/common 的路径引导：把项目 scripts 各子目录插到 sys.path（无 __init__.py 约定）。

任何 common/ 模块（含 tests）首先 import _bootstrap，随后可直接
`from strict_online_env import ...` 等。路径由本文件位置推导（与 cwd 无关），
并调用项目的 project_paths.validate_layout() 做布局契约校验。
"""
import os
import sys

_COMMON_ROOT = os.path.dirname(os.path.abspath(__file__))
_CC_COMPARE_ROOT = os.path.dirname(_COMMON_ROOT)
_WORKSPACE_ROOT = os.path.dirname(_CC_COMPARE_ROOT)
_EXTENSION_ROOT = os.path.join(_WORKSPACE_ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem')
_SCRIPTS_ROOT = os.path.join(_EXTENSION_ROOT, 'scripts')

for p in (_SCRIPTS_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from project_paths import validate_layout  # noqa: E402

validate_layout()

for _sub in ('simulation', 'evaluation', 'coldchain'):
    _p = os.path.join(_SCRIPTS_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROJECT_SCRIPTS_ROOT = _SCRIPTS_ROOT
PROJECT_EXTENSION_ROOT = _EXTENSION_ROOT
