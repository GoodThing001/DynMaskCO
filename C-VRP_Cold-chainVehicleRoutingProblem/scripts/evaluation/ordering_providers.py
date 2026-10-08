"""P2–P4 学习排序 provider 注册表（2026-09-29）。

solver=ordering 时 run_a1_external_accept.py 经此把方法名映射到 provider 构造。
每个方法的 provider 是其独立文件（保留方法自身机制，只实现 order(sp) 接口），
放在方法各自的 CC_Compare/<方法>/a1_accept_v1/ 下；这里只插入 sys.path 并构造。
新增方法 = 写好 provider 文件后在 _REGISTRY 加一行，不改动 driver。
"""
import importlib
import os
import sys

_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

_REGISTRY = {
    # name: (相对仓库根的 provider 目录, 模块名, 类名)
    'routefinder': ('CC_Compare/RouteFinder/a1_accept_v1',
                    'routefinder_provider', 'RouteFinderProvider'),
    'cada': ('CC_Compare/CaDA/a1_accept_v1', 'cada_provider', 'CaDAProvider'),
    'mvmoe': ('CC_Compare/MVMoE/a1_accept_v1', 'mvmoe_provider', 'MVMoEProvider'),
    'pomo': ('CC_Compare/POMO/a1_accept_v1', 'pomo_provider', 'POMOProvider'),
    'attention': ('CC_Compare/AttentionModel/a1_accept_v1',
                  'am_provider', 'AMProvider'),
    'symnco': ('CC_Compare/Sym-NCO/a1_accept_v1',
               'symnco_provider', 'SymNCOProvider'),
    'omnivrp': ('CC_Compare/Omni-VRP/a1_accept_v1',
                'omnivrp_provider', 'OmniVRPProvider'),
    'sgbs': ('CC_Compare/SGBS/a1_accept_v1', 'sgbs_provider', 'SGBSProvider'),
    'deepaco': ('CC_Compare/DeepACO/a1_accept_v1',
                'deepaco_provider', 'DeepACOProvider'),
    'lih': ('CC_Compare/Learn-Improvement-Heuristics/a1_accept_v1',
            'lih_provider', 'LIHProvider'),
}


def build_ordering_provider(name, ckpt, device='cuda', **kw):
    if name not in _REGISTRY:
        raise SystemExit('unknown --provider: %r (known: %s)'
                         % (name, sorted(_REGISTRY)))
    rel_dir, mod, cls = _REGISTRY[name]
    d = os.path.normpath(os.path.join(_ROOT, *rel_dir.split('/')))
    if not os.path.isdir(d):
        raise SystemExit('provider dir missing: %s（先落 provider 文件并上传）' % d)
    if d not in sys.path:
        sys.path.insert(0, d)
    m = importlib.import_module(mod)
    return getattr(m, cls)(ckpt, device=device, **kw)


_EXT_ROOT = os.path.normpath(os.path.join(
    _ROOT, 'C-VRP_Cold-chainVehicleRoutingProblem'))


def provider_file(name):
    """provider 源文件路径（相对扩展根，含 ../.. 前缀）——供 run_a1_external_accept
    的源码封存（Q-04 身份）把方法 provider 纳入启动/结束 hash，防「磁盘版本代替
    实际加载版本」。"""
    rel_dir, mod, _cls = _REGISTRY[name]
    p = os.path.normpath(os.path.join(_ROOT, *rel_dir.split('/'), mod + '.py'))
    return os.path.relpath(p, _EXT_ROOT).replace(os.sep, '/')
