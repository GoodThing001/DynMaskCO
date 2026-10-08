"""LIH 臂 1 天诊断：统计 provider 调用/缓存命中/墙钟（workers=1 同进程，仅调试）。"""
import os
import sys
import time

_BASE = '/home/hzeng/project/MASKCO-Main'
_CVRP = os.path.join(_BASE, 'C-VRP_Cold-chainVehicleRoutingProblem')
sys.path.insert(0, os.path.join(_CVRP, 'scripts', 'evaluation'))
sys.path.insert(0, os.path.join(_BASE, 'CC_Compare', 'Learn-Improvement-Heuristics',
                                 'a1_accept_v1'))

import run_a1_external_accept as drv  # noqa: E402

from lih_provider import LIHProvider  # noqa: E402

_orig = LIHProvider.order
STATS = {'calls': 0, 'hits': 0, 'solve_wall': 0.0}


def counted(self, sp):
    STATS['calls'] += 1
    t0 = time.perf_counter()
    r = _orig(self, sp)
    STATS['solve_wall'] += time.perf_counter() - t0
    STATS['hits'] = self.n_cache_hits
    return r


LIHProvider.order = counted
t_start = time.time()
rep = drv.main([
    '--solver', 'ordering', '--provider', 'lih',
    '--provider-ckpt', os.path.join(_BASE, 'CC_Compare',
                                    'Learn-Improvement-Heuristics', 'CVRP',
                                    'CVRP50', 'outputs', 'cvrp_50', 'run',
                                    'epoch-199.pt'),
    '--provider-device', 'cpu',
    '--baseline-from', os.path.join(_CVRP, 'results',
                                    'a1_step2_gate_c1_20260926', 'gate.json'),
    '--dev-instances', '1', '--workers', '1',
    '--penalty', 'p_c=0',
    '--out', os.path.join(_CVRP, 'results', 'a1_ext_extra', '_diag_lih'),
])
g = rep['results']['20260926']['p_c=0']
a = g['arms']['ordering_accept']
print('DIAG: wall=%.1fs decisions=%s calls=%s cache_hits=%s solve_wall=%.1fs '
      'mean_decide=%.2fs fail=%s timeouts=%s'
      % (time.time() - t_start, g['solver_meta']['n_solves'],
         STATS['calls'], STATS['hits'], STATS['solve_wall'],
         g['solver_meta']['mean_solve_time_s'], a['fail_counts'],
         a['mean_timeouts']))
