#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rr3 冻结源码运行器：从 results/rr3_src_frozen 的隔离副本加载步骤 2 驱动并执行。

用法（由 tools_remote/tools_remote_run_rr3_frozen.sh 调用）：
  RR3_FROZEN_SRC=<冻结树> [RR3_EVIDENCE_OUT=<证据文件>] python3 rr3_run_frozen.py <驱动参数>
冻结树路径优先插入 sys.path，驱动与其 10 个封存文件全部从冻结副本加载。
RR3_EVIDENCE_OUT 设置时，把 10 个封存模块的实际导入路径 + 本运行器路径 + sys.path 前缀
写入该文件（启动时的可复查 import 证据，2026-10-01 用户决策补齐）。
"""
import os
import sys

frozen = os.environ.get('RR3_FROZEN_SRC')
if not frozen or not os.path.isdir(frozen):
    raise SystemExit('RR3_FROZEN_SRC 未设置或不存在')

for p in (frozen,
          os.path.join(frozen, 'scripts'),
          os.path.join(frozen, 'scripts', 'evaluation'),
          os.path.join(frozen, 'scripts', 'coldchain'),
          os.path.join(frozen, 'scripts', 'simulation')):
    sys.path.insert(0, p)

from run_a1_step2_gate import main  # noqa: E402  （冻结副本导入）

_SEALED = ('run_a1_step2_gate', 'run_identity', 'scenario_saa', 'run_exp_energy_c0',
           'coldchain_evaluator_a1', 'run_exp_reserve', 'run_exp_encoder_v3',
           'strict_online_env', 'coldchain_state', 'coldchain_contract')


def _write_evidence():
    out = os.environ.get('RR3_EVIDENCE_OUT')
    if not out:
        return
    lines = []
    for name in _SEALED:
        m = sys.modules.get(name)
        lines.append('%s -> %s' % (name, m.__file__ if m is not None else 'MISSING'))
    lines.append('runner -> %s' % os.path.abspath(__file__))
    lines.append('sys.path[0:6] -> %s' % sys.path[:6])
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    _write_evidence()
    main(sys.argv[1:])
