# -*- coding: utf-8 -*-
"""S3-2 密度单因素分析（硬门版 v2，2026-09-30 路线图验收清单 #3 修订）。

输入：同修订版 reveal 与 density 两个步骤 2 gate.json。
硬门（失败即停止计算并退出 2）：
  - 两报告 config.gate_instances==40、time_limit==10、K==10、capacity/num_vehicles/
    n_orders/energy_pricing/shadow_mode/standby_orders 一致，仅 future_policy 可不同；
  - 两报告 identity 存在且 source_stable==True；共享源码/合同/数据 hash 一致；
  - 两报告 seeds.gate 相同（bootstrap 种子必须一致）；
  - 四臂（cond/uncond × reveal/density）每臂 day_id 恰为 0..39 且唯一、效用全有限。
v2 修复：按 day_id 唯一映射重排效用向量后配对（原按 JSON 行序相减在行序不同时错配）；
bootstrap 种子读实际 gate seed（不再硬编码 20260926）。
输出（每档 p_c，写入 --out JSON）：
  1) 各配置自身原条件门（cond−uncond mean/CI/passed）；
  2) 同臂绝对差分：cond_density−cond_reveal 与 uncond_density−uncond_reveal
     （密度策略在两臂内的绝对效用作用，整天联合重采样 CI）；
  3) 交互差 (cond_d−uncond_d)−(cond_r−uncond_r)（密度是否改变条件信息优势）。
后两项仅机制归因；density 自身过门与否只看第 1 项。
"""
import argparse
import json
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'evaluation'),
           os.path.join(_SCRIPTS, 'coldchain')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_exp_reserve import _stat

TIERS = ['p_c=0', 'p_c=(5,10,15)', 'p_c=(10,20,30)']
EXPECTED = 40


def _fail(msg):
    print('FATAL:', msg)
    sys.exit(2)


def _arm(d, pn, arm):
    """按 day_id 唯一映射返回 0..39 有序效用向量（v2：不依赖 JSON 行序）。"""
    rows = d['gate'][pn]['per_day'][arm]
    ids = sorted(int(r['i']) for r in rows)
    if len(rows) != EXPECTED or ids != list(range(EXPECTED)) or len(set(ids)) != EXPECTED:
        _fail('%s/%s day_ids 不完整或重复: n=%d' % (pn, arm, len(rows)))
    by_id = {}
    for r in rows:
        if int(r['i']) in by_id:
            _fail('%s/%s 重复 day_id=%s' % (pn, arm, r['i']))
        by_id[int(r['i'])] = r
        if r['utility'] is None or not np.isfinite(r['utility']):
            _fail('%s/%s 存在非有限效用 day=%s' % (pn, arm, r['i']))
    return np.array([by_id[i]['utility'] for i in range(EXPECTED)], float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('reveal_json')
    ap.add_argument('density_json')
    ap.add_argument('--out', default=None,
                    help='输出 JSON（默认写入 density_json 同目录 s3_density_analysis.json）')
    args = ap.parse_args(argv)
    dr = json.load(open(args.reveal_json, encoding='utf-8'))
    dd = json.load(open(args.density_json, encoding='utf-8'))

    cfg_r, cfg_d = dr.get('config', {}), dd.get('config', {})
    for k in ('gate_instances', 'time_limit', 'K', 'capacity', 'num_vehicles',
              'n_orders', 'energy_pricing', 'shadow_mode', 'standby_orders'):
        if cfg_r.get(k) != cfg_d.get(k):
            _fail('共享配置不一致: %s reveal=%s density=%s'
                  % (k, cfg_r.get(k), cfg_d.get(k)))
    if cfg_r.get('gate_instances') != EXPECTED:
        _fail('gate_instances != 40')
    for name, d in (('reveal', dr), ('density', dd)):
        idt = d.get('identity')
        if not idt or idt.get('source_stable') is not True:
            _fail('%s 身份缺失或不稳定' % name)
        for k in ('shared_source_sha256', 'contract_sha256', 'data_sha256'):
            if not idt.get(k):
                _fail('%s 缺身份字段 %s' % (name, k))
    for k in ('shared_source_sha256', 'contract_sha256', 'data_sha256'):
        if dr['identity'][k] != dd['identity'][k]:
            _fail('身份不一致: %s' % k)
    try:
        seed_r = int(dr['seeds']['gate'])
        seed_d = int(dd['seeds']['gate'])
    except (KeyError, TypeError, ValueError):
        _fail('seeds.gate 缺失')
    if seed_r != seed_d:
        _fail('两报告 gate seed 不一致: %s vs %s' % (seed_r, seed_d))
    gate_seed = seed_r

    print('== 身份/配置核验通过（gate seed=%d）==' % gate_seed)
    out = {'gate_seed': gate_seed, 'tiers': {}}

    def _st(x):
        s = _stat(x, gate_seed)
        return dict(mean=s['mean'], ci_lo=s['ci_lo'], ci_hi=s['ci_hi'], n=s['n'])

    for pn in TIERS:
        cond_r = _arm(dr, pn, 'cond_hist')
        uncond_r = _arm(dr, pn, 'uncond_hist')
        cond_d = _arm(dd, pn, 'cond_hist')
        uncond_d = _arm(dd, pn, 'uncond_hist')
        gr = dr['gate'][pn]['main_comparison']['cond_minus_uncond']
        gd = dd['gate'][pn]['main_comparison']['cond_minus_uncond']

        abs_cond = _st(cond_d - cond_r)          # 同臂绝对差分（cond）
        abs_uncond = _st(uncond_d - uncond_r)    # 同臂绝对差分（uncond）
        inter = _st((cond_d - uncond_d) - (cond_r - uncond_r))  # 交互差
        block = {
            'reveal_original_gate': {'mean': gr['mean'], 'ci_lo': gr['ci_lo'],
                                     'ci_hi': gr['ci_hi'],
                                     'passed': bool(dr['gate'][pn]['main_comparison']
                                                    .get('passed'))},
            'density_original_gate': {'mean': gd['mean'], 'ci_lo': gd['ci_lo'],
                                      'ci_hi': gd['ci_hi'],
                                      'passed': bool(dd['gate'][pn]['main_comparison']
                                                     .get('passed'))},
            'abs_diff_cond_density_minus_reveal': abs_cond,
            'abs_diff_uncond_density_minus_reveal': abs_uncond,
            'interaction_four_arm': inter,
        }
        out['tiers'][pn] = block
        print('  [%s]' % pn)
        print('    原条件门 reveal : cond-uncond mean=%s ci=[%s, %s] passed=%s'
              % (gr['mean'], gr['ci_lo'], gr['ci_hi'],
                 dr['gate'][pn]['main_comparison'].get('passed')))
        print('    原条件门 density: cond-uncond mean=%s ci=[%s, %s] passed=%s'
              % (gd['mean'], gd['ci_lo'], gd['ci_hi'],
                 dd['gate'][pn]['main_comparison'].get('passed')))
        print('    绝对差分 cond_d−cond_r   : mean=%s ci=[%s, %s]'
              % (abs_cond['mean'], abs_cond['ci_lo'], abs_cond['ci_hi']))
        print('    绝对差分 uncond_d−uncond_r: mean=%s ci=[%s, %s]'
              % (abs_uncond['mean'], abs_uncond['ci_lo'], abs_uncond['ci_hi']))
        print('    交互差 (四臂)            : mean=%s ci=[%s, %s]'
              % (inter['mean'], inter['ci_lo'], inter['ci_hi']))
        print('    （绝对差分与交互差仅机制归因；density 过门只看原条件门）')

    out_path = args.out or os.path.join(os.path.dirname(
        os.path.abspath(args.density_json)), 's3_density_analysis.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('saved', out_path)


if __name__ == '__main__':
    main()
