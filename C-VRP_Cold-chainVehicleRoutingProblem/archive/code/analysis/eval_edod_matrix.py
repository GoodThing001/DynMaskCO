"""
EDoD 全矩阵评估 + 结果汇总。

用法:
    bash run_eval_matrix.sh
    或逐个:
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    python analysis/eval_edod_matrix.py
"""

import sys, os, subprocess, re, time

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))

CKPT_MAP = {
    # R1: use st_round2 for edod05 (best), phaseb for others
    ('r1', '02'): 'phaseb_r1_edod02/step50000.ckpt',
    ('r1', '05'): 'st_round2/step5000.ckpt',
    ('r1', '08'): 'phaseb_r1_edod08/step50000.ckpt',
    # C1: use phaseb
    ('c1', '02'): 'phaseb_c1_edod02/step50000.ckpt',
    ('c1', '05'): 'phaseb_c1_edod05/step50000.ckpt',
    ('c1', '08'): 'phaseb_c1_edod08/step50000.ckpt',
    # RC1: use phaseb
    ('rc1', '02'): 'phaseb_rc1_edod02/step50000.ckpt',
    ('rc1', '05'): 'phaseb_rc1_edod05/step50000.ckpt',
    ('rc1', '08'): 'phaseb_rc1_edod08/step50000.ckpt',
}

def run_eval(data_type, edod):
    ckpt_rel = CKPT_MAP[(data_type, edod)]
    ckpt_path = os.path.join(_CVRPTW, 'ckpts', ckpt_rel)
    data_path = os.path.join(_CVRPTW, 'data', f'dcc_50_{data_type}_edod{edod}_test.npz')
    script = os.path.join(_CVRPTW, 'decoding', 'cvrptw.py')

    cmd = (
        f'cd /home/hzeng/project/MASKCO-Main && '
        f'source /home/hzeng/envs/MASKCO_env/bin/activate && '
        f'CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 '
        f'python -u "{script}" '
        f'--capacity 50 --penalty 3. --data "{data_path}" --ckpt "{ckpt_path}" '
        f'--keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 '
        f'--sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 '
        f'--threads_over_batches 1 '
        f'--enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py'
    )

    try:
        result = subprocess.run(cmd, shell=True, executable='/bin/bash',
                               capture_output=True, text=True, timeout=300)
        output = result.stdout + '\n' + result.stderr
        if result.returncode != 0:
            print(f"  [CMD FAILED rc={result.returncode}]")
    except subprocess.TimeoutExpired:
        output = "TIMEOUT"
        print(f"  [TIMEOUT]")
    return parse_output(output)


def parse_output(text):
    """从 cvrptw.py 输出中提取指标。"""
    cost = feas = viol = gap = -1.0
    for line in text.split('\n'):
        if 'mean cost:' in line:
            try: cost = float(line.split()[-1])
            except: pass
        if 'TW feas rate:' in line:
            try: feas = float(line.strip().split()[-1].replace('%', ''))
            except: pass
        if 'Avg TW viol/inst:' in line:
            try: viol = float(line.strip().split()[-1])
            except: pass
        if 'Gap:' in line:
            try: gap = float(line.strip().split()[-1].replace('%', ''))
            except: pass
    # 若运行失败，打印 stderr
    if feas < 0:
        print(f"  [PARSE ERROR] output preview: {text[-500:]}")
    return {'cost': cost, 'feas': feas, 'viol': viol, 'gap': gap}


def main():
    results = {}
    for (dt, ed), ckpt in CKPT_MAP.items():
        label = f'{dt.upper()} EDoD=0.{ed}'
        print(f'\n===== {label} =====')
        t0 = time.time()
        r = run_eval(dt, ed)
        elapsed = time.time() - t0
        results[(dt, ed)] = r
        print(f"  feas={r['feas']:.1f}% cost={r['cost']:.2f} viol={r['viol']:.1f} gap={r['gap']:.1f}% ({elapsed:.0f}s)")

    # 汇总表
    print(f"\n{'='*80}")
    print(f"EDoD Full Matrix Summary")
    print(f"{'='*80}")
    print(f"{'Type':<6} {'EDoD=0.2':>18} {'EDoD=0.5':>18} {'EDoD=0.8':>18}")
    print(f"{'-'*80}")
    for dt in ['r1', 'c1', 'rc1']:
        row = f"{dt.upper():<6}"
        for ed in ['02', '05', '08']:
            r = results.get((dt, ed), {})
            f = r.get('feas', 0) or 0
            g = r.get('gap', 0) or 0
            row += f" {f:>5.1f}% g={g:>+6.1f}%  "
        print(row)
    print(f"{'='*80}")

    # 保存
    import numpy as np
    np.savez(os.path.join(_CVRPTW, 'analysis', 'edod_matrix.npz'),
             results={f'{dt}_{ed}': r for (dt, ed), r in results.items()})


if __name__ == '__main__':
    main()
