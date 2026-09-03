"""
EDoD 全矩阵评估 (9组) → 自动汇总表。
用法: python analysis/run_edod_matrix.py
"""
import subprocess, sys, os, re, time

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))

CKPTS = {
    ('r1','02'): 'phaseb_r1_edod02/step50000.ckpt',
    ('r1','05'): 'st_round2/step5000.ckpt',
    ('r1','08'): 'phaseb_r1_edod08/step50000.ckpt',
    ('c1','02'): 'phaseb_c1_edod02/step50000.ckpt',
    ('c1','05'): 'phaseb_c1_edod05/step50000.ckpt',
    ('c1','08'): 'phaseb_c1_edod08/step50000.ckpt',
    ('rc1','02'): 'phaseb_rc1_edod02/step50000.ckpt',
    ('rc1','05'): 'phaseb_rc1_edod05/step50000.ckpt',
    ('rc1','08'): 'phaseb_rc1_edod08/step50000.ckpt',
}

def run_one(dt, ed):
    ckpt = os.path.join(_CVRPTW, 'ckpts', CKPTS[(dt, ed)])
    data = os.path.join(_CVRPTW, 'data', f'dcc_50_{dt}_edod{ed}_test.npz')
    script = os.path.join(_CVRPTW, 'decoding', 'cvrptw.py')

    cmd = (
        f'cd /home/hzeng/project/MASKCO-Main && '
        f'source /home/hzeng/envs/MASKCO_env/bin/activate && '
        f'CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 '
        f'python -u {script} '
        f'--capacity 50 --penalty 3. --data {data} --ckpt {ckpt} '
        f'--keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 '
        f'--sampling_steps 2 --augment_level 0 --gumbel_scale_factor 0. --seed 42 '
        f'--threads_over_batches 1 '
        f'--enable_tw_filter --enable_tw_repair_edd --enable_tw_aware_2opt_py'
    )

    r = subprocess.run(cmd, shell=True, executable='/bin/bash',
                       capture_output=True, text=True, timeout=300)
    text = r.stdout + '\n' + r.stderr

    # 从 tqdm 进度条和 Overall Results 中提取
    feas = re.search(r'TW feas rate:\s+([\d.]+)%', text)
    cost = re.search(r'mean cost:\s+([\d.]+)', text)
    viol = re.search(r'Avg TW viol/inst:\s+([\d.]+)', text)
    gap  = re.search(r'Gap:\s+([\d.-]+)\s*%', text)

    return {
        'feas': float(feas.group(1)) if feas else -1,
        'cost': float(cost.group(1)) if cost else -1,
        'viol': float(viol.group(1)) if viol else -1,
        'gap':  float(gap.group(1)) if gap else -999,
    }


def main():
    results = {}
    for (dt, ed) in CKPTS:
        label = f'{dt.upper()} EDoD=0.{ed}'
        print(f'{label}...', end=' ', flush=True)
        t0 = time.time()
        r = run_one(dt, ed)
        results[(dt, ed)] = r
        print(f'feas={r["feas"]:.1f}% gap={r["gap"]:+.1f}% ({time.time()-t0:.0f}s)')

    # 汇总表
    print(f'\n{"="*72}')
    print(f'  EDoD FULL MATRIX')
    print(f'{"="*72}')
    header = f'{"Type":<6}'
    for ed_label in ['EDoD=0.2', 'EDoD=0.5', 'EDoD=0.8']:
        header += f' {ed_label:>20}'
    print(header)
    print(f'{"-"*72}')
    for dt in ['r1', 'c1', 'rc1']:
        row = f'{dt.upper():<6}'
        for ed in ['02', '05', '08']:
            r = results[(dt, ed)]
            row += f' {r["feas"]:>5.1f}% g={r["gap"]:+6.1f}% '
        print(row)
    print(f'{"="*72}')

    # 平均 EDoD
    print(f'\n  Average by EDoD:')
    for ed in ['02', '05', '08']:
        avg_f = sum(results[(dt,ed)]['feas'] for dt in ['r1','c1','rc1']) / 3
        avg_g = sum(results[(dt,ed)]['gap'] for dt in ['r1','c1','rc1']) / 3
        print(f'    EDoD=0.{ed}: avg feas={avg_f:.1f}% avg gap={avg_g:+.1f}%')


if __name__ == '__main__':
    main()
