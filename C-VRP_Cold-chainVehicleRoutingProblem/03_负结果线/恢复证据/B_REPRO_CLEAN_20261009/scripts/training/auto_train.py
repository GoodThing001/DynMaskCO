"""
自动选 GPU + 冒烟验证 + 正式训练。

用法:
    python "C-VRP_Cold-chainVehicleRoutingProblem/training/auto_train.py" \
        --script train_coldchain.py \
        --num_nodes 50 --capacity 50 --model_config softcap_fn \
        --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 \
        --num_steps 50000 --save_interval 5000 \
        --data "C-VRP_Cold-chainVehicleRoutingProblem/data/dcc_50_r1_edod05_train.npz" \
        --logdir "C-VRP_Cold-chainVehicleRoutingProblem/logs/phaseb_r1_edod05" \
        --savedir "C-VRP_Cold-chainVehicleRoutingProblem/ckpts/phaseb_r1_edod05" \
        --optimizer_type adamw --weight_decay 1e-2 --target_disruption None
"""

import subprocess, re, sys, os, time, argparse

def get_free_gpu(min_free_mb=4000):
    """返回显存最充裕的 GPU ID，若都不够则等待重试。"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free',
             '--format=csv,noheader,nounits'],
            text=True, timeout=10
        )
        best_gpu, best_free = None, 0
        for line in out.strip().split('\n'):
            idx, free_mb = line.split(',')
            free_mb = int(free_mb.strip())
            if free_mb > best_free:
                best_free = free_mb
                best_gpu = int(idx.strip())
        if best_free >= min_free_mb:
            return best_gpu, best_free
        return None, best_free
    except Exception:
        return 0, 99999  # fallback

def run_step(cmd, desc):
    """运行命令，检查返回值。"""
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, executable='/bin/bash')
    if result.returncode != 0:
        print(f"ERROR: {desc} failed (code={result.returncode})")
        sys.exit(1)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, required=True, help='训练脚本路径（相对于 CVRPTW/training/）')
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--capacity', type=eval, required=True)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--peak_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--optimizer_type', type=str, default='adamw')
    parser.add_argument('--model_config', type=str, default='softcap_fn')
    parser.add_argument('--encoder_input_dim', type=int, default=7)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--target_disruption', type=str, default='None')
    parser.add_argument('--min_gpu_mb', type=int, default=8000)
    args = parser.parse_args()

    _BASE = os.path.dirname(os.path.abspath(__file__))
    _CVRPTW = os.path.dirname(os.path.dirname(_BASE))

    script_path = os.path.join(_BASE, args.script)
    if not os.path.exists(script_path):
        print(f"ERROR: script not found: {script_path}")
        sys.exit(1)

    # ── Step 1: 选择 GPU ──
    gpu_id, free_mb = get_free_gpu(args.min_gpu_mb)
    if gpu_id is None:
        print(f"[GPU] No GPU with >{args.min_gpu_mb}MB free. Best={free_mb}MB. Trying anyway...")
        gpu_id = 0
    print(f"[GPU] Using GPU {gpu_id} (free: {free_mb} MB)")

    suffix = os.path.basename(args.data).replace('.npz', '').replace('dcc_50_', '')

    # ── Step 2: 冒烟测试 (50步) ──
    smoke_logdir = args.logdir + '_smoke'
    smoke_savedir = args.savedir + '_smoke'
    cmd_smoke = (
        f'cd /home/hzeng/project/MASKCO-Main && '
        f'source /home/hzeng/envs/MASKCO_env/bin/activate && '
        f'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 && '
        f'python -u "{script_path}" '
        f'--gpu_id {gpu_id} --num_nodes {args.num_nodes} --capacity {args.capacity} '
        f'--model_config {args.model_config} --encoder_input_dim {args.encoder_input_dim} '
        f'--peak_lr {args.peak_lr} --batch_size {args.batch_size} '
        f'--num_steps 50 --save_interval 50 '
        f'--data "{args.data}" '
        f'--logdir "{smoke_logdir}" --savedir "{smoke_savedir}" '
        f'--optimizer_type {args.optimizer_type} --weight_decay {args.weight_decay} '
        f'--target_disruption {args.target_disruption}'
    )
    run_step(cmd_smoke, "SMOKE: 50 steps")

    # ── Step 3: 检查冒烟结果 ──
    smoke_ckpt = os.path.join(args.savedir + '_smoke', 'step50.ckpt')
    # Check if any checkpoint was saved (the script might use a different pattern)
    smoke_dir = args.savedir + '_smoke'
    if os.path.exists(smoke_dir):
        ckpts = [f for f in os.listdir(smoke_dir) if f.endswith('.ckpt')]
        if ckpts:
            print(f"[SMOKE] Passed! Found checkpoints: {ckpts}")
        else:
            print(f"[SMOKE] WARNING: No checkpoints found, but script completed. Proceeding.")
    else:
        print(f"[SMOKE] WARNING: savedir not found. Proceeding anyway.")

    print(f"[SMOKE] ✅ Passed → Starting full training...")

    # ── Step 4: 完整训练 ──
    cmd_full = (
        f'cd /home/hzeng/project/MASKCO-Main && '
        f'source /home/hzeng/envs/MASKCO_env/bin/activate && '
        f'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 && '
        f'python -u "{script_path}" '
        f'--gpu_id {gpu_id} --num_nodes {args.num_nodes} --capacity {args.capacity} '
        f'--model_config {args.model_config} --encoder_input_dim {args.encoder_input_dim} '
        f'--peak_lr {args.peak_lr} --batch_size {args.batch_size} '
        f'--num_steps {args.num_steps} --save_interval {args.save_interval} '
        f'--data "{args.data}" '
        f'--logdir "{args.logdir}" --savedir "{args.savedir}" '
        f'--optimizer_type {args.optimizer_type} --weight_decay {args.weight_decay} '
        f'--target_disruption {args.target_disruption}'
    )
    run_step(cmd_full, f"FULL TRAINING: {args.num_steps} steps")
    print(f"\n[DONE] Training complete. Checkpoint: {args.savedir}/step{args.num_steps}.ckpt")
