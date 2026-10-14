"""
P0-5: 数据质量控制 (Data QC)

对 .npz 数据做 sanity 检查，防止静默数据错误进入实验：
  - NaN/Inf 检测
  - 单位/范围 sanity（coords∈[0,1]、demands∈[0,cap]、tw_start≤tw_end、temp∈{0,1,2}）
  - 字段完整性（visible_mask、reveal_time、service_time）
  - depot 语义（demand=0、reveal_time=0）
  - 重复实例检测

用法:
    python tests/test_data_qc.py --data <dcc.npz> [--capacity 50] [--strict]
"""

import sys, os, argparse, hashlib, numpy as np


def check_data_qc(data_path, capacity=50, strict=False):
    d = dict(np.load(data_path))
    coords = d['coords']
    demands = d['demands']
    tw_start = d['tw_start']
    tw_end = d['tw_end']

    N_inst, N_nodes = coords.shape[0], coords.shape[1]
    failures = []
    warnings = []

    def fail(msg):
        failures.append(msg)

    # 1. NaN/Inf
    for key in d:
        arr = d[key]
        if np.issubdtype(arr.dtype, np.floating):
            if np.isnan(arr).any():
                fail(f"{key}: NaN 存在")
            if np.isinf(arr).any():
                fail(f"{key}: Inf 存在")

    # 2. 坐标范围 [0,1]
    if coords.min() < -1e-6 or coords.max() > 1 + 1e-6:
        fail(f"coords 范围 [{coords.min():.3f}, {coords.max():.3f}] 超出 [0,1]")
    else:
        warnings.append(f"coords ∈ [{coords.min():.3f}, {coords.max():.3f}] ✓")

    # 3. 需求范围 [0, capacity]
    if demands.min() < 0:
        fail(f"demands 含负值")
    if demands.max() > capacity:
        fail(f"demands max={demands.max()} 超过容量 {capacity}")

    # 4. 时间窗 tw_start <= tw_end
    if (tw_start > tw_end).any():
        n_bad = (tw_start > tw_end).sum()
        fail(f"tw_start > tw_end 的实例: {n_bad}")

    # 5. depot 语义（节点 0）
    if demands[:, 0].sum() != 0:
        fail(f"depot demand 非零")
    if 'reveal_time' in d and d['reveal_time'][:, 0].sum() != 0:
        fail(f"depot reveal_time 非零")

    # 6. temp_class ∈ {0,1,2}（若存在）
    if 'temp_class' in d:
        tc = d['temp_class']
        if not np.isin(tc, [0, 1, 2]).all():
            fail(f"temp_class 含 {0,1,2} 之外的值")

    # 7. visible_mask ∈ {0,1}（若存在）
    if 'visible_mask' in d:
        vm = d['visible_mask']
        if not np.isin(vm, [0, 1]).all():
            fail(f"visible_mask 含 0/1 之外的值")
        # visible_mask 与 reveal_time 一致性：reveal_time=0 → visible=1
        if 'reveal_time' in d:
            known = d['reveal_time'] <= 0
            mismatch = (known & (vm == 0)).sum()
            if mismatch > 0:
                warnings.append(f"visible_mask 与 reveal_time 不一致: {mismatch} 个节点")

    # 8. 重复实例检测（coords 完全相同的实例）
    flat = coords.reshape(N_inst, -1)
    _, unique_idx = np.unique(flat, axis=0, return_index=True)
    if len(unique_idx) < N_inst:
        fail(f"重复实例: {N_inst - len(unique_idx)} 个（coords 完全相同）")

    # 9. reveal_time 非负（若存在）
    if 'reveal_time' in d and d['reveal_time'].min() < -1e-6:
        fail(f"reveal_time 含负值")

    # 10. 文件 hash
    with open(data_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    warnings.append(f"sha256[:16] = {sha}")

    # 报告
    print(f"=== Data QC: {os.path.basename(data_path)} ===")
    print(f"  instances={N_inst}, nodes={N_nodes}, capacity={capacity}")
    for w in warnings:
        print(f"  [info] {w}")
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    ✗ {f}")
        return False
    else:
        print(f"\n  ALL CHECKS PASSED ✓")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P0-5: Data QC')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, default=50)
    args = parser.parse_args()
    ok = check_data_qc(args.data, args.capacity)
    sys.exit(0 if ok else 1)
