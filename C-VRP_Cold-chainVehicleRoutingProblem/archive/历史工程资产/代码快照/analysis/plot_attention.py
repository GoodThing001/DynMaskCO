"""
Phase E-03: 绘制 Attention Heatmap (3 EDoD × 3 instances = 9 张图)。

用法:
    python "C-VRP_Cold-chainVehicleRoutingProblem/analysis/plot_attention.py" \
        --input_dir "C-VRP_Cold-chainVehicleRoutingProblem/analysis/attention_maps/" \
        --output_dir "docs/figures/"
"""

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, argparse


def plot_single(attn_weights, title, save_path, head_idx=0, topk=51):
    """绘制单张 attention heatmap。"""
    # attn_weights: (heads, nodes, nodes) → 取指定 head 或平均
    if attn_weights.ndim == 3:
        weights = attn_weights[head_idx]  # (nodes, nodes)
    else:
        weights = attn_weights

    nodes = weights.shape[0]
    # 聚焦 top-k 节点
    if nodes > topk:
        # 选 attention 最高的节点
        importance = weights.sum(axis=0)
        top_indices = np.argsort(importance)[-topk:]
        weights = weights[top_indices][:, top_indices]
        nodes = topk

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(weights, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=weights.max())

    ax.set_xlabel('Key (Target Node)', fontsize=12)
    ax.set_ylabel('Query (Source Node)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # 标注 depot 节点 (index 0)
    if topk >= nodes:
        ax.annotate('Depot', xy=(0, 0), xytext=(10, 10),
                     arrowprops=dict(arrowstyle='->', color='black'),
                     fontsize=10, color='red', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary(all_data, save_path):
    """绘制 3×3 汇总图。"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    edod_labels = {'02': 'EDoD=0.2 (Weak)', '05': 'EDoD=0.5 (Medium)', '08': 'EDoD=0.8 (Strong)'}

    for row, edod in enumerate(['02', '05', '08']):
        for col in range(3):
            ax = axes[row, col]
            key = f'attn_edod{edod}_inst{col}'
            if key in all_data:
                weights = all_data[key]
                if weights.ndim == 3:
                    weights = weights[0]  # first head
                im = ax.imshow(weights, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=weights.max() * 0.8)
                if col == 2:
                    plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f'{edod_labels[edod]} - Inst {col+1}', fontsize=11)
            if row == 2:
                ax.set_xlabel('Key Node')
            if col == 0:
                ax.set_ylabel('Query Node')

    plt.suptitle('Decoder Attention Maps: EDoD Comparison (R1, 50-node)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Summary saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='docs/figures/')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_data = {}
    for fname in sorted(os.listdir(args.input_dir)):
        if fname.endswith('.npy'):
            key = fname.replace('.npy', '')
            all_data[key] = np.load(os.path.join(args.input_dir, fname))
    print(f"Loaded {len(all_data)} attention maps from {args.input_dir}")

    # 单张图
    for key, weights in all_data.items():
        edod = key.split('_')[1].replace('edod', '')
        inst = key.split('_')[2].replace('inst', '')
        edod_val = float(edod) / 10.0
        title = f'Decoder Attention: EDoD={edod_val}, Instance {int(inst)+1} (R1, 50-node)'
        save_path = os.path.join(args.output_dir, f'{key}.pdf')
        plot_single(weights, title, save_path)

    # 汇总图
    if len(all_data) >= 9:
        plot_summary(all_data, os.path.join(args.output_dir, 'attention_summary_3x3.pdf'))


if __name__ == '__main__':
    main()
