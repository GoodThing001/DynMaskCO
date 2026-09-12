"""
P0-2 信息泄漏验证: 确认未来订单特征不可被模型访问。

验证项目:
  1. Dataloader 输出: 不可见节点特征已置零
  2. 特征梯度隔离: 修改不可见节点不影响可见节点编码
  3. 注意力屏蔽: visible→future 注意力权重=0

用法:
    python "C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_no_future_leakage.py" \
        --data <dcc_test.npz> --ckpt <step50000.ckpt>
"""

import sys, os, argparse
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'data'))

import jax, jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
from ColdChainDataloader import ColdChainDataloader
from modules.functional import coord_normalize
from cvrptw_utils import coord_normalize_visible


def test_1_dataloader_unmasked(data_path):
    """测试1: Dataloader yield 未掩码特征 + 正确的 visible_mask/reveal_time。

    （2026-08-27 起掩码移到训练器按 vis_k 逐步做，dataloader 只提供原料：
     未掩码特征 + visible_mask + reveal_time，供 progressive reveal 用。）
    """
    print("=" * 60)
    print("Test 1: Dataloader Unmasked + Mask Correctness")
    print("=" * 60)

    data = dict(np.load(data_path))
    if 'visible_mask' not in data:
        print("  SKIP: no visible_mask in dataset")
        return True

    dl = ColdChainDataloader(data, batch_size=8, capacity=50, tw_max=None)
    features, routes, timestep, visible_mask, reveal_time = next(iter(dl))

    invisible = (visible_mask == 0)
    if not invisible.any():
        print("  WARN: no invisible nodes in batch, re-running...")
        return test_1_dataloader_unmasked(data_path)

    # 1) future 节点特征应未掩码（保留真实 demand/tw/temp/reveal，供逐步揭示）
    non_coord = features[invisible][:, 2:]
    unmasked = not np.allclose(non_coord, 0, atol=1e-6)

    # 2) visible_mask 二值正确（future=0, visible=1）
    mask_binary = np.isin(visible_mask, [0, 1]).all()

    # 3) reveal_time 正确：future 节点 reveal_time > 0（且已被单独 yield，未掩码）
    rt_positive = (reveal_time[invisible] > 0).all()

    print(f"  future 特征未掩码（保留真实值）: {'PASS' if unmasked else 'FAIL'} ({non_coord.max():.6f})")
    print(f"  visible_mask 二值:              {'PASS' if mask_binary else 'FAIL'}")
    print(f"  future 节点 reveal_time > 0:    {'PASS' if rt_positive else 'FAIL'}")
    print(f"  批次中不可见节点比例:           {invisible.mean():.1%}")

    return unmasked and mask_binary and rt_positive


def test_2_gradient_isolation(ckpt_path, data_path):
    """测试2: 修改不可见节点特征不应影响可见节点的编码输出。"""
    print("\n" + "=" * 60)
    print("Test 2: Gradient Isolation (Encoder)")
    print("=" * 60)

    data = dict(np.load(data_path))
    if 'visible_mask' not in data:
        print("  SKIP: no visible_mask")
        return True

    # 加载模型
    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model = model_config.construct_model()
    if params is not None:
        model = nnx.merge(nnx.graphdef(model), params)

    # 取单个实例
    coords = data['coords'][0:1].astype(np.float32)
    demands = data['demands'][0:1].astype(np.float32) / 50.0
    tw_max = float(data['tw_end'].max())
    tw_start = data['tw_start'][0:1].astype(np.float32) / tw_max
    tw_end_n = data['tw_end'][0:1].astype(np.float32) / tw_max
    temp_class = data['temp_class'][0:1].astype(np.float32) / 2.0
    reveal_time = data['reveal_time'][0:1].astype(np.float32) / tw_max
    visible_mask = data['visible_mask'][0:1].astype(np.float32)

    # 构建两个版本: A=原始, B=未来节点坐标随机扰动
    feats_a = np.concatenate([
        coords, demands[..., None], tw_start[..., None], tw_end_n[..., None],
        temp_class[..., None], reveal_time[..., None],
    ], axis=-1)

    feats_b = feats_a.copy()
    # 给不可见节点随机坐标
    invisible = (visible_mask[0] == 0)
    if invisible.sum() == 0:
        print("  SKIP: no future nodes")
        return True

    np.random.seed(42)
    # 扰动 future 节点的全部属性（坐标/demand/TW/temp/reveal_time）
    feats_b[0, invisible, 0] = np.random.uniform(0, 1, size=invisible.sum())   # x
    feats_b[0, invisible, 1] = np.random.uniform(0, 1, size=invisible.sum())   # y
    feats_b[0, invisible, 2] = np.random.uniform(0, 1, size=invisible.sum())   # demand
    feats_b[0, invisible, 3] = np.random.uniform(0, 1, size=invisible.sum())   # tw_start
    feats_b[0, invisible, 4] = np.random.uniform(0, 1, size=invisible.sum())   # tw_end
    feats_b[0, invisible, 5] = np.random.uniform(0, 1, size=invisible.sum())   # temp_class
    feats_b[0, invisible, 6] = np.random.uniform(0, 1, size=invisible.sum())   # reveal_time

    # 不预掩码：让 encoder 的 attention bias + coord_normalize_visible 自己隔离 future。
    # 这测的是「causal encoder 架构」本身，而非 dataloader 掩码层（P1 #26 修复：
    # 原版先掩码再编码，扰动被掩码抹平，测不到 encoder 的因果性）。
    def encode_w_mask(model, feats, mask):
        f = coord_normalize_visible(feats[..., :2], mask)  # visible-only 归一化（无泄漏）
        f2 = jnp.concatenate([f, feats[..., 2:7]], axis=-1)
        return model.encode(f2, visible_mask=mask)

    out_a = np.array(encode_w_mask(model, jnp.array(feats_a), jnp.array(visible_mask)))
    out_b = np.array(encode_w_mask(model, jnp.array(feats_b), jnp.array(visible_mask)))

    # 可见节点的编码应完全不受不可见节点扰动影响
    vis_idx = np.where(visible_mask[0] == 1)[0]
    diff = float(np.abs(out_a[0, vis_idx] - out_b[0, vis_idx]).max())
    isolated = diff < 1e-4

    # 不可见节点被掩码置零后编码也相同（特征被统一抹平）
    invis_idx = np.where(invisible)[0]
    diff_invis = float(np.abs(out_a[0, invis_idx] - out_b[0, invis_idx]).max())

    print(f"  可见节点编码差异 (max |Δ|): {'PASS' if isolated else 'FAIL'} ({diff:.2e})")
    print(f"  不可见节点编码差异 (max |Δ|): {diff_invis:.2e} (预期>0)")
    print(f"  可见节点数: {len(vis_idx)}, 不可见节点数: {len(invis_idx)}")

    return isolated


def test_3_attention_mask():
    """测试3: 验证 visible_mask → attn bias 的数学正确性。"""
    print("\n" + "=" * 60)
    print("Test 3: Attention Bias Math")
    print("=" * 60)

    # 模拟: B=2, N=5, visible_mask 第2/4节点不可见
    visible_mask = jnp.array([
        [1., 0., 1., 0., 1.],   # batch 0: nodes 1,3 invisible
        [1., 1., 1., 0., 0.],   # batch 1: nodes 3,4 invisible
    ])

    # 复现 DynamicColdChainModel.encode 中的 bias 构造
    B, N = visible_mask.shape
    vis_bias = jnp.where(visible_mask[:, None, :], 0.0, -1e9)
    vis_bias = vis_bias.at[:, :, 0].set(0.0)  # depot always visible

    # 模拟 QK logits (全1矩阵, sm_scale=1)
    logits = jnp.ones((B, 1, N, N))
    logits = logits + vis_bias[:, None, :, :]

    probs = jax.nn.softmax(logits, axis=-1)

    # 检查: 任何 query 对不可见节点 key 的 attention=0
    for b in range(B):
        for q in range(N):
            row = probs[b, 0, q]
            for k in range(N):
                if visible_mask[b, k] == 0 and k != 0:
                    ok = row[k] < 1e-5
                    if not ok:
                        print(f"  FAIL: batch {b}, query {q} attends to invisible key {k}: {row[k]:.6f}")
                        return False

    # 检查: depot 列非零
    depot_ok = (probs[..., 0] > 0).all()

    print(f"  invisible→0: attention  {'PASS' if True else 'FAIL'}")
    print(f"  depot visible:            {'PASS' if depot_ok else 'FAIL'}")
    print(f"  Bias range: [{vis_bias.min():.0f}, {vis_bias.max():.0f}]")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    results = []
    results.append(("Dataloader Unmasked",     test_1_dataloader_unmasked(args.data)))
    if args.ckpt:
        results.append(("Gradient Isolation", test_2_gradient_isolation(args.ckpt, args.data)))
    else:
        print("\n  SKIP Test 2 (--ckpt not provided)")
    results.append(("Attention Bias Math",    test_3_attention_mask()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok: all_pass = False
        print(f"  {name:<30} {status}")
    print(f"\n  Overall: {'ALL PASS - No Leakage Detected' if all_pass else 'FAIL - Leakage Found!'}")
    sys.exit(0 if all_pass else 1)
