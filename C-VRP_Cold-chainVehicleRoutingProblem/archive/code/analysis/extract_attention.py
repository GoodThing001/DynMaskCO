"""
Phase E-02: 提取 DynamicColdChainModel 在不同 EDoD 下的 Attention Weights。

用法:
    cd /home/hzeng/project/MASKCO-Main/
    python -u "C-VRP_Cold-chainVehicleRoutingProblem/analysis/extract_attention.py" \
        --data_prefix "C-VRP_Cold-chainVehicleRoutingProblem/data/dcc_50_r1_edod" \
        --ckpt "C-VRP_Cold-chainVehicleRoutingProblem/ckpts/phaseb_r1_edod{edod}/step50000.ckpt" \
        --output_dir "C-VRP_Cold-chainVehicleRoutingProblem/analysis/attention_maps/"
"""

import sys, os, argparse, numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))
_MASKCO = os.path.dirname(_CVRPTW)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'data'))

import jax, jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
from ColdChainDataloader import ColdChainDataloader


def extract_attention_weights(model, features, timestep, adjmat, layer_idx=-1):
    """
    从指定 decoder layer 提取 multi-head attention 权重。

    绕过 Triton flash-attn（不产出 weights），用纯 JAX softmax 重新计算。

    Returns:
        attn_weights: (num_heads, nodes, nodes) float32
    """
    # Encode
    encoded = model.encode(features)
    x = model.mid_proj(encoded)
    x = x + model.timestep_embedder(timestep).astype(x.dtype)[:, None]

    num_layers = len(model.decoder.layers)
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    # 前向传播到 target layer
    attn_options = {'softcap': model.softcap, 'sm_scale': model.sm_scale, 'bias': adjmat}
    target_layer = model.decoder.layers[layer_idx]

    # 先过前面的层
    for i in range(layer_idx):
        x = model.decoder.layers[i](x, attn_options=attn_options)

    # 在 target layer 手动计算 attention
    x_norm = target_layer.ln1(x)

    # Q, K, V from packed qkv projection
    mha = target_layer.attn
    batch, nodes, embed = x_norm.shape
    num_heads = mha.num_heads
    head_dim = embed // num_heads

    qkv_proj = mha.qkv_proj_params.value.astype(x_norm.dtype)
    qkv = jnp.dot(x_norm, qkv_proj)  # (batch, nodes, 3*embed)
    qkv = qkv.reshape(batch, nodes, 3, num_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    # (batch, nodes, num_heads, head_dim)

    # QK^T: (batch, num_heads, nodes, nodes)
    sm_scale = head_dim ** -0.5
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale

    # 加入 softcap（如果配置了）
    softcap = model.softcap if hasattr(model, 'softcap') else None
    if softcap is not None:
        logits = softcap * jnp.tanh(logits / softcap)

    # 加入 attention bias (adjmat)
    if adjmat is not None:
        bias = adjmat.astype(jnp.float32)
        if bias.ndim == 3:
            bias = bias[:, None, :, :]  # (batch, 1, nodes, nodes)
        logits = logits + bias

    # Softmax → attention weights
    attn_weights = jax.nn.softmax(logits, axis=-1)  # (batch, heads, nodes, nodes)

    return attn_weights[0]  # first batch element


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_prefix', type=str, required=True)
    parser.add_argument('--ckpt_template', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./attention_maps/')
    parser.add_argument('--edod_list', type=str, default='02,05,08')
    parser.add_argument('--num_instances', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    edods = args.edod_list.split(',')

    for edod_str in edods:
        edod_val = float(edod_str) / 10.0  # '02'→0.2, '05'→0.5, '08'→0.8
        print(f"\n{'='*60}")
        print(f"EDoD = {edod_val}")
        print(f"{'='*60}")

        # 加载数据
        data_path = f"{args.data_prefix}{edod_str}_test.npz"
        data = dict(np.load(data_path))
        dl = ColdChainDataloader(data, batch_size=args.num_instances, capacity=50, tw_max=None)
        batch = next(iter(dl))
        features, target, timestep, _visible = batch

        # 构建 adjmat（全零 = 无已知边，从头解码）
        num_nodes = features.shape[1]
        adjmat = jnp.zeros((args.num_instances, num_nodes, num_nodes), dtype=jnp.float32)

        # 加载模型
        ckpt_path = args.ckpt_template.replace('{edod}', edod_str)
        params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
        if model_config is None:
            model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
            model_config.encoder_input_dim = 7
        model_config.dtype = 'float32'
        model = model_config.construct_model()
        try:
            model = nnx.merge(nnx.graphdef(model), params)
            print(f"  Model loaded: {ckpt_path}")
        except Exception as e:
            print(f"  WARNING: partial load: {e}")

        # 提取 attention
        for inst_idx in range(min(args.num_instances, features.shape[0])):
            print(f"  Instance {inst_idx}...")
            attn = extract_attention_weights(
                model,
                features[inst_idx:inst_idx+1],
                timestep[inst_idx:inst_idx+1],
                adjmat[inst_idx:inst_idx+1],
                layer_idx=-1,  # 最后一层 decoder
            )
            # attn shape: (num_heads, nodes, nodes)
            attn_np = np.array(attn)
            out_path = os.path.join(
                args.output_dir,
                f"attn_edod{edod_str}_inst{inst_idx}.npy"
            )
            np.save(out_path, attn_np)
            print(f"    Saved: {out_path} ({attn_np.shape})")


if __name__ == '__main__':
    main()
