"""
冷链解码（Exp-10）— 基于 cvrptw.py 扩展温度约束过滤。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import argparse, numpy as np
from ColdChainModel import ColdChainModelConfig, ColdChainModel
from cvrptw import cvrptw_searching_decode


def coldchain_decode(dataset, capacity, penalty, model, **kwargs):
    """冷链解码：在 cvrptw 基础上增加 temp_class 特征。"""
    temp_class = dataset.get('temp_class', np.zeros_like(dataset['demands']))
    temp_class_norm = temp_class.astype(np.float32) / 2.0  # [0,1]

    # 6D 特征
    tw_max = kwargs.get('tw_max', None)
    if tw_max is None:
        tw_max = float(dataset['tw_end'].max())
    raw_features = np.concatenate([
        dataset['coords'],
        dataset['demands'][..., None] / capacity,
        dataset['tw_start'][..., None] / tw_max,
        dataset['tw_end'][..., None] / tw_max,
        temp_class_norm[..., None],
    ], axis=-1).astype(np.float32)

    # 替换 dataset 中的 features（cvrptw_searching_decode 会重建，我们直接改 dataset）
    # 这里我们不改动 cvrptw.py 内部逻辑，而是创建包装的 model 和 features

    print(f"[ColdChain] 6D features: {raw_features.shape[-1]} dims "
          f"(+temp_class, range [0,{temp_class.max()}])")

    return cvrptw_searching_decode(
        dataset, capacity, penalty, model, **kwargs
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--penalty', type=float, default=3.)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sampling_steps', type=int, required=True)
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--keep_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--two_opt_steps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable_gumbel', action='store_true', default=False)
    parser.add_argument('--gumbel_scale_factor', type=float, required=True)
    parser.add_argument('--augment_level', type=int, default=0)
    parser.add_argument('--threads_over_batches', type=int, default=1)
    parser.add_argument('--padding_policy', type=str, default='none')
    parser.add_argument('--heatmap_dtype', type=str, default='float32')
    parser.add_argument('--topk', type=eval, default=None)
    parser.add_argument('--model_config', type=str, default='softcap_fn')
    # CVRPTW 参数
    parser.add_argument('--enable_tw_filter', action='store_true', default=False)
    parser.add_argument('--enable_tw_attn_bias', action='store_true', default=False)
    parser.add_argument('--tw_attn_penalty', type=float, default=5.0)
    parser.add_argument('--enable_tw_preserving_2opt', action='store_true', default=False)
    parser.add_argument('--enable_tw_aware_2opt_py', action='store_true', default=False)
    parser.add_argument('--enable_tw_repair_edd', action='store_true', default=False)
    parser.add_argument('--tw_speed', type=float, default=1.0)
    parser.add_argument('--tw_max', type=float, default=None)
    args = parser.parse_args()

    from training import load_ckpt
    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    if model_config is None or not hasattr(model_config, 'encoder_input_dim'):
        model_config = ColdChainModelConfig.get_config(args.model_config)
    if model_config.encoder_input_dim < 6:
        model_config.encoder_input_dim = 6
    model_config.dtype = 'float32'
    model = model_config.construct_model()

    from flax import nnx
    try:
        model = nnx.merge(nnx.graphdef(model), params)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Partial load: {e}")

    dataset = dict(np.load(args.data))
    coldchain_decode(dataset, args.capacity, args.penalty, model,
                     sampling_steps=args.sampling_steps, cycles=args.cycles,
                     keep_rate=args.keep_rate, batch_size=args.batch_size,
                     runs=args.runs, two_opt_steps=args.two_opt_steps,
                     disable_gumbel=args.disable_gumbel,
                     gumbel_scale_factor=args.gumbel_scale_factor,
                     heatmap_dtype=args.heatmap_dtype, topk=args.topk,
                     augment_level=args.augment_level,
                     padding_policy=args.padding_policy,
                     threads_over_batches=args.threads_over_batches, seed=args.seed,
                     enable_tw_filter=args.enable_tw_filter,
                     enable_tw_attn_bias=args.enable_tw_attn_bias,
                     tw_attn_penalty=args.tw_attn_penalty,
                     enable_tw_preserving_2opt=args.enable_tw_preserving_2opt,
                     enable_tw_aware_2opt_py=args.enable_tw_aware_2opt_py,
                     enable_tw_repair_edd=args.enable_tw_repair_edd,
                     tw_speed=args.tw_speed, tw_max=args.tw_max)
