"""
ColdChain Model — 冷链 CVRPTW（Exp-10）。
扩展 CVRPTWModel: 5D→6D [x, y, demand, tw_start, tw_end, temp_class]
"""

import sys, os
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import MASKCO_ROOT
sys.path.insert(0, str(MASKCO_ROOT))

import jax, jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass

from models.CVRPModel import CVRPModel
from CVRPTWModel import CVRPTWModel, CVRPTWModelConfig


@dataclass
class ColdChainModelConfig(CVRPTWModelConfig):
    encoder_input_dim: int = 6  # +temp_class

    @staticmethod
    def get_config(which=''):
        match which:
            case '' | 'default':
                return ColdChainModelConfig()
            case 'softcap_fn':
                return ColdChainModelConfig(
                    embed_dim=256, num_heads=8,
                    qk_norm=False, softcap=30.,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 256,
                )
            case 'softcap_fn_512':
                return ColdChainModelConfig(
                    embed_dim=512, num_heads=8,
                    qk_norm=False, softcap=30.,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 512,
                )
            case _:
                raise ValueError(f"Unknown ColdChain config: {which}")

    def construct_model(self):
        return ColdChainModel(**vars(self))


class ColdChainModel(CVRPTWModel):
    """冷链模型 — 继承 CVRPTWModel，增加温度特征。"""

    def __init__(self, encoder_input_dim=6, **kwargs):
        if 'rngs' in kwargs and isinstance(kwargs['rngs'], int):
            kwargs['rngs'] = nnx.Rngs(kwargs['rngs'])
        super().__init__(encoder_input_dim=encoder_input_dim, **kwargs)

    def encode(self, raw_features, attn_options=None):
        """编码 6D 特征 [x, y, demand, tw_start, tw_end, temp_class]."""
        if attn_options is None:
            attn_options = {}
        return super().encode(raw_features, attn_options)
