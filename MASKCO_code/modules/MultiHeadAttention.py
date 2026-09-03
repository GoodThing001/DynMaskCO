import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Any

def attention_fn(
    q: jax.Array | None = None, k: jax.Array | None = None, v: jax.Array | None = None,
    qkv: jax.Array | None = None, bias: jax.Array | None = None,
    causal: bool = False, sm_scale: float | None = None,
    query_seq_lengths: jax.typing.ArrayLike | None = None,
    key_value_seq_lengths: jax.typing.ArrayLike | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    softcap: float | None = None,
    **kwargs,
):
    del kwargs, query_seq_lengths, key_value_seq_lengths, local_window_size, softcap

    if qkv is not None:
        assert q is None and k is None and v is None
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = jax.tree.map(lambda x: jnp.squeeze(x, axis=2), [q, k, v])
    else:
        assert q is not None and k is not None and v is not None
    del qkv

    assert q.dtype == k.dtype == v.dtype
    out_dtype = q.dtype

    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)

    B, S, H, D = q.shape
    if sm_scale is None:
        sm_scale = D ** -0.5

    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale

    if bias is not None:
        if bias.ndim == 3:
            bias = jnp.expand_dims(bias, axis=1)
        logits = logits + bias.astype(jnp.float32)

    if causal:
        causal_mask = jnp.tril(jnp.ones((S, S), dtype=bool))
        logits = jnp.where(causal_mask[None, None, :, :], logits, -jnp.inf)

    probs = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", probs, v)

    return out.astype(out_dtype)


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dtype: jax.typing.DTypeLike,
        param_dtype: jax.typing.DTypeLike = jnp.float32,
        qkv_packed: bool = True,        # qkv-packed attn has not been impl yet, so now qkv_packed=False will be faster
        attention_fn: Callable[..., jax.Array] = attention_fn,
        normalize_qk: bool = False,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dtype = dtype
        self.qkv_packed = True
        self.normalize_qk = normalize_qk

        qkv_kernel_init = nnx.initializers.xavier_uniform()
        out_kernel_init = nnx.initializers.xavier_uniform()

        self.qkv_proj_params: nnx.Param | list[nnx.Param]
        self.qkv_proj_params = nnx.Param(
            qkv_kernel_init(
                rngs.params(), (embed_dim, 3 * embed_dim), param_dtype,
            )
        )
        self.set_qkv_packed(qkv_packed)

        self.out_proj_params = nnx.Param(
            out_kernel_init(
                rngs.params(), (embed_dim, embed_dim), param_dtype,
            )
        )
        self.attention_fn = attention_fn

        self.norm_fn = nnx.RMSNorm(embed_dim // num_heads, use_scale=False, rngs=rngs)

    def set_qkv_packed(self, qkv_packed: bool):
        if qkv_packed == self.qkv_packed:
            return
        self.qkv_packed = qkv_packed
        if not qkv_packed:
            # split
            self.qkv_proj_params = [
                nnx.Param(
                    self.qkv_proj_params.value[:, i * self.embed_dim:(i + 1) * self.embed_dim]
                ) for i in range(3)
            ]
        else:
            # join
            self.qkv_proj_params = nnx.Param(
                jnp.concat([p.value for p in self.qkv_proj_params], axis=-1)
            )

    def __call__(self, x: jax.Array, attn_options: dict[str, Any] = {}):
        assert x.ndim == 3

        batch_size, seqlen, embed_dim = x.shape
        num_heads = self.num_heads
        x = x.astype(self.dtype)
        
        out_proj_params = self.out_proj_params.value.astype(self.dtype)
        head_dim = embed_dim // num_heads


        out: jax.Array
        if self.qkv_packed:
            qkv_proj_params = self.qkv_proj_params.value.astype(self.dtype)

            qkv = jnp.dot(x, qkv_proj_params)
            qkv = qkv.reshape(batch_size, seqlen, 3, self.num_heads, head_dim)
           
            if self.normalize_qk:
                qkv = qkv.at[:, :, :2].set(self.norm_fn(qkv[:, :, :2]))
            out = self.attention_fn(qkv=qkv, **attn_options)
        else:
            q_proj_param, k_proj_param, v_proj_param = [p.value.astype(self.dtype) for p in self.qkv_proj_params]
            q, k, v = jax.tree.map(
                lambda proj_param: jnp.dot(x, proj_param).reshape(batch_size, seqlen, num_heads, head_dim), 
                [q_proj_param, k_proj_param, v_proj_param],
            )
            if self.normalize_qk:
                q = self.norm_fn(q)
                k = self.norm_fn(k)
            out = self.attention_fn(q=q, k=k, v=v, **attn_options)
                
        out = out.reshape(batch_size, seqlen, -1)
        out = jnp.dot(out, out_proj_params)
        return out
