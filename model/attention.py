import functools
import math
import jax.lax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.attention import dot_product_attention as test
from typing import Any
import einops


def dot_product_attention(query, key, value):
    scale = query.shape[-1] ** -0.5
    attn_weights = jnp.einsum('...ihd,...jhd->...hij', query, key) * scale
    attn_weights = nn.softmax(attn_weights)
    out = jnp.einsum('...hij,...jhd->...ihd', attn_weights, value)
    return out


def _query_chunk_attention(query, key, value, precision, key_chunk_size=4096):
    """Multi-head dot product attention with a limited number of queries."""
    b, num_kv, num_heads, k_features = key.shape
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value):
        attn_weights = jnp.einsum('...qhd,...khd->...qhk', query, key, precision=precision)
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum('...vhf,...qhv->...qhf', value, exp_weights, precision=precision)
        return (exp_values, exp_weights.sum(axis=-1), max_score.reshape((b, query.shape[1], num_heads)))

    def chunk_scanner(chunk_idx):
        key_chunk = jax.lax.dynamic_slice(key, (0, chunk_idx, 0, 0),
                                          slice_sizes=(b, key_chunk_size, num_heads, k_features))
        value_chunk = jax.lax.dynamic_slice(value, (0, chunk_idx, 0, 0),
                                            slice_sizes=(b, key_chunk_size, num_heads, v_features))

        return summarize_chunk(query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights


def memory_efficient_attention(query, key, value, precision=jax.lax.Precision.HIGHEST,
                               query_chunk_size=49*8):
    """Memory-efficient multi-head dot product attention."""
    b, num_q, num_heads, q_features = query.shape
    def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(
            query, (0, chunk_idx, 0, 0),
            slice_sizes=(b, min(query_chunk_size, num_q), num_heads, q_features))
        print(query_chunk.shape)
        return (chunk_idx + query_chunk_size, _query_chunk_attention(query_chunk, key, value, precision=precision))

    _, res = jax.lax.scan(chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size))
    return res.reshape(b, num_q, num_heads, value.shape[-1])


class MultiHeadSelfAttention(nn.Module):
    dim: int
    num_heads: int
    attention_type: Any = 'math'
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.Dense(self.dim * 3, self.dtype)(x)
        q, k, v = tuple(einops.rearrange(x, 'b n (k h d)->k b n h d', k=3, h=self.num_heads))
        if self.attention_type == 'math':
            out = dot_product_attention(q, k, v)
        elif self.attention_type == 'memory_efficient':
            out = memory_efficient_attention(q, k, v)
        out = einops.rearrange(out, 'b n h d->b n (h d)')
        return nn.Dense(self.dim, dtype=self.dtype)(out)


if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(42)
    shape = (4, 2048, 2, 48)
    x = jax.random.normal(rng_key, shape) * 2
    x = jnp.ones(shape)
    # x=jnp.array([[[1,10,100],[1,20,200]]])

    num_kv = 13
    key_chunk_size = 2
    print(dot_product_attention(x, x, x) - test(x, x, x))
