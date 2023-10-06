import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.attention import dot_product_attention


# def my_attention(q, k, v, ):
#     scale = q.shape[-1] ** -0.5
#     q = q * scale
#     attn = jnp.einsum('b i h d ,b j h d ->b h i j', q, k)
#     attn = nn.softmax(attn, axis=-1)
#     out = jnp.einsum('b h i j,b j h d ->b h i d', attn, v)
#     return out


def my_attention(
        q, k, v,
        **kwargs
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    sim = jnp.einsum('...qhd, ...khd -> ...hqk', q, k)
    # sim = sim - sim.amax(dim = -1, keepdim = True).detach()
    attn = nn.softmax(sim, axis=-1)
    out = jnp.einsum('...hqk,...khd ->...qhd', attn, v)
    return out


# def my_attention(q, k, v, ):
#     scale = q.shape[-1] ** -0.5
#     q = q * scale
#     attn = jnp.einsum('b  i d ,b  j d ->b  i j', q, k)
#     attn = nn.softmax(attn, axis=-1)
#     out = jnp.einsum('b  i j,b  j d ->b  i d', attn, v)
#     return out


import functools, jax, math
from jax import numpy as jnp, lax


# def _query_chunk_attention(query, key, value, precision, key_chunk_size=1024):
#     """Multi-head dot product attention with a limited number of queries."""
#     num_kv, num_heads, k_features = key.shape
#     v_features = value.shape[-1]
#     key_chunk_size = min(key_chunk_size, num_kv)
#     query = query / jnp.sqrt(k_features)
#
#     @functools.partial(jax.checkpoint, prevent_cse=False)
#     def summarize_chunk(query, key, value):
#         attn_weights = jnp.einsum('qhd,khd->qhk', query, key, precision=precision)
#         max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
#         max_score = jax.lax.stop_gradient(max_score)
#         exp_weights = jnp.exp(attn_weights - max_score)
#         exp_values = jnp.einsum('vhf,qhv->qhf', value, exp_weights, precision=precision)
#         return (exp_values, exp_weights.sum(axis=-1), max_score.reshape((query.shape[0], num_heads)))
#
#     def chunk_scanner(chunk_idx):
#         key_chunk = jax.lax.dynamic_slice(key, (chunk_idx, 0, 0), slice_sizes=(key_chunk_size, num_heads, k_features))
#         value_chunk = jax.lax.dynamic_slice(value, (chunk_idx, 0, 0),
#                                             slice_sizes=(key_chunk_size, num_heads, v_features))
#
#         return summarize_chunk(query, key_chunk, value_chunk)
#
#     chunk_values, chunk_weights, chunk_max = jax.lax.map(
#         chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))
#
#     print(chunk_values.shape,chunk_weights.shape,chunk_max.shape)
#
#     global_max = jnp.max(chunk_max, axis=0, keepdims=True)
#     print(global_max.shape,chunk_max.shape)
#     max_diffs = jnp.exp(chunk_max - global_max)
#     chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
#     chunk_weights *= max_diffs
#
#     all_values = chunk_values.sum(axis=0)
#     all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
#     return all_values / all_weights
#
#
# def attention(query, key, value, precision=jax.lax.Precision.HIGHEST,
#               query_chunk_size=1024):
#     """Memory-efficient multi-head dot product attention."""
#     num_q, num_heads, q_features = query.shape
#
#     def chunk_scanner(chunk_idx, _):
#         query_chunk = lax.dynamic_slice(
#             query, (chunk_idx, 0, 0),
#             slice_sizes=(min(query_chunk_size, num_q), num_heads, q_features))
#         print(query_chunk.shape)
#         return (chunk_idx + query_chunk_size, _query_chunk_attention(query_chunk, key, value, precision=precision))
#
#     _, res = jax.lax.scan(chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size))
#     return res.reshape(num_q, num_heads, value.shape[-1])

""""""
def _query_chunk_attention(query, key, value, precision, key_chunk_size=1024):
    """Multi-head dot product attention with a limited number of queries."""
    b,num_kv, num_heads, k_features = key.shape
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
        return (exp_values, exp_weights.sum(axis=-1), max_score.reshape((b,query.shape[1], num_heads)))

    def chunk_scanner(chunk_idx):
        key_chunk = jax.lax.dynamic_slice(key, (0,chunk_idx, 0, 0), slice_sizes=(b,key_chunk_size, num_heads, k_features))
        value_chunk = jax.lax.dynamic_slice(value, (0,chunk_idx, 0, 0),
                                            slice_sizes=(b,key_chunk_size, num_heads, v_features))

        return summarize_chunk(query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

    print(chunk_values.shape, chunk_weights.shape, chunk_max.shape)

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights


def attention(query, key, value, precision=jax.lax.Precision.HIGHEST,
              query_chunk_size=1024):
    """Memory-efficient multi-head dot product attention."""
    b,num_q, num_heads, q_features = query.shape

    def chunk_scanner(chunk_idx, _):
        query_chunk = lax.dynamic_slice(
            query, (0,chunk_idx, 0, 0),
            slice_sizes=(b,min(query_chunk_size, num_q), num_heads, q_features))
        print(query_chunk.shape)
        return (chunk_idx + query_chunk_size, _query_chunk_attention(query_chunk, key, value, precision=precision))

    _, res = jax.lax.scan(chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size))
    return res.reshape(b,num_q, num_heads, value.shape[-1])

def _query_chunk_attention_test(query_chunk, k, v):
    attn = jnp.einsum('qhd,khd->qhk', query_chunk, k)
    print(query_chunk.shape, k.shape, attn.shape)
    attn=nn.softmax(attn,axis=-1)
    out = jnp.einsum('qhk,khd->qhd', attn, v)
    return out


def attention_test(query, key, value, query_chunk_size=1):
    num_q, num_heads, q_features = query.shape

    def chunk_scanner(chunk_idx, init):
        print(chunk_idx, init)

        query_chunk = jax.lax.dynamic_slice(query, start_indices=(chunk_idx, 0, 0),
                                            slice_sizes=(query_chunk_size, num_heads, q_features))

        return chunk_idx + query_chunk_size, _query_chunk_attention_test(query_chunk, key, value)

    _, res = jax.lax.scan(chunk_scanner, init=0, xs=None, length=num_q // query_chunk_size)

    print(res.shape)
    res = res.reshape(num_q, num_heads, value.shape[-1])
    print(res.shape)
    # print(res.shape)
    return res


if __name__ == "__main__":

    rng_key = jax.random.PRNGKey(42)
    shape = (4,2048, 2, 48)
    # x = jax.random.normal(rng_key, shape) * 2
    x = jnp.ones(shape)
    #x=jnp.array([[[1,10,100],[1,20,200]]])

    num_kv = 13
    key_chunk_size = 2
    print(attention(x, x, x)-dot_product_attention(x,x,x))
    # print(my_attention(x, x, x)-attention(x, x, x))

    # print(jnp.arange(0, num_kv, key_chunk_size))

    # dot_product_attention=jax.jit(dot_product_attention)
    # attention=jax.jit(attention)
    # flax_attn = dot_product_attention(x, x, x)
    # flax_attn = attention(x, x, x)
    # start = time.time()
    # for _ in range(1):
    #     flax_attn = dot_product_attention(x, x, x)
    # end=time.time()
    # print(end-start)
    #
    # # my_attn = my_attention(x, x, x)
    #
    # # print(my_attn - flax_attn)
    #
    # start=time.time()
    # for _ in range(1):
    #     flax_attn =
    # end = time.time()
    # print(end - start)
    # print(attention(x, x, x)-dot_product_attention(x, x, x))

    # attn=
    """
    dtype = jnp.float32
    x = jnp.array([[3, 3, 30], [1, 1, 1]], dtype=dtype)

    out = nn.softmax(x)
    print(out)

    exp_value = jnp.exp(x)

    row_max = jnp.max(exp_value, -1, keepdims=True)

    exp_sum = jnp.sum(exp_value, -1)
    exp_sum = jnp.expand_dims(exp_sum, -1)

    print(exp_value, exp_sum)
    print(exp_value / exp_sum)
    print(row_max)
    print()

    rows_max = []
    chunks = []
    for row in x:
        row_max = jnp.max(row, -1, keepdims=True)
        print(row)
        rows_max.append(row_max)
        chunk = row - row_max
        chunk = jnp.exp(chunk)
        chunks.append(chunk)

    global_max = jnp.max(jnp.array(rows_max), axis=0, keepdims=True)

    chunks=jnp.array(chunks)

    global_sum = jnp.sum(chunks, axis=-1, keepdims=True)

    print(global_max)

    print(global_sum)
    print(chunks)
    print(chunks/global_sum)
    """