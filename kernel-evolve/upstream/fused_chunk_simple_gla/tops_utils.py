import types

import jax
import jax.numpy as jnp
from functools import singledispatch

def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n

@singledispatch
def cdiv(x: int, y: int):
    return (x + y - 1) // y


@cdiv.register
def cdiv(x: jax.Array, y: int):
    return (x + y - 1) // y


def align_up(x: int, align: int):
    return cdiv(x, align) * align


@singledispatch
def pad_to_multiple(x, multiple: int, axis: int, val):
    raise NotImplementedError(f"pad_to_multiple is not implemented for type {type(x)}")


@pad_to_multiple.register
def pad_to_multiple(x: jax.Array, multiple: int | list, axis: int | list, val):
    if isinstance(multiple, int):
        multiple = [multiple]
    if isinstance(axis, int):
        axis = [axis]

    assert len(multiple) == len(axis), (
        f"Length of multiple {len(multiple)} must match length of axis {len(axis)}"
    )

    shape = list(x.shape)
    pad_width = [(0, 0)] * len(shape)
    for idx in range(0, len(axis)):
        ax = axis[idx]
        mu = multiple[idx]
        length = shape[ax]
        remainder = length % mu
        if remainder == 0:
            continue
        pad_len = mu - remainder
        pad_width[ax] = (0, pad_len)
    return jnp.pad(x, pad_width, constant_values=val)


def prepare_lens(cu_seqlens: jax.Array) -> jax.Array:
    """
    Compute the actual length of each sequence.
    [0, 48, 64] -> [48, 16]
    """
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
) -> jax.Array:
    """Build a mapping from physical chunks to logical sequences for varlen inputs."""
    lens = prepare_lens(cu_seqlens)
    n_chunks = cdiv(lens, chunk_size)
    total_nt = int(jnp.sum(n_chunks))
    num_seqs = len(lens)

    seq_ids = jnp.repeat(
        jnp.arange(num_seqs, dtype=jnp.int32), n_chunks, total_repeat_length=total_nt
    )
    prefix_chunks = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(n_chunks)]
    )
    seq_offsets = jnp.repeat(prefix_chunks[:-1], n_chunks, total_repeat_length=total_nt)
    block_ids = jnp.arange(total_nt, dtype=jnp.int32) - seq_offsets

    return jnp.stack([seq_ids, block_ids], axis=1)

def assert_shape_or_none(x: jax.Array | list[jax.Array | None] | tuple[jax.Array | None, ...] | None,
                         expected_shape: tuple[int, ...], name: str | list[str] | tuple[str, ...] = "tensor"):
    """
    Concise helper to assert tensor shapes.
    Skips assertion for any element that is None.
    Supports a single array or an iterable of arrays that should all match the expected shape.
    """
    if x is None:
        return

    if isinstance(x, (list, tuple)):
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            if tensor is not None:
                curr_name = name[i] if has_names else f"{name}_{i}"
                assert tensor.shape == expected_shape, f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"

def assert_shape(x: jax.Array | list[jax.Array] | tuple[jax.Array, ...],
                 expected_shape: tuple[int, ...], name: str | list[str] | tuple[str, ...] = "tensor"):
    """
    Concise helper to assert tensor shapes.
    Supports a single array or an iterable of arrays that should all match the expected shape.
    """
    if isinstance(x, (list, tuple)):
        # If name is not a sequence or has mismatched length, use a generic numbered name
        has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
        for i, tensor in enumerate(x):
            curr_name = name[i] if has_names else f"{name}_{i}"
            assert tensor.shape == expected_shape, f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"

def export_public(current_globals):
    """
    """
    public_members = []

    module_name = current_globals.get('__name__')

    for name, obj in current_globals.items():
        if name.startswith('_'):
            continue

        if hasattr(obj, '__module__') and obj.__module__ != module_name:
            continue

        if isinstance(obj, types.ModuleType):
            continue

        if name == 'export_public':
            continue

        public_members.append(name)

    return public_members
