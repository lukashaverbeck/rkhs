from typing import Optional, Iterable, Sequence

import jax.numpy as jnp
from jax import Array


def make_arg_signature(ndim: int, var_symbol: str, prefixes: Optional[Iterable[str]] = None) -> str:
    if prefixes is None:
        prefixes = []

    symbols = prefixes + [f"{var_symbol}{dim + 1}" for dim in range(ndim)]

    return ",".join(symbols)


def is_broadcastable(x: Sequence[int] | Array, shape: Sequence[int]) -> bool:
    try:
        if isinstance(x, Array):
            jnp.broadcast_to(x, shape)
        else:
            jnp.broadcast_shapes(x, shape)
        return True
    except ValueError:
        return False


def expand_shape(array: Array, dims: int, prefix: bool = True) -> Array:
    if dims < array.ndim:
        raise ValueError(f"Array of shape {array.shape} has already more than {dims} dimensions.")

    added_dims = (1,) * (dims - array.ndim)

    if prefix:
        return array.reshape(*added_dims, *array.shape)
    else:
        return array.reshape(*array.shape, *added_dims)
