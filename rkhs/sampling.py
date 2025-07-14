from functools import partial
from typing import Sequence, Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from rkhs.base import Fn, CMO, distance, FnArray


def _extract_submatrix(matrix: Array, indices_1: Array, indices_2: Array) -> Array:
    return matrix[indices_2[..., None, :], indices_1[..., None]]


def _resample_fn(fn: Fn, indices: Array) -> Fn:
    bootstrap_points = jnp.take_along_axis(fn.points[None], indices[..., None], axis=1)
    bootstrap_coefficients = jnp.take_along_axis(fn.coefficients[None], indices, axis=1)
    mask = jnp.broadcast_to(fn.mask, shape=indices.shape)

    return fn.kernel.fn(bootstrap_points, bootstrap_coefficients, mask=mask)


def _resample_cme(cme: CMO, indices: Array) -> CMO:
    xs = jnp.take_along_axis(cme.xs[None], indices[..., None], axis=1)
    ys = jnp.take_along_axis(cme.ys[None], indices[..., None], axis=1)
    mask = jnp.broadcast_to(cme.mask, indices.shape)
    gram = _extract_submatrix(cme.gram, indices, indices)

    return cme.kernel.cme(xs, ys, mask=mask, gram=gram)


def _bootstrap_masked_indices(mask: Array, key: Array, shape: Sequence[int]) -> Array:
    if mask.ndim != 1:
        raise TypeError(f"Expected 1D mask. Got mask shape {mask.shape}.")

    return jax.random.choice(
        key=key,
        a=jnp.arange(mask.size),
        shape=shape,
        replace=True,
        p=(1 - mask) / (1 - mask).sum()
    )


def _batch_bootstrap[T: FnArray, O: FnArray](
        batch: T, key: Array, bootstrap: Callable[[T, Array], O]
) -> O:
    batch_shape = batch.shape
    batch = batch.reshape(-1)
    keys = jax.random.split(key, len(batch))

    bootstrapped: T = jax.lax.map(
        f=lambda inp: bootstrap(*inp),
        xs=(batch, keys)
    )

    return bootstrapped.transpose(1, 0).reshape(-1, *batch_shape)


@partial(jax.jit, static_argnums={2})
def bootstrap_fn(fn: Fn, key: Array, n: int) -> Fn:
    def bootstrap(fn_: Fn, key_: Array) -> Fn:
        resampled_indices = _bootstrap_masked_indices(fn_.mask, key_, shape=(n, fn_.n_points))
        return _resample_fn(fn_, resampled_indices)

    return _batch_bootstrap(fn, key, bootstrap)


@partial(jax.jit, static_argnums={2})
def bootstrap_cme(cme: CMO, key: Array, n: int) -> CMO:
    def bootstrap(cme_: CMO, key_: Array) -> CMO:
        resampled_indices = _bootstrap_masked_indices(cme_.mask, key_, shape=(n, cme_.n_points))
        return _resample_cme(cme_, resampled_indices)

    return _batch_bootstrap(cme, key, bootstrap)


@partial(jax.jit, static_argnums={3})
def bootstrap_distance(
        fn_1: Fn, fn_2: Fn,
        key: Array,
        n: int,
        gram_11: Optional[Array] = None, gram_22: Optional[Array] = None, gram_12: Optional[Array] = None,
) -> Array:
    if gram_11 is None:
        gram_11 = fn_1.kernel.gram(fn_1.points)
    if gram_22 is None:
        gram_22 = fn_2.kernel.gram(fn_2.points)
    if gram_12 is None:
        gram_12 = fn_2.kernel.gram(fn_1.points, fn_2.points)

    def bootstrap(fn_1_: Fn, fn_2_: Fn, key_: Array) -> Array:
        key_1, key_2 = jax.random.split(key_)
        resampled_indices_1 = _bootstrap_masked_indices(fn_1_.mask, key_1, shape=(n, fn_1_.n_points))
        resampled_indices_2 = _bootstrap_masked_indices(fn_1_.mask, key_2, shape=(n, fn_1_.n_points))

        bootstrap_gram_11 = _extract_submatrix(gram_11, resampled_indices_1, resampled_indices_1)
        bootstrap_gram_22 = _extract_submatrix(gram_22, resampled_indices_2, resampled_indices_2)
        bootstrap_gram_12 = _extract_submatrix(gram_12, resampled_indices_1, resampled_indices_2)

        bootstrap_fn_1 = _resample_fn(fn_1_, resampled_indices_1)
        bootstrap_fn_2 = _resample_fn(fn_2_, resampled_indices_2)

        return distance(
            bootstrap_fn_1, bootstrap_fn_2,
            kernel_matrix_11=bootstrap_gram_11,
            kernel_matrix_22=bootstrap_gram_22,
            kernel_matrix_12=bootstrap_gram_12,
        )

    batch_shape = jnp.broadcast_shapes(fn_1.shape, fn_2.shape)
    fn_1 = fn_1.broadcast_to(batch_shape).reshape(-1)
    fn_2 = fn_2.broadcast_to(batch_shape).reshape(-1)
    keys = jax.random.split(key, len(fn_1.shape))

    bootstrap_distances = jax.lax.map(
        f=lambda inp: bootstrap(*inp),
        xs=(fn_1, fn_2, keys)
    )

    return bootstrap_distances.transpose(1, 0).reshape(n, *batch_shape)
