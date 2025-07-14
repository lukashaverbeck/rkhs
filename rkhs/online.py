from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from rkhs.base import KME, CMO


@partial(jax.jit)
def _cholesky_update(cholesky: Array, update_vector: Array) -> Array:
    cholesky = jnp.asarray(cholesky)
    update_vector = jnp.asarray(update_vector, dtype=cholesky.dtype)
    n = cholesky.shape[-1]

    omega_init = update_vector
    beta_init = jnp.asarray(1)
    row_idx = jnp.arange(n)

    def step(carry: tuple[Array, Array, Array], j: Array) -> tuple[tuple[Array, Array, Array], None]:
        cholesky_, omega, beta = carry

        l_jj = cholesky_[..., j, j]
        omega_j = omega[..., j]

        new_ljj = jnp.sqrt(l_jj ** 2 + 1 / beta * omega_j ** 2)
        scale = l_jj ** 2 * beta + omega_j ** 2

        col = cholesky_[..., :, j]

        omega = omega - (omega_j / l_jj)[..., None] * col

        mask = (row_idx >= j).astype(cholesky_.dtype)
        new_col = new_ljj[..., None] * (col / l_jj[..., None] + (omega_j / scale)[..., None] * omega * mask)

        cholesky_ = cholesky_.at[..., :, j].set(new_col)
        cholesky_ = cholesky_.at[..., j, j].set(new_ljj)

        beta = beta + (omega_j / l_jj) ** 2

        return (cholesky_, omega, beta), None

    (updated_cholesky, _, _), _ = jax.lax.scan(
        f=step,
        init=(cholesky, omega_init, beta_init),
        xs=jnp.arange(n)
    )
    return updated_cholesky


def _decrease_matrix(matrix: Array, index: int) -> Array:
    matrix = jnp.delete(matrix, index, axis=-1, assume_unique_indices=True)
    return jnp.delete(matrix, index, axis=-2, assume_unique_indices=True)


def _extend_matrix(matrix: Array, row: Array, column: Array, corner: Array) -> Array:
    column = jnp.append(column, jnp.expand_dims(corner, axis=-1), axis=-1)
    matrix = jnp.append(matrix, jnp.expand_dims(row, axis=-2), axis=-2)
    matrix = jnp.append(matrix, jnp.expand_dims(column, axis=-1), axis=-1)
    return matrix


def _decrease_gram_cholesky(cholesky: Array, index: int) -> Array:
    removed_row = jnp.delete(cholesky[..., :, index], index, axis=-1, assume_unique_indices=True)
    cholesky = _decrease_matrix(cholesky, index)

    batch_shape = cholesky.shape[:-2]
    cholesky = cholesky.reshape(-1, *cholesky.shape[-2:])
    removed_row = removed_row.reshape(-1, removed_row.shape[-1])

    def update_cholesky(cholesky_: Array, removed_row_: Array) -> Array:
        new_cholesky_block = _cholesky_update(cholesky_, removed_row_)
        return jax.lax.dynamic_update_slice(cholesky_, new_cholesky_block, start_indices=(index, index))

    return jax.lax.map(
        f=lambda inp: update_cholesky(*inp),
        xs=(cholesky, removed_row)
    ).reshape(*batch_shape, *cholesky.shape[-2:])


def _extend_gram_cholesky(cholesky: Array, kernel_xs: Array, kernel_x: Array, regularization: float) -> Array:
    row = jax.scipy.linalg.solve_triangular(cholesky, kernel_xs, lower=True)
    corner_value = jnp.sqrt(kernel_x + regularization - jnp.einsum("...i,...i->...", row, row))
    column = jnp.zeros_like(kernel_xs)

    return _extend_matrix(cholesky, row, column, corner_value)


@partial(jax.jit)
def kme_remove_data(kme: KME, index: int) -> KME:
    def delete_in_data_axis(array: Array) -> Array:
        return jnp.delete(array, index, axis=kme.dataset_axis, assume_unique_indices=True)

    points = delete_in_data_axis(kme.points)
    mask = delete_in_data_axis(kme.mask)

    return kme.kernel.kme(points, mask=mask)


@partial(jax.jit)
def kme_add_data(kme: KME, x: Array, mask: Optional[Array] = None) -> KME:
    if mask is None:
        mask = jnp.full(shape=kme.shape, fill_value=False)

    x = jnp.expand_dims(x, axis=kme.dataset_axis)
    mask = jnp.expand_dims(mask, axis=kme.dataset_axis)

    points = jnp.append(kme.points, x, axis=kme.dataset_axis)
    mask = jnp.append(kme.mask, mask, axis=kme.dataset_axis)

    return kme.kernel.kme(points, mask=mask)


@partial(jax.jit)
def cme_remove_data(cme: CMO, index: int) -> CMO:
    def delete_in_data_axis(array: Array) -> Array:
        return jnp.delete(array, index, axis=cme.dataset_axis, assume_unique_indices=True)

    xs = delete_in_data_axis(cme.xs)
    ys = delete_in_data_axis(cme.ys)
    mask = delete_in_data_axis(cme.mask)
    gram = _decrease_matrix(cme.gram, index)

    cholesky = _decrease_gram_cholesky(cme.cholesky, index)

    return CMO(
        kernel=cme.kernel,
        xs=xs,
        ys=ys,
        gram=gram,
        cholesky=cholesky,
        mask=mask
    )


@partial(jax.jit)
def cme_add_data(cme: CMO, x: Array, y: Array, mask: Optional[Array] = None) -> CMO:
    if mask is None:
        mask = jnp.full(shape=cme.shape, fill_value=False)

    kernel_xs = cme.kernel.x.kernel_vector(cme.xs, x)
    kernel_x = cme.kernel.x(x, x)

    x = jnp.expand_dims(x, axis=cme.dataset_axis)
    y = jnp.expand_dims(y, axis=cme.dataset_axis)
    mask = jnp.expand_dims(mask, axis=cme.dataset_axis)

    xs = jnp.append(cme.xs, x, axis=cme.dataset_axis)
    ys = jnp.append(cme.ys, y, axis=cme.dataset_axis)
    mask = jnp.append(cme.mask, mask, axis=cme.dataset_axis)

    gram = _extend_matrix(cme.gram, kernel_xs, kernel_xs, kernel_x)
    cholesky = _extend_gram_cholesky(cme.cholesky, kernel_xs, kernel_x, cme.kernel.regularization)

    return CMO(
        kernel=cme.kernel,
        xs=xs,
        ys=ys,
        gram=gram,
        cholesky=cholesky,
        mask=mask
    )
