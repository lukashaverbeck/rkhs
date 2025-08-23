from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Sequence, Final, Optional, Self, Literal, Concatenate

import jax
import jax.numpy as jnp
from jax import Array

from rkhs._util import make_arg_signature, is_broadcastable, expand_shape

type RKHSDim = int | Literal["inf"]


class Kernel(ABC):
    data_shape: Final[tuple[int, ...]]
    rkhs_dim: Final[RKHSDim]

    @property
    def data_ndim(self) -> int:
        return len(self.data_shape)

    def __init__(self, data_shape: Sequence[int], rkhs_dim: RKHSDim = "inf"):
        if any(shape < 0 for shape in data_shape):
            raise ValueError(f"Got negative data shape: {data_shape}.")

        self.data_shape = tuple(data_shape)
        self.rkhs_dim = rkhs_dim

    @abstractmethod
    def _dot(self, x_1: Array, x_2: Array) -> Array:
        raise NotImplementedError

    def batch_shape(self, array: Array) -> tuple[int, ...]:
        return array.shape[:array.ndim - self.data_ndim]

    @staticmethod
    def __no_mask(batch_shape: Sequence[int]) -> Array:
        dataset_size = batch_shape[-1]
        mask = jnp.full(shape=(dataset_size,), fill_value=False, dtype=bool)
        mask = expand_shape(mask, dims=len(batch_shape))
        return mask

    def check_shape(self, x: Array, batch: bool = False):
        data_shape_str = ", ".join(map(str, self.data_shape))

        if batch:
            expected_shape_str = f"(..., *, {data_shape_str})"
        else:
            expected_shape_str = f"(..., {data_shape_str})"

        error = TypeError(f"Expected shape {expected_shape_str}. Got {x.shape}.")

        if (batch and x.ndim < self.data_ndim + 1) or (not batch and x.ndim < self.data_ndim):
            raise error

        if x.shape[x.ndim - self.data_ndim:] != self.data_shape:
            raise error

    @partial(jax.jit, static_argnums={0})
    def __call__(self, x_1: Array, x_2: Array) -> Array:
        self.check_shape(x_1)
        self.check_shape(x_2)

        x_signature = make_arg_signature(self.data_ndim, var_symbol="x")

        @partial(jnp.vectorize, signature=f"({x_signature}),({x_signature})->()")
        def vectorized(x_1_: Array, x_2_: Array) -> Array:
            return self._dot(x_1_, x_2_)

        return vectorized(x_1, x_2)

    def gram(self, xs: Array, xs_2: Optional[Array] = None) -> Array:
        if xs_2 is None:
            xs_2 = xs

        self.check_shape(xs, batch=True)
        self.check_shape(xs_2, batch=True)

        xs = jnp.expand_dims(xs, axis=xs.ndim - self.data_ndim)
        xs_2 = jnp.expand_dims(xs_2, axis=xs_2.ndim - self.data_ndim - 1)

        return self(xs, xs_2)

    def vector(self, xs: Array, x: Array) -> Array:
        self.check_shape(xs, batch=True)
        self.check_shape(x)

        x = jnp.expand_dims(x, axis=x.ndim - self.data_ndim)

        return self.gram(xs, x).squeeze(-1)

    def fn(self, points: Array, coefficients: Array, mask: Optional[Array] = None) -> Fn:
        self.check_shape(points, batch=True)

        point_batch_shape = self.batch_shape(points)

        if mask is None:
            mask = self.__no_mask(point_batch_shape)

        fn_shape = jnp.broadcast_shapes(point_batch_shape, coefficients.shape, mask.shape)[:-1]
        ndim = len(fn_shape)

        points = expand_shape(points, ndim + self.data_ndim + 1)
        coefficients = expand_shape(coefficients, ndim + 1)
        mask = expand_shape(mask, ndim + 1)

        return Fn(kernel=self, points=points, coefficients=coefficients, mask=mask)

    @partial(jax.jit, static_argnums={0})
    def kme(self, xs: Array, mask: Optional[Array] = None) -> Fn:
        self.check_shape(xs, batch=True)

        batch_shape = self.batch_shape(xs)

        if mask is None:
            mask = self.__no_mask(batch_shape)

        mask_degenerate = jnp.all(mask, axis=-1)

        coefficients = 1 / ((~mask).sum(axis=-1, keepdims=True) + mask_degenerate) * ~mask_degenerate
        coefficients = jnp.broadcast_to(coefficients, mask.shape)

        return self.fn(xs, coefficients, mask)


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class _MaskedShape(ABC):
    mask: Array

    @property
    def shape(self) -> tuple[int, ...]:
        shapes = self._batch_shapes()

        return tuple(
            max(sizes)
            for sizes in zip(*shapes)
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dataset_axis(self) -> int:
        return self.ndim

    @property
    def dataset_shape_size(self) -> int:
        return self.mask.shape[self.dataset_axis]

    @partial(jax.jit)
    def dataset_size(self) -> Array:
        return self.dataset_shape_size - self.mask.sum(axis=-1)

    @abstractmethod
    def _batch_shapes(self) -> tuple[tuple[int, ...], ...]:
        raise NotImplementedError

    def _check_ndim(self, array: Array, excess_dims: int, name: str):
        if array.ndim != self.ndim + 1 + excess_dims:
            raise TypeError(f"Dimension {array.ndim} of {name} does not match dimension {self.ndim} of CME.")

    def _check_dataset_dim(self, array: Array, name: str):
        if array.shape[self.dataset_axis] != self.dataset_shape_size:
            raise TypeError(f"Dimension {array.shape[self.dataset_axis]} of {name} in axis {self.dataset_axis} of shape"
                            f"{array.shape} does not match dataset size {self.dataset_shape_size}.")

    def _check_broadcastable(self, array: Array, excess_shape: Sequence[int], name: str):
        target_shape = (*self.shape, self.dataset_shape_size, *excess_shape)
        if not is_broadcastable(array, target_shape):
            raise TypeError(f"Can't broadcast shape {array.shape} of {name} to shape {target_shape}.")

    @partial(jax.jit)
    def _take_from_dataset(self, dataset: Array, indices: Array) -> Array:
        broadcasted_shape = (*self.shape, indices.shape[-1])

        if not is_broadcastable(indices, shape=broadcasted_shape):
            raise TypeError(f"Indices must be broadcastable to cme shape {self.shape} in first {self.ndim} axes. "
                            f"Got shape {indices.shape}.")

        indices = jnp.broadcast_to(indices, shape=broadcasted_shape)
        indices = expand_shape(indices, dims=dataset.ndim, prefix=False)

        return jnp.take_along_axis(dataset, indices, axis=self.dataset_axis)

    @abstractmethod
    def materialize(self) -> Self:
        raise NotImplementedError

    @staticmethod
    def _pre_materialize[T: _MaskedShape, **P, R](
            op: Callable[Concatenate[T, P], R]
    ) -> Callable[Concatenate[Fn, P], R]:
        def wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R:
            self = self.materialize()
            return op(self, *args, **kwargs)

        return wrapper

    def _reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return array.reshape(*shape, *array.shape[self.ndim:])

    def _expand_shape(self, axis: int | Sequence[int]) -> tuple[int, ...]:
        return jnp.expand_dims(jnp.ones(self.shape), axis=axis).shape

    def _expand_dim(self, array: Array, axis: int | Sequence[int]) -> Array:
        if isinstance(axis, int):
            axis = (axis,)

        axis = tuple(
            ax - (array.ndim - self.ndim) if ax < 0 else ax
            for ax in axis
        )

        return jnp.expand_dims(array, axis)

    @abstractmethod
    def expand_dims(self, axis: int | Sequence[int]) -> Array:
        raise NotImplementedError

    @abstractmethod
    def reshape(self, *shape: int) -> Self:
        raise NotImplementedError

    def _transpose(self, array: Array, axes: tuple[int, ...]) -> Array:
        if set(axes) != set(range(self.ndim)):
            raise ValueError(f"Dimensions must be a permutation of {tuple(range(self.ndim))}. Found {axes}.")

        return array.transpose(*axes, *range(self.ndim, array.ndim))

    @abstractmethod
    def transpose(self, *axes: int) -> Self:
        raise NotImplementedError

    def _broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
        return jnp.broadcast_to(array, shape=(*shape, *array.shape[self.ndim:]))

    @abstractmethod
    def broadcast_to(self, shape: Sequence[int]) -> Self:
        raise NotImplementedError


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class Fn(_MaskedShape):
    kernel: Kernel = field(metadata=dict(static=True))
    points: Array
    coefficients: Array

    @property
    def rkhs_dim(self) -> int:
        if self.kernel.rkhs_dim == "inf":
            return self.dataset_shape_size

        return min(self.kernel.rkhs_dim, self.dataset_shape_size)

    @property
    def point_shape(self) -> tuple[int, ...]:
        return self.kernel.data_shape

    def __post_init__(self):
        self.kernel.check_shape(self.points, batch=True)

        self._check_ndim(self.points, excess_dims=self.kernel.data_ndim, name="points")
        self._check_ndim(self.coefficients, excess_dims=0, name="coefficients")
        self._check_ndim(self.mask, excess_dims=0, name="mask")

        self._check_dataset_dim(self.points, name="points")
        self._check_dataset_dim(self.coefficients, name="coefficients")
        self._check_dataset_dim(self.mask, name="mask")

        self._check_broadcastable(self.points, excess_shape=self.kernel.data_shape, name="points")
        self._check_broadcastable(self.coefficients, excess_shape=(), name="coefficients")
        self._check_broadcastable(self.mask, excess_shape=(), name="mask")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, kernel={self.kernel.__class__.__name__})"

    def _batch_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            self.mask.shape[:-1],
            self.kernel.batch_shape(self.points)[:-1],
            self.coefficients.shape[:-1],
        )

    def materialize(self) -> Self:
        points = jnp.broadcast_to(self.points, shape=(*self.shape, self.dataset_shape_size, *self.point_shape))
        coefficients = jnp.broadcast_to(self.coefficients, shape=(*self.shape, self.dataset_shape_size))
        mask = jnp.broadcast_to(self.mask, shape=(*self.shape, self.dataset_shape_size))

        return Fn(kernel=self.kernel, points=points, coefficients=coefficients, mask=mask)

    @partial(jax.jit)
    def __call__(self, x: Array) -> Array:
        self.kernel.check_shape(x)
        k_x = self.kernel.vector(self.points, x)
        return (self.coefficients * k_x * ~self.mask).sum(axis=-1)

    @_MaskedShape._pre_materialize
    def reshape(self, *shape: int) -> Self:
        points = self._reshape(self.points, shape)
        coefficients = self._reshape(self.coefficients, shape)
        mask = self._reshape(self.mask, shape=shape)

        return Fn(kernel=self.kernel, points=points, coefficients=coefficients, mask=mask)

    def expand_dims(self, axis: int | Sequence[int]) -> Self:
        points = self._expand_dim(self.points, axis)
        coefficients = self._expand_dim(self.coefficients, axis)
        mask = self._expand_dim(self.mask, axis)

        return Fn(kernel=self.kernel, points=points, coefficients=coefficients, mask=mask)

    def transpose(self, *axes: int) -> Self:
        points = self._transpose(self.points, axes)
        coefficients = self._transpose(self.coefficients, axes)
        mask = self._transpose(self.mask, axes)

        return Fn(kernel=self.kernel, points=points, coefficients=coefficients, mask=mask)

    def broadcast_to(self, shape: Sequence[int]) -> Self:
        shape = tuple(shape)

        points = self._broadcast_to(self.points, shape)
        coefficients = self._broadcast_to(self.coefficients, shape)
        mask = self._broadcast_to(self.mask, shape)

        return Fn(kernel=self.kernel, points=points, coefficients=coefficients, mask=mask)

    def __add__(self, other: Fn) -> Self:
        points_1, points_2 = jnp.broadcast_arrays(self.points, other.points)
        coefficients_1, coefficients_2 = jnp.broadcast_arrays(self.coefficients, other.coefficients)
        mask_1, mask_2 = jnp.broadcast_arrays(self.mask, other.mask)

        points = jnp.concatenate([points_1, points_2], axis=self.ndim)
        coefficients = jnp.concatenate([coefficients_1, coefficients_2], axis=self.ndim)
        mask = jnp.concatenate([mask_1, mask_2], axis=self.ndim)

        return Fn(mask=mask, kernel=self.kernel, points=points, coefficients=coefficients)


class VectorKernel:
    x: Kernel
    y: Kernel
    regularization: float

    def __init__(self, x: Kernel, y: Kernel, regularization: float):
        if regularization < 0:
            raise ValueError(f"Regularization must be positive. Got {regularization}.")

        self.x = x
        self.y = y
        self.regularization = regularization

    @partial(jax.jit, static_argnums={0})
    def cme(self, xs: Array, ys: Array, mask: Optional[Array] = None, gram: Optional[Array] = None) -> CME:
        self.x.check_shape(xs, batch=True)
        self.y.check_shape(ys, batch=True)

        batch_shape_xs = self.x.batch_shape(xs)
        batch_shape_ys = self.y.batch_shape(ys)

        dataset_size = batch_shape_xs[-1]

        if mask is None:
            mask = jnp.full(shape=(dataset_size,), fill_value=False, dtype=bool)

        if gram is None:
            gram = self.x.gram(xs)

        regularized_gram = gram + self.regularization * jnp.eye(batch_shape_xs[-1])
        cholesky, _ = jax.scipy.linalg.cho_factor(regularized_gram, lower=True, overwrite_a=True)

        cme_shape = jnp.broadcast_shapes(batch_shape_xs, batch_shape_ys, cholesky.shape[:-1], mask.shape)[:-1]

        xs = expand_shape(xs, dims=len(cme_shape) + self.x.data_ndim + 1)
        ys = expand_shape(ys, dims=len(cme_shape) + self.y.data_ndim + 1)
        cholesky = expand_shape(cholesky, dims=len(cme_shape) + 2)
        mask = expand_shape(mask, dims=len(cme_shape) + 1)

        return CME(kernel=self, xs=xs, ys=ys, cholesky=cholesky, mask=mask)


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class CME(_MaskedShape):
    kernel: VectorKernel = field(metadata=dict(static=True))
    xs: Array
    ys: Array
    cholesky: Array

    @property
    def rkhs_dim_y(self) -> int:
        if self.kernel.y.rkhs_dim == "inf":
            return self.dataset_shape_size

        return min(self.kernel.y.rkhs_dim, self.dataset_shape_size)

    @property
    def point_x_shape(self) -> tuple[int, ...]:
        return self.kernel.x.data_shape

    @property
    def point_y_shape(self) -> tuple[int, ...]:
        return self.kernel.y.data_shape

    def __post_init__(self):
        self.kernel.x.check_shape(self.xs, batch=True)
        self.kernel.y.check_shape(self.ys, batch=True)

        self._check_ndim(self.xs, excess_dims=self.kernel.x.data_ndim, name="xs")
        self._check_ndim(self.ys, excess_dims=self.kernel.y.data_ndim, name="ys")
        self._check_ndim(self.cholesky, excess_dims=1, name="cholesky")
        self._check_ndim(self.mask, excess_dims=0, name="mask")

        self._check_dataset_dim(self.xs, name="xs")
        self._check_dataset_dim(self.ys, name="ys")
        self._check_dataset_dim(self.cholesky, name="cholesky")
        self._check_dataset_dim(self.mask, name="mask")

        self._check_broadcastable(self.xs, excess_shape=self.point_x_shape, name="xs")
        self._check_broadcastable(self.ys, excess_shape=self.point_y_shape, name="ys")
        self._check_broadcastable(self.cholesky, excess_shape=(self.dataset_shape_size,), name="cholesky")
        self._check_broadcastable(self.mask, excess_shape=(), name="mask")

        if self.cholesky.shape[-1] != self.cholesky.shape[-2]:
            raise TypeError(f"Cholesky should be symmetric. Got shape {self.cholesky.shape}.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, kernel={self.kernel.__class__.__name__})"

    def _batch_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            self.mask.shape[:-1],
            self.kernel.x.batch_shape(self.xs)[:-1],
            self.kernel.y.batch_shape(self.ys)[:-1],
            self.cholesky.shape[:-2]
        )

    def materialize(self) -> Self:
        dataset_shape = (*self.shape, self.dataset_shape_size)

        xs = jnp.broadcast_to(self.xs, shape=(*dataset_shape, *self.point_x_shape))
        ys = jnp.broadcast_to(self.ys, shape=(*dataset_shape, *self.point_y_shape))
        cholesky = jnp.broadcast_to(self.cholesky, shape=(*dataset_shape, self.dataset_shape_size))
        mask = jnp.broadcast_to(self.mask, shape=dataset_shape)

        return CME(kernel=self.kernel, xs=xs, ys=ys, cholesky=cholesky, mask=mask)

    def take_xs(self, indices: Array) -> Array:
        return self._take_from_dataset(self.xs, indices)

    def take_ys(self, indices: Array) -> Array:
        return self._take_from_dataset(self.ys, indices)

    def take_data(self, indices: Array) -> tuple[Array, Array]:
        xs = self.take_xs(indices)
        ys = self.take_ys(indices)
        return xs, ys

    @partial(jax.jit)
    def influence(self, x: Array) -> Array:
        self.kernel.x.check_shape(x)

        @partial(jnp.vectorize, signature="(n),(n,n)->(n)")
        def vectorized_solve(k_x_: Array, cholesky: Array):
            return jax.scipy.linalg.cho_solve((cholesky, True), k_x_, overwrite_b=True)

        k_x = self.kernel.x.vector(self.xs, x)

        return vectorized_solve(k_x, self.cholesky) * ~self.mask

    @partial(jax.jit)
    def __call__(self, x: Array) -> Fn:
        self.kernel.x.check_shape(x)

        coefficients = self.influence(x)

        return self.kernel.y.fn(self.ys, coefficients, self.mask)

    def reshape(self, *shape: int) -> Self:
        xs = self._reshape(self.xs, shape)
        ys = self._reshape(self.ys, shape)
        cholesky = self._reshape(self.cholesky, shape)
        mask = self._reshape(self.mask, shape)

        return CME(kernel=self.kernel, xs=xs, ys=ys, cholesky=cholesky, mask=mask)

    def expand_dims(self, axis: int | Sequence[int]) -> Self:
        xs = self._expand_dim(self.xs, axis)
        ys = self._expand_dim(self.ys, axis)
        cholesky = self._expand_dim(self.cholesky, axis)
        mask = self._expand_dim(self.mask, axis)

        return CME(kernel=self.kernel, xs=xs, ys=ys, cholesky=cholesky, mask=mask)

    def transpose(self, *axes: int) -> Self:
        xs = self._transpose(self.xs, axes)
        ys = self._transpose(self.ys, axes)
        cholesky = self._transpose(self.cholesky, axes)
        mask = self._transpose(self.mask, axes)

        return CME(kernel=self.kernel, xs=xs, ys=ys, cholesky=cholesky, mask=mask)

    def broadcast_to(self, shape: Sequence[int]) -> Self:
        shape = tuple(shape)

        xs = self._broadcast_to(self.xs, shape)
        ys = self._broadcast_to(self.ys, shape)
        cholesky = self._broadcast_to(self.cholesky, shape)
        mask = self._broadcast_to(self.mask, shape)

        return CME(kernel=self.kernel, xs=xs, ys=ys, cholesky=cholesky, mask=mask)


@partial(jax.jit)
def dot(fn_1: Fn, fn_2: Fn, gram: Optional[Array] = None) -> Array:
    if gram is None:
        gram = fn_1.kernel.gram(fn_1.points, fn_2.points)

    return jnp.einsum("...i,...ij,...j->...", fn_1.coefficients, gram, fn_2.coefficients)


@partial(jax.jit)
def squared_distance(
        fn_1: Fn, fn_2: Fn,
        gram_11: Optional[Array] = None,
        gram_22: Optional[Array] = None,
        gram_12: Optional[Array] = None
) -> Array:
    dp_11 = dot(fn_1, fn_1, gram=gram_11)
    dp_22 = dot(fn_2, fn_2, gram=gram_22)
    dp_12 = dot(fn_1, fn_2, gram=gram_12)

    return jnp.clip(dp_11 + dp_22 - 2 * dp_12, min=0)  # clip to avoid numerical errors


def distance(
        fn_1: Fn, fn_2: Fn,
        gram_11: Optional[Array] = None,
        gram_22: Optional[Array] = None,
        gram_12: Optional[Array] = None
) -> Array:
    return jnp.sqrt(squared_distance(fn_1, fn_2, gram_11=gram_11, gram_22=gram_22, gram_12=gram_12))


def squared_norm(fn: Fn, gram: Optional[Array] = None) -> Array:
    return jnp.clip(dot(fn, fn, gram=gram), min=0)  # clip to avoid numerical errors


def norm(fn: Fn, gram: Optional[Array] = None) -> Array:
    return jnp.sqrt(squared_norm(fn, gram=gram))


@partial(jax.jit, static_argnums={0})
def kme_dot(
        kernel: Kernel, xs_1: Array, xs_2: Array,
        mask_1: Optional[Array] = None,
        mask_2: Optional[Array] = None
) -> Array:
    kme_1 = kernel.kme(xs_1, mask=mask_1)
    kme_2 = kernel.kme(xs_2, mask=mask_2)
    return dot(kme_1, kme_2)


@partial(jax.jit, static_argnums={0})
def squared_mmd(
        kernel: Kernel,
        xs_1: Array, xs_2: Array,
        mask_1: Optional[Array] = None,
        mask_2: Optional[Array] = None,
        gram_11: Optional[Array] = None,
        gram_22: Optional[Array] = None,
        gram_12: Optional[Array] = None
) -> Array:
    kme_1 = kernel.kme(xs_1, mask=mask_1)
    kme_2 = kernel.kme(xs_2, mask=mask_2)
    return squared_distance(kme_1, kme_2, gram_11, gram_22, gram_12)


@partial(jax.jit, static_argnums={0})
def mmd(
        kernel: Kernel,
        xs_1: Array, xs_2: Array,
        mask_1: Optional[Array] = None,
        mask_2: Optional[Array] = None,
        gram_11: Optional[Array] = None,
        gram_22: Optional[Array] = None,
        gram_12: Optional[Array] = None
) -> Array:
    kme_1 = kernel.kme(xs_1, mask=mask_1)
    kme_2 = kernel.kme(xs_2, mask=mask_2)
    return distance(kme_1, kme_2, gram_11=gram_11, gram_22=gram_22, gram_12=gram_12)


@partial(jax.jit, static_argnums={0})
def cme_dot(
        kernel: VectorKernel,
        xs_1: Array, xs_2: Array, ys_1: Array, ys_2: Array, e_1: Array, e_2: Array,
        mask_1: Optional[Array] = None,
        mask_2: Optional[Array] = None,
        gram_y: Optional[Array] = None,
) -> Array:
    kme_1 = kernel.cme(xs_1, ys_1, mask=mask_1)(e_1)
    kme_2 = kernel.cme(xs_2, ys_2, mask=mask_2)(e_2)

    if gram_y is None:
        gram_y = kernel.y.gram(ys_1, ys_2)

    return dot(kme_1, kme_2, gram=gram_y)


@partial(jax.jit, static_argnums={0})
def squared_cmmd(
        kernel: VectorKernel,
        xs_1: Array, xs_2: Array, ys_1: Array, ys_2: Array, e_1: Array, e_2: Array,
        mask_1: Optional[Array] = None,
        mask_2: Optional[Array] = None,
        gram_y_11: Optional[Array] = None,
        gram_y_22: Optional[Array] = None,
        gram_y_12: Optional[Array] = None
) -> Array:
    dp_11 = cme_dot(kernel, xs_1, xs_1, ys_1, ys_1, e_1, e_2, mask_1=mask_1, mask_2=mask_1, gram_y=gram_y_11)
    dp_22 = cme_dot(kernel, xs_2, xs_2, ys_2, ys_2, e_1, e_2, mask_1=mask_2, mask_2=mask_2, gram_y=gram_y_22)
    dp_12 = cme_dot(kernel, xs_1, xs_2, ys_1, ys_2, e_1, e_2, mask_1=mask_1, mask_2=mask_2, gram_y=gram_y_12)

    return jnp.clip(dp_11 + dp_22 - 2 * dp_12, min=0)  # clip to avoid numerical errors


@partial(jax.jit, static_argnums={0})
def cmmd(
        kernel: VectorKernel,
        xs_1: Array, xs_2: Array, ys_1: Array, ys_2: Array, e_1: Array, e_2: Array,
        mask_1: Optional[Array] = None,
        mask_2: Optional[Array] = None,
        gram_y_11: Optional[Array] = None,
        gram_y_22: Optional[Array] = None,
        gram_y_12: Optional[Array] = None
) -> Array:
    return jnp.sqrt(squared_cmmd(
        kernel, xs_1, xs_2, ys_1, ys_2, e_1, e_2,
        mask_1=mask_1, mask_2=mask_2,
        gram_y_11=gram_y_11, gram_y_22=gram_y_22, gram_y_12=gram_y_12
    ))
