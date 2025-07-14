from __future__ import annotations

from abc import abstractmethod, ABC
from collections.abc import Callable
from functools import partial
from typing import NamedTuple, Optional, Self, Mapping, Iterator, Final, Sequence, Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from rkhs._util import _make_arg_signature

type KernelFn = Callable[[Array, Array], Array]


class Kernel(ABC):
    ndim: Final[int]

    def __init__(self, ndim: int):
        if ndim < 0:
            raise ValueError(f"Number of dimensions must be non-negative. Got {ndim}.")

        self.ndim = ndim

    @abstractmethod
    def _dot(self, x_1: Array, x_2: Array) -> Array:
        raise NotImplementedError

    def __check_shape(self, x: Array, batched: bool = False):
        expected_dimensions = self.ndim + 1 if batched else self.ndim

        if x.ndim < expected_dimensions:
            raise ValueError(f"Expected at least {expected_dimensions} dimensions. Got shape {x.shape}.")

    @partial(jax.jit, static_argnums={0})
    def __call__(self, x_1: Array, x_2: Array, mask_1: Optional[Array] = None, mask_2: Optional[Array] = None) -> Array:
        self.__check_shape(x_1)
        self.__check_shape(x_2)

        if mask_1 is None:
            mask_1 = self._no_mask(x_1)
        if mask_2 is None:
            mask_2 = self._no_mask(x_2)

        # TODO: check shape of mask

        arg_signature = _make_arg_signature(self.ndim, var_symbol='x')

        @partial(jnp.vectorize, signature=f"({arg_signature}),({arg_signature}),(),()->()")
        def vectorized(x_1_: Array, x_2_: Array, mask_1_: Array, mask_2_: Array) -> Array:
            return self._dot(x_1_, x_2_) * (1 - mask_1_) * (1 - mask_2_)

        return vectorized(x_1, x_2, mask_1, mask_2)

    def _no_mask(self, x: Array) -> Array:
        return jnp.full(shape=x.shape[:x.ndim - self.ndim], fill_value=False)

    @partial(jax.jit, static_argnums={0})
    def gram(
            self,
            xs: Array, xs_2: Optional[Array] = None,
            mask: Optional[Array] = None, mask_2: Optional[Array] = None
    ) -> Array:
        if xs_2 is None:
            xs_2 = xs
        if mask is None:
            mask = self._no_mask(xs)
        if mask_2 is None:
            mask_2 = self._no_mask(xs_2)

        self.__check_shape(xs, batched=True)
        self.__check_shape(xs_2, batched=True)

        arg_signature_1 = _make_arg_signature(self.ndim, var_symbol="x1_")
        arg_signature_2 = _make_arg_signature(self.ndim, var_symbol="x2_")

        @partial(jnp.vectorize, signature=f"(n,{arg_signature_1}),(m,{arg_signature_2}),(n),(m)->(n,m)")
        def vectorized(xs_1_: Array, xs_2_: Array, mask_1_: Array, mask_2_: Array) -> Array:
            return self(xs_1_[:, None], xs_2_[None, :], mask_1_[:, None], mask_2_[None, :])

        return vectorized(xs, xs_2, mask, mask_2)

    @partial(jax.jit, static_argnums={0})
    def kernel_vector(
            self,
            xs: Array, x: Array,
            mask_xs: Optional[Array] = None, mask_x: Optional[Array] = None
    ) -> Array:
        if mask_xs is None:
            mask_xs = self._no_mask(xs)
        if mask_x is None:
            mask_x = self._no_mask(x)

        # TODO: add shape checks

        x = jnp.expand_dims(x, axis=x.ndim - self.ndim)
        mask_x = mask_x[..., None]

        return self.gram(xs, x, mask=mask_xs, mask_2=mask_x).squeeze(-1)

    def fn(self, points: Array, coefficients: Array, mask: Optional[Array] = None) -> Fn:
        return Fn(kernel=self, points=points, coefficients=coefficients, mask=mask)

    @partial(jax.jit, static_argnums={0})
    def kme(self, xs: Array, mask: Optional[Array] = None) -> KME:
        self.__check_shape(xs, batched=True)

        batch_shape = xs.shape[:xs.ndim - self.ndim]

        if mask is None:
            mask = jnp.full(shape=batch_shape, fill_value=False)

        if mask.shape != batch_shape:
            raise ValueError(f"Kernel mask for point of shape {xs.shape} must have shape {batch_shape}. "
                             f"Got mask with shape {mask.shape}.")

        n_points = batch_shape[-1] - mask.sum(axis=-1, keepdims=True)
        coefficients = (1 - mask) * jnp.ones(shape=batch_shape) / n_points

        return KME(kernel=self, points=xs, coefficients=coefficients, mask=mask)


class VectorKernel(NamedTuple):
    x: Kernel
    y: Kernel
    regularization: float

    @partial(jax.jit, static_argnums={0})
    def cme(self, xs: Array, ys: Array, mask: Optional[Array] = None, gram: Optional[Array] = None) -> CMO:
        if xs.ndim < self.x.ndim + 1:
            raise ValueError(f"Expected at least {xs.ndim} dimensions. Got shape {xs.shape}.")
        if ys.ndim < self.y.ndim + 1:
            raise ValueError(f"Expected at least {ys.shape} dimensions. Got shape {ys.shape}.")
        if ys.shape[:ys.ndim - self.y.ndim] != xs.shape[:xs.ndim - self.x.ndim]:
            raise ValueError(f"Inconsistent batch dimensions. "
                             f"Got {ys.shape[:ys.ndim - self.x.ndim]} and {xs.shape[:xs.ndim - self.y.ndim]}.")

        if gram is None:
            gram = self.x.gram(xs)

        if mask is None:
            mask = jnp.full(shape=xs.shape[:xs.ndim - self.x.ndim], fill_value=False)

        # TODO: add shape checks for gram and mask

        diagonal_indices = jnp.arange(xs.shape[xs.ndim - self.x.ndim - 1])

        regularized_gram = gram.at[..., diagonal_indices, diagonal_indices].add(self.regularization)
        cholesky, _ = jax.scipy.linalg.cho_factor(regularized_gram, lower=True)

        return CMO(kernel=self, xs=xs, ys=ys, gram=gram, cholesky=cholesky, mask=mask)


@partial(jax.tree_util.register_pytree_node_class)
class FnArray(Mapping, ABC):
    @property
    @abstractmethod
    def shape(self) -> Sequence[int]:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Self:
        raise NotImplementedError

    @abstractmethod
    def tree_flatten(self) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def transpose(self, *args: int) -> Self:
        raise NotImplementedError

    @abstractmethod
    def reshape(self, *args: int) -> Self:
        raise NotImplementedError

    @abstractmethod
    def broadcast_to(self, shape: Sequence[int]) -> Self:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item: int) -> Self:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Self]:
        return (self[i] for i in range(len(self)))

    def __len__(self) -> int:
        if self.ndim == 0:
            return 0
        else:
            return self.shape[0]


@partial(jax.tree_util.register_pytree_node_class)
class Fn(FnArray):
    kernel: Final[Kernel]
    points: Final[Array]
    coefficients: Final[Array]
    mask: Final[Array]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coefficients.shape[:-1]

    @property
    def shape_point(self) -> tuple[int, ...]:
        return self.points.shape[self.ndim + 1:]

    @property
    def n_points(self) -> int:
        return self.coefficients.shape[-1]

    @property
    def dataset_axis(self) -> int:
        return self.ndim

    def __init__(self, kernel: Kernel, points: Array, coefficients: Array, mask: Array):
        self.kernel = kernel
        self.points = points
        self.coefficients = coefficients
        self.mask = mask

    def tree_flatten(self):
        children = (self.points, self.coefficients, self.mask)
        aux_data = self.kernel
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Kernel, children: tuple) -> Self:
        points, coefficients, mask = children
        return Fn(kernel=aux_data, points=points, coefficients=coefficients, mask=mask)

    def reshape(self, *shape: int) -> Self:
        points = self.points.reshape(*shape, self.n_points, *self.shape_point)
        coefficients = self.coefficients.reshape(*shape, self.n_points)
        mask = self.mask.reshape(*shape, self.n_points)
        return Fn(kernel=self.kernel, points=points, coefficients=coefficients, mask=mask)

    def transpose(self, *axes: int) -> Self:
        if set(axes) != set(range(self.ndim)):
            raise ValueError(f"Dimensions must be a permutation of {tuple(range(self.ndim))}. Found {axes}.")

        points = self.points.transpose(*axes, *(i for i in range(self.ndim, self.points.ndim)))
        coefficients = self.coefficients.transpose(*axes, self.ndim)
        mask = self.mask.transpose(*axes, self.ndim)
        return Fn(kernel=self.kernel, coefficients=coefficients, points=points, mask=mask)

    def broadcast_to(self, shape: Sequence[int]) -> Self:
        points = jnp.broadcast_to(self.points, shape=(*shape, self.n_points, *self.shape_point))
        coefficients = jnp.broadcast_to(self.coefficients, shape=(*shape, self.n_points))
        mask = jnp.broadcast_to(self.mask, shape=(*shape, self.n_points))
        return Fn(kernel=self.kernel, points=points, coefficients=coefficients, mask=mask)

    def dataset_size(self) -> Array:
        return self.n_points - self.mask.sum(axis=-1)

    def take_data(self, indices: ArrayLike) -> Array:
        return jnp.take(self.points, indices, axis=self.dataset_axis)

    def bootstrap(self, key: Array, n: int) -> Fn:
        from rkhs.sampling import bootstrap_fn
        return bootstrap_fn(self, key, n)

    @partial(jax.jit)
    def __call__(self, x: Array) -> Array:
        if x.ndim < self.ndim:
            raise ValueError(f"Input must have {self.ndim} dimensions. Got shape {x.shape}.")

        arg_signature_xs = _make_arg_signature(self.kernel.ndim, var_symbol="xs_")
        arg_signature_x = _make_arg_signature(self.kernel.ndim, "x_")

        @partial(jnp.vectorize, signature=f"(n,{arg_signature_xs}),(n),(n),({arg_signature_x})->()")
        def vectorized(xs: Array, coefficients: Array, mask: Array, x_: Array) -> Array:
            kernel_vector = self.kernel(xs, x_) * (1 - mask)
            return jnp.dot(coefficients, kernel_vector)

        return vectorized(self.points, self.coefficients, x)

    @partial(jax.jit)
    def __add__(self, other: Self) -> Self:
        points = jnp.concatenate([self.points, other.points], axis=self.ndim)
        coefficients = jnp.concatenate([self.coefficients, other.coefficients], axis=self.ndim)
        mask = jnp.concatenate([self.mask, other.mask], axis=self.ndim)

        return Fn(kernel=self.kernel, points=points, coefficients=coefficients, mask=mask)

    def __getitem__(self, item) -> Self:
        indexing = jnp.indices(self.shape)
        indexing = tuple(index[item] for index in indexing)

        points = self.points[*indexing]
        coefficients = self.coefficients[*indexing]
        mask = self.mask[*indexing]

        return Fn(kernel=self.kernel, points=points, coefficients=coefficients, mask=mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, kernel={self.kernel.__class__.__name__})"


@partial(jax.tree_util.register_pytree_node_class)
class KME(Fn):
    def add_data(self, x: Array, mask: Optional[Array] = None) -> Self:
        from rkhs.online import kme_add_data
        return kme_add_data(self, x, mask)

    def removed_data(self, index: int) -> Self:
        from rkhs.online import kme_remove_data
        return kme_remove_data(self, index)


@partial(jax.tree_util.register_pytree_node_class)
class CMO(FnArray):
    kernel: Final[VectorKernel]
    xs: Final[Array]
    ys: Final[Array]
    gram: Final[Array]
    cholesky: Final[Array]
    mask: Final[Array]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.xs.shape[:-self.kernel.x.ndim - 1]

    @property
    def shape_x(self) -> tuple[int, ...]:
        return self.xs.shape[self.ndim + 1:]

    @property
    def shape_y(self) -> tuple[int, ...]:
        return self.ys.shape[self.ndim + 1:]

    @property
    def n_points(self) -> int:
        return self.xs.shape[self.ndim]

    @property
    def dataset_axis(self) -> int:
        return self.ndim

    def __init__(self, kernel: VectorKernel, xs: Array, ys: Array, gram: Array, cholesky: Array, mask: Array):
        if xs.ndim < kernel.x.ndim + 1:
            raise ValueError(f"Expected at least {kernel.x.ndim + 1} dimensions. Got shape {xs.shape}.")
        if ys.ndim < kernel.y.ndim + 1:
            raise ValueError(f"Expected at least {kernel.y.ndim + 1} dimensions. Got shape {ys.shape}.")
        if xs.ndim - kernel.x.ndim != ys.ndim - kernel.y.ndim:
            raise ValueError(f"Inconsistent batch dimensions. "
                             f"Got xs with shape {xs.shape} and ys with shape {ys.shape}.")
        if gram.ndim != xs.ndim - kernel.x.ndim + 1:
            raise ValueError(f"Gram matrix must have {xs.ndim - kernel.x.ndim + 1} dimensions. Got shape {gram.shape}.")
        if gram.shape[-1] != gram.shape[-2]:
            raise ValueError(f"Gram matrix must be square. Got shape {gram.shape}.")
        if cholesky.shape != gram.shape:
            raise ValueError(f"Cholesky matrix must have same shape as gram matrix. "
                             f"Got {cholesky.shape} vs {gram.shape}.")
        if mask.shape != gram.shape[:-1]:
            raise ValueError(f"Mask shape {mask.shape} must match gram matrix shape {gram.shape[:-1]}.")

        self.kernel = kernel
        self.xs = xs
        self.ys = ys
        self.gram = gram
        self.cholesky = cholesky
        self.mask = mask

    def tree_flatten(self):
        children = (self.xs, self.ys, self.gram, self.cholesky, self.mask)
        aux_data = self.kernel
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: VectorKernel, children: tuple) -> Self:
        xs, ys, gram, cholesky, mask = children
        return CMO(kernel=aux_data, xs=xs, ys=ys, gram=gram, cholesky=cholesky, mask=mask)

    def reshape(self, *shape: int) -> Self:
        xs = self.xs.reshape(*shape, self.n_points, *self.shape_x)
        ys = self.ys.reshape(*shape, self.n_points, *self.shape_y)
        gram = self.gram.reshape(*shape, self.n_points, self.n_points)
        cholesky = self.cholesky.reshape(*shape, self.n_points, self.n_points)
        mask = self.mask.reshape(*shape, self.n_points)

        return CMO(kernel=self.kernel, xs=xs, ys=ys, gram=gram, cholesky=cholesky, mask=mask)

    def transpose(self, *axes: int) -> Self:
        if set(axes) != set(range(self.ndim)):
            raise TypeError(f"Transpose permutation isn't a permutation of operand dimensions. "
                            f"Got permutation {axes} for operand shape {self.shape}.")

        xs = self.xs.transpose(*axes, *(i for i in range(self.ndim, self.xs.ndim)))
        ys = self.ys.transpose(*axes, *(i for i in range(self.ndim, self.ys.ndim)))
        gram = self.gram.transpose(*axes, self.ndim, self.ndim + 1)
        cholesky = self.cholesky.transpose(*axes, self.ndim, self.ndim + 1)
        mask = self.mask.transpose(*axes, self.ndim)

        return CMO(kernel=self.kernel, xs=xs, ys=ys, gram=gram, cholesky=cholesky, mask=mask)

    def broadcast_to(self, shape: Sequence[int]) -> Self:
        dataset_shape = (*shape, self.n_points)

        xs = jnp.broadcast_to(self.xs, (*dataset_shape, *self.shape_x))
        ys = jnp.broadcast_to(self.ys, (*dataset_shape, *self.shape_y))
        gram = jnp.broadcast_to(self.gram, (*dataset_shape, self.n_points))
        cholesky = jnp.broadcast_to(self.cholesky, (*dataset_shape, self.n_points))
        mask = jnp.broadcast_to(self.mask, dataset_shape)

        return CMO(kernel=self.kernel, xs=xs, ys=ys, gram=gram, cholesky=cholesky, mask=mask)

    def dataset_size(self) -> Array:
        return self.mask.sum(axis=-1)

    def influence(self, x: Array) -> Array:
        arg_signature_xs = _make_arg_signature(self.kernel.x.ndim, "xs_")
        arg_signature_x = _make_arg_signature(self.kernel.x.ndim, "x_")

        @partial(jnp.vectorize, signature=f"(n,{arg_signature_xs}),(n,n),(n),({arg_signature_x})->(n)")
        def vectorized_coefficients(xs: Array, cholesky: Array, mask_: Array, x_: Array):
            kernel_vector = self.kernel.x(xs, x_) * (1 - mask_)
            return jax.scipy.linalg.cho_solve((cholesky, True), kernel_vector) * (1 - mask_)

        return vectorized_coefficients(self.xs, self.cholesky, self.mask, x)

    def bootstrap(self, key: Array, n: int) -> CMO:
        from rkhs.sampling import bootstrap_cme
        return bootstrap_cme(self, key, n)

    @partial(jax.jit)
    def __call__(self, x: Array) -> Fn:
        if x.ndim < self.kernel.x.ndim:
            raise ValueError(f"Cannot handle input of shape {x.shape}. Kernel has dimension {self.kernel.x.ndim}.")

        coefficients = self.influence(x)
        ys = jnp.broadcast_to(self.ys, coefficients.shape[:-1] + (self.n_points,) + self.shape_y)

        return Fn(kernel=self.kernel.y, coefficients=coefficients, points=ys, mask=self.mask)

    def __getitem__(self, item) -> Self:
        indexing = jnp.indices(self.shape)
        indexing = tuple(index[item] for index in indexing)

        xs = self.xs[*indexing]
        ys = self.ys[*indexing]
        gram = self.gram[*indexing]
        cholesky = self.cholesky[*indexing]
        mask = self.mask[*indexing]

        return CMO(kernel=self.kernel, xs=xs, ys=ys, gram=gram, cholesky=cholesky, mask=mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, kernel={self.kernel.__class__.__name__})"


@partial(jax.jit)
def dot(fn_1: Fn, fn_2: Fn, kernel_matrix: Optional[Array] = None) -> Array:
    if kernel_matrix is None:
        kernel_matrix = fn_1.kernel.gram(fn_1.points, fn_2.points)

    return jnp.einsum("...i,...ij,...j->...", fn_1.coefficients, kernel_matrix, fn_2.coefficients)


@partial(jax.jit)
def squared_distance(
        fn_1: Fn, fn_2: Fn,
        kernel_matrix_11: Optional[Array] = None,
        kernel_matrix_22: Optional[Array] = None,
        kernel_matrix_12: Optional[Array] = None
) -> Array:
    dp_11 = dot(fn_1, fn_1, kernel_matrix=kernel_matrix_11)
    dp_22 = dot(fn_2, fn_2, kernel_matrix=kernel_matrix_22)
    dp_12 = dot(fn_1, fn_2, kernel_matrix=kernel_matrix_12)

    return dp_11 + dp_22 - 2 * dp_12


def distance(
        fn_1: Fn, fn_2: Fn,
        kernel_matrix_11: Optional[Array] = None,
        kernel_matrix_22: Optional[Array] = None,
        kernel_matrix_12: Optional[Array] = None
) -> Array:
    squared_distance_ = squared_distance(
        fn_1, fn_2,
        kernel_matrix_11=kernel_matrix_11, kernel_matrix_22=kernel_matrix_22, kernel_matrix_12=kernel_matrix_12
    )

    return jnp.sqrt(jnp.clip(squared_distance_, min=0))  # clip to avoid numerical errors


def squared_norm(fn: Fn, kernel_matrix: Optional[Array] = None) -> Array:
    return dot(fn, fn, kernel_matrix=kernel_matrix)


def norm(fn: Fn, kernel_matrix: Optional[Array] = None) -> Array:
    squared_norm_ = squared_norm(fn, kernel_matrix=kernel_matrix)
    return jnp.sqrt(jnp.clip(squared_norm_, min=0))  # clip to avoid numerical errors


@partial(jax.jit, static_argnums={0})
def kme_dot(
        kernel: Kernel,
        xs_1: Array, xs_2: Array,
        mask_1: Optional[Array] = None, mask_2: Optional[Array] = None
) -> Array:
    kme_1 = kernel.kme(xs_1, mask=mask_1)
    kme_2 = kernel.kme(xs_2, mask=mask_2)
    return dot(kme_1, kme_2)


@partial(jax.jit, static_argnums={0})
def squared_mmd(
        kernel: Kernel,
        xs_1: Array, xs_2: Array,
        mask_1: Optional[Array] = None, mask_2: Optional[Array] = None
) -> Array:
    kme_1 = kernel.kme(xs_1, mask=mask_1)
    kme_2 = kernel.kme(xs_2, mask=mask_2)
    return squared_distance(kme_1, kme_2)


@partial(jax.jit, static_argnums={0})
def mmd(
        kernel: Kernel,
        xs_1: Array, xs_2: Array,
        mask_1: Optional[Array] = None, mask_2: Optional[Array] = None
) -> Array:
    kme_1 = kernel.kme(xs_1, mask=mask_1)
    kme_2 = kernel.kme(xs_2, mask=mask_2)
    return distance(kme_1, kme_2)


@partial(jax.jit, static_argnums={0})
def cme_dot(
        kernel: VectorKernel,
        xs_1: Array, xs_2: Array,
        ys_1: Array, ys_2: Array,
        e_1: Array, e_2: Array,
        mask_1: Optional[Array] = None, mask_2: Optional[Array] = None
) -> Array:
    kme_1 = kernel.cme(xs_1, ys_1, mask=mask_1)(e_1)
    kme_2 = kernel.cme(xs_2, ys_2, mask=mask_2)(e_2)

    return dot(kme_1, kme_2)


@partial(jax.jit, static_argnums={0})
def squared_cmmd(
        kernel: VectorKernel,
        xs_1: Array, xs_2: Array,
        ys_1: Array, ys_2: Array,
        e_1: Array, e_2: Array,
        mask_1: Optional[Array] = None, mask_2: Optional[Array] = None
) -> Array:
    kme_1 = kernel.cme(xs_1, ys_1, mask=mask_1)(e_1)
    kme_2 = kernel.cme(xs_2, ys_2, mask=mask_2)(e_2)

    return squared_distance(kme_1, kme_2)


@partial(jax.jit, static_argnums={0})
def cmmd(
        kernel: VectorKernel,
        xs_1: Array, xs_2: Array,
        ys_1: Array, ys_2: Array,
        e_1: Array, e_2: Array,
        mask_1: Optional[Array] = None, mask_2: Optional[Array] = None
) -> Array:
    kme_1 = kernel.cme(xs_1, ys_1, mask=mask_1)(e_1)
    kme_2 = kernel.cme(xs_2, ys_2, mask=mask_2)(e_2)

    return distance(kme_1, kme_2)
