from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Self, ClassVar, Optional, Literal, Sequence, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import rkhs
from rkhs import Fn, CME, Kernel, VectorKernel
from rkhs._util import is_broadcastable, expand_shape


def _resample_masked_population(population: Array, mask: Array, shape: Sequence[int], key: Array) -> Array:
    if population.ndim != 1:
        raise TypeError(f"Only supports 1D arrays. Got shape {population.shape}.")

    if mask.shape[-1] != population.shape[0]:
        raise TypeError(f"Mask shape {mask.shape} does not match population shape in last axis. "
                        f"Got mask of shape {mask.shape} for population shape {population.shape}.")

    if len(shape) == 0:
        raise ValueError("Mask shape must not be empty.")

    population_size = population.shape[0]
    batch_shape = shape[:-1]
    sample_size = shape[-1]

    def sample(key_: Array, mask_: Array) -> Array:
        mask_degenerate = jnp.all(mask_)
        return jax.random.choice(key_, population, shape=(sample_size,), replace=True, p=1 - mask_ + mask_degenerate)

    mask = jnp.broadcast_to(mask, shape=(*batch_shape, population_size)).reshape(-1, population_size)
    keys = jax.random.split(key, num=len(mask))
    sample_batch = jax.vmap(sample)(keys, mask)

    return sample_batch.reshape(shape)


def _extract_submatrix(matrix: Array, indices_1: Array, indices_2: Array) -> Array:
    return jnp.take_along_axis(
        jnp.take_along_axis(matrix, indices_1[..., :, None], axis=-2),
        indices_2[..., None, :],
        axis=-1,
    )


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class MarginalTestEmbedding(ABC):
    analytical: ClassVar[type[AnalyticalMarginalTestEmbedding]]
    bootstrap: ClassVar[type[BootstrapMarginalTestEmbedding]]

    kme: Fn

    @abstractmethod
    def __call__(self, level: ArrayLike) -> Array:
        raise NotImplementedError


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class AnalyticalMarginalTestEmbedding(MarginalTestEmbedding):
    kernel_bound: float = field(metadata=dict(static=True))

    @classmethod
    def from_data(cls, kernel: Kernel, xs: Array, kernel_bound: float, mask: Optional[Array] = None) -> Self:
        kme = kernel.kme(xs, mask=mask)
        return cls.from_kme(kme, kernel_bound)

    @classmethod
    def from_kme(cls, kme: Fn, kernel_bound: float) -> Self:
        return cls(kme, kernel_bound)

    def __call__(self, level: ArrayLike) -> Array:
        dataset_size = self.kme.dataset_size()
        return jnp.sqrt(8 * self.kernel_bound * jnp.log(2 / level) / dataset_size)


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class BootstrapMarginalTestEmbedding(MarginalTestEmbedding):
    threshold_null: Array

    @classmethod
    def from_data(
            cls, kernel: Kernel, xs: Array, key: Array, n_bootstrap: int,
            mask: Optional[Array] = None
    ) -> Self:
        kme = kernel.kme(xs, mask=mask)
        return cls.from_kme(kme, key, n_bootstrap)

    @classmethod
    @partial(jax.jit, static_argnums={0, 3})
    def from_kme(cls, kme: Fn, key: Array, n_bootstrap: int) -> Self:
        dataset_size = kme.dataset_size()

        bootstrap_multiplicities = jax.random.multinomial(
            key, kme.dataset_shape_size,
            p=(1 - kme.mask) / dataset_size,
            shape=(n_bootstrap, *kme.shape, kme.dataset_shape_size),
        )

        bootstrap_kmes = kme.kernel.fn(
            points=kme.points, coefficients=bootstrap_multiplicities / dataset_size, mask=kme.mask
        )

        bootstrap_mmd = rkhs.distance(bootstrap_kmes, kme)
        bootstrap_mmd = bootstrap_mmd.transpose(*range(1, bootstrap_mmd.ndim), 0)

        return BootstrapMarginalTestEmbedding(kme, bootstrap_mmd)

    def __post_init__(self):
        if self.threshold_null.shape[:-1] != self.kme.shape:
            raise TypeError(f"Inconsistent shapes for threshold null distribution and KME. "
                            f"Got {self.threshold_null.shape} and {self.kme.shape}.")

    def __call__(self, level: ArrayLike) -> Array:
        return jnp.quantile(self.threshold_null, q=1 - level, axis=-1)


MarginalTestEmbedding.analytical = AnalyticalMarginalTestEmbedding
MarginalTestEmbedding.bootstrap = BootstrapMarginalTestEmbedding


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class ConditionedTestEmbedding(MarginalTestEmbedding, ABC):
    std: Array

    @abstractmethod
    def beta(self, level: ArrayLike) -> Array:
        raise NotImplementedError

    def __batch_beta(self, level: Array) -> Array:
        @partial(jax.vmap)
        def batch(level_: Array) -> Array:
            return self.beta(level_)

        return batch(level)

    def __call__(self, level: ArrayLike) -> ArrayLike:
        level = jnp.asarray(level)
        beta = self.__batch_beta(level.reshape(-1))
        beta = beta.reshape(*level.shape, *beta.shape[1:])

        return beta * self.std


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class AnalyticalConditionedTestEmbedding(ConditionedTestEmbedding):
    log_determinant: Array
    rkhs_norm_bound: float = field(metadata=dict(static=True))
    sub_gaussian_std: float = field(metadata=dict(static=True))
    regularization: float = field(metadata=dict(static=True))

    def beta(self, level: ArrayLike) -> Array:
        return self.rkhs_norm_bound + self.sub_gaussian_std * jnp.sqrt(
            (self.log_determinant - 2 * jnp.log(level)) / self.regularization
        )


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class BootstrapConditionedTestEmbedding(ConditionedTestEmbedding):
    beta_null: Array

    def beta(self, level: ArrayLike) -> Array:
        return jnp.quantile(self.beta_null, q=1 - level, axis=-1)


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class ConditionalTestEmbedding(ABC):
    analytical: ClassVar[type[AnalyticalConditionalTestEmbedding]]
    bootstrap: ClassVar[type[BootstrapConditionalTestEmbedding]]

    cme: CME

    @classmethod
    def _std(cls, cme: CME, x: Array, influence: Optional[Array] = None) -> Array:
        if influence is None:
            influence = cme.influence(x)

        k_x = cme.kernel.x.vector(cme.xs, x)
        return jnp.sqrt(cme.kernel.x(x, x) - (k_x * influence).sum(axis=-1))

    def std(self, x: Array, influence: Optional[Array] = None) -> Array:
        return self._std(self.cme, x, influence)

    @abstractmethod
    def __call__(self, x: Array) -> MarginalTestEmbedding:
        raise NotImplementedError


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class AnalyticalConditionalTestEmbedding(ConditionalTestEmbedding):
    log_determinant: Array
    rkhs_norm_bound: float = field(metadata=dict(static=True))
    sub_gaussian_std: float = field(metadata=dict(static=True))

    @classmethod
    def from_data(
            cls, kernel: VectorKernel, xs: Array, ys: Array, rkhs_norm_bound: float, sub_gaussian_std: float,
            mask: Optional[Array] = None
    ) -> Self:
        cme = kernel.cme(xs, ys, mask=mask)
        return cls.from_cme(cme, rkhs_norm_bound, sub_gaussian_std)

    @classmethod
    def from_cme(cls, cme: CME, rkhs_norm_bound: float, sub_gaussian_std: float) -> Self:
        gram = cme.kernel.x.gram(cme.xs)
        _, log_determinant = jnp.linalg.slogdet(jnp.eye(cme.dataset_shape_size) + gram / cme.kernel.regularization)

        return AnalyticalConditionalTestEmbedding(cme, log_determinant, rkhs_norm_bound, sub_gaussian_std)

    def __post_init__(self):
        if not is_broadcastable(self.cme.shape, self.log_determinant.shape):
            raise TypeError(f"Inconsistent shape for cme and log determinant. Got shapes {self.cme.shape} and "
                            f"{self.log_determinant.shape}.")

    def __call__(self, x: Array) -> AnalyticalConditionedTestEmbedding:
        kme = self.cme(x)

        return AnalyticalConditionedTestEmbedding(
            kme=kme,
            std=self.std(x, kme.coefficients),
            log_determinant=self.log_determinant,
            rkhs_norm_bound=self.rkhs_norm_bound,
            sub_gaussian_std=self.sub_gaussian_std,
            regularization=self.cme.kernel.regularization
        )


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class BootstrapConditionalTestEmbedding(ConditionalTestEmbedding):
    type Mode = Literal["resample", "resample-double", "wild"]
    DEFAULT_MODE: ClassVar[Mode] = "resample-double"

    beta_null: Array

    @classmethod
    def from_data(
            cls, kernel: VectorKernel, xs: Array, ys: Array, grid: Array, key: Array, n_bootstrap: int,
            mask: Optional[Array] = None,
            mode: Mode = DEFAULT_MODE
    ) -> Self:
        cme = kernel.cme(xs, ys, mask=mask)
        return cls.from_cme(cme, grid, key, n_bootstrap, mode=mode)

    @classmethod
    def from_cme(cls, cme: CME, grid: Array, key: Array, n_bootstrap: int, mode: Mode = DEFAULT_MODE) -> Self:
        if mode == "resample":
            return cls.resample(cme, grid, key, n_bootstrap)
        elif mode == "resample-double":
            return cls.resample_double(cme, grid, key, n_bootstrap)
        elif mode == "wild":
            return cls.wild(cme, grid, key, n_bootstrap)
        else:
            raise NotImplementedError(f"Unsupported mode {mode}.")

    @classmethod
    def __resample_beta(cls, cme: CME, grid: Array, indices_1: Array, indices_2: Array, gram_y: Array) -> Array:
        xs_1, ys_1 = cme.take_data(indices_1)
        xs_2, ys_2 = cme.take_data(indices_2)

        cme_1 = cme.kernel.cme(xs_1, ys_1, mask=cme.mask).expand_dims(-1)
        cme_2 = cme.kernel.cme(xs_2, ys_2, mask=cme.mask).expand_dims(-1)

        kmes_1 = cme_1(grid)
        kmes_2 = cme_2(grid)

        std_1 = cls._std(cme_1, grid, kmes_1.coefficients)
        std_2 = cls._std(cme_2, grid, kmes_2.coefficients)

        gram_y_11 = _extract_submatrix(gram_y, indices_1, indices_1)
        gram_y_22 = _extract_submatrix(gram_y, indices_2, indices_2)
        gram_y_12 = _extract_submatrix(gram_y, indices_1, indices_2)

        gram_y_11 = jnp.expand_dims(gram_y_11, axis=cme.dataset_axis)
        gram_y_22 = jnp.expand_dims(gram_y_22, axis=cme.dataset_axis)
        gram_y_12 = jnp.expand_dims(gram_y_12, axis=cme.dataset_axis)

        cmmd = rkhs.distance(kmes_1, kmes_2, gram_11=gram_y_11, gram_22=gram_y_22, gram_12=gram_y_12)

        return cmmd / (std_1 + std_2)

    @classmethod
    @partial(jax.jit, static_argnums={0, 4})
    def resample(cls, cme: CME, grid: Array, key: Array, n_bootstrap: int) -> Self:
        indices = jnp.arange(cme.dataset_shape_size)
        gram = cme.kernel.y.gram(cme.ys)

        def resample(bootstrap_indices_: Array) -> Array:
            identity = expand_shape(indices, dims=cme.ndim + 1)
            return cls.__resample_beta(cme, grid, bootstrap_indices_, identity, gram).max(axis=-1)

        bootstrap_indices = _resample_masked_population(
            indices, cme.mask,
            shape=(n_bootstrap, *cme.shape, cme.dataset_shape_size,),
            key=key
        )

        bootstrap_beta = jax.lax.map(resample, bootstrap_indices)
        bootstrap_beta = bootstrap_beta.transpose(*range(1, bootstrap_beta.ndim), 0)

        return BootstrapConditionalTestEmbedding(cme, bootstrap_beta)

    @classmethod
    @partial(jax.jit, static_argnums={0, 4})
    def resample_double(cls, cme: CME, grid: Array, key: Array, n_bootstrap: int) -> Self:
        indices = jnp.arange(cme.dataset_shape_size)
        gram = cme.kernel.y.gram(cme.ys)

        def resample(bootstrap_indices_1_: Array, bootstrap_indices_2_: Array) -> Array:
            return cls.__resample_beta(cme, grid, bootstrap_indices_1_, bootstrap_indices_2_, gram).max(axis=-1)

        def resample_indices(key_: Array) -> Array:
            return _resample_masked_population(
                indices, cme.mask,
                shape=(n_bootstrap, *cme.shape, cme.dataset_shape_size,),
                key=key_
            )

        key_1, key_2 = jax.random.split(key)
        bootstrap_indices_1 = resample_indices(key_1)
        bootstrap_indices_2 = resample_indices(key_2)

        bootstrap_beta = jax.lax.map(
            f=lambda bootstrap_indices_pair: resample(*bootstrap_indices_pair),
            xs=(bootstrap_indices_1, bootstrap_indices_2)
        )

        bootstrap_beta = bootstrap_beta.transpose(*range(1, bootstrap_beta.ndim), 0)

        return BootstrapConditionalTestEmbedding(cme, bootstrap_beta)

    @classmethod
    @partial(jax.jit, static_argnums={0, 4})
    def wild(cls, cme: CME, grid: Array, key: Array, n_bootstrap: int) -> Self:
        gram_y = cme.kernel.y.gram(cme.ys)

        influence_grid = cme.expand_dims(-1).influence(grid)
        influence_xs = cme.expand_dims(-1).influence(cme.xs)

        residual_coefficients = jnp.eye(cme.dataset_shape_size) - influence_xs
        gram_residual = residual_coefficients @ gram_y @ residual_coefficients

        def compute_squared_norm(noise_: Array) -> Array:
            weights = influence_grid * noise_ * ~cme.mask
            return jnp.einsum("...i,...ij,...j->...", weights, gram_residual[..., None, :, :], weights)

        noise = jax.random.normal(key, shape=(n_bootstrap, cme.dataset_shape_size))
        wild_norm_squared = jax.lax.map(compute_squared_norm, noise)
        wild_norm = jnp.sqrt(jnp.clip(wild_norm_squared, min=0))

        std = cls._std(cme.expand_dims(-1), grid, influence_grid)
        bootstrap_beta = (wild_norm / std).max(axis=-1)
        bootstrap_beta = bootstrap_beta.transpose(*range(1, bootstrap_beta.ndim), 0)

        return BootstrapConditionalTestEmbedding(cme, bootstrap_beta)

    def __post_init__(self):
        if self.beta_null.shape[:-1] != self.cme.shape:
            raise TypeError(f"Shape of beta_null does not match shape of cme. Got shapes {self.cme.shape} and "
                            f"{self.cme.shape}.")

    def __call__(self, x: Array) -> BootstrapConditionedTestEmbedding:
        kme = self.cme(x)
        std = self.std(x, kme.coefficients)
        return BootstrapConditionedTestEmbedding(kme, std, self.beta_null)


ConditionalTestEmbedding.analytical = AnalyticalConditionalTestEmbedding
ConditionalTestEmbedding.bootstrap = BootstrapConditionalTestEmbedding


class TwoSampleTest(NamedTuple):
    distance: Array
    threshold: Array

    @classmethod
    def from_embeddings(
            cls, embedding_1: MarginalTestEmbedding, embedding_2: MarginalTestEmbedding, level: ArrayLike
    ) -> Self:
        distance = rkhs.distance(embedding_1.kme, embedding_2.kme)

        threshold_1 = embedding_1(level / 2)
        threshold_2 = embedding_2(level / 2)
        threshold = threshold_1 + threshold_2

        return TwoSampleTest(distance, threshold)

    def rejection(self) -> Array:
        return self.distance > self.threshold


MarginalTestEmbedding.bootstrap.from_data()