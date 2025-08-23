from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from rkhs import CME, Fn
from rkhs.testing._base import ConfidenceRadius, ConfidenceTube


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class ConditionedConfidenceRadius(ConfidenceRadius, ABC):
    std: Array

    @abstractmethod
    def _beta(self, level: ArrayLike) -> Array:
        raise NotImplementedError

    def _confidence(self, level: ArrayLike) -> Array:
        return self._beta(level) * self.std


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class AnalyticalConfidenceRadius(ConditionedConfidenceRadius):
    log_determinant: Array
    rkhs_norm_bound: float = field(metadata=dict(static=True))
    sub_gaussian_std: float = field(metadata=dict(static=True))
    regularization: float = field(metadata=dict(static=True))

    def _beta(self, level: ArrayLike) -> Array:
        return self.rkhs_norm_bound + self.sub_gaussian_std * jnp.sqrt(
            (self.log_determinant - 2 * jnp.log(level)) / self.regularization
        )


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class BootstrapConfidenceRadius(ConditionedConfidenceRadius):
    beta_null: Array

    def _beta(self, level: ArrayLike) -> Array:
        return jnp.quantile(self.beta_null, q=1 - level, axis=-1)


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class AnalyticalConfidenceTube(ConfidenceTube):
    log_determinant: Array
    rkhs_norm_bound: float = field(metadata=dict(static=True))
    sub_gaussian_std: float = field(metadata=dict(static=True))

    @classmethod
    @partial(jax.jit, static_argnums={0, 2, 3})
    def from_cme(cls, cme: CME, rkhs_norm_bound: float, sub_gaussian_std: float) -> Self:
        gram = cme.kernel.x.gram(cme.xs)
        _, log_determinant = jnp.linalg.slogdet(jnp.eye(cme.dataset_shape_size) + gram / cme.kernel.regularization)

        return cls(log_determinant=log_determinant, rkhs_norm_bound=rkhs_norm_bound, sub_gaussian_std=sub_gaussian_std)

    def _confidence(self, cme: CME, kme: Fn, x: Array) -> ConfidenceRadius:
        return AnalyticalConfidenceRadius(
            std=self._std(cme, kme, x),
            log_determinant=self.log_determinant,
            rkhs_norm_bound=self.rkhs_norm_bound,
            sub_gaussian_std=self.sub_gaussian_std,
            regularization=cme.kernel.regularization,
        )


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class BootstrapConfidenceTube(ConfidenceTube):
    beta_null: Array

    @classmethod
    @partial(jax.jit, static_argnums={0, 4})
    def from_cme(cls, cme: CME, grid: Array, key: Array, n_bootstrap: int) -> Self:
        gram_y = cme.kernel.y.gram(cme.ys)

        kmes_grid = cme.expand_dims(-1)(grid)
        kmes_xs = cme.expand_dims(-1)(cme.xs)

        residual_coefficients = jnp.eye(cme.dataset_shape_size) - kmes_xs.coefficients
        gram_residual = residual_coefficients @ gram_y @ residual_coefficients

        def compute_squared_norm(noise_: Array) -> Array:
            weights = kmes_grid.coefficients * noise_ * ~cme.mask
            return jnp.einsum("...i,...ij,...j->...", weights, gram_residual[..., None, :, :], weights)

        noise = jax.random.normal(key, shape=(n_bootstrap, cme.dataset_shape_size))
        wild_norm_squared = jax.lax.map(compute_squared_norm, noise)
        wild_norm = jnp.sqrt(jnp.clip(wild_norm_squared, min=0))

        std = cls._std(cme.expand_dims(-1), kmes_grid, grid)
        bootstrap_beta = (wild_norm / std).max(axis=-1)

        bootstrap_beta = bootstrap_beta.transpose(*range(1, bootstrap_beta.ndim), 0)

        return cls(beta_null=bootstrap_beta)

    def _confidence(self, cme: CME, kme: Fn, x: Array) -> BootstrapConfidenceRadius:
        return BootstrapConfidenceRadius(
            std=self._std(cme, kme, x),
            beta_null=self.beta_null
        )
