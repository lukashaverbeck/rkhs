from dataclasses import field, dataclass
from functools import partial
from typing import Self

import jax.numpy as jnp
import jax.tree_util
from jax import Array
from jax.typing import ArrayLike

import rkhs
from rkhs import Fn
from rkhs.testing._base import ConfidenceRadius


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class AnalyticalConfidenceRadius(ConfidenceRadius):
    dataset_size: Array
    kernel_bound: float = field(metadata=dict(static=True))

    @classmethod
    def from_kme(cls, kme: Fn, kernel_bound: float) -> Self:
        dataset_size = kme.dataset_size()
        return cls(dataset_size=dataset_size, kernel_bound=kernel_bound)

    def _confidence(self, level: ArrayLike) -> Array:
        return jnp.sqrt(8 * self.kernel_bound * jnp.log(2 / level) / self.dataset_size)


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class BootstrapConfidenceRadius(ConfidenceRadius):
    threshold_null: Array

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

        return cls(threshold_null=bootstrap_mmd)

    def _confidence(self, level: ArrayLike) -> Array:
        return jnp.quantile(self.threshold_null, q=1 - level, axis=-1)
