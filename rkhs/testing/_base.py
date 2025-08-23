from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import rkhs
from rkhs import Fn, CME
from rkhs._util import expand_shape


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class ConfidenceRadius(ABC):
    @abstractmethod
    def _confidence(self, level: ArrayLike) -> Array:
        raise NotImplementedError

    def __call__(self, level: ArrayLike) -> Array:
        level = jnp.asarray(level)

        @partial(jax.vmap)
        def batch_confidence(level_: ArrayLike) -> Array:
            return self._confidence(level_)

        confidence = batch_confidence(level.reshape(-1))
        return confidence.reshape(*level.shape, *confidence.shape[1:])


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class ConfidenceTube(ABC):
    @abstractmethod
    def _confidence(self, cme: CME, kme: Fn, x: Array) -> ConfidenceRadius:
        raise NotImplementedError

    @staticmethod
    def _std(cme: CME, kme: Fn, x: Array) -> Array:
        k_x = cme.kernel.x.vector(cme.xs, x)
        return jnp.sqrt(cme.kernel.x(x, x) - (k_x * kme.coefficients).sum(axis=-1))

    def __call__(self, cme: CME, kme: Fn, x: Array) -> ConfidenceRadius:
        return self._confidence(cme, kme, x)


class TestEmbedding(NamedTuple):
    kme: Fn
    confidence: ConfidenceRadius

    @classmethod
    def analytical(cls, kme: Fn, kernel_bound: float) -> Self:
        from rkhs.testing._marginal import AnalyticalConfidenceRadius

        return cls(
            kme=kme,
            confidence=AnalyticalConfidenceRadius.from_kme(kme, kernel_bound)
        )

    @classmethod
    def bootstrap(cls, kme: Fn, key: Array, n_bootstrap: int) -> Self:
        from rkhs.testing._marginal import BootstrapConfidenceRadius

        return cls(
            kme=kme,
            confidence=BootstrapConfidenceRadius.from_kme(kme, key, n_bootstrap)
        )

    def __call__(self, level: ArrayLike) -> Array:
        return self.confidence(level)


class ConditionalTestEmbedding(NamedTuple):
    cme: CME
    confidence: ConfidenceTube

    @classmethod
    def analytical(cls, cme: CME, rkhs_norm_bound: float, sub_gaussian_std: float) -> Self:
        from rkhs.testing._conditional import AnalyticalConfidenceTube

        return ConditionalTestEmbedding(
            cme=cme,
            confidence=AnalyticalConfidenceTube.from_cme(cme, rkhs_norm_bound, sub_gaussian_std),
        )

    @classmethod
    def bootstrap(cls, cme: CME, grid: Array, key: Array, n_bootstrap: int) -> Self:
        from rkhs.testing._conditional import BootstrapConfidenceTube

        return ConditionalTestEmbedding(
            cme=cme,
            confidence=BootstrapConfidenceTube.from_cme(cme, grid, key, n_bootstrap)
        )

    def __call__(self, x: Array) -> TestEmbedding:
        kme = self.cme(x)
        confidence_bound = self.confidence(self.cme, kme, x)
        return TestEmbedding(kme=kme, confidence=confidence_bound)


@partial(jax.tree_util.register_dataclass)
@dataclass(frozen=True)
class TwoSampleTest:
    level: Array
    level_1: Array
    level_2: Array
    threshold_1: Array
    threshold_2: Array
    distance: Array

    @classmethod
    @partial(jax.jit, static_argnums={0})
    def from_embeddings(cls, embedding_1: TestEmbedding, embedding_2: TestEmbedding, level: ArrayLike) -> Self:
        level = jnp.asarray(level)

        ts = jnp.linspace(start=0, stop=1, num=102)[1:-1]

        thresholds_1 = jax.vmap(embedding_1, in_axes=-1, out_axes=0)(ts * level[..., None])
        thresholds_2 = jax.vmap(embedding_2, in_axes=-1, out_axes=0)((1 - ts) * level[..., None])

        thresholds_1 = expand_shape(thresholds_1, dims=max(thresholds_1.ndim, thresholds_2.ndim), prefix=False)
        thresholds_2 = expand_shape(thresholds_2, dims=max(thresholds_1.ndim, thresholds_2.ndim), prefix=False)
        thresholds_1, thresholds_2 = jnp.broadcast_arrays(thresholds_1, thresholds_2)

        best_t_index = jnp.argmin(thresholds_1 + thresholds_2, axis=0)
        best_t = ts[best_t_index]

        level_1 = best_t * level[..., None]
        level_2 = (1 - best_t) * level[..., None]

        threshold_1 = jnp.take_along_axis(thresholds_1, indices=best_t_index[None], axis=0).squeeze(0)
        threshold_2 = jnp.take_along_axis(thresholds_2, indices=best_t_index[None], axis=0).squeeze(0)

        distance = rkhs.distance(embedding_1.kme, embedding_2.kme)

        return TwoSampleTest(
            level=level, level_1=level_1, level_2=level_2,
            threshold_1=threshold_1, threshold_2=threshold_2,
            distance=distance,
        )

    @property
    def threshold(self) -> Array:
        return self.threshold_1 + self.threshold_2

    @property
    def reject(self) -> Array:
        return self.distance > self.threshold
