from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import partial
from typing import Final, Self, NamedTuple, Optional, Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import rkhs.base as rkhs
from rkhs._util import _make_arg_signature
from rkhs.base import VectorKernel, Kernel, KME, Fn, CMO
from rkhs.sampling import bootstrap_distance, _bootstrap_masked_indices, _resample_cme, _extract_submatrix


@partial(jax.jit)
def _posterior_std(cme: CMO, x: Array, influence_vector: Optional[Array] = None) -> Array:
    if influence_vector is None:
        influence_vector = cme.influence(x)

    arg_signature_x = _make_arg_signature(cme.kernel.x.ndim, 'x')

    @partial(jnp.vectorize, signature=f"(n,{arg_signature_x}),({arg_signature_x}),(n)->()")
    def vectorized(xs_: Array, x_: Array, influence_vector_: Array) -> Array:
        kernel_vector = cme.kernel.x(xs_, x_)
        return jnp.sqrt(cme.kernel.x(x_, x_) - jnp.dot(kernel_vector, influence_vector_))

    return vectorized(cme.xs, x, influence_vector)


class TestEmbedding[T: Fn](ABC):
    kme: Final[T]

    def __init__(self, kme: Fn):
        self.kme = kme

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Self:
        raise NotImplementedError

    @abstractmethod
    def tree_flatten(self) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def _threshold(self, level: Array) -> Array:
        raise NotImplementedError

    def threshold(self, level: ArrayLike) -> Array:
        original_shape = jnp.shape(level)
        level = jnp.asarray(level).reshape(-1)

        threshold_batch = jax.lax.map(self._threshold, level)

        target_batch_shape = (math.prod(original_shape), *self.kme.shape)
        if threshold_batch.shape != target_batch_shape:
            raise TypeError(f"Threshold batch should have shape {target_batch_shape}. Got {threshold_batch.shape}.")

        return threshold_batch.reshape(*original_shape, *self.kme.shape)


class MarginalTest(TestEmbedding[KME], ABC):
    @classmethod
    def analytical(cls, kme: KME, kernel_bound: float) -> AnalyticalMarginalTest:
        return AnalyticalMarginalTest.from_kme(kme, kernel_bound)

    @classmethod
    def bootstrap(cls, kme: KME, n_bootstrap: int, key: Array) -> BootstrapMarginalTest:
        return BootstrapMarginalTest.from_kme(kme, key, n_bootstrap)


@partial(jax.tree_util.register_pytree_node_class)
class AnalyticalMarginalTest(MarginalTest):
    kernel_bound: Final[float]
    dataset_size: Final[int]

    @classmethod
    def from_data(cls, kernel: Kernel, kernel_bound: float, xs: Array, mask: Optional[Array] = None) -> Self:
        kme = kernel.kme(xs, mask=mask)
        return AnalyticalMarginalTest.from_kme(kme, kernel_bound)

    @classmethod
    def from_kme(cls, kme: KME, kernel_bound: float) -> Self:
        return AnalyticalMarginalTest(kme=kme, kernel_bound=kernel_bound, dataset_size=kme.n_points)

    def __init__(self, kme: KME, kernel_bound: float, dataset_size: int):
        super().__init__(kme)
        self.kernel_bound = kernel_bound
        self.dataset_size = dataset_size

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Self:
        kernel_bound, dataset_size = children
        kme, = children
        return AnalyticalMarginalTest(kme=kme, kernel_bound=kernel_bound, dataset_size=dataset_size)

    def tree_flatten(self) -> tuple[Any, Any]:
        return (self.kme,), (self.kernel_bound, self.dataset_size),

    def _threshold(self, level: Array) -> Array:
        kernel_bound = jnp.broadcast_to(self.kernel_bound, self.kme.shape)
        dataset_size = jnp.broadcast_to(self.dataset_size, self.kme.shape)
        return jnp.sqrt(kernel_bound / dataset_size) + jnp.sqrt(-2 * kernel_bound * jnp.log(level) / dataset_size)


@partial(jax.tree_util.register_pytree_node_class)
class BootstrapMarginalTest(MarginalTest):
    threshold_null: Final[Array]

    @classmethod
    def from_data(cls, kernel: Kernel, xs: Array, key: Array, n_bootstrap: int, mask: Optional[Array] = None) -> Self:
        kme = kernel.kme(xs, mask=mask)
        return BootstrapMarginalTest.from_kme(kme, key, n_bootstrap)

    @classmethod
    def from_kme(cls, kme: KME, key: Array, n_bootstrap: int) -> Self:
        threshold_null = bootstrap_distance(kme, kme, key, n_bootstrap)
        return BootstrapMarginalTest(kme=kme, threshold_null=threshold_null)

    def __init__(self, kme: KME, threshold_null: Array):
        if threshold_null.shape[1:] != kme.shape:
            raise TypeError(f"Expected threshold null distribution to hold one array for every mean embedding. "
                            f"Got shapes {kme.shape} and {threshold_null.shape}.")

        super().__init__(kme)
        self.threshold_null = threshold_null

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Self:
        kme, threshold_null = children
        return BootstrapMarginalTest(kme=kme, threshold_null=threshold_null)

    def tree_flatten(self) -> tuple[Any, Any]:
        return (self.kme, self.threshold_null), None

    def _threshold(self, level: Array) -> Array:
        return jnp.quantile(self.threshold_null, 1 - level, axis=0)


class ConditionedTest(TestEmbedding[Fn], ABC):
    std: Final[Array]

    def __init__(self, kme: Fn, std: Array):
        if kme.shape != std.shape:
            raise TypeError(f"Expect standard deviation to match mean embedding. "
                            f"Got shapes {std.shape} and {kme.shape}.")

        super().__init__(kme)
        self.std = std

    @abstractmethod
    def beta(self, level: Array) -> Array:
        raise NotImplementedError

    def _threshold(self, level: Array) -> Array:
        beta = self.beta(level)
        beta, level = jnp.broadcast_arrays(beta, level)
        return beta * self.std


@partial(jax.tree_util.register_pytree_node_class)
class AnalyticalConditionedTest(ConditionedTest):
    log_determinant: Final[Array]
    rkhs_norm: Final[Array]
    sub_gaussian_std: Final[Array]
    regularization: Final[float]

    def __init__(
            self,
            kme: Fn,
            std: Array,
            log_determinant: Array, rkhs_norm: ArrayLike, sub_gaussian_std: ArrayLike, regularization: float
    ):
        super().__init__(kme, std)

        if log_determinant.shape != kme.shape:
            raise TypeError(f"Expect log-determinant to match mean embedding. "
                            f"Got shapes {log_determinant.shape} and {kme.shape}.")

        rkhs_norm = jnp.broadcast_to(rkhs_norm, self.kme.shape)
        sub_gaussian_std = jnp.broadcast_to(sub_gaussian_std, self.kme.shape)

        self.log_determinant = log_determinant
        self.rkhs_norm = rkhs_norm
        self.sub_gaussian_std = sub_gaussian_std
        self.regularization = regularization

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Self:
        regularization = aux_data
        kme, std, log_determinant, rkhs_norm, sub_gaussian_std = children
        return AnalyticalConditionedTest(
            kme=kme,
            std=std,
            log_determinant=log_determinant,
            rkhs_norm=rkhs_norm,
            sub_gaussian_std=sub_gaussian_std,
            regularization=regularization
        )

    def tree_flatten(self) -> tuple[Any, Any]:
        return (
            self.kme,
            self.std,
            self.log_determinant, self.rkhs_norm, self.sub_gaussian_std
        ), self.regularization

    def beta(self, level: Array) -> Array:
        return self.rkhs_norm + self.sub_gaussian_std * jnp.sqrt(
            (self.log_determinant - 2 * jnp.log(level)) / self.regularization
        )


@partial(jax.tree_util.register_pytree_node_class)
class BootstrapConditionedTest(ConditionedTest):
    beta_null: Final[Array]

    def __init__(self, kme: Fn, std: Array, beta_null: Array):
        super().__init__(kme, std)
        beta_null = beta_null.reshape(-1, *(1,) * len(kme.shape))
        self.beta_null = jnp.broadcast_to(beta_null, (beta_null.shape[0], *kme.shape))

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Self:
        kme, std, beta_null = children
        return BootstrapConditionedTest(kme=kme, std=std, beta_null=beta_null)

    def tree_flatten(self) -> tuple[Any, Any]:
        return (self.kme, self.std, self.beta_null), None

    def beta(self, level: Array) -> Array:
        return jnp.quantile(self.beta_null, 1 - level, axis=0)


class ConditionalTest(ABC):
    cme: Final[CMO]

    def __init__(self, cme: CMO):
        self.cme = cme

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Self:
        raise NotImplementedError

    @abstractmethod
    def tree_flatten(self) -> tuple[Any, Any]:
        raise NotImplementedError

    def std(self, x: Array) -> Array:
        return _posterior_std(self.cme, x)

    @abstractmethod
    def __call__(self, x: Array) -> TestEmbedding:
        raise NotImplementedError


@partial(jax.tree_util.register_pytree_node_class)
class AnalyticalConditionalTest(ConditionalTest):
    log_determinant: Array
    rkhs_norm: Array
    sub_gaussian_std: Array

    @classmethod
    def from_data(
            cls, kernel: VectorKernel, xs: Array, ys: Array, rkhs_norm: ArrayLike, sub_gaussian_std: ArrayLike
    ) -> Self:
        cme = kernel.cme(xs, ys)
        return cls.from_cme(cme, rkhs_norm, sub_gaussian_std)

    @classmethod
    def from_cme(cls, cme: CMO, rkhs_norm: ArrayLike, sub_gaussian_std: ArrayLike) -> Self:
        _, log_determinant = jnp.linalg.slogdet(jnp.eye(cme.n_points) + cme.gram / cme.kernel.regularization)

        rkhs_norm = jnp.broadcast_to(rkhs_norm, log_determinant.shape)
        sub_gaussian_std = jnp.broadcast_to(sub_gaussian_std, log_determinant.shape)

        return AnalyticalConditionalTest(
            cme=cme,
            log_determinant=log_determinant,
            rkhs_norm=rkhs_norm,
            sub_gaussian_std=sub_gaussian_std
        )

    def __init__(self, cme: CMO, log_determinant: ArrayLike, rkhs_norm: ArrayLike, sub_gaussian_std: ArrayLike):
        super().__init__(cme)

        self.log_determinant = jnp.asarray(log_determinant)
        self.rkhs_norm = jnp.asarray(rkhs_norm)
        self.sub_gaussian_std = jnp.asarray(sub_gaussian_std)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Self:
        cme, log_determinant, sub_gaussian_std = children
        return AnalyticalConditionalTest(
            cme=cme,
            log_determinant=log_determinant,
            sub_gaussian_std=sub_gaussian_std
        )

    def tree_flatten(self) -> tuple[Any, Any]:
        return (self.cme, self.log_determinant, self.sub_gaussian_std), None

    def __call__(self, x: Array) -> TestEmbedding:
        return AnalyticalConditionedTest(
            kme=self.cme(x),
            std=self.std(x),
            log_determinant=self.log_determinant,
            rkhs_norm=self.rkhs_norm,
            sub_gaussian_std=self.sub_gaussian_std,
            regularization=self.cme.kernel.regularization
        )


@partial(jax.tree_util.register_pytree_node_class)
class BootstrapConditionalTest(ConditionalTest):
    beta_null: Final[Array]

    @classmethod
    @partial(jax.jit, static_argnums={0, 3})
    def __bootstrap_beta(cls, cme: CMO, key: Array, n_bootstrap: int, es: Optional[Array] = None) -> Array:
        if es is None:
            es = cme.xs

        if es.shape[-cme.kernel.x.ndim:] != cme.shape_x:
            raise TypeError(f"Expected shape of es to match shape of cme. Got shape {es.shape} and {cme.shape}.")

        def bootstrap_single_beta(cme_: CMO, kmes: Fn, es_: Array, gram_y: Array, stds: Array, key_: Array) -> Array:
            resampled_indices = _bootstrap_masked_indices(cme_.mask, key_, shape=(1, cme_.n_points))
            bootstrap_cme = _resample_cme(cme_, resampled_indices)[0]
            bootstrap_kmes = bootstrap_cme(es_)

            identity_indices = jnp.arange(cme_.n_points)
            bootstrap_gram_y_22 = _extract_submatrix(gram_y, resampled_indices, resampled_indices)
            bootstrap_gram_y_12 = _extract_submatrix(gram_y, identity_indices, resampled_indices)

            bootstrap_cmmds = rkhs.distance(
                kmes, bootstrap_kmes,
                kernel_matrix_11=gram_y,
                kernel_matrix_22=bootstrap_gram_y_22,
                kernel_matrix_12=bootstrap_gram_y_12,
            )

            bootstrap_stds = _posterior_std(bootstrap_cme, es_, bootstrap_kmes.coefficients)

            return (bootstrap_cmmds / (stds + bootstrap_stds)).max()

        def bootstrap_beta(cme_: CMO, es_: Array, key_: Array) -> Array:
            kmes = cme_(es_)
            stds = _posterior_std(cme_, es_, kmes.coefficients)
            gram_y = cme.kernel.y.gram(cme_.ys, mask=cme_.mask)

            keys_ = jax.random.split(key_, n_bootstrap)

            return jax.lax.map(
                f=lambda key__: bootstrap_single_beta(cme_, kmes, es_, gram_y, stds, key__),
                xs=keys_,
            )

        batch_shape = cme.shape
        cme = cme.reshape(-1)
        es = es.reshape(len(cme), -1, *cme.shape_x)
        keys = jax.random.split(key, len(cme))

        bootstrap_betas = jax.lax.map(
            f=lambda inp: bootstrap_beta(*inp),
            xs=(cme, es, keys)
        )

        return bootstrap_betas.transpose(1, 0).reshape(n_bootstrap, *batch_shape)

    @classmethod
    def from_data(
            cls,
            kernel: VectorKernel,
            xs: Array, ys: Array,
            key: Array,
            n_bootstrap: int,
            mask: Optional[Array] = None,
            es: Optional[Array] = None
    ) -> Self:
        cme = kernel.cme(xs, ys, mask=mask)
        return BootstrapConditionalTest.from_cme(cme, key, n_bootstrap, es=es)

    @classmethod
    def from_cme(cls, cme: CMO, key: Array, n_bootstrap: int, es: Optional[Array] = None) -> Self:
        bootstrap_betas = cls.__bootstrap_beta(cme, key, n_bootstrap, es=es)
        return BootstrapConditionalTest(cme, bootstrap_betas)

    def __init__(self, cme: CMO, beta_null: Array):
        super().__init__(cme)

        if beta_null.shape[1:] != cme.shape:
            raise TypeError(f"Expected beta null distribution to hold one array for every mean embedding. "
                            f"Got shapes {cme.shape} and {beta_null.shape}.")

        self.beta_null = beta_null

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Self:
        cme, beta_null = children
        return BootstrapConditionalTest(cme=cme, beta_null=beta_null)

    def tree_flatten(self) -> tuple[Any, Any]:
        return (self.cme, self.beta_null), None

    def __call__(self, x: Array) -> BootstrapConditionedTest:
        return BootstrapConditionedTest(
            kme=self.cme(x),
            std=self.std(x),
            beta_null=self.beta_null,
        )


class TwoSampleTest(NamedTuple):
    t: Array
    threshold: Array
    distance: Array

    @classmethod
    def from_embeddings(cls, embedding_1: TestEmbedding, embedding_2: TestEmbedding, level: ArrayLike) -> Self:
        def threshold_combination(t: Array):
            threshold_1 = embedding_1.threshold(t * level)
            threshold_2 = embedding_2.threshold((1 - t) * level)
            return threshold_1 + threshold_2

        ts = jnp.linspace(start=0, stop=1, num=100)
        thresholds = jax.vmap(threshold_combination)(ts)

        optimal_t_index = thresholds.argmin(axis=0)
        optimal_t = ts[optimal_t_index]
        threshold = thresholds[optimal_t_index]

        distance = rkhs.distance(embedding_1.kme, embedding_2.kme)

        return TwoSampleTest(optimal_t, threshold, distance)

    def rejection_region(self) -> Array:
        return self.distance > self.threshold
