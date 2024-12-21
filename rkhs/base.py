from __future__ import annotations

from functools import partial, update_wrapper
from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp

type KernelFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
type RKHSKernelFn = Callable[[RKHSFn, RKHSFn], jnp.ndarray]
type ConditioningFn = Callable[[CKME, jnp.ndarray], RKHSFn]


class RKHSFn(NamedTuple):
    coefficients: jnp.ndarray
    points: jnp.ndarray

    def assert_sane(self, kernel_ndim: int):
        assert self.points.ndim == self.coefficients.ndim + kernel_ndim


class CKME(NamedTuple):
    xs: jnp.ndarray
    ys: jnp.ndarray
    regularization: float
    cholesky: jnp.ndarray

    def assert_sane_x(self, x_kernel_ndim: int):
        assert self.xs.ndim == self.cholesky.ndim
        assert self.cholesky.shape[-1] == self.cholesky.shape[-2]
        assert self.xs.shape[-x_kernel_ndim - 1] == self.cholesky.shape[-1]


class VecDistributionalKernelFn:
    __kernel_ndim: int
    __vectorized: KernelFn
    __batch: KernelFn
    __many_one: KernelFn
    __many_many: KernelFn

    def __init__(self, fn: KernelFn, kernel_ndim: int):
        assert kernel_ndim >= 0

        self.__kernel_ndim = kernel_ndim

        kernel_arguments = [f"d{i}" for i in range(kernel_ndim)]
        arg_shape_1 = ",".join(["n"] + kernel_arguments)
        arg_shape_2 = ",".join(["m"] + kernel_arguments)

        @partial(jax.jit)
        @partial(jnp.vectorize, signature=f"({arg_shape_1}),({arg_shape_2})->()")
        def vectorized(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            return fn(x_1, x_2)

        @partial(jax.jit)
        @partial(jax.vmap)
        def batch(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
            return fn(xs_1, xs_2)

        @partial(jax.jit)
        @partial(jax.vmap, in_axes=(0, None))
        def many_one(xs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            return fn(xs, x)

        @partial(jax.jit)
        def many_many(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
            def single_argument_fn(x: jnp.ndarray) -> jnp.ndarray:
                return many_one(xs_2, x)

            return jax.lax.map(single_argument_fn, xs_1)

        self.__vectorized = vectorized
        self.__batch = batch
        self.__many_one = many_one
        self.__many_many = many_many

    def batch(self, xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
        assert xs_1.ndim == self.__kernel_ndim + 2
        assert xs_2.ndim == self.__kernel_ndim + 2

        return self.__batch(xs_1, xs_2)

    def many_one(self, xs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        assert xs.ndim == self.__kernel_ndim + 2
        assert x.ndim == self.__kernel_ndim + 1

        return self.__many_one(xs, x)

    def many_many(self, xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
        assert xs_1.ndim == self.__kernel_ndim + 2
        assert xs_2.ndim == self.__kernel_ndim + 2

        return self.__many_many(xs_1, xs_2)

    def __call__(self, x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        assert x_1.ndim >= self.__kernel_ndim + 1
        assert x_2.ndim >= self.__kernel_ndim + 1

        return self.__vectorized(x_1, x_2)


class VecRKHSKernelFn:
    __kernel_ndim: int
    __vectorized: RKHSKernelFn
    __batch: RKHSKernelFn
    __many_one: RKHSKernelFn
    __many_many: RKHSKernelFn

    def __init__(self, kernel_fn: RKHSKernelFn, kernel_ndim: int):
        assert kernel_ndim >= 0
        self.__kernel_ndim = kernel_ndim

        kernel_arguments = [f"d{i}" for i in range(kernel_ndim)]
        arg_shape_points_1 = ",".join(["n"] + kernel_arguments)
        arg_shape_points_2 = ",".join(["m"] + kernel_arguments)

        @partial(jax.jit)
        @partial(jnp.vectorize, signature=f"({arg_shape_points_1}),({arg_shape_points_2}),(n),(m)->()")
        def array_decomposition(
                points_1: jnp.ndarray, points_2: jnp.ndarray,
                coefficients_1: jnp.ndarray, coefficients_2: jnp.ndarray
        ) -> jnp.ndarray:
            fn_1 = RKHSFn(coefficients_1, points_1)
            fn_2 = RKHSFn(coefficients_2, points_2)
            return kernel_fn(fn_1, fn_2)

        @partial(jax.jit)
        def vectorized(fn_1: RKHSFn, fn_2: RKHSFn) -> jnp.ndarray:
            return array_decomposition(fn_1.points, fn_2.points, fn_1.coefficients, fn_2.coefficients)

        @partial(jax.jit)
        @partial(jax.vmap)
        def batch(fns_1: RKHSFn, fns_2: RKHSFn) -> jnp.ndarray:
            return kernel_fn(fns_1, fns_2)

        @partial(jax.jit)
        @partial(jax.vmap, in_axes=(0, None))
        def many_one(fns: RKHSFn, fn: RKHSFn) -> jnp.ndarray:
            return kernel_fn(fns, fn)

        @partial(jax.jit)
        @partial(jax.vmap, in_axes=(0, None))
        @partial(jax.vmap, in_axes=(None, 0))
        def many_many(fns_1: RKHSFn, fns_2: RKHSFn) -> jnp.ndarray:
            return kernel_fn(fns_1, fns_2)

        self.__vectorized = vectorized
        self.__batch = batch
        self.__many_one = many_one
        self.__many_many = many_many

    def batch(self, fns_1: RKHSFn, fns_2: RKHSFn) -> jnp.ndarray:
        assert fns_1.assert_sane(self.__kernel_ndim)
        assert fns_2.assert_sane(self.__kernel_ndim)
        assert fns_1.points.ndim == self.__kernel_ndim + 2
        assert fns_2.points.ndim == self.__kernel_ndim + 2
        assert fns_1.coefficients.ndim == 2
        assert fns_2.coefficients.ndim == 2

        return self.__batch(fns_1, fns_2)

    def many_one(self, fns: RKHSFn, fn: RKHSFn) -> jnp.ndarray:
        fns.assert_sane(self.__kernel_ndim)
        fn.assert_sane(self.__kernel_ndim)
        assert fns.points.ndim == self.__kernel_ndim + 2
        assert fn.points.ndim == self.__kernel_ndim + 1
        assert fns.coefficients.ndim == 2
        assert fn.coefficients.ndim == 1

        return self.__many_one(fns, fn)

    def many_many(self, fns_1: RKHSFn, fns_2: RKHSFn) -> jnp.ndarray:
        fns_1.assert_sane(self.__kernel_ndim)
        fns_2.assert_sane(self.__kernel_ndim)
        assert fns_1.points.ndim == self.__kernel_ndim + 2
        assert fns_2.points.ndim == self.__kernel_ndim + 2
        assert fns_1.coefficients.ndim == 2
        assert fns_2.coefficients.ndim == 2

        return self.__many_many(fns_1, fns_2)

    def __call__(self, fn_1: RKHSFn, fn_2: RKHSFn) -> jnp.ndarray:
        fn_1.assert_sane(self.__kernel_ndim)
        fn_2.assert_sane(self.__kernel_ndim)
        assert fn_1.points.ndim >= self.__kernel_ndim + 1
        assert fn_2.points.ndim >= self.__kernel_ndim + 1
        assert fn_1.coefficients.ndim >= 1
        assert fn_2.coefficients.ndim >= 1

        return self.__vectorized(fn_1, fn_2)


class VecConditioning:
    __kernel_ndim: int
    __fn: ConditioningFn
    __batch: ConditioningFn
    __many_one: ConditioningFn
    __one_many: ConditioningFn
    __many_many: ConditioningFn

    def __init__(self, fn: ConditioningFn, kernel_ndim: int):
        assert kernel_ndim >= 0

        self.__fn = jax.jit(fn)
        self.__kernel_ndim = kernel_ndim

        @partial(jax.jit)
        def array_decomposition(
                xs: jnp.ndarray,
                ys: jnp.ndarray,
                cholesky: jnp.ndarray,
                e: jnp.ndarray,
                regularization: float
        ) -> RKHSFn:
            ckme = CKME(xs, ys, regularization, cholesky)
            return fn(ckme, e)

        @partial(jax.jit)
        def batch(ckmes: CKME, es: jnp.ndarray) -> RKHSFn:
            return jax.vmap(array_decomposition, in_axes=(0, 0, 0, 0, None))(
                ckmes.xs, ckmes.ys, ckmes.cholesky, es, ckmes.regularization
            )

        @partial(jax.jit)
        def many_one(ckme: CKME, e: jnp.ndarray) -> RKHSFn:
            return jax.vmap(array_decomposition, in_axes=(0, 0, 0, None, None))(
                ckme.xs, ckme.ys, ckme.cholesky, e, ckme.regularization
            )

        @partial(jax.jit)
        @partial(jax.vmap, in_axes=(None, 0))
        def one_many(ckme: CKME, es: jnp.ndarray) -> RKHSFn:
            return fn(ckme, es)

        @partial(jax.jit)
        def many_many(ckmes: CKME, es: jnp.ndarray) -> RKHSFn:
            return jax.vmap(
                jax.vmap(array_decomposition, in_axes=(None, None, None, 0, None)),
                in_axes=(0, 0, 0, None, None)
            )(ckmes.xs, ckmes.ys, ckmes.cholesky, es, ckmes.regularization)

        self.__batch = batch
        self.__many_one = many_one
        self.__one_many = one_many
        self.__many_many = many_many

    def batch(self, ckmes: CKME, es: jnp.ndarray) -> RKHSFn:
        ckmes.assert_sane_x(self.__kernel_ndim)
        assert es.ndim == self.__kernel_ndim + 1

        return self.__batch(ckmes, es)

    def many_one(self, ckme: CKME, e: jnp.ndarray) -> RKHSFn:
        ckme.assert_sane_x(self.__kernel_ndim)
        assert e.ndim == self.__kernel_ndim

        return self.__many_one(ckme, e)

    def one_many(self, ckme: CKME, es: jnp.ndarray) -> RKHSFn:
        ckme.assert_sane_x(self.__kernel_ndim)
        assert es.ndim == self.__kernel_ndim + 1

        return self.__one_many(ckme, es)

    def many_many(self, ckmes: CKME, es: jnp.ndarray) -> RKHSFn:
        ckmes.assert_sane_x(self.__kernel_ndim)
        assert es.ndim == self.__kernel_ndim + 1

        return self.__many_many(ckmes, es)

    def __call__(self, ckme: CKME, e: jnp.ndarray) -> RKHSFn:
        ckme.assert_sane_x(self.__kernel_ndim)
        assert e.ndim == self.__kernel_ndim

        return self.__fn(ckme, e)


class Kernel:
    ndim: int

    __vectorized: KernelFn
    __batch: KernelFn
    __many_one: KernelFn
    __many_many: KernelFn

    kme_dp: VecDistributionalKernelFn
    squared_mmd: VecDistributionalKernelFn
    mmd: VecDistributionalKernelFn

    __kme: Callable[[jnp.ndarray], RKHSFn]
    __kmes: Callable[[jnp.ndarray], RKHSFn]
    __ckme: Callable[[jnp.ndarray, jnp.ndarray, float], CKME]
    __ckmes: Callable[[jnp.ndarray, jnp.ndarray, float], CKME]

    dp: VecRKHSKernelFn
    squared_distance: VecRKHSKernelFn
    distance: VecRKHSKernelFn

    def __init__(self, fn: KernelFn, ndim: int):
        assert ndim >= 0
        self.ndim = ndim

        update_wrapper(self, fn)
        fn = jax.jit(fn)

        argument_shape = ",".join([f"d{i}" for i in range(ndim)])

        @partial(jax.jit)
        @partial(jnp.vectorize, signature=f"({argument_shape}),({argument_shape})->()")
        def vectorized(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            return fn(x_1, x_2)

        @partial(jax.jit)
        @partial(jax.vmap)
        def batch(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
            return fn(xs_1, xs_2)

        @partial(jax.jit)
        @partial(jax.vmap, in_axes=(0, None))
        def many_one(xs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            return fn(xs, x)

        @partial(jax.jit)
        @partial(jax.vmap, in_axes=(0, None))
        @partial(jax.vmap, in_axes=(None, 0))
        def many_many(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
            return fn(xs_1, xs_2)

        self.__vectorized = vectorized
        self.__batch = batch
        self.__many_one = many_one
        self.__many_many = many_many

        def kme_dp_fn(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
            kernel_matrix = many_many(xs_1, xs_2)
            return kernel_matrix.mean()

        def squared_mmd_fn(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
            squared_mmd = kme_dp_fn(xs_1, xs_1) - 2 * kme_dp_fn(xs_1, xs_2) + kme_dp_fn(xs_2, xs_2)
            return jnp.clip(squared_mmd, 0)

        def mmd_fn(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(squared_mmd_fn(xs_1, xs_2))

        self.kme_dp = VecDistributionalKernelFn(kme_dp_fn, ndim)
        self.squared_mmd = VecDistributionalKernelFn(squared_mmd_fn, ndim)
        self.mmd = VecDistributionalKernelFn(mmd_fn, ndim)

        def dp_fn(fn_1: RKHSFn, fn_2: RKHSFn) -> jnp.ndarray:
            kernel_matrix = many_many(fn_1.points, fn_2.points)
            return fn_1.coefficients @ kernel_matrix @ fn_2.coefficients

        def squared_distance_fn(fn_1: RKHSFn, fn_2: RKHSFn) -> jnp.ndarray:
            squared_distance = dp_fn(fn_1, fn_1) - 2 * dp_fn(fn_1, fn_2) + dp_fn(fn_2, fn_2)
            return jnp.clip(squared_distance, 0)

        def distance_fn(fn_1: RKHSFn, fn_2: RKHSFn) -> jnp.ndarray:
            return jnp.sqrt(squared_distance_fn(fn_1, fn_2))

        self.dp = VecRKHSKernelFn(dp_fn, ndim)
        self.squared_distance = VecRKHSKernelFn(squared_distance_fn, ndim)
        self.distance = VecRKHSKernelFn(distance_fn, ndim)

        @partial(jax.jit, static_argnums=0)
        def kme_fn(xs: jnp.ndarray) -> RKHSFn:
            n = xs.shape[-ndim - 1]
            coefficients = jnp.ones(xs.shape[:-ndim]) / n

            return RKHSFn(coefficients, xs)

        @partial(jax.jit)
        @partial(jax.vmap)
        def kmes_fn(xs: jnp.ndarray) -> RKHSFn:
            return kme_fn(xs)

        @partial(jax.jit, static_argnums=2)
        def ckme_fn(xs: jnp.ndarray, ys: jnp.ndarray, regularization: float) -> CKME:
            n = xs.shape[-ndim - 1]

            x_gram = many_many(xs, xs)
            regularized_gram = x_gram + regularization * n * jnp.eye(n)
            cholesky, _ = jax.scipy.linalg.cho_factor(regularized_gram, lower=True)

            return CKME(xs, ys, regularization, cholesky)

        @partial(jax.jit)
        @partial(jax.vmap, in_axes=(0, 0, None))
        def ckmes_fn(xs: jnp.ndarray, ys: jnp.ndarray, regularization: float) -> CKME:
            return ckme_fn(xs, ys, regularization)

        self.__kme = kme_fn
        self.__kmes = kme_fn
        self.__ckme = ckme_fn
        self.__ckmes = ckme_fn

        def condition(ckme: CKME, e: jnp.ndarray) -> RKHSFn:
            evaluations = many_one(ckme.xs, e)
            coefficients = jax.scipy.linalg.cho_solve((ckme.cholesky, True), evaluations)

            return RKHSFn(coefficients, ckme.ys)

        self.condition = VecConditioning(condition, ndim)

    def kme(self, xs: jnp.ndarray) -> RKHSFn:
        assert xs.ndim == self.ndim + 1

        return self.__kme(xs)

    def kmes(self, xs_batch: jnp.ndarray) -> RKHSFn:
        assert xs_batch.ndim == self.ndim + 2

        return self.__kmes(xs_batch)

    def ckme(self, xs: jnp.ndarray, ys: jnp.ndarray, regularization: float) -> CKME:
        assert regularization > 0
        assert xs.ndim == self.ndim + 1

        return self.__ckme(xs, ys, regularization)

    def ckmes(self, xs_batch: jnp.ndarray, ys_batch: jnp.ndarray, regularization: float) -> CKME:
        assert regularization > 0
        assert xs_batch.ndim == self.ndim + 2

        return self.__ckmes(xs_batch, ys_batch, regularization)

    def batch(self, xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
        assert xs_1.ndim == self.ndim + 1
        assert xs_2.ndim == self.ndim + 1

        return self.__batch(xs_1, xs_2)

    def many_one(self, xs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        assert xs.ndim == self.ndim + 1
        assert x.ndim == self.ndim

        return self.__many_one(xs, x)

    def many_many(self, xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
        assert xs_1.ndim == self.ndim + 1
        assert xs_2.ndim == self.ndim + 1

        return self.__many_many(xs_1, xs_2)

    def __call__(self, x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        assert x_1.ndim >= self.ndim
        assert x_2.ndim >= self.ndim

        return self.__vectorized(x_1, x_2)
