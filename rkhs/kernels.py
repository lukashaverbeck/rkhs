from __future__ import annotations

import jax.random
from jax import numpy as jnp

from rkhs.base import Kernel


class LinearKernel(Kernel):
    def __init__(self):
        def transform(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            return jnp.dot(x_1, x_2.T)

        super().__init__(fn=transform, ndim=1)


class GaussianKernel(Kernel):
    def __init__(self, bandwidth: float | jnp.ndarray):
        bandwidth = jnp.array(bandwidth)
        assert bandwidth.ndim <= 1, "Bandwidth must be a scalar or a vector."

        def transform(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            difference = (x_1 - x_2) / bandwidth
            return jnp.exp(-jnp.dot(difference, difference) / 2)

        super().__init__(fn=transform, ndim=1)


if __name__ == "__main__":
    XS_1 = jax.random.normal(jax.random.PRNGKey(0), (500, 5))
    YS_1 = 2 * XS_1
    XS_2 = jax.random.normal(jax.random.PRNGKey(2), (500, 5))
    YS_2 = 2 * XS_2
    E = jax.random.normal(jax.random.PRNGKey(2), (5,))

    KERNEL = GaussianKernel(1)
    REGULARIZATION = 0.0001


    def dp(
            xs_1: jnp.ndarray, xs_2: jnp.ndarray,
            ys_1: jnp.ndarray, ys_2: jnp.ndarray,
            e: jnp.ndarray,
            regularization: float
    ) -> jnp.ndarray:
        n = xs_1.shape[0]
        m = xs_2.shape[0]

        x_gram_11 = KERNEL.many_many(xs_1, xs_1) + regularization * n * jnp.eye(n)
        x_gram_22 = KERNEL.many_many(xs_2, xs_2) + regularization * m * jnp.eye(m)

        kernel_vector_1 = KERNEL.many_one(xs_1, e)
        kernel_vector_2 = KERNEL.many_one(xs_2, e)

        chol_11, _ = jax.scipy.linalg.cho_factor(x_gram_11, lower=True)
        chol_22, _ = jax.scipy.linalg.cho_factor(x_gram_22, lower=True)

        y_gram_12 = KERNEL.many_many(ys_1, ys_2)

        return (
                jax.scipy.linalg.cho_solve((chol_11, True), kernel_vector_1)
                @ y_gram_12
                @ jax.scipy.linalg.cho_solve((chol_22, True), kernel_vector_2)
        )


    CKME_1 = KERNEL.ckme(XS_1, YS_1, REGULARIZATION)
    CKME_2 = KERNEL.ckme(XS_2, YS_2, REGULARIZATION)

    KME_1 = KERNEL.condition(CKME_1, E)
    KME_2 = KERNEL.condition(CKME_2, E)

    print(KERNEL.squared_distance(KME_1, KME_2))
    print(
        dp(XS_1, XS_1, YS_1, YS_1, E, REGULARIZATION)
        - 2 * dp(XS_1, XS_2, YS_1, YS_2, E, REGULARIZATION)
        + dp(XS_2, XS_2, YS_2, YS_2, E, REGULARIZATION)
    )
