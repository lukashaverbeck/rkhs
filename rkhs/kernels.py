from __future__ import annotations

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
