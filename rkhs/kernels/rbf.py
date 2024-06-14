from __future__ import annotations

from typing import Callable

from jax import numpy as jnp

from rkhs import Kernel
from rkhs.base import Function

Metric = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def generalized_rbf_transformation(sigma: float | jnp.ndarray, squared_metric: Metric) -> Function:
    if not isinstance(sigma, jnp.ndarray):
        sigma = jnp.array(sigma)

    assert sigma.ndim <= 1
    assert jnp.all(sigma > 0)

    def transformation(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        x_1 = x_1 / sigma
        x_2 = x_2 / sigma
        distance = squared_metric(x_1, x_2)
        return jnp.exp(-0.5 * distance)

    return transformation


class RBFKernel(Kernel):
    def __init__(self, sigma: float | jnp.ndarray):
        def squared_euclidean(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            difference = x_1 - x_2
            return jnp.inner(difference, difference)

        transformation = generalized_rbf_transformation(sigma, squared_euclidean)
        super().__init__(transformation, input_dim=1)
