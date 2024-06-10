from __future__ import annotations

from typing import Callable

from jax import numpy as jnp

from rkhs import Kernel
from rkhs.base import Function

Metric = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def generalized_rbf_transformation(sigma: float, squared_metric: Metric) -> Function:
    def transformation(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        distance = squared_metric(x_1, x_2)
        return jnp.exp(-distance / (2 * sigma ** 2))

    return transformation


class RBFKernel(Kernel):
    def __init__(self, sigma: float):
        assert sigma > 0

        def squared_euclidean(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            difference = x_1 - x_2
            return jnp.inner(difference, difference)

        transformation = generalized_rbf_transformation(sigma, squared_euclidean)
        super().__init__(transformation, input_dim=1)
