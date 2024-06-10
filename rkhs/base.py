from __future__ import annotations

from functools import update_wrapper, partial
from typing import Final, Callable

import jax
from jax import numpy as jnp, jit

Function = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def vectorize_function(fn: Function, input_dim: int) -> Function:
    def input_shape(argument_index: int) -> str:
        character_offset = 97 + argument_index * input_dim
        return ",".join([chr(character_offset + i) for i in range(input_dim)])

    signature = f"({input_shape(0)}),({input_shape(1)})->()"
    return jnp.vectorize(fn, signature=signature)


class ExpandableFunction:
    input_dim: Final[int]
    __vectorized_function: Final[Function]

    def __init__(self, fn: Function, input_dim: int):
        update_wrapper(self, fn)
        fn = partial(jit)(fn)
        self.input_dim = input_dim
        self.__vectorized_function = vectorize_function(fn, input_dim)

    def batch(self, xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
        assert xs_1.ndim == xs_2.ndim
        assert xs_1.ndim == self.input_dim + 1
        return self.__vectorized_function(xs_1, xs_2)

    def many_one(self, xs_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        assert xs_1.ndim == x_2.ndim + 1
        assert x_2.ndim == self.input_dim
        return self.__vectorized_function(xs_1, x_2)

    def many_many(self, xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
        assert xs_1.ndim == xs_2.ndim
        assert xs_1.ndim == self.input_dim + 1

        def transform(reduced_x_2: jnp.ndarray) -> jnp.ndarray:
            return self.__vectorized_function(xs_1, reduced_x_2)

        return jax.lax.map(transform, xs_2).T

    def __call__(self, x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        return self.__vectorized_function(x_1, x_2)


def make_kme_dot_product(kernel: Kernel) -> ExpandableFunction:
    def dot_product(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
        kernel_matrix = kernel.many_many(xs_1, xs_2)
        return kernel_matrix.sum() / (xs_1.shape[0] * xs_2.shape[0])

    return ExpandableFunction(dot_product, input_dim=kernel.input_dim + 1)


def make_squared_mmd(kernel: Kernel) -> ExpandableFunction:
    dot_product = make_kme_dot_product(kernel)

    def squared_mmd(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
        dot_product_11 = dot_product(xs_1, xs_1)
        dot_product_22 = dot_product(xs_2, xs_2)
        dot_product_12 = dot_product(xs_1, xs_2)
        return dot_product_11 + dot_product_22 - 2 * dot_product_12

    return ExpandableFunction(squared_mmd, input_dim=kernel.input_dim + 1)


def make_mmd(kernel: Kernel) -> ExpandableFunction:
    squared_mmd = make_squared_mmd(kernel)

    def mmd(xs_1: jnp.ndarray, xs_2: jnp.ndarray) -> jnp.ndarray:
        squared_mmd_value = squared_mmd(xs_1, xs_2)
        clipped_squared_mmd = jnp.clip(squared_mmd_value, a_min=0)
        return jnp.sqrt(clipped_squared_mmd)

    return ExpandableFunction(mmd, input_dim=kernel.input_dim + 1)


class Kernel(ExpandableFunction):
    kme_dot_product: Final[ExpandableFunction]
    squared_mmd: Final[ExpandableFunction]
    mmd: Final[ExpandableFunction]

    def __init__(self, fn: Function, input_dim: int = 1):
        super().__init__(fn, input_dim)
        self.kme_dot_product = make_kme_dot_product(self)
        self.squared_mmd = make_squared_mmd(self)
        self.mmd = make_mmd(self)

    def __matmul__(self, other: Kernel) -> Kernel:
        assert isinstance(other, Kernel)
        assert self.input_dim == other.input_dim

        def product_transformation(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            return self(x_1, x_2) * other(x_1, x_2)

        return Kernel(product_transformation, input_dim=self.input_dim)

    def __add__(self, other: Kernel) -> Kernel:
        assert isinstance(other, Kernel)
        assert self.input_dim == other.input_dim

        def sum_transformation(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            return self(x_1, x_2) + other(x_1, x_2)

        return Kernel(sum_transformation, input_dim=self.input_dim)

    def __pow__(self, power: float) -> Kernel:
        assert power > 0, "Raising a kernel to a non-positive power results in an invalid kernel"

        def power_transformation(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
            return self(x_1, x_2) ** power

        return Kernel(power_transformation, input_dim=self.input_dim)
