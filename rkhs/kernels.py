from typing import Final, Literal

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from rkhs.base import Kernel


class LinearKernel(Kernel):
    def __init__(self, ndim: int = 1):
        if ndim not in {0, 1}:
            raise ValueError(f"Linear kernel only supports scalar or 1D inputs. Got {ndim}.")

        super().__init__(ndim=ndim)

    def _dot(self, x_1: Array, x_2: Array) -> Array:
        return jnp.dot(x_1, x_2)


class PolynomialKernel(Kernel):
    degree: Final[float]

    def __init__(self, degree: int, ndim: int = 1):
        if ndim not in {0, 1}:
            raise ValueError(f"Polynomial kernel only supports scalar or 1D inputs. Got {ndim}.")

        super().__init__(ndim=ndim)
        self.degree = degree

    def _dot(self, x_1: Array, x_2: Array) -> Array:
        return (1 + jnp.dot(x_1, x_2)) ** self.degree


class GaussianKernel(Kernel):
    bandwidth: Final[Array]

    def __init__(self, bandwidth: ArrayLike, ndim: Literal[0, 1] = 1):
        if ndim not in {0, 1}:
            raise ValueError(f"Gaussian kernel only supports scalar or 1D inputs. Got {ndim}.")

        bandwidth = jnp.broadcast_to(bandwidth, ndim)

        if jnp.any(bandwidth <= 0):
            raise ValueError(f"Bandwidth must be positive. Got {bandwidth}.")
        if bandwidth.ndim > 1:
            raise ValueError(f"Bandwidth must be a scalar or a vector. Got {bandwidth.ndim}.")

        self.bandwidth = bandwidth

        super().__init__(ndim=ndim)

    def _dot(self, x_1: Array, x_2: Array) -> Array:
        difference = (x_1 - x_2) / self.bandwidth
        return jnp.exp(-jnp.dot(difference, difference) / 2)


class MaternKernel(Kernel):
    bandwidth: Final[float]
    length_scale: Final[float]
    nu: Final[float]

    SUPPORTED_NU = {1 / 2, 3 / 2, 5 / 2}

    def __init__(self, bandwidth: float, length_scale: float, nu: float, ndim: Literal[0, 1] = 1):
        super().__init__(ndim)

        if ndim not in {0, 1}:
            raise ValueError(f"Matérn kernel only supports scalar or 1D inputs. Got {ndim}.")

        if bandwidth <= 0:
            raise ValueError(f"Bandwidth must be positive. Got {bandwidth}.")
        if length_scale <= 0:
            raise ValueError(f"Length scale must be positive. Got {length_scale}.")
        if nu not in self.SUPPORTED_NU:
            raise ValueError(f"Only support {{{self.SUPPORTED_NU}}} for nu. Got {nu}.")

        self.bandwidth = bandwidth
        self.length_scale = length_scale
        self.nu = nu

    def _dot(self, x_1: Array, x_2: Array) -> Array:
        distance = jnp.linalg.norm(x_1 - x_2)
        scaled_distance = jnp.sqrt(self.nu * 2) * distance

        if self.nu == 1 / 2:
            scale_term = 1
        elif self.nu == 3 / 2:
            scale_term = 1 + scaled_distance / self.length_scale
        elif self.nu == 5 / 2:
            scale_term = 1 + scaled_distance / self.length_scale + self.nu * ((distance / self.length_scale) ** 2)
        else:
            raise ValueError(f"Only support {{{self.SUPPORTED_NU}}} for nu. Got {self.nu}.")

        return self.bandwidth * scale_term * jnp.exp(-scaled_distance / self.length_scale)


class LaplacianKernel(Kernel):
    length_scale: Final[float]

    def __init__(self, length_scale: float, ndim: Literal[0, 1] = 1):
        if ndim not in {0, 1}:
            raise ValueError(f"Gaussian kernel only supports scalar or 1D inputs. Got {ndim}.")

        if jnp.any(length_scale <= 0):
            raise ValueError(f"Length scale must be positive. Got {length_scale}.")
        if length_scale <= 0:
            raise ValueError(f"Length scale must be positive. Got {length_scale}.")

        self.length_scale = length_scale

        super().__init__(ndim=ndim)

    def _dot(self, x_1: Array, x_2: Array) -> Array:
        return jnp.exp(-jnp.linalg.norm(x_1 - x_2, ord=1) / self.length_scale)
