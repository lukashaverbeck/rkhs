from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from rkhs import Fn, Kernel


def uniform_fn(key: Array, domain: Array, kernel: Kernel, norm: ArrayLike, shape: Sequence[int] = ()) -> Fn:
    if domain.ndim < kernel.data_ndim + 1:
        raise TypeError(f"Domain must have at least one more dimension than the kernel. Got domain with shape "
                        f"{domain.shape} for kernel with data ndim {kernel.data_ndim}.")

    norm = jnp.asarray(norm)

    key_coefficients, key_norm = jax.random.split(key)
    coefficient_shape = (*shape, *kernel.batch_shape(domain))

    gram = kernel.gram(domain)
    u, s, _ = jnp.linalg.svd(gram, hermitian=True)
    mask = s >= 1e-5

    random_direction_orthonormal = jax.random.normal(key_coefficients, coefficient_shape) * mask
    actual_norm = jnp.linalg.norm(random_direction_orthonormal, axis=-1, keepdims=True)

    coefficients_orthonormal_basis = random_direction_orthonormal / actual_norm * norm[..., None]
    coefficients = jnp.einsum("...ij,...j->...i", u, coefficients_orthonormal_basis / jnp.sqrt(s) * mask)

    return kernel.fn(domain, coefficients)
