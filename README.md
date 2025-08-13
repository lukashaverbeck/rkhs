# 🌱 rkhs

`rkhs` is a small Python framework for marginal and conditional two-sample testing with kernels in JAX.

---

## ⚙️ Installation
**_Coming soon_**

---

## 🚀 Features
`rkhs` provides JAX-native two-sample tests (marginal, conditional, mixed) based on kernel embeddings with analytical or bootstrap confidence bounds, a simple API, and pluggable kernels.

- **Three test modes — one API**. You can test the following two-sample hypothesis with one common set of primitives: 
  - **Marginal:** $H_0: P = Q$
  - **Conditional:** $H_0(x_1,x_2): P(\cdot\mid X=x_1)=Q(\cdot\mid X=x_2)$
  - **Mixed:** $H_0(x): P(\cdot\mid X=x)=Q$

The test compares kernel embeddings in RKHS norm and rejects $H_0$ at level $\alpha$ if

$$
\|\widehat\mu_P - \widehat\mu_Q\|_\mathcal{H} > \beta_P + \beta_Q,
$$

where $\beta_\ast$ are finite-sample confidence radii from the selected regime.

- **Confidence regimes**
  - **Analytical bounds:** distribution-free guarantees under the stated assumptions (conservative; no extra computation).
  - **Bootstrap bounds:** data-driven thresholds with typically higher power (cost scales with the number of resamples).

- **JAX integration**
  Works with `jit`/`vmap`, runs on CPU/GPU/TPU, and uses explicit `PRNGKey` for reproducibility. Kernels must be JIT-friendly.

- **Kernels**
  Built-ins: `Gaussian`, `Matern`, `Laplacian`, `Polynomial`, `Linear`.  
  Conditional tests use `VectorKernel(x=..., y=..., regularization=...)` to specify input and output kernels.

- **Decision interface**
  `test.reject()` (boolean at level $\alpha$), `test.distance` (empirical RKHS distance), `test.threshold` ($\beta_P+\beta_Q$).

---

## 🧩 Usage

### 1) Marginal two-sample test (analytical bounds)

```python
import jax
from rkhs.testing import MarginalTestEmbedding, TwoSampleTest
from rkhs.kernels import GaussianKernel

key1, key2 = jax.random.key(1), jax.random.key(2)

# toy data: two 3D Gaussians with different scale
xs_1 = jax.random.normal(key=key1, shape=(200, 3))
xs_2 = jax.random.normal(key=key2, shape=(200, 3)) * 1.25

# kernel on the sample space
kernel = GaussianKernel(bandwidth=0.5, data_shape=(3,))

# sup_x k(x, x); for normalized RBF this is often 1.0
k_bound = 1.0

# embeddings + analytical confidence radii
kme_1 = MarginalTestEmbedding.analytical.from_data(kernel, xs_1, kernel_bound=k_bound)
kme_2 = MarginalTestEmbedding.analytical.from_data(kernel, xs_2, kernel_bound=k_bound)

# level-α test
test = TwoSampleTest.from_embedding(kme_1, kme_2, level=0.05)

decision = test.reject()    # bool
dist     = test.distance    # RKHS distance
thr      = test.threshold   # β_P + β_Q
print(decision, dist, thr)
```
### 2) Marginal test (bootstrap bounds)

```python
import jax
from rkhs.testing import MarginalTestEmbedding, TwoSampleTest
from rkhs.kernels import GaussianKernel

xs_1 = jax.random.normal(key=jax.random.key(1), shape=(200, 3))
xs_2 = jax.random.normal(key=jax.random.key(2), shape=(200, 3)) * 1.25

kernel = GaussianKernel(bandwidth=0.5, data_shape=(3,))

kme_1 = MarginalTestEmbedding.bootstrap.from_data(
    kernel, xs_1, key=jax.random.key(3), n_bootstrap=500
)
kme_2 = MarginalTestEmbedding.bootstrap.from_data(
    kernel, xs_2, key=jax.random.key(4), n_bootstrap=500
)

test = TwoSampleTest.from_embedding(kme_1, kme_2, level=0.05)
print(test.reject(), test.distance, test.threshold)
```

### 3) Conditional two-sample test at selected covariates

Compute conditional mean embeddings (CMEs) for each dataset, evaluate them at covariates \(x\), and compare the resulting embeddings in the output space.

```python
import jax
from rkhs import VectorKernel
from rkhs.testing import ConditionalTestEmbedding, TwoSampleTest
from rkhs.kernels import GaussianKernel

# synthetic x,y pairs with additive noise
xs_1 = jax.random.normal(key=jax.random.key(1), shape=(300, 3))
ys_1 = xs_1 + jax.random.normal(key=jax.random.key(2), shape=(300, 3))

xs_2 = jax.random.normal(key=jax.random.key(3), shape=(300, 3))
ys_2 = xs_2 + jax.random.normal(key=jax.random.key(4), shape=(300, 3))

# vector-valued kernel over inputs X and outputs Y
kernel = VectorKernel(
    x=GaussianKernel(bandwidth=0.5, data_shape=(3,)),
    y=GaussianKernel(bandwidth=1.0, data_shape=(3,)),
    regularization=0.1
)

# fit CMEs (bootstrap for tighter thresholds)
cme_1 = ConditionalTestEmbedding.bootstrap.from_data(
    kernel, xs_1, ys_1,
    grid=xs_1,
    key=jax.random.key(5),
    n_bootstrap=300
)

cme_2 = ConditionalTestEmbedding.bootstrap.from_data(
    kernel, xs_2, ys_2,
    grid=xs_2,
    key=jax.random.key(6),
    n_bootstrap=300
)

# covariates to test equality at
covariates = jax.numpy.array([[0., 0., 0.], [-1., -1., -1.], [1., 1., 1.]])

# evaluate CMEs at covariates → embeddings over Y
kme_1 = cme_1(covariates)
kme_2 = cme_2(covariates)

# vectorized test across covariates
test = TwoSampleTest.from_embeddings(kme_1, kme_2, level=0.05)
reject_per_x = test.reject()   # Boolean array

print(reject_per_x)
```

### 4) Mixed test: $P(\cdot\mid X=x)$ vs. $Q$

```python
import jax
from rkhs import VectorKernel
from rkhs.testing import MarginalTestEmbedding, ConditionalTestEmbedding, TwoSampleTest
from rkhs.kernels import GaussianKernel

# data for P(X,Y) and Q over Y
xs = jax.random.normal(key=jax.random.key(1), shape=(400, 3))
ys = xs + 0.5 * jax.random.normal(key=jax.random.key(2), shape=(400, 3))
ys_q = jax.random.normal(key=jax.random.key(3), shape=(400, 3))  # i.i.d. from Q

# CME for P
c_kernel = VectorKernel(
    x=GaussianKernel(bandwidth=0.5, data_shape=(3,)),
    y=GaussianKernel(bandwidth=1.0, data_shape=(3,)),
    regularization=0.1
)

cme_p = ConditionalTestEmbedding.bootstrap.from_data(c_kernel, xs, ys, grid=xs, key=jax.random.key(4), n_bootstrap=300)

# KME for Q
y_kernel = GaussianKernel(bandwidth=1.0, data_shape=(3,))

kme_q = MarginalTestEmbedding.bootstrap.from_data(y_kernel, ys_q, key=jax.random.key(5), n_bootstrap=300)

# covariates x to test at
covariates = jax.numpy.array([[0., 0., 0.], [1., 0., -1.]])

# compare P(.|X=x) to Q for each x
kme_p_at_x = cme_p(covariates)
test = TwoSampleTest.from_embeddings(kme_p_at_x, kme_q, level=0.05)
print(test.reject())  # Boolean per covariate
```

---

## 🔍 Kernel quick reference

- `LinearKernel` — compares means (first moment).
- `PolynomialKernel(degree=d)` — compares moments up to degree $d$.
- `Gaussian`, `Matern`, `Laplacian` — characteristic; compare full distributions.

For conditional tests:
- **Input kernel (`x`)**: used to learn the conditional embedding (not for comparison).
- **Output kernel (`y`)**: determines what aspects of the conditional law are compared.

---

## 🧠 Notes
- Embeddings preserve batch axes; passing a batch of covariates returns a batch of embeddings.
- All randomness is via explicit `PRNGKey`.
- You can use your own custom kernel by inheriting from `rkhs.Kernel`:

    ```python
    from jax import Array
    from rkhs import Kernel
    
    class MyCustomKernel(Kernel):
        def __init__(self, data_shape: tuple[int, ...]):
            super().__init__(data_shape, rkhs_dim="inf")
            ...
        
        def _dot(self, x1: Array, x2: Array) -> Array:
            ...  # your logic here (must be jit-compilable)
    ```

---

## 📚 References

- Marginal test: Gretton, A., et al. (2012). *A Kernel Two-Sample Test*. [JMLR page](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html) · [PDF](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)

- Conditional test: Massiani, P.-F., et al. (2025). *A Kernel Conditional Two-Sample Test*. [arXiv](https://arxiv.org/abs/2506.03898) · [PDF](https://arxiv.org/pdf/2506.03898)
