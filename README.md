# 🌱 rkhs

`rkhs` is a small Python framework for marginal and conditional two-sample testing with kernels in JAX.

---

## ⚙️ Installation
```bash
pip install rkhs
```

---

## 🚀 Features
`rkhs` provides JAX-native two-sample tests (marginal, conditional, mixed) based on kernel embeddings with analytical or bootstrap confidence bounds, a simple API, and pluggable kernels.

### Three test modes — one API 
You can test the following two-sample hypotheses with one common set of primitives: 
  - **Marginal:** $H_0: P = Q$
  - **Conditional:** $H_0(x_1,x_2): P(\cdot\mid X=x_1) = Q(\cdot\mid X=x_2)$
  - **Mixed:** $H_0(x): P(\cdot\mid X=x) = Q$

The test compares kernel embeddings in RKHS norm and rejects $H_0$ at level $\alpha$ if

$$
  \|\hat\mu_P - \hat\mu_Q\|_\mathcal{H} > \beta_P + \beta_Q \quad,
$$

where $\beta_\ast$ are finite-sample confidence radii from the selected regime.

### Confidence regimes
**Analytical bounds.** Finite-sample guarantees under the stated assumptions (conservative, little overhead).

**Bootstrap bounds.** Data-driven thresholds with typically higher power (cost scales with the number of resamples).

### JAX integration
Works with `jit`/`vmap`, runs on CPU/GPU/TPU, and uses explicit `PRNGKey` for reproducibility.

### Kernels
Popular kernels are built in: `Gaussian`, `Matern`, `Laplacian`, `Polynomial`, `Linear`.

Conditional tests use a scalar kernel on the input domain and a separate kernel on the output domain: 
`VectorKernel(x=..., y=..., regularization=...)`.

---

## 🧩 Usage

### 1) Marginal two-sample test (analytical bounds)

```python
import jax
from rkhs.testing import TestEmbedding, TwoSampleTest
from rkhs.kernels import GaussianKernel

# toy data: two 3D Gaussians with different means
xs_1 = jax.random.normal(key=jax.random.key(1), shape=(200, 3))
xs_2 = jax.random.normal(key=jax.random.key(2), shape=(200, 3)) + 1.0

# kernel on the sample space
kernel = GaussianKernel(bandwidth=1.5, data_shape=(3,))

# embedding + analytical confidence radius
kme_1 = TestEmbedding.analytical(
    kme=kernel.kme(xs_1),   # embed dataset in RKHS
    kernel_bound=1.0        # sup_x k(x, x)
)
kme_2 = TestEmbedding.analytical(
    kme=kernel.kme(xs_2),   # embed dataset in RKHS
    kernel_bound=1.0        # sup_x k(x, x)
) 

# level-α test
test = TwoSampleTest.from_embeddings(kme_1, kme_2, level=0.05)

decision = test.reject      # boolean (reject H_0?)
distance = test.distance    # RKHS distance
threshold = test.threshold  # β_P + β_Q
print(decision, distance, threshold)
```
### 2) Marginal test (bootstrap bounds)

```python
import jax
from rkhs.testing import TestEmbedding, TwoSampleTest
from rkhs.kernels import GaussianKernel

# toy data: two 3D Gaussians with different means
xs_1 = jax.random.normal(key=jax.random.key(1), shape=(200, 3))
xs_2 = jax.random.normal(key=jax.random.key(2), shape=(200, 3)) + 1.0

# kernel on the sample space
kernel = GaussianKernel(bandwidth=1.5, data_shape=(3,))

# embedding + analytical confidence radius
kme_1 = TestEmbedding.bootstrap(
    kme=kernel.kme(xs_1),   # embed dataset in RKHS
    key=jax.random.key(3),  # random key
    n_bootstrap=1000        # number of bootstrap resamples
)
kme_2 = TestEmbedding.bootstrap(
    kme=kernel.kme(xs_2),   # embed dataset in RKHS
    key=jax.random.key(4),  # random key
    n_bootstrap=1000        # number of bootstrap resamples
) 

# level-α test
test = TwoSampleTest.from_embeddings(kme_1, kme_2, level=0.05)

decision = test.reject      # boolean (reject H_0?)
distance = test.distance    # RKHS distance
threshold = test.threshold  # β_P + β_Q
print(decision, distance, threshold)
```

### 3) Conditional two-sample test at selected covariates

```python
import jax
from rkhs import VectorKernel
from rkhs.testing import ConditionalTestEmbedding, TwoSampleTest
from rkhs.kernels import GaussianKernel

# synthetic x,y pairs with additive noise
xs_1 = jax.random.normal(key=jax.random.key(1), shape=(1000, 3))
ys_1 = xs_1 + jax.random.normal(key=jax.random.key(2), shape=(1000, 3)) * 0.05

xs_2 = jax.random.normal(key=jax.random.key(3), shape=(1000, 3))
ys_2 = xs_2 + jax.random.normal(key=jax.random.key(4), shape=(1000, 3)) * 0.05 + 0.5

# inputs at which to test for distributional equality: H_0(x): P(. | x) =? Q(. | x), for x in grid
covariates = jax.numpy.linspace(jax.numpy.array([-3, -3, -3]), jax.numpy.array([3, 3, 3]), num=100)

# vector-valued kernel over inputs X and outputs Y
kernel = VectorKernel(
    x=GaussianKernel(bandwidth=0.5, data_shape=(3,)),  # kernel used for ridge regression
    y=GaussianKernel(bandwidth=1.0, data_shape=(3,)),  # kernel used for embedding marginal distribution at each covariate
    regularization=0.1
)

# conditional embedding + bootstrap confidence radius
cme_1 = ConditionalTestEmbedding.bootstrap(
    cme=kernel.cme(xs_1, ys_1), # embed dataset in vector-valued RKHS
    grid=covariates,            # covariates used in bootstrap of threshold parameters
    key=jax.random.key(5),      # random key
    n_bootstrap=100             # number of bootstrap resamples
)

cme_2 = ConditionalTestEmbedding.bootstrap(
    cme=kernel.cme(xs_2, ys_2), # embed dataset in vector-valued RKHS
    grid=covariates,            # covariates used in bootstrap of threshold parameters
    key=jax.random.key(6),      # random key
    n_bootstrap=100             # number of bootstrap resamples
)

# evaluate CMEs at covariates -> embeds each distribution over Y in RKHS of `kernel.y`
kme_1 = cme_1(covariates)
kme_2 = cme_2(covariates)

# batched test across all covariates
test = TwoSampleTest.from_embeddings(kme_1, kme_2, level=0.05)
reject_per_x = test.reject   # Boolean array

decision = test.reject      # boolean (reject H_0(x)?, individually for each covariate). shape: covariates.shape
distance = test.distance    # RKHS distance. shape: covariates.shape
threshold = test.threshold  # β_P + β_Q
print(decision, distance, threshold)
```

### 4) Mixed test: $P(\cdot\mid X=x)$ vs. $Q$

```python
import jax
from rkhs import VectorKernel
from rkhs.testing import TestEmbedding, ConditionalTestEmbedding, TwoSampleTest
from rkhs.kernels import GaussianKernel

# dataset from marginal distribution over Y
ys_1 = jax.random.normal(key=jax.random.key(1), shape=(1000, 3)) * 0.05

# synthetic x,y pairs with additive noise
xs_2 = jax.random.normal(key=jax.random.key(2), shape=(1000, 3))
ys_2 = xs_2 + jax.random.normal(key=jax.random.key(3), shape=(1000, 3)) * 0.05

# inputs at which to test for distributional equality: H_0(x): P =? Q(. | x), for x in grid
covariates = jax.numpy.linspace(jax.numpy.array([-3, -3, -3]), jax.numpy.array([3, 3, 3]), num=200)

y_kernel = GaussianKernel(bandwidth=1.0, data_shape=(3,))

# vector-valued kernel over inputs X and outputs Y
vector_kernel = VectorKernel(
    x=GaussianKernel(bandwidth=0.5, data_shape=(3,)),  # kernel used for ridge regression
    y=y_kernel,                                        # kernel used for embedding marginal distribution at each covariate
    regularization=0.1
)

# embedding + analytical confidence radius (can be drop-in replaced with bootstrap radius)
kme_1 = TestEmbedding.analytical(
    kme=y_kernel.kme(ys_1), # embed dataset in RKHS
    kernel_bound=1.0        # sup_x k(x, x)
)

# conditional embedding + analytical confidence radius
cme_2 = ConditionalTestEmbedding.bootstrap(
    cme=vector_kernel.cme(xs_2, ys_2),  # embed dataset in vector-valued RKHS
    grid=covariates,                    # covariates used in bootstrap of threshold parameters
    key=jax.random.key(4),              # random key
    n_bootstrap=100                     # number of bootstrap resamples
)

# evaluate CME at covariates -> embeds each distribution over Y in RKHS of `kernel.y`
kme_2 = cme_2(covariates)

# batched test across all covariates
test = TwoSampleTest.from_embeddings(kme_1, kme_2, level=0.05)
reject_per_x = test.reject   # Boolean array

decision = test.reject      # boolean (reject H_0(x)?, individually for each covariate). shape: covariates.shape
distance = test.distance    # RKHS distance. shape: covariates.shape
threshold = test.threshold  # β_P + β_Q
print(decision, distance, threshold)
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
- All randomness is explicit via `jax.random.PRNGKey`.
- You can use your own custom kernel by extending `rkhs.Kernel`:

```python
from jax import Array
from rkhs import Kernel
import jax

class MyCustomKernel(Kernel):
    def __init__(self, data_shape: tuple[int, ...]):
        super().__init__(data_shape)
        ...
    
    def _dot(self, x1: Array, x2: Array) -> Array:
        ...  # your logic here (must be jit-compilable)
```

---

## 📚 References

- Marginal test: Gretton, A., et al. (2012). *A Kernel Two-Sample Test*. [JMLR page](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html) · [PDF](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)

- Conditional test: Massiani, P.-F., et al. (2025). *A Kernel Conditional Two-Sample Test*. [arXiv](https://arxiv.org/abs/2506.03898) · [PDF](https://arxiv.org/pdf/2506.03898)
