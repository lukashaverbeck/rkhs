from rkhs import Kernel

from rkhs.kernels.rbf import generalized_rbf_transformation


class KMERBFKernel(Kernel):
    def __init__(self, sample_kernel: Kernel, sigma: float):
        kernel_function = generalized_rbf_transformation(sigma, sample_kernel.mmd)
        super().__init__(kernel_function, input_dim=sample_kernel.input_dim + 1)
