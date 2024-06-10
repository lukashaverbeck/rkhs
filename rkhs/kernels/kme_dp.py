from rkhs import Kernel


class KMEDotProductKernel(Kernel):
    def __init__(self, sample_kernel: Kernel):
        super().__init__(sample_kernel.kme_dot_product, input_dim=sample_kernel.input_dim + 1)
