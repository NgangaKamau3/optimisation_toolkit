import numpy as np
from .base import BaseOptimizer


class Adam(BaseOptimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grad):
        if self.m is None or self.m.shape != params.shape:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.t = 0

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
