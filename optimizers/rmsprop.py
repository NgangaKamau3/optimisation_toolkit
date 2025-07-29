import numpy as np
from .base import BaseOptimizer


class RMSProp(BaseOptimizer):
    def __init__(self, lr=0.001, alpha=0.99, eps=1e-8, **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.alpha = alpha
        self.eps = eps
        self.v = None

    def step(self, params, grad):
        if self.v is None or self.v.shape != params.shape:
            self.v = np.zeros_like(params)

        self.v = self.alpha * self.v + (1 - self.alpha) * grad**2
        return params - self.lr * grad / (np.sqrt(self.v) + self.eps)
