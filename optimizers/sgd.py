import numpy as np
from .base import BaseOptimizer


class SGD(BaseOptimizer):
    def __init__(self, lr=0.01, momentum=0.0, **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grad):
        if self.velocity is None or self.velocity.shape != params.shape:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity - self.lr * grad
        return params + self.velocity
