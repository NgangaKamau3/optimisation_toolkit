import numpy as np
from .base import BaseFunction


class Rastrigin(BaseFunction):
    def __init__(self, A=10, n=2):
        super().__init__(bounds=[(-5.12, 5.12)] * n)
        self.A = A
        self.n = n
        self.global_minimum = np.zeros(n)

    def evaluate(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return self.A * self.n + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))

    def gradient(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return 2 * x + 2 * np.pi * self.A * np.sin(2 * np.pi * x)
