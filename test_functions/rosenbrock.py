import numpy as np
from .base import BaseFunction


class Rosenbrock(BaseFunction):
    def __init__(self, a=1, b=100):
        super().__init__(bounds=[(-2, 2), (-1, 3)])
        self.a = a
        self.b = b
        self.global_minimum = np.array([a, a**2])

    def evaluate(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return (self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2

    def gradient(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        dx = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0] ** 2)
        dy = 2 * self.b * (x[1] - x[0] ** 2)
        return np.array([dx, dy])
