import numpy as np
from .base import BaseFunction


class Beale(BaseFunction):
    def __init__(self):
        super().__init__(bounds=[(-4.5, 4.5), (-4.5, 4.5)])
        self.global_minimum = np.array([3, 0.5])

    def evaluate(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        term1 = (1.5 - x[0] + x[0] * x[1]) ** 2
        term2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return term1 + term2 + term3

    def gradient(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        dx = (
            2 * (1.5 - x[0] + x[0] * x[1]) * (x[1] - 1)
            + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (x[1] ** 2 - 1)
            + 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * (x[1] ** 3 - 1)
        )

        dy = (
            2 * (1.5 - x[0] + x[0] * x[1]) * x[0]
            + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * 2 * x[0] * x[1]
            + 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * 3 * x[0] * x[1] ** 2
        )

        return np.array([dx, dy])
