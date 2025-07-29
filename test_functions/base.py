import numpy as np
from abc import ABC, abstractmethod


class BaseFunction(ABC):
    def __init__(self, bounds=None):
        self.bounds = bounds

    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    def hessian(self, x, eps=1e-8):
        """Numerical Hessian approximation"""
        n = len(x)
        H = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()

                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps

                H[i, j] = (
                    self.evaluate(x_pp)
                    - self.evaluate(x_pm)
                    - self.evaluate(x_mp)
                    + self.evaluate(x_mm)
                ) / (4 * eps**2)

        return H
