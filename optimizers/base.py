import numpy as np
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    @abstractmethod
    def step(self, params, grad):
        pass

    def minimize(self, func, initial_point, callback=None):
        params = np.array(initial_point, dtype=float)
        self.history = [params.copy()]

        for i in range(self.max_iter):
            grad = func.gradient(params)

            if np.linalg.norm(grad) < self.tol:
                break

            params = self.step(params, grad)
            self.history.append(params.copy())

            if callback:
                callback(i, params, func.evaluate(params))

        return {
            "x": params,
            "fun": func.evaluate(params),
            "nit": len(self.history),
            "history": np.array(self.history),
        }
