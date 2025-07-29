import numpy as np
from .base import BaseOptimizer


class LBFGS(BaseOptimizer):
    def __init__(self, lr=1.0, m=10, **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.m = m
        self.s_list = []
        self.y_list = []
        self.rho_list = []
        self.prev_params = None
        self.prev_grad = None

    def step(self, params, grad):
        if self.prev_params is not None:
            s = params - self.prev_params
            y = grad - self.prev_grad

            if np.dot(y, s) > 1e-10:
                if len(self.s_list) >= self.m:
                    self.s_list.pop(0)
                    self.y_list.pop(0)
                    self.rho_list.pop(0)

                self.s_list.append(s)
                self.y_list.append(y)
                self.rho_list.append(1.0 / np.dot(y, s))

        direction = self._two_loop_recursion(grad)

        self.prev_params = params.copy()
        self.prev_grad = grad.copy()

        return params - self.lr * direction

    def _two_loop_recursion(self, grad):
        q = grad.copy()
        alpha = []

        for i in reversed(range(len(self.s_list))):
            a = self.rho_list[i] * np.dot(self.s_list[i], q)
            alpha.append(a)
            q = q - a * self.y_list[i]

        if len(self.y_list) > 0:
            gamma = np.dot(self.s_list[-1], self.y_list[-1]) / np.dot(
                self.y_list[-1], self.y_list[-1]
            )
            r = gamma * q
        else:
            r = q

        for i in range(len(self.s_list)):
            beta = self.rho_list[i] * np.dot(self.y_list[i], r)
            r = r + self.s_list[i] * (alpha[-(i + 1)] - beta)

        return r
