import numpy as np
from typing import Union, List
from .base import BaseFunction


class Sphere(BaseFunction):
    def __init__(self, n: int = 2) -> None:
        super().__init__(bounds=[(-5, 5)] * n)
        self.n = n
        self.global_minimum = np.zeros(n)

    def evaluate(self, x: Union[List[float], np.ndarray]) -> float:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return float(np.sum(x**2))

    def gradient(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return 2 * x
