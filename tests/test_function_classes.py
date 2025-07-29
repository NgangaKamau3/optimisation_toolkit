import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_functions import Rosenbrock, Rastrigin, Sphere, Beale


class TestFunctions:
    def test_sphere_properties(self):
        func = Sphere(n=2)

        # Test global minimum
        assert func.evaluate([0, 0]) == 0
        assert np.allclose(func.gradient([0, 0]), [0, 0])

        # Test other points
        assert func.evaluate([1, 1]) == 2
        assert np.allclose(func.gradient([1, 1]), [2, 2])

    def test_rosenbrock_properties(self):
        func = Rosenbrock()

        # Test global minimum at [1, 1]
        assert func.evaluate([1, 1]) == 0
        assert np.allclose(func.gradient([1, 1]), [0, 0], atol=1e-10)

        # Test gradient computation
        grad = func.gradient([0, 0])
        assert len(grad) == 2

    def test_rastrigin_properties(self):
        func = Rastrigin(n=2)

        # Test global minimum at [0, 0]
        assert func.evaluate([0, 0]) == 0
        assert np.allclose(func.gradient([0, 0]), [0, 0])

        # Test multimodal nature
        assert func.evaluate([1, 1]) > 0

    def test_beale_properties(self):
        func = Beale()

        # Test global minimum at [3, 0.5]
        min_val = func.evaluate([3, 0.5])
        assert min_val < 1e-10

        # Test gradient at minimum
        grad = func.gradient([3, 0.5])
        assert np.allclose(grad, [0, 0], atol=1e-6)

    def test_gradient_accuracy(self):
        """Test gradient accuracy using finite differences"""
        func = Rosenbrock()
        x = np.array([0.5, 0.5])

        # Analytical gradient
        grad_analytical = func.gradient(x)

        # Numerical gradient
        eps = 1e-8
        grad_numerical = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad_numerical[i] = (func.evaluate(x_plus) - func.evaluate(x_minus)) / (
                2 * eps
            )

        assert np.allclose(grad_analytical, grad_numerical, rtol=1e-5)
