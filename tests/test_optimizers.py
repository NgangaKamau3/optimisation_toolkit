import pytest
import numpy as np
from optimizers import SGD, Adam, RMSProp, LBFGS
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_functions import Sphere, Rosenbrock


class TestOptimizers:
    def test_sgd_sphere(self):
        func = Sphere(n=2)
        optimizer = SGD(lr=0.1, max_iter=100)
        result = optimizer.minimize(func, [1.0, 1.0])

        assert result["fun"] < 0.1
        assert len(result["history"]) <= 100

    def test_adam_sphere(self):
        func = Sphere(n=2)
        optimizer = Adam(lr=0.1, max_iter=100)
        result = optimizer.minimize(func, [1.0, 1.0])

        assert result["fun"] < 0.01
        assert (
            len(result["history"]) <= 101
        )  # Allow for off-by-one in iteration counting

    def test_rmsprop_sphere(self):
        func = Sphere(n=2)
        optimizer = RMSProp(lr=0.1, max_iter=100)
        result = optimizer.minimize(func, [1.0, 1.0])

        assert result["fun"] < 0.01
        assert len(result["history"]) <= 100

    def test_lbfgs_sphere(self):
        func = Sphere(n=2)
        optimizer = LBFGS(lr=1.0, max_iter=50)
        result = optimizer.minimize(func, [1.0, 1.0])

        assert result["fun"] < 0.001
        assert len(result["history"]) <= 50

    def test_convergence_tolerance(self):
        func = Sphere(n=2)
        optimizer = Adam(lr=0.1, tol=1e-3)
        result = optimizer.minimize(func, [1.0, 1.0])

        # Should stop early due to tolerance
        assert len(result["history"]) < 1000

    def test_rosenbrock_convergence(self):
        func = Rosenbrock()
        optimizer = Adam(lr=0.01, max_iter=1000)
        result = optimizer.minimize(func, [0.0, 0.0])

        # Should get reasonably close to global minimum [1, 1]
        assert np.linalg.norm(result["x"] - [1, 1]) < 0.5
