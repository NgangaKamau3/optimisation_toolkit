import pytest
import numpy as np
import time
from optimizers import Adam, SGD
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_functions import Sphere


class TestPerformance:
    """Performance regression tests"""

    @pytest.mark.benchmark
    def test_adam_sphere_benchmark(self, benchmark):
        """Benchmark Adam optimizer on Sphere function"""
        func = Sphere(n=10)
        optimizer = Adam(lr=0.1, max_iter=100)
        initial_point = np.ones(10)

        result = benchmark(optimizer.minimize, func, initial_point)
        assert result["fun"] < 0.01

    @pytest.mark.benchmark
    def test_sgd_sphere_benchmark(self, benchmark):
        """Benchmark SGD optimizer on Sphere function"""
        func = Sphere(n=10)
        optimizer = SGD(lr=0.1, max_iter=100)
        initial_point = np.ones(10)

        result = benchmark(optimizer.minimize, func, initial_point)
        assert result["fun"] < 0.1

    @pytest.mark.slow
    def test_memory_usage_bounded(self):
        """Test that memory usage doesn't grow unbounded"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        func = Sphere(n=100)
        optimizer = Adam(lr=0.1, max_iter=1000)
        initial_point = np.random.uniform(-1, 1, 100)

        optimizer.minimize(func, initial_point)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100

    def test_convergence_time_reasonable(self):
        """Test that convergence happens in reasonable time"""
        func = Sphere(n=2)
        optimizer = Adam(lr=0.1, max_iter=1000)
        initial_point = [1.0, 1.0]

        start_time = time.time()
        result = optimizer.minimize(func, initial_point)
        end_time = time.time()

        # Should converge quickly on simple function
        assert end_time - start_time < 1.0  # Less than 1 second
        assert result["fun"] < 1e-6
