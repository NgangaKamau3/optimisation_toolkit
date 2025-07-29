import time
import memory_profiler
import numpy as np
import os
from typing import Dict, List
import matplotlib.pyplot as plt

from optimizers import SGD, Adam, RMSProp, LBFGS
from test_functions import Rosenbrock, Rastrigin, Sphere


class PerformanceProfiler:
    """Profile optimizer performance across dimensions and problem types"""

    def __init__(self):
        self.results = {}

    def profile_scalability(self, dimensions: List[int] = [2, 5, 10, 20, 50]) -> Dict:
        """Profile optimizer performance across different dimensions"""
        optimizers = {
            "SGD": SGD(lr=0.01, momentum=0.9, max_iter=1000),
            "Adam": Adam(lr=0.01, max_iter=1000),
            "RMSProp": RMSProp(lr=0.01, max_iter=1000),
        }

        results = {
            name: {"dimensions": [], "times": [], "memory": []}
            for name in optimizers.keys()
        }

        for dim in dimensions:
            func = Sphere(n=dim)
            initial_point = np.random.uniform(-1, 1, dim)

            for name, optimizer in optimizers.items():
                # Time profiling
                start_time = time.time()
                result = optimizer.minimize(func, initial_point)
                end_time = time.time()

                # Memory profiling
                mem_usage = memory_profiler.memory_usage(
                    (optimizer.minimize, (func, initial_point)), interval=0.1
                )

                results[name]["dimensions"].append(dim)
                results[name]["times"].append(end_time - start_time)
                results[name]["memory"].append(max(mem_usage) - min(mem_usage))

        return results

    def profile_convergence_speed(self) -> Dict:
        """Profile convergence speed on different problem types"""
        functions = {
            "Sphere": Sphere(n=2),
            "Rosenbrock": Rosenbrock(),
            "Rastrigin": Rastrigin(n=2),
        }

        optimizers = {
            "SGD": SGD(lr=0.01, momentum=0.9, max_iter=2000),
            "Adam": Adam(lr=0.01, max_iter=1000),
            "RMSProp": RMSProp(lr=0.01, max_iter=1000),
            "L-BFGS": LBFGS(lr=1.0, max_iter=500),
        }

        results = {}

        for func_name, func in functions.items():
            results[func_name] = {}
            initial_point = np.random.uniform(-1, 1, 2)

            for opt_name, optimizer in optimizers.items():
                start_time = time.time()
                result = optimizer.minimize(func, initial_point)
                end_time = time.time()

                # Calculate convergence metrics
                values = [func.evaluate(point) for point in result["history"]]
                convergence_iter = self._find_convergence_iteration(values)

                results[func_name][opt_name] = {
                    "time": end_time - start_time,
                    "iterations": result["nit"],
                    "final_value": result["fun"],
                    "convergence_iter": convergence_iter,
                    "convergence_rate": (
                        len(values) / convergence_iter
                        if convergence_iter
                        else float("inf")
                    ),
                }

        return results

    def _find_convergence_iteration(
        self, values: List[float], threshold: float = 1e-6
    ) -> int:
        """Find iteration where convergence threshold is reached"""
        for i, val in enumerate(values):
            if val < threshold:
                return i + 1
        return len(values)

    def plot_scalability_results(self, results: Dict):
        """Plot scalability profiling results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Time scaling
        for name, data in results.items():
            ax1.loglog(data["dimensions"], data["times"], "o-", label=name)
        ax1.set_xlabel("Problem Dimension")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("Time Complexity Scaling")
        ax1.legend()
        ax1.grid(True)

        # Memory scaling
        for name, data in results.items():
            ax2.loglog(data["dimensions"], data["memory"], "s-", label=name)
        ax2.set_xlabel("Problem Dimension")
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.set_title("Memory Complexity Scaling")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig


def main():
    """Run performance profiling suite"""
    profiler = PerformanceProfiler()

    print("Running scalability profiling...")
    scalability_results = profiler.profile_scalability()

    print("Running convergence speed profiling...")
    convergence_results = profiler.profile_convergence_speed()

    # Plot results
    os.makedirs("output", exist_ok=True)
    profiler.plot_scalability_results(scalability_results)
    plt.savefig("output/scalability_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print convergence results
    print("\nConvergence Speed Results:")
    print("=" * 50)
    for func_name, func_results in convergence_results.items():
        print(f"\n{func_name}:")
        for opt_name, metrics in func_results.items():
            print(
                f"  {opt_name:10}: {metrics['time']:.3f}s, "
                f"{metrics['iterations']:4d} iter, "
                f"final: {metrics['final_value']:.2e}"
            )

    print("\nPlots saved to output/ folder:")
    print("- output/scalability_profile.png")


if __name__ == "__main__":
    main()
