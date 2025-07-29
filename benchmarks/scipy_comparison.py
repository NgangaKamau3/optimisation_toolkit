import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from optimizers import SGD, Adam, RMSProp, LBFGS
from test_functions import Rosenbrock, Rastrigin, Sphere, Beale


def benchmark_optimizers():
    """Benchmark custom optimizers against SciPy"""
    functions = [Rosenbrock(), Rastrigin(), Sphere(), Beale()]
    custom_optimizers = {
        "SGD": SGD(lr=0.01, momentum=0.9),
        "Adam": Adam(lr=0.01),
        "RMSProp": RMSProp(lr=0.01),
        "L-BFGS": LBFGS(lr=1.0),
    }

    results = {}

    for func in functions:
        func_name = func.__class__.__name__
        results[func_name] = {}

        initial_point = np.random.uniform(-1, 1, 2)

        # Test custom optimizers
        for opt_name, optimizer in custom_optimizers.items():
            start_time = time.time()
            result = optimizer.minimize(func, initial_point)
            end_time = time.time()

            results[func_name][f"Custom_{opt_name}"] = {
                "final_value": result["fun"],
                "iterations": result["nit"],
                "time": end_time - start_time,
                "success": result["fun"] < 1e-6,
            }

        # Test SciPy optimizers
        scipy_methods = ["BFGS", "L-BFGS-B", "CG"]

        for method in scipy_methods:
            start_time = time.time()
            result = minimize(
                func.evaluate, initial_point, jac=func.gradient, method=method
            )
            end_time = time.time()

            results[func_name][f"SciPy_{method}"] = {
                "final_value": result.fun,
                "iterations": result.nit if hasattr(result, "nit") else 0,
                "time": end_time - start_time,
                "success": result.success,
            }

    return results


def print_benchmark_results(results):
    """Print formatted benchmark results"""
    for func_name, func_results in results.items():
        print(f"\n{func_name} Function:")
        print("-" * 50)
        print(f"{'Method':<15} {'Value':<12} {'Iter':<6} {'Time':<8} {'Success'}")
        print("-" * 50)

        for method, result in func_results.items():
            print(
                f"{method:<15} {result['final_value']:<12.2e} "
                f"{result['iterations']:<6} {result['time']:<8.3f} "
                f"{result['success']}"
            )


def plot_benchmark_results(results):
    """Plot benchmark comparison"""
    os.makedirs("output", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (func_name, func_results) in enumerate(results.items()):
        methods = list(func_results.keys())
        values = [func_results[method]["final_value"] for method in methods]

        axes[i].bar(range(len(methods)), values)
        axes[i].set_yscale("log")
        axes[i].set_title(f"{func_name} Function")
        axes[i].set_xticks(range(len(methods)))
        axes[i].set_xticklabels(methods, rotation=45, ha="right")
        axes[i].set_ylabel("Final Value (log scale)")

    plt.tight_layout()
    plt.savefig("output/scipy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Main function for command line execution"""
    results = benchmark_optimizers()
    print_benchmark_results(results)
    plot_benchmark_results(results)
    print("\nBenchmark plot saved to output/scipy_comparison.png")
    return results


if __name__ == "__main__":
    main()
