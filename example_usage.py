#!/usr/bin/env python3
"""
Example usage of the optimization toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from optimizers import SGD, Adam, RMSProp, LBFGS
from test_functions import Rosenbrock, Rastrigin, Sphere, Beale
from visualization import (
    plot_convergence,
    plot_comparison,
    plot_landscape,
    analyze_convergence,
)


def main():
    print("Optimization Toolkit Demo")
    print("=" * 50)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # 1. Single optimizer example
    print("\n1. Single Optimizer Example (Adam on Rosenbrock)")
    func = Rosenbrock()
    optimizer = Adam(lr=0.01, max_iter=1000)
    initial_point = [1.5, 1.5]

    result = optimizer.minimize(func, initial_point)
    print(f"Final point: {result['x']}")
    print(f"Final value: {result['fun']:.6f}")
    print(f"Iterations: {result['nit']}")

    # Plot single result
    plot_convergence(result, func, "Adam on Rosenbrock Function")
    plt.savefig("output/single_optimization.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Optimizer comparison
    print("\n2. Optimizer Comparison")
    optimizers = {
        "SGD": SGD(lr=0.001, momentum=0.9, max_iter=2000),
        "Adam": Adam(lr=0.01, max_iter=1000),
        "RMSProp": RMSProp(lr=0.01, max_iter=1000),
        "L-BFGS": LBFGS(lr=1.0, max_iter=500),
    }

    results = {}
    for name, opt in optimizers.items():
        print(f"Running {name}...")
        results[name] = opt.minimize(func, initial_point)

    # Plot comparison
    plot_comparison(results, func)
    plt.savefig("output/optimizer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Convergence analysis
    print("\n3. Convergence Analysis")
    analysis_df, analysis_fig = analyze_convergence(results, func)
    print(analysis_df)
    plt.savefig("output/convergence_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Function landscapes
    print("\n4. Function Landscapes")
    functions = [Rosenbrock(), Rastrigin(), Sphere(), Beale()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, func in enumerate(functions):
        plt.subplot(2, 2, i + 1)
        plot_landscape(func)

    plt.tight_layout()
    plt.savefig("output/function_landscapes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Performance on different functions
    print("\n5. Adam Performance on Different Functions")
    optimizer = Adam(lr=0.01, max_iter=1000)

    for func in functions:
        print(f"\n{func.__class__.__name__} Function:")
        initial_point = np.random.uniform(-1, 1, 2)
        result = optimizer.minimize(func, initial_point)

        error = (
            abs(result["fun"] - func.evaluate(func.global_minimum))
            if hasattr(func, "global_minimum")
            else result["fun"]
        )
        print(f"  Final value: {result['fun']:.6f}")
        print(f"  Error: {error:.6f}")
        print(f"  Iterations: {result['nit']}")

    print("\nAll plots saved to output/ folder:")
    print("- output/single_optimization.png")
    print("- output/optimizer_comparison.png")
    print("- output/convergence_analysis.png")
    print("- output/function_landscapes.png")


if __name__ == "__main__":
    main()
