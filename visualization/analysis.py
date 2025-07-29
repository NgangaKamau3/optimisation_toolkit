import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyze_convergence(results_dict, func):
    """Analyze convergence properties of different optimizers"""
    analysis = {}

    for name, result in results_dict.items():
        values = [func.evaluate(point) for point in result["history"]]

        # Calculate convergence metrics
        final_error = (
            abs(values[-1] - func.evaluate(func.global_minimum))
            if hasattr(func, "global_minimum")
            else values[-1]
        )

        analysis[name] = {
            "iterations": len(values),
            "final_value": values[-1],
            "final_error": final_error,
            "convergence_rate": _estimate_convergence_rate(values),
            "path_length": _calculate_path_length(result["history"]),
        }

    # Create summary DataFrame
    df = pd.DataFrame(analysis).T

    # Plot analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Iterations comparison
    df["iterations"].plot(kind="bar", ax=axes[0, 0], title="Iterations to Convergence")
    axes[0, 0].set_ylabel("Iterations")

    # Final error comparison
    df["final_error"].plot(kind="bar", ax=axes[0, 1], title="Final Error", logy=True)
    axes[0, 1].set_ylabel("Error (log scale)")

    # Convergence rate comparison
    df["convergence_rate"].plot(kind="bar", ax=axes[1, 0], title="Convergence Rate")
    axes[1, 0].set_ylabel("Rate")

    # Path length comparison
    df["path_length"].plot(kind="bar", ax=axes[1, 1], title="Path Length")
    axes[1, 1].set_ylabel("Distance")

    plt.tight_layout()

    return df, fig


def _estimate_convergence_rate(values):
    """Estimate linear convergence rate"""
    if len(values) < 10:
        return np.nan

    log_values = np.log(np.maximum(values[-10:], 1e-15))
    if len(log_values) < 2:
        return np.nan

    # Fit linear regression to log values
    x = np.arange(len(log_values))
    slope = np.polyfit(x, log_values, 1)[0]
    return -slope


def _calculate_path_length(history):
    """Calculate total path length"""
    if len(history) < 2:
        return 0

    distances = [
        np.linalg.norm(history[i + 1] - history[i]) for i in range(len(history) - 1)
    ]
    return sum(distances)
