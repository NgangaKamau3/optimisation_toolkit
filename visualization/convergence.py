import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(result, func, title=None):
    """Plot convergence path on function landscape"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot landscape with path
    if hasattr(func, "bounds") and len(func.bounds) == 2:
        x_range = np.linspace(func.bounds[0][0], func.bounds[0][1], 100)
        y_range = np.linspace(func.bounds[1][0], func.bounds[1][1], 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array(
            [[func.evaluate(np.array([x, y])) for x in x_range] for y in y_range]
        )

        ax1.contour(X, Y, Z, levels=20, alpha=0.6)
        ax1.plot(result["history"][:, 0], result["history"][:, 1], "ro-", markersize=3)
        ax1.plot(result["x"][0], result["x"][1], "g*", markersize=15, label="Final")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_title("Convergence Path")
        ax1.legend()

    # Plot function value over iterations
    values = [func.evaluate(np.array(point)) for point in result["history"]]
    ax2.semilogy(values)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Function Value (log scale)")
    ax2.set_title("Function Value Convergence")
    ax2.grid(True)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_comparison(results_dict, func):
    """Compare multiple optimization results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    # Plot all paths
    if hasattr(func, "bounds") and len(func.bounds) == 2:
        x_range = np.linspace(func.bounds[0][0], func.bounds[0][1], 100)
        y_range = np.linspace(func.bounds[1][0], func.bounds[1][1], 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array(
            [[func.evaluate(np.array([x, y])) for x in x_range] for y in y_range]
        )

        ax1.contour(X, Y, Z, levels=20, alpha=0.3)

        for (name, result), color in zip(results_dict.items(), colors):
            ax1.plot(
                result["history"][:, 0],
                result["history"][:, 1],
                "o-",
                color=color,
                markersize=2,
                label=name,
            )

        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_title("Convergence Paths Comparison")
        ax1.legend()

    # Plot convergence curves
    for (name, result), color in zip(results_dict.items(), colors):
        values = [func.evaluate(np.array(point)) for point in result["history"]]
        ax2.semilogy(values, color=color, label=name)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Function Value (log scale)")
    ax2.set_title("Convergence Rate Comparison")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig
