import numpy as np
import matplotlib.pyplot as plt


def plot_landscape(func, resolution=100, plot_3d=False):
    """Plot function landscape"""
    if not hasattr(func, "bounds") or len(func.bounds) != 2:
        raise ValueError("Function must have 2D bounds for landscape plotting")

    x_range = np.linspace(func.bounds[0][0], func.bounds[0][1], resolution)
    y_range = np.linspace(func.bounds[1][0], func.bounds[1][1], resolution)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[func.evaluate([x, y]) for x in x_range] for y in y_range])

    if plot_3d:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x1, x2)")
        plt.colorbar(surf)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
        ax.contour(X, Y, Z, levels=20, colors="black", alpha=0.3, linewidths=0.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.colorbar(contour)

    if hasattr(func, "global_minimum"):
        if plot_3d:
            ax.scatter(
                func.global_minimum[0],
                func.global_minimum[1],
                func.evaluate(func.global_minimum),
                color="red",
                s=100,
                label="Global Minimum",
            )
        else:
            ax.plot(
                func.global_minimum[0],
                func.global_minimum[1],
                "r*",
                markersize=15,
                label="Global Minimum",
            )
        ax.legend()

    ax.set_title(f"{func.__class__.__name__} Function")
    return fig
