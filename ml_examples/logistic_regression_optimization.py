import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers import SGD, Adam, RMSProp, LBFGS
from visualization import plot_comparison


class LogisticRegression:
    """Logistic regression with L2 regularization"""

    def __init__(self, X_train, y_train, lambda_reg=0.01):
        self.X_train = X_train
        self.y_train = y_train
        self.lambda_reg = lambda_reg
        self.n_features = X_train.shape[1]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def evaluate(self, params):
        """Compute logistic loss with L2 regularization"""
        z = self.X_train @ params
        predictions = self.sigmoid(z)

        # Cross-entropy loss
        loss = -np.mean(
            self.y_train * np.log(predictions + 1e-15)
            + (1 - self.y_train) * np.log(1 - predictions + 1e-15)
        )

        # L2 regularization
        reg_loss = self.lambda_reg * np.sum(params**2) / 2

        return loss + reg_loss

    def gradient(self, params):
        """Compute gradient of logistic loss"""
        z = self.X_train @ params
        predictions = self.sigmoid(z)

        # Gradient of cross-entropy
        grad = self.X_train.T @ (predictions - self.y_train) / len(self.y_train)

        # L2 regularization gradient
        grad += self.lambda_reg * params

        return grad

    def predict(self, X, params):
        """Make predictions"""
        z = X @ params
        return self.sigmoid(z)

    def accuracy(self, X, y, params):
        """Calculate accuracy"""
        predictions = self.predict(X, params)
        return np.mean((predictions > 0.5) == y)


def test_on_dataset(dataset_name, X, y):
    """Test optimizers on a specific dataset"""
    print(f"\n{dataset_name} Dataset")
    print("-" * 30)

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

    # Create logistic regression problem
    problem = LogisticRegression(X_train, y_train, lambda_reg=0.01)

    # Initialize parameters
    initial_params = np.random.randn(X_train.shape[1]) * 0.01

    # Test optimizers
    optimizers = {
        "SGD": SGD(lr=0.1, momentum=0.9, max_iter=1000),
        "Adam": Adam(lr=0.01, max_iter=500),
        "RMSProp": RMSProp(lr=0.01, max_iter=500),
        "L-BFGS": LBFGS(lr=1.0, max_iter=200),
    }

    results = {}

    for name, optimizer in optimizers.items():
        print(f"Training with {name}...")
        result = optimizer.minimize(problem, initial_params)
        results[name] = result

        # Evaluate performance
        train_acc = problem.accuracy(X_train, y_train, result["x"])
        test_acc = problem.accuracy(X_test, y_test, result["x"])

        print(f"  Final loss: {result['fun']:.4f}")
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")
        print(f"  Iterations: {result['nit']}")

    return results, problem


def main():
    print("Logistic Regression Optimization on Real Datasets")
    print("=" * 55)

    # Test on Breast Cancer dataset
    cancer_data = load_breast_cancer()
    X_cancer, y_cancer = cancer_data.data, cancer_data.target

    cancer_results, cancer_problem = test_on_dataset(
        "Breast Cancer", X_cancer, y_cancer
    )

    # Test on Wine dataset (convert to binary)
    wine_data = load_wine()
    X_wine, y_wine = wine_data.data, (wine_data.target == 0).astype(
        int
    )  # Class 0 vs others

    wine_results, wine_problem = test_on_dataset("Wine (Binary)", X_wine, y_wine)

    # Plot comparison for both datasets
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Cancer dataset plots
    axes[0, 0].set_title("Breast Cancer - Loss Curves")
    for name, result in cancer_results.items():
        losses = [cancer_problem.evaluate(params) for params in result["history"]]
        axes[0, 0].plot(losses, label=name)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].set_title("Breast Cancer - Loss Curves (Log Scale)")
    for name, result in cancer_results.items():
        losses = [cancer_problem.evaluate(params) for params in result["history"]]
        axes[0, 1].semilogy(losses, label=name)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Loss (log scale)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Wine dataset plots
    axes[1, 0].set_title("Wine - Loss Curves")
    for name, result in wine_results.items():
        losses = [wine_problem.evaluate(params) for params in result["history"]]
        axes[1, 0].plot(losses, label=name)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].set_title("Wine - Loss Curves (Log Scale)")
    for name, result in wine_results.items():
        losses = [wine_problem.evaluate(params) for params in result["history"]]
        axes[1, 1].semilogy(losses, label=name)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Loss (log scale)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("logistic_regression_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nComparison plots saved as 'logistic_regression_comparison.png'")


if __name__ == "__main__":
    main()
