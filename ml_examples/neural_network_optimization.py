import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from optimizers import SGD, Adam, RMSProp
from visualization import plot_comparison


class NeuralNetwork:
    """Simple 2-layer neural network for binary classification"""

    def __init__(self, input_size, hidden_size=10):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_params(self, params):
        idx = 0
        W1_size = self.W1.size
        self.W1 = params[idx : idx + W1_size].reshape(self.W1.shape)
        idx += W1_size

        b1_size = self.b1.size
        self.b1 = params[idx : idx + b1_size]
        idx += b1_size

        W2_size = self.W2.size
        self.W2 = params[idx : idx + W2_size].reshape(self.W2.shape)
        idx += W2_size

        self.b2 = params[idx:]


class MLOptimizationProblem:
    """Wrapper to make ML training compatible with our optimizers"""

    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def evaluate(self, params):
        self.model.set_params(params)
        predictions = self.model.forward(self.X_train)
        # Binary cross-entropy loss
        loss = -np.mean(
            self.y_train * np.log(predictions + 1e-15)
            + (1 - self.y_train) * np.log(1 - predictions + 1e-15)
        )
        return loss

    def gradient(self, params):
        self.model.set_params(params)
        m = self.X_train.shape[0]

        # Forward pass
        predictions = self.model.forward(self.X_train)

        # Backward pass
        dz2 = predictions - self.y_train.reshape(-1, 1)
        dW2 = self.model.a1.T @ dz2 / m
        db2 = np.mean(dz2, axis=0)

        da1 = dz2 @ self.model.W2.T
        dz1 = da1 * self.model.a1 * (1 - self.model.a1)
        dW1 = self.X_train.T @ dz1 / m
        db1 = np.mean(dz1, axis=0)

        return np.concatenate([dW1.flatten(), db1, dW2.flatten(), db2])


def main():
    print("Neural Network Optimization with Real Dataset")
    print("=" * 50)

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Dataset: {X_train.shape[0]} training samples, {X_train.shape[1]} features")

    # Create neural network
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=10)
    problem = MLOptimizationProblem(model, X_train, y_train)

    # Test different optimizers
    optimizers = {
        "SGD": SGD(lr=0.1, momentum=0.9, max_iter=500),
        "Adam": Adam(lr=0.01, max_iter=500),
        "RMSProp": RMSProp(lr=0.01, max_iter=500),
    }

    results = {}
    initial_params = model.get_params()

    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name}...")
        result = optimizer.minimize(problem, initial_params)
        results[name] = result

        # Evaluate final model
        model.set_params(result["x"])
        train_pred = model.forward(X_train)
        train_acc = np.mean((train_pred > 0.5) == y_train.reshape(-1, 1))

        test_pred = model.forward(X_test)
        test_acc = np.mean((test_pred > 0.5) == y_test.reshape(-1, 1))

        print(f"  Final loss: {result['fun']:.4f}")
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")
        print(f"  Iterations: {result['nit']}")

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, result in results.items():
        losses = [problem.evaluate(params) for params in result["history"]]
        plt.plot(losses, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, result in results.items():
        losses = [problem.evaluate(params) for params in result["history"]]
        plt.semilogy(losses, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss Curves (Log Scale)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("neural_network_training.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nTraining curves saved as 'neural_network_training.png'")


if __name__ == "__main__":
    main()
