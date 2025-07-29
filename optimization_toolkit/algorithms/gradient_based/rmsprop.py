"""Production-ready RMSProp optimizer."""

from typing import Optional
import numpy as np

from ...core.base_optimizer import BaseOptimizer


class RMSProp(BaseOptimizer):
    """
    RMSProp (Root Mean Square Propagation) optimizer.
    
    RMSProp maintains a moving average of the squared gradients to normalize
    the gradient, which helps with convergence.
    
    Parameters:
        learning_rate: Step size for parameter updates (default: 0.001)
        alpha: Smoothing constant for moving average (default: 0.99)
        epsilon: Small constant for numerical stability (default: 1e-8)
        max_iterations: Maximum number of optimization iterations (default: 1000)
        tolerance: Convergence tolerance for gradient norm (default: 1e-6)
        verbose: Enable detailed logging (default: False)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        alpha: float = 0.99,
        epsilon: float = 1e-8,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False
    ):
        """Initialize RMSProp optimizer with validated parameters."""
        if not 0 <= alpha < 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        
        super().__init__(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose
        )
        
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self._v: Optional[np.ndarray] = None
    
    def _reset_state(self) -> None:
        """Reset optimizer internal state including moving average."""
        super()._reset_state()
        self._v = None
    
    def _update_parameters(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Update parameters using RMSProp algorithm."""
        # Initialize moving average on first call
        if self._v is None or self._v.shape != x.shape:
            self._v = np.zeros_like(x)
        
        # Update moving average of squared gradients
        self._v = self.alpha * self._v + (1 - self.alpha) * gradient**2
        
        # Update parameters
        update = self.learning_rate * gradient / (np.sqrt(self._v) + self.epsilon)
        
        return x - update
    
    def __repr__(self) -> str:
        """String representation of RMSProp optimizer."""
        return (f"RMSProp(learning_rate={self.learning_rate}, "
                f"alpha={self.alpha}, epsilon={self.epsilon})")