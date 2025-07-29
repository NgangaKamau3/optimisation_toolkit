"""Production-ready SGD optimizer with momentum."""

from typing import Optional
import numpy as np

from ...core.base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.
    
    SGD with momentum helps accelerate gradients vectors in the right directions,
    thus leading to faster converging.
    
    Parameters:
        learning_rate: Step size for parameter updates (default: 0.01)
        momentum: Momentum factor (default: 0.0)
        max_iterations: Maximum number of optimization iterations (default: 1000)
        tolerance: Convergence tolerance for gradient norm (default: 1e-6)
        verbose: Enable detailed logging (default: False)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False
    ):
        """Initialize SGD optimizer with validated parameters."""
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        
        super().__init__(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose
        )
        
        self.momentum = float(momentum)
        self._velocity: Optional[np.ndarray] = None
    
    def _reset_state(self) -> None:
        """Reset optimizer internal state including velocity."""
        super()._reset_state()
        self._velocity = None
    
    def _update_parameters(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Update parameters using SGD with momentum."""
        # Initialize velocity on first call
        if self._velocity is None or self._velocity.shape != x.shape:
            self._velocity = np.zeros_like(x)
        
        # Update velocity
        self._velocity = self.momentum * self._velocity - self.learning_rate * gradient
        
        # Update parameters
        return x + self._velocity
    
    def __repr__(self) -> str:
        """String representation of SGD optimizer."""
        return (f"SGD(learning_rate={self.learning_rate}, "
                f"momentum={self.momentum})")