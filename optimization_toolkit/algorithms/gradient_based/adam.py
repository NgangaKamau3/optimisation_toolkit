"""
Production-ready Adam optimizer implementation.

References:
    Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
    arXiv preprint arXiv:1412.6980.
"""

from typing import Optional
import numpy as np

from ...core.base_optimizer import BaseOptimizer


class Adam(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Adam combines the advantages of AdaGrad and RMSProp by computing adaptive
    learning rates for each parameter using estimates of first and second moments
    of the gradients.
    
    Parameters:
        learning_rate: Step size for parameter updates (default: 0.001)
        beta1: Exponential decay rate for first moment estimates (default: 0.9)
        beta2: Exponential decay rate for second moment estimates (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
        max_iterations: Maximum number of optimization iterations (default: 1000)
        tolerance: Convergence tolerance for gradient norm (default: 1e-6)
        verbose: Enable detailed logging (default: False)
    
    Example:
        >>> from optimization_toolkit import Adam
        >>> optimizer = Adam(learning_rate=0.01, verbose=True)
        >>> result = optimizer.minimize(rosenbrock, [1.5, 1.5])
        >>> print(f"Minimum found at: {result.x}")
        >>> print(f"Function value: {result.fun:.6e}")
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False
    ):
        """Initialize Adam optimizer with validated parameters."""
        # Validate Adam-specific parameters
        if not 0 <= beta1 < 1:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        
        super().__init__(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose
        )
        
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        
        # Initialize moment estimates
        self._m: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._t = 0  # Time step
    
    def _reset_state(self) -> None:
        """Reset optimizer internal state including moment estimates."""
        super()._reset_state()
        self._m = None
        self._v = None
        self._t = 0
    
    def _update_parameters(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Update parameters using Adam algorithm.
        
        Parameters:
            x: Current parameter values
            gradient: Gradient at current parameters
            
        Returns:
            Updated parameter values
        """
        # Initialize moment estimates on first call
        if self._m is None or self._m.shape != x.shape:
            self._m = np.zeros_like(x)
            self._v = np.zeros_like(x)
            self._t = 0
        
        # Increment time step
        self._t += 1
        
        # Update biased first moment estimate
        self._m = self.beta1 * self._m + (1 - self.beta1) * gradient
        
        # Update biased second raw moment estimate
        self._v = self.beta2 * self._v + (1 - self.beta2) * gradient**2
        
        # Compute bias-corrected first moment estimate
        m_hat = self._m / (1 - self.beta1**self._t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self._v / (1 - self.beta2**self._t)
        
        # Update parameters
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return x - update
    
    def __repr__(self) -> str:
        """String representation of Adam optimizer."""
        return (f"Adam(learning_rate={self.learning_rate}, "
                f"beta1={self.beta1}, beta2={self.beta2}, "
                f"epsilon={self.epsilon})")