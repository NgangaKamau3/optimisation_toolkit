"""Production-ready L-BFGS optimizer."""

from typing import Optional, List
import numpy as np

from ...core.base_optimizer import BaseOptimizer


class LBFGS(BaseOptimizer):
    """
    Limited-memory BFGS (L-BFGS) optimizer.
    
    L-BFGS is a quasi-Newton method that approximates the Broyden–Fletcher–
    Goldfarb–Shanno algorithm using a limited amount of computer memory.
    
    Parameters:
        learning_rate: Step size for parameter updates (default: 1.0)
        memory_size: Number of previous gradients to store (default: 10)
        max_iterations: Maximum number of optimization iterations (default: 1000)
        tolerance: Convergence tolerance for gradient norm (default: 1e-6)
        verbose: Enable detailed logging (default: False)
    """
    
    def __init__(
        self,
        learning_rate: float = 1.0,
        memory_size: int = 10,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False
    ):
        """Initialize L-BFGS optimizer with validated parameters."""
        if memory_size <= 0:
            raise ValueError(f"memory_size must be positive, got {memory_size}")
        
        super().__init__(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose
        )
        
        self.memory_size = int(memory_size)
        
        # L-BFGS memory
        self._s_history: List[np.ndarray] = []  # Parameter differences
        self._y_history: List[np.ndarray] = []  # Gradient differences
        self._rho_history: List[float] = []     # 1 / (y^T s)
        
        self._prev_x: Optional[np.ndarray] = None
        self._prev_grad: Optional[np.ndarray] = None
    
    def _reset_state(self) -> None:
        """Reset optimizer internal state including L-BFGS memory."""
        super()._reset_state()
        self._s_history.clear()
        self._y_history.clear()
        self._rho_history.clear()
        self._prev_x = None
        self._prev_grad = None
    
    def _compute_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        """
        Compute search direction using L-BFGS two-loop recursion.
        
        Parameters:
            gradient: Current gradient
            
        Returns:
            Search direction
        """
        if not self._s_history:
            # No history available, use steepest descent
            return -gradient
        
        # Two-loop recursion
        q = gradient.copy()
        alphas = []
        
        # First loop (backward)
        for i in reversed(range(len(self._s_history))):
            alpha = self._rho_history[i] * np.dot(self._s_history[i], q)
            q -= alpha * self._y_history[i]
            alphas.append(alpha)
        
        alphas.reverse()
        
        # Initial Hessian approximation (identity scaled)
        if self._y_history:
            gamma = (np.dot(self._s_history[-1], self._y_history[-1]) / 
                    np.dot(self._y_history[-1], self._y_history[-1]))
            r = gamma * q
        else:
            r = q
        
        # Second loop (forward)
        for i in range(len(self._s_history)):
            beta = self._rho_history[i] * np.dot(self._y_history[i], r)
            r += self._s_history[i] * (alphas[i] - beta)
        
        return -r
    
    def _update_memory(self, x: np.ndarray, gradient: np.ndarray) -> None:
        """
        Update L-BFGS memory with new parameter and gradient information.
        
        Parameters:
            x: Current parameter values
            gradient: Current gradient
        """
        if self._prev_x is not None and self._prev_grad is not None:
            # Compute differences
            s = x - self._prev_x
            y = gradient - self._prev_grad
            
            # Check curvature condition
            sy = np.dot(s, y)
            if sy > 1e-10:  # Positive curvature
                # Add to memory
                self._s_history.append(s)
                self._y_history.append(y)
                self._rho_history.append(1.0 / sy)
                
                # Maintain memory size limit
                if len(self._s_history) > self.memory_size:
                    self._s_history.pop(0)
                    self._y_history.pop(0)
                    self._rho_history.pop(0)
        
        # Update previous values
        self._prev_x = x.copy()
        self._prev_grad = gradient.copy()
    
    def _update_parameters(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Update parameters using L-BFGS algorithm."""
        # Compute search direction
        direction = self._compute_search_direction(gradient)
        
        # Update memory
        self._update_memory(x, gradient)
        
        # Update parameters
        return x + self.learning_rate * direction
    
    def __repr__(self) -> str:
        """String representation of L-BFGS optimizer."""
        return (f"LBFGS(learning_rate={self.learning_rate}, "
                f"memory_size={self.memory_size})")