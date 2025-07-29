"""Production-ready base optimizer with comprehensive validation and logging."""

from typing import Callable, Optional, Dict, Any, List
import numpy as np
import logging
from abc import ABC, abstractmethod

from .result import OptimizationResult
from .validation import validate_inputs, safe_function_call
from .exceptions import ConvergenceError, OptimizationError

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    
    Provides production-ready foundation with:
    - Comprehensive input validation
    - Convergence monitoring
    - Error handling and logging
    - Standardized result format
    - Type safety throughout
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False
    ):
        """
        Initialize base optimizer.
        
        Parameters:
            learning_rate: Step size for parameter updates
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance for gradient norm
            verbose: Enable detailed logging
        """
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.verbose = bool(verbose)
        
        # Initialize state
        self._reset_state()
        
        # Configure logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
    
    def _reset_state(self) -> None:
        """Reset optimizer internal state."""
        self._iteration = 0
        self._function_evaluations = 0
        self._gradient_evaluations = 0
        self._history: List[np.ndarray] = []
        self._function_history: List[float] = []
    
    @abstractmethod
    def _update_parameters(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Algorithm-specific parameter update.
        
        Parameters:
            x: Current parameter values
            gradient: Gradient at current parameters
            
        Returns:
            Updated parameter values
        """
        pass
    
    def _compute_gradient(self, func: Callable, x: np.ndarray) -> np.ndarray:
        """
        Compute numerical gradient using finite differences.
        
        Parameters:
            func: Objective function
            x: Parameter values
            
        Returns:
            Gradient vector
        """
        eps = np.sqrt(np.finfo(float).eps)
        gradient = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            f_plus = safe_function_call(func, x_plus)
            f_minus = safe_function_call(func, x_minus)
            
            gradient[i] = (f_plus - f_minus) / (2 * eps)
            self._gradient_evaluations += 2
        
        return gradient
    
    def _check_convergence(
        self, 
        x_new: np.ndarray, 
        x_old: np.ndarray, 
        gradient: np.ndarray
    ) -> tuple[bool, str]:
        """
        Check convergence criteria.
        
        Parameters:
            x_new: New parameter values
            x_old: Previous parameter values  
            gradient: Current gradient
            
        Returns:
            Tuple of (converged, message)
        """
        # Gradient norm criterion
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < self.tolerance:
            return True, f"Gradient norm {grad_norm:.2e} below tolerance {self.tolerance:.2e}"
        
        # Parameter change criterion
        param_change = np.linalg.norm(x_new - x_old)
        if param_change < self.tolerance:
            return True, f"Parameter change {param_change:.2e} below tolerance {self.tolerance:.2e}"
        
        return False, ""
    
    def minimize(
        self,
        func: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Minimize objective function.
        
        Parameters:
            func: Objective function to minimize
            x0: Initial parameter values
            bounds: Optional parameter bounds
            
        Returns:
            OptimizationResult with solution and metadata
            
        Raises:
            OptimizationError: If optimization fails
            ConvergenceError: If convergence criteria not met
        """
        # Validate inputs
        x0, bounds = validate_inputs(func, x0, bounds)
        
        # Reset state for new optimization
        self._reset_state()
        
        # Initialize
        x = x0.copy()
        
        if self.verbose:
            logger.info(f"Starting optimization with {self.__class__.__name__}")
            logger.info(f"Initial point: {x}")
        
        try:
            for iteration in range(self.max_iterations):
                self._iteration = iteration
                
                # Evaluate function
                f_val = safe_function_call(func, x)
                self._function_evaluations += 1
                
                # Store history
                self._history.append(x.copy())
                self._function_history.append(f_val)
                
                # Compute gradient
                gradient = self._compute_gradient(func, x)
                
                # Check convergence
                if iteration > 0:
                    converged, message = self._check_convergence(x, x_old, gradient)
                    if converged:
                        if self.verbose:
                            logger.info(f"Converged after {iteration} iterations: {message}")
                        
                        return OptimizationResult(
                            x=x,
                            fun=f_val,
                            nit=iteration,
                            success=True,
                            message=message,
                            nfev=self._function_evaluations,
                            njev=self._gradient_evaluations
                        )
                
                # Update parameters
                x_old = x.copy()
                x = self._update_parameters(x, gradient)
                
                # Apply bounds if provided
                if bounds is not None:
                    x = np.clip(x, bounds[:, 0], bounds[:, 1])
                
                # Log progress
                if self.verbose and iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: f={f_val:.6e}, ||grad||={np.linalg.norm(gradient):.6e}")
            
            # Max iterations reached
            final_f = safe_function_call(func, x)
            message = f"Maximum iterations ({self.max_iterations}) reached"
            
            if self.verbose:
                logger.warning(message)
            
            return OptimizationResult(
                x=x,
                fun=final_f,
                nit=self.max_iterations,
                success=False,
                message=message,
                nfev=self._function_evaluations,
                njev=self._gradient_evaluations
            )
            
        except Exception as e:
            error_msg = f"Optimization failed: {e}"
            logger.error(error_msg)
            raise OptimizationError(error_msg) from e
    
    @property
    def history(self) -> List[np.ndarray]:
        """Get optimization history."""
        return self._history.copy()
    
    @property
    def function_history(self) -> List[float]:
        """Get function value history."""
        return self._function_history.copy()