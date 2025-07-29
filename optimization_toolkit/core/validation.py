"""Input validation utilities."""

from typing import Callable, Optional, Tuple, Union, List
import numpy as np
from .exceptions import ValidationError, FunctionEvaluationError

ArrayLike = Union[List[float], np.ndarray]


def validate_inputs(
    func: Callable,
    x0: ArrayLike,
    bounds: Optional[ArrayLike] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Validate optimization inputs.
    
    Parameters:
        func: Objective function to minimize
        x0: Initial parameter values
        bounds: Optional parameter bounds
        
    Returns:
        Tuple of validated (x0, bounds)
        
    Raises:
        ValidationError: If inputs are invalid
    """
    # Validate function
    if not callable(func):
        raise ValidationError(f"func must be callable, got {type(func)}")
    
    # Validate initial point
    try:
        x0 = np.asarray(x0, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"x0 must be array-like of floats: {e}")
    
    if x0.ndim != 1:
        raise ValidationError(f"x0 must be 1-D, got shape {x0.shape}")
    
    if x0.size == 0:
        raise ValidationError("x0 cannot be empty")
    
    if not np.all(np.isfinite(x0)):
        raise ValidationError("x0 must contain only finite values")
    
    # Validate bounds if provided
    validated_bounds = None
    if bounds is not None:
        try:
            bounds_array = np.asarray(bounds, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"bounds must be array-like of floats: {e}")
        
        if bounds_array.shape != (len(x0), 2):
            raise ValidationError(
                f"bounds shape {bounds_array.shape} incompatible with "
                f"x0 length {len(x0)}"
            )
        
        if not np.all(bounds_array[:, 0] <= bounds_array[:, 1]):
            raise ValidationError("All lower bounds must be <= upper bounds")
        
        validated_bounds = bounds_array
    
    return x0, validated_bounds


def safe_function_call(func: Callable, x: np.ndarray) -> float:
    """
    Safely evaluate objective function with error handling.
    
    Parameters:
        func: Objective function
        x: Parameter values
        
    Returns:
        Function value
        
    Raises:
        FunctionEvaluationError: If function evaluation fails
    """
    try:
        result = func(x)
        if not np.isscalar(result):
            raise FunctionEvaluationError(
                f"Function must return scalar, got {type(result)}"
            )
        if not np.isfinite(result):
            raise FunctionEvaluationError(
                f"Function returned non-finite value: {result}"
            )
        return float(result)
    except Exception as e:
        raise FunctionEvaluationError(
            f"Function evaluation failed at x={x}: {e}"
        )