"""Core optimization framework components."""

from .result import OptimizationResult
from .exceptions import OptimizationError, ConvergenceError, ValidationError, FunctionEvaluationError
from .base_optimizer import BaseOptimizer
from .validation import validate_inputs, safe_function_call

__all__ = ["OptimizationResult", "OptimizationError", "ConvergenceError", 
           "ValidationError", "FunctionEvaluationError", "BaseOptimizer", 
           "validate_inputs", "safe_function_call"]