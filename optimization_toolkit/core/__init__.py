"""Core optimization framework components."""

from .result import OptimizationResult
from .exceptions import OptimizationError, ConvergenceError
from .base_optimizer import BaseOptimizer
from .validation import validate_inputs

__all__ = ["OptimizationResult", "OptimizationError", "ConvergenceError", 
           "BaseOptimizer", "validate_inputs"]