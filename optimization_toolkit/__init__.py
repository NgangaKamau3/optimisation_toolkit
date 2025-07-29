"""
Optimization Toolkit - Production-Ready Optimization Library

A modular, type-safe optimization framework with comprehensive validation,
error handling, and scientific rigor.
"""

from .core.result import OptimizationResult
from .core.exceptions import OptimizationError, ConvergenceError
from .algorithms.gradient_based import Adam, SGD, RMSProp, LBFGS

__version__ = "2.0.0"
__all__ = ["OptimizationResult", "OptimizationError", "ConvergenceError", 
           "Adam", "SGD", "RMSProp", "LBFGS"]