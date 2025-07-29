"""
Optimization Toolkit - Production-Ready Optimization Library

A modular, type-safe optimization framework with comprehensive validation,
error handling, and scientific rigor.
"""

from .core.result import OptimizationResult
from .core.exceptions import OptimizationError, ConvergenceError, ValidationError
from .algorithms.gradient_based import Adam, SGD, RMSProp, LBFGS

__version__ = "2.0.0"
__author__ = "Optimization Toolkit Contributors"
__description__ = "Production-ready optimization library with comprehensive validation and error handling"

__all__ = ["OptimizationResult", "OptimizationError", "ConvergenceError", 
           "ValidationError", "Adam", "SGD", "RMSProp", "LBFGS"]