"""Custom exceptions for optimization toolkit."""


class OptimizationError(Exception):
    """Base exception for optimization-related errors."""
    pass


class ConvergenceError(OptimizationError):
    """Raised when optimization fails to converge."""
    pass


class ValidationError(OptimizationError):
    """Raised when input validation fails."""
    pass


class FunctionEvaluationError(OptimizationError):
    """Raised when objective function evaluation fails."""
    pass