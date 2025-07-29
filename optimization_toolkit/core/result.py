"""Standardized optimization result container."""

from typing import Optional
import numpy as np


class OptimizationResult:
    """
    Standardized container for optimization results.
    
    Attributes:
        x (np.ndarray): The solution array
        fun (float): The objective function value at the solution
        nit (int): Number of iterations performed
        success (bool): Whether optimization was successful
        message (str): Termination message
        nfev (int): Number of function evaluations
        njev (int): Number of jacobian evaluations
    """
    
    def __init__(
        self,
        x: np.ndarray,
        fun: float,
        nit: int,
        success: bool,
        message: str,
        nfev: Optional[int] = None,
        njev: Optional[int] = None
    ):
        self.x = np.asarray(x)
        self.fun = float(fun)
        self.nit = int(nit)
        self.success = bool(success)
        self.message = str(message)
        self.nfev = nfev or nit
        self.njev = njev or nit
    
    def __repr__(self) -> str:
        return (f"OptimizationResult(fun={self.fun:.6e}, nit={self.nit}, "
                f"success={self.success})")
    
    def __str__(self) -> str:
        return (f"Optimization Result:\n"
                f"  Success: {self.success}\n"
                f"  Function value: {self.fun:.6e}\n"
                f"  Iterations: {self.nit}\n"
                f"  Message: {self.message}")