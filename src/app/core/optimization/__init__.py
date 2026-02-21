"""
Bayesian optimization for crutch geometry optimization.

This module provides Bayesian optimization capabilities using Gaussian Processes
to find optimal crutch configurations for individual users.
"""

from .bayesian_optimizer import BayesianOptimizer, OptimizationResult
from .loss_functions import LossFunction, CombinedLossFunction

__all__ = [
    "BayesianOptimizer",
    "OptimizationResult",
    "LossFunction", 
    "CombinedLossFunction",
]
