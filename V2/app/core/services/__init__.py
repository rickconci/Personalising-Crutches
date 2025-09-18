"""
Service layer for business logic.

This module contains service classes that encapsulate business logic
and provide a clean interface between the API layer and data layer.
"""

# Import only the essential services that don't have heavy dependencies
from .experiment_service import ExperimentService

# Lazy imports for services with heavy dependencies to avoid startup issues
def get_data_processing_service():
    """Lazy import for DataProcessingService to avoid NumPy/SciPy import issues."""
    from .data_processing_service import DataProcessingService
    return DataProcessingService

def get_optimization_service():
    """Lazy import for OptimizationService to avoid heavy dependency imports."""
    from .optimization_service import OptimizationService
    return OptimizationService

__all__ = [
    "ExperimentService",
    "get_data_processing_service",
    "get_optimization_service",
]
