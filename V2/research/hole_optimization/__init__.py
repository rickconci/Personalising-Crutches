"""Professional hole optimization framework for crutch design.

This package provides a comprehensive framework for optimizing hole placements
on crutch rods to maximize geometry vocabulary while minimizing manufacturing complexity.

Key Features:
- Multiple optimization algorithms (differentiable, genetic, simulated annealing, etc.)
- Professional configuration management
- Comprehensive benchmarking and visualization
- Modular architecture for easy extension

Example Usage:
    from hole_optimization import ConfigManager, create_optimizer
    from hole_optimization.config import OptimizerType
    
    # Load configuration
    config = ConfigManager.load_config("config.yaml")
    
    # Create optimizer
    optimizer = create_optimizer(
        OptimizerType.DIFFERENTIABLE,
        config.constraints,
        config.objectives,
        config.differentiable
    )
    
    # Run optimization
    result = optimizer.optimize()
    
    # Visualize results
    from hole_optimization.visualization import plot_optimization_results
    plot_optimization_results(result, "output/")
"""

from .config import (
    ConfigManager,
    ExperimentConfig,
    OptimizerType,
    CrutchConstraints,
    OptimizationObjectives
)
from .geometry import HoleLayout, Geometry, CrutchGeometry, create_uniform_holes
from .optimizers import create_optimizer, BaseOptimizer, OptimizationResult
from .visualization import plot_optimization_results, save_results_summary
from .benchmarking import run_benchmark, compare_optimizers

__version__ = "1.0.0"
__author__ = "Crutch Optimization Team"

__all__ = [
    # Configuration
    'ConfigManager',
    'ExperimentConfig', 
    'OptimizerType',
    'CrutchConstraints',
    'OptimizationObjectives',
    
    # Geometry
    'HoleLayout',
    'Geometry',
    'CrutchGeometry',
    'create_uniform_holes',
    
    # Optimization
    'create_optimizer',
    'BaseOptimizer',
    'OptimizationResult',
    
    # Visualization
    'plot_optimization_results',
    'save_results_summary',
    
    # Benchmarking
    'run_benchmark',
    'compare_optimizers',
]
