"""Factory for creating optimizers."""

from typing import Type, Dict, Any
from ..config import (
    OptimizerType, 
    CrutchConstraints, 
    OptimizationObjectives,
    DifferentiableConfig,
    GeneticAlgorithmConfig,
    SimulatedAnnealingConfig,
    IntegerProgrammingConfig,
    HybridConfig
)
from .base import BaseOptimizer
from .differentiable import DifferentiableOptimizer


def create_optimizer(
    optimizer_type: OptimizerType,
    constraints: CrutchConstraints,
    objectives: OptimizationObjectives,
    config: Dict[str, Any],
    random_seed: int = 42
) -> BaseOptimizer:
    """Create optimizer instance based on type and configuration.
    
    Args:
        optimizer_type: Type of optimizer to create
        constraints: Physical constraints
        objectives: Optimization objectives
        config: Optimizer-specific configuration
        random_seed: Random seed
        
    Returns:
        Optimizer instance
        
    Raises:
        ValueError: If optimizer type is not supported
        ImportError: If required dependencies are missing
    """
    if optimizer_type == OptimizerType.DIFFERENTIABLE:
        diff_config = DifferentiableConfig(**config) if isinstance(config, dict) else config
        return DifferentiableOptimizer(constraints, objectives, diff_config, random_seed)
    
    elif optimizer_type == OptimizerType.GENETIC_ALGORITHM:
        # Import here to avoid dependency issues if not available
        try:
            from .genetic import GeneticAlgorithmOptimizer
        except ImportError as e:
            raise ImportError(f"Genetic algorithm dependencies not available: {e}")
        
        ga_config = GeneticAlgorithmConfig(**config) if isinstance(config, dict) else config
        return GeneticAlgorithmOptimizer(constraints, objectives, ga_config, random_seed)
    
    elif optimizer_type == OptimizerType.SIMULATED_ANNEALING:
        try:
            from .simulated_annealing import SimulatedAnnealingOptimizer
        except ImportError as e:
            raise ImportError(f"Simulated annealing dependencies not available: {e}")
        
        sa_config = SimulatedAnnealingConfig(**config) if isinstance(config, dict) else config
        return SimulatedAnnealingOptimizer(constraints, objectives, sa_config, random_seed)
    
    elif optimizer_type == OptimizerType.INTEGER_PROGRAMMING:
        try:
            from .integer_programming import IntegerProgrammingOptimizer
        except ImportError as e:
            raise ImportError(f"Integer programming dependencies not available: {e}")
        
        ip_config = IntegerProgrammingConfig(**config) if isinstance(config, dict) else config
        return IntegerProgrammingOptimizer(constraints, objectives, ip_config, random_seed)
    
    elif optimizer_type == OptimizerType.HYBRID:
        try:
            from .hybrid import HybridOptimizer
        except ImportError as e:
            raise ImportError(f"Hybrid optimizer dependencies not available: {e}")
        
        hybrid_config = HybridConfig(**config) if isinstance(config, dict) else config
        return HybridOptimizer(constraints, objectives, hybrid_config, random_seed)
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def get_available_optimizers() -> Dict[OptimizerType, bool]:
    """Check which optimizers are available based on installed dependencies.
    
    Returns:
        Dictionary mapping optimizer types to availability
    """
    availability = {}
    
    # Differentiable is always available (uses JAX which should be installed)
    availability[OptimizerType.DIFFERENTIABLE] = True
    
    # Check genetic algorithm
    try:
        import deap  # or whatever GA library we use
        availability[OptimizerType.GENETIC_ALGORITHM] = True
    except ImportError:
        availability[OptimizerType.GENETIC_ALGORITHM] = False
    
    # Check simulated annealing (usually just numpy/scipy)
    try:
        import scipy.optimize
        availability[OptimizerType.SIMULATED_ANNEALING] = True
    except ImportError:
        availability[OptimizerType.SIMULATED_ANNEALING] = False
    
    # Check integer programming
    try:
        import pyscipopt  # or pulp, or ortools
        availability[OptimizerType.INTEGER_PROGRAMMING] = True
    except ImportError:
        availability[OptimizerType.INTEGER_PROGRAMMING] = False
    
    # Hybrid depends on at least one other being available
    availability[OptimizerType.HYBRID] = any(
        availability.get(opt_type, False) 
        for opt_type in [OptimizerType.GENETIC_ALGORITHM, OptimizerType.SIMULATED_ANNEALING]
    )
    
    return availability


def list_optimizer_configs() -> Dict[OptimizerType, Type]:
    """Get configuration classes for each optimizer type.
    
    Returns:
        Dictionary mapping optimizer types to config classes
    """
    return {
        OptimizerType.DIFFERENTIABLE: DifferentiableConfig,
        OptimizerType.GENETIC_ALGORITHM: GeneticAlgorithmConfig,
        OptimizerType.SIMULATED_ANNEALING: SimulatedAnnealingConfig,
        OptimizerType.INTEGER_PROGRAMMING: IntegerProgrammingConfig,
        OptimizerType.HYBRID: HybridConfig,
    }
