"""Configuration management for hole optimization framework."""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None


class OptimizerType(Enum):
    """Available optimizer types."""
    DIFFERENTIABLE = "differentiable"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    INTEGER_PROGRAMMING = "integer_programming"
    HYBRID = "hybrid"


class CrutchConstraints(BaseModel):
    """Physical constraints for the crutch geometry."""
    # Rod lengths (cm)
    vertical_length: float = Field(default=20.0, gt=0, description="Length of vertical rod in cm")
    handle_length: float = Field(default=38.0, gt=0, description="Length of handle rod in cm")
    forearm_length: float = Field(default=17.0, gt=0, description="Length of forearm rod in cm")
    
    # Pivot positions (cm from back of handle)
    vertical_pivot_length: float = Field(default=19.0, gt=0, description="Vertical pivot position from back of handle")
    forearm_pivot_length: float = Field(default=19.0, gt=0, description="Forearm pivot position from back of handle")
    
    # Hole constraints
    min_hole_distance: float = Field(default=2.0, gt=0, description="Minimum distance between holes")
    hole_margin: float = Field(default=0.5, ge=0, description="Margin from rod ends")
    
    # Angle constraints (degrees)
    alpha_min: float = Field(default=85.0, ge=0, le=180, description="Minimum alpha angle")
    alpha_max: float = Field(default=115.0, ge=0, le=180, description="Maximum alpha angle")
    beta_min: float = Field(default=95.0, ge=0, le=180, description="Minimum beta angle")
    beta_max: float = Field(default=140.0, ge=0, le=180, description="Maximum beta angle")
    gamma_min: float = Field(default=-9.0, ge=-90, le=90, description="Minimum gamma distance in cm: calculated as vertical_pivot_length - forearm_pivot_length")
    gamma_max: float = Field(default=9.0, ge=-90, le=90, description="Maximum gamma distance in cm")
    
    # Usability constraint
    require_alpha_beta_sum_ge_180: bool = Field(default=True, description="Require alpha + beta >= 180 degrees")
    
    class Config:
        frozen = True  # Equivalent to @dataclass(frozen=True)
    
    @model_validator(mode='after')
    def validate_constraints(self):
        """Validate physical constraints are reasonable."""
        if self.vertical_pivot_length > self.handle_length:
            raise ValueError("Vertical pivot must be within handle length")
        
        if self.forearm_pivot_length > self.handle_length:
            raise ValueError("Forearm pivot must be within handle length")
        
        if self.alpha_min >= self.alpha_max:
            raise ValueError("Invalid alpha range")
        
        if self.beta_min >= self.beta_max:
            raise ValueError("Invalid beta range")
        
        return self
    
    def calculate_max_holes_for_rod(self, rod_length: float) -> int:
        """Calculate maximum holes that can physically fit on a rod.
        
        Args:
            rod_length: Length of the rod in cm
            
        Returns:
            Maximum number of holes that can fit
        """
        available_length = rod_length - 2 * self.hole_margin
        if available_length <= 0:
            return 0
        return max(1, int(available_length // self.min_hole_distance) + 1)
    
    @property
    def max_handle_holes(self) -> int:
        """Maximum holes that can fit on handle rod."""
        return self.calculate_max_holes_for_rod(self.handle_length)
    
    @property  
    def max_vertical_holes(self) -> int:
        """Maximum holes that can fit on vertical rod."""
        return self.calculate_max_holes_for_rod(self.vertical_length)
    
    @property
    def max_forearm_holes(self) -> int:
        """Maximum holes that can fit on forearm rod."""
        return self.calculate_max_holes_for_rod(self.forearm_length)
    
    @property
    def total_max_holes(self) -> int:
        """Total maximum holes across all rods."""
        return self.max_handle_holes + self.max_vertical_holes + self.max_forearm_holes


class OptimizationObjectives(BaseModel):
    """Multi-objective optimization weights and targets."""
    # Primary objectives
    vocabulary_weight: float = Field(default=1.0, ge=0, description="Weight for maximizing geometry diversity")
    truss_complexity_weight: float = Field(default=0.5, ge=0, description="Weight for minimizing unique truss lengths")
    
    # Secondary objectives
    manufacturability_weight: float = Field(default=0.1, ge=0, description="Weight for preferring standard hole spacings")
    robustness_weight: float = Field(default=0.1, ge=0, description="Weight for preferring geometries robust to tolerances")
    
    # Constraints
    max_unique_trusses: Optional[int] = Field(default=15, gt=0, description="Maximum number of unique trusses")
    min_vocabulary_size: Optional[int] = Field(default=50, gt=0, description="Minimum vocabulary size")
    
    # Tolerances
    length_tolerance: float = Field(default=0.25, gt=0, description="Length tolerance in cm")
    angle_tolerance: float = Field(default=1.0, gt=0, description="Angle tolerance in degrees")
    
    @model_validator(mode='after')
    def validate_objectives(self):
        """Validate optimization objectives are reasonable."""
        if self.vocabulary_weight < 0 or self.truss_complexity_weight < 0:
            raise ValueError("Objective weights must be non-negative")
        
        if self.vocabulary_weight + self.truss_complexity_weight == 0:
            raise ValueError("At least one objective weight must be positive")
        
        return self


class DifferentiableConfig(BaseModel):
    """Configuration for differentiable optimizer."""
    learning_rate: float = Field(default=0.01, gt=0, le=1, description="Learning rate for optimization")
    max_iterations: int = Field(default=1000, gt=0, description="Maximum number of iterations")
    convergence_threshold: float = Field(default=1e-6, gt=0, description="Convergence threshold")
    
    # Soft selection parameters
    temperature: float = Field(default=0.1, gt=0, description="Temperature for soft selection")
    temperature_schedule: str = Field(default="constant", description="Temperature schedule: constant, linear, exponential")
    
    # Architecture
    hidden_dims: Tuple[int, ...] = Field(default=(64, 32), description="Hidden layer dimensions")
    activation: str = Field(default="relu", description="Activation function")
    
    # Regularization
    l2_reg: float = Field(default=1e-4, ge=0, description="L2 regularization strength")
    dropout_rate: float = Field(default=0.1, ge=0, le=1, description="Dropout rate")
    
    # Sampling parameters for differentiable functions
    vocab_angle_samples: int = Field(default=20, gt=0, description="Number of (α, β) angle combinations to sample for vocabulary")
    vocab_hole_samples: int = Field(default=50, gt=0, description="Number of hole combinations per angle for vocabulary")
    truss_angle_samples: int = Field(default=20, gt=0, description="Number of α angles to sample for truss complexity")
    truss_hole_samples: int = Field(default=50, gt=0, description="Number of hole combinations per angle for truss complexity")
    
    # Feasibility parameters
    truss_length_min: float = Field(default=3.0, gt=0, description="Minimum feasible truss length (cm)")
    truss_length_max: float = Field(default=15.0, gt=0, description="Maximum feasible truss length (cm)")
    
    class Config:
        frozen = True  # Equivalent to @dataclass(frozen=True)
    
    @field_validator('hidden_dims', mode='before')
    @classmethod
    def convert_list_to_tuple(cls, v):
        """Ensure hashable types for JAX compatibility."""
        if isinstance(v, list):
            return tuple(v)
        return v
    
    @field_validator('temperature_schedule')
    @classmethod
    def validate_temperature_schedule(cls, v):
        """Validate temperature schedule is valid."""
        valid_schedules = ["constant", "linear", "exponential"]
        if v not in valid_schedules:
            raise ValueError(f"Temperature schedule must be one of {valid_schedules}")
        return v
    
    @field_validator('activation')
    @classmethod
    def validate_activation(cls, v):
        """Validate activation function is valid."""
        valid_activations = ["relu", "tanh", "sigmoid", "gelu"]
        if v not in valid_activations:
            raise ValueError(f"Activation must be one of {valid_activations}")
        return v


class GeneticAlgorithmConfig(BaseModel):
    """Configuration for genetic algorithm optimizer."""
    population_size: int = Field(default=100, gt=0, description="Population size")
    num_generations: int = Field(default=500, gt=0, description="Number of generations")
    crossover_rate: float = Field(default=0.8, ge=0, le=1, description="Crossover rate")
    mutation_rate: float = Field(default=0.1, ge=0, le=1, description="Mutation rate")
    
    # Selection
    selection_method: str = Field(default="tournament", description="Selection method: tournament, roulette, rank")
    tournament_size: int = Field(default=3, gt=0, description="Tournament size for selection")
    
    # Multi-objective
    use_nsga2: bool = Field(default=True, description="Use NSGA-II for multi-objective optimization")
    crowding_distance_weight: float = Field(default=0.1, ge=0, description="Crowding distance weight")
    
    @field_validator('selection_method')
    @classmethod
    def validate_selection_method(cls, v):
        """Validate selection method is valid."""
        valid_methods = ["tournament", "roulette", "rank"]
        if v not in valid_methods:
            raise ValueError(f"Selection method must be one of {valid_methods}")
        return v


class SimulatedAnnealingConfig(BaseModel):
    """Configuration for simulated annealing optimizer."""
    initial_temperature: float = Field(default=100.0, gt=0, description="Initial temperature")
    final_temperature: float = Field(default=0.01, gt=0, description="Final temperature")
    max_iterations: int = Field(default=10000, gt=0, description="Maximum iterations")
    
    # Cooling schedule
    cooling_schedule: str = Field(default="exponential", description="Cooling schedule: linear, exponential, logarithmic")
    cooling_rate: float = Field(default=0.95, gt=0, lt=1, description="Cooling rate")
    
    # Neighborhood
    perturbation_strength: float = Field(default=0.1, gt=0, description="Perturbation strength")
    adaptive_perturbation: bool = Field(default=True, description="Use adaptive perturbation")
    
    @field_validator('cooling_schedule')
    @classmethod
    def validate_cooling_schedule(cls, v):
        """Validate cooling schedule is valid."""
        valid_schedules = ["linear", "exponential", "logarithmic"]
        if v not in valid_schedules:
            raise ValueError(f"Cooling schedule must be one of {valid_schedules}")
        return v


class IntegerProgrammingConfig(BaseModel):
    """Configuration for integer programming optimizer."""
    solver: str = Field(default="SCIP", description="Solver: SCIP, CBC, GUROBI")
    time_limit: float = Field(default=3600.0, gt=0, description="Time limit in seconds")
    gap_tolerance: float = Field(default=0.01, ge=0, le=1, description="Gap tolerance")
    
    # Problem formulation
    linearization_method: str = Field(default="big_m", description="Linearization method: big_m, sos2")
    presolve: bool = Field(default=True, description="Use presolve")
    
    @field_validator('solver')
    @classmethod
    def validate_solver(cls, v):
        """Validate solver is valid."""
        valid_solvers = ["SCIP", "CBC", "GUROBI"]
        if v not in valid_solvers:
            raise ValueError(f"Solver must be one of {valid_solvers}")
        return v
    
    @field_validator('linearization_method')
    @classmethod
    def validate_linearization_method(cls, v):
        """Validate linearization method is valid."""
        valid_methods = ["big_m", "sos2"]
        if v not in valid_methods:
            raise ValueError(f"Linearization method must be one of {valid_methods}")
        return v


class HybridConfig(BaseModel):
    """Configuration for hybrid optimizer."""
    # Stage 1: Differentiable pre-training
    differentiable_iterations: int = Field(default=500, gt=0, description="Number of differentiable iterations")
    differentiable_config: DifferentiableConfig = Field(default_factory=DifferentiableConfig, description="Differentiable optimizer config")
    
    # Stage 2: Discrete refinement
    discrete_optimizer: OptimizerType = Field(default=OptimizerType.GENETIC_ALGORITHM, description="Discrete optimizer type")
    discrete_config: Optional[Dict[str, Any]] = Field(default=None, description="Discrete optimizer config")
    
    # Transfer settings
    transfer_top_k: int = Field(default=10, gt=0, description="Number of top solutions to transfer")
    refinement_radius: float = Field(default=2.0, gt=0, description="Refinement radius in cm")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    # Problem setup
    constraints: CrutchConstraints = Field(default_factory=CrutchConstraints, description="Physical constraints")
    objectives: OptimizationObjectives = Field(default_factory=OptimizationObjectives, description="Optimization objectives")
    
    # Optimizer selection
    optimizer_type: OptimizerType = Field(default=OptimizerType.DIFFERENTIABLE, description="Optimizer type")
    
    # Optimizer-specific configs
    differentiable: DifferentiableConfig = Field(default_factory=DifferentiableConfig, description="Differentiable optimizer config")
    genetic_algorithm: GeneticAlgorithmConfig = Field(default_factory=GeneticAlgorithmConfig, description="Genetic algorithm config")
    simulated_annealing: SimulatedAnnealingConfig = Field(default_factory=SimulatedAnnealingConfig, description="Simulated annealing config")
    integer_programming: IntegerProgrammingConfig = Field(default_factory=IntegerProgrammingConfig, description="Integer programming config")
    hybrid: HybridConfig = Field(default_factory=HybridConfig, description="Hybrid optimizer config")
    
    # Experiment settings
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    output_dir: str = Field(default="results", description="Output directory")
    save_intermediate: bool = Field(default=True, description="Save intermediate results")
    verbose: bool = Field(default=True, description="Verbose output")
    use_initial_layout: bool = Field(default=False, description="Use initial layout")
    create_plots: bool = Field(default=True, description="Create plots")
    
    # Validation
    cross_validation_folds: int = Field(default=5, gt=1, description="Cross-validation folds")
    test_split: float = Field(default=0.2, gt=0, lt=1, description="Test split ratio")


class ConfigManager:
    """Manages loading and saving of configurations."""
    
    @staticmethod
    def load_config(config_path: str | Path) -> ExperimentConfig:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ExperimentConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if yaml is None:
            raise ImportError("PyYAML is required for loading YAML configs. Install with: pip install PyYAML")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def save_config(config: ExperimentConfig, config_path: str | Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: ExperimentConfig object
            config_path: Path to save YAML file
        """
        if yaml is None:
            raise ImportError("PyYAML is required for saving YAML configs. Install with: pip install PyYAML")
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = ConfigManager._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        # Pydantic handles all type conversions automatically!
        return ExperimentConfig(**config_dict)
    
    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        # Pydantic provides built-in dict conversion
        return config.dict()
    
    @staticmethod
    def create_default_configs() -> Dict[str, ExperimentConfig]:
        """Create default configurations for each optimizer type.
        
        Returns:
            Dictionary mapping optimizer names to default configs
        """
        configs = {}
        
        for optimizer_type in OptimizerType:
            config = ExperimentConfig(optimizer_type=optimizer_type)
            configs[optimizer_type.value] = config
        
        return configs


# Note: Validation functions are now built into the Pydantic models
# using @root_validator and @validator decorators
