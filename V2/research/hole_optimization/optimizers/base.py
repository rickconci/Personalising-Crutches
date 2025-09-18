"""Base classes for hole optimization algorithms."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import numpy as np
import jax.numpy as jnp

from ..config import CrutchConstraints, OptimizationObjectives
from ..geometry import HoleLayout, Geometry, CrutchGeometry


@dataclass
class OptimizationMetrics:
    """Metrics for evaluating optimization quality."""
    vocabulary_size: int  # Number of unique geometries
    unique_truss_count: int  # Number of unique truss lengths
    pareto_efficiency: float  # How well objectives are balanced
    convergence_rate: float  # Speed of convergence
    robustness_score: float  # Sensitivity to perturbations
    manufacturability_score: float  # Ease of manufacturing


@dataclass
class OptimizationResult:
    """Result of hole optimization."""
    # Best solution
    best_hole_layout: HoleLayout
    best_geometries: List[Geometry]
    best_metrics: OptimizationMetrics
    
    # Multi-objective solutions (Pareto front)
    pareto_solutions: List[Tuple[HoleLayout, OptimizationMetrics]]
    
    # Optimization history
    objective_history: List[float] = field(default_factory=list)
    parameter_history: List[jnp.ndarray] = field(default_factory=list)
    
    # Runtime information
    total_time: float = 0.0
    iterations: int = 0
    converged: bool = False
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """Abstract base class for hole optimization algorithms."""
    
    def __init__(
        self,
        constraints: CrutchConstraints,
        objectives: OptimizationObjectives,
        random_seed: int = 42
    ):
        """Initialize optimizer.
        
        Args:
            constraints: Physical constraints for crutch geometry
            objectives: Multi-objective optimization weights
            random_seed: Random seed for reproducibility
        """
        self.constraints = constraints
        self.objectives = objectives
        self.random_seed = random_seed
        self.geometry_calculator = CrutchGeometry(constraints)
        
        # Set random seeds
        np.random.seed(random_seed)
        
        # Optimization state
        self.iteration = 0
        self.start_time = 0.0
        self.best_objective = float('inf')
        self.history = {'objectives': [], 'parameters': []}
        
    @abstractmethod
    def optimize(
        self,
        initial_layout: Optional[HoleLayout] = None,
        **kwargs
    ) -> OptimizationResult:
        """Run optimization algorithm.
        
        Args:
            initial_layout: Optional initial hole layout
            **kwargs: Algorithm-specific parameters
            
        Returns:
            OptimizationResult containing best solutions and metrics
        """
        pass
    
    def evaluate_layout(self, hole_layout: HoleLayout) -> Tuple[float, OptimizationMetrics]:
        """Evaluate a hole layout against objectives.
        
        Args:
            hole_layout: Hole positions to evaluate
            
        Returns:
            Tuple of (objective_value, metrics)
        """
        # Generate all possible geometries
        geometries = self.geometry_calculator.enumerate_geometries(hole_layout)
        
        # Calculate metrics
        metrics = self._calculate_metrics(geometries)
        
        # Calculate weighted objective
        objective = self._calculate_objective(metrics)
        
        return objective, metrics
    
    def _calculate_metrics(self, geometries: List[Geometry]) -> OptimizationMetrics:
        """Calculate optimization metrics from geometries.
        
        Args:
            geometries: List of valid geometries
            
        Returns:
            OptimizationMetrics object
        """
        if not geometries:
            return OptimizationMetrics(
                vocabulary_size=0,
                unique_truss_count=0,
                pareto_efficiency=0.0,
                convergence_rate=0.0,
                robustness_score=0.0,
                manufacturability_score=0.0
            )
        
        # Vocabulary size (unique angle combinations)
        unique_geometries = set(geometries)
        vocabulary_size = len(unique_geometries)
        
        # Unique truss count
        all_trusses = []
        for geom in geometries:
            all_trusses.extend([geom.truss_1, geom.truss_2, geom.truss_3])
        
        # Round to tolerance and count unique
        rounded_trusses = set(round(t / self.objectives.length_tolerance) * self.objectives.length_tolerance 
                            for t in all_trusses)
        unique_truss_count = len(rounded_trusses)
        
        # Pareto efficiency (how well we balance objectives)
        vocab_score = vocabulary_size / max(self.objectives.min_vocabulary_size or 1, 1)
        truss_score = 1.0 / max(unique_truss_count, 1)
        pareto_efficiency = min(vocab_score, truss_score)
        
        # Robustness (variance in angle scores)
        angle_scores = [g.score_alpha + g.score_beta for g in geometries]
        robustness_score = 1.0 / (1.0 + np.std(angle_scores)) if angle_scores else 0.0
        
        # Manufacturability (preference for standard hole spacings)
        manufacturability_score = self._calculate_manufacturability_score(geometries)
        
        return OptimizationMetrics(
            vocabulary_size=vocabulary_size,
            unique_truss_count=unique_truss_count,
            pareto_efficiency=pareto_efficiency,
            convergence_rate=0.0,  # Will be set by optimizer
            robustness_score=robustness_score,
            manufacturability_score=manufacturability_score
        )
    
    def _calculate_objective(self, metrics: OptimizationMetrics) -> float:
        """Calculate weighted objective value.
        
        Args:
            metrics: OptimizationMetrics object
            
        Returns:
            Weighted objective value (lower is better)
        """
        # Primary objectives (to minimize)
        vocab_penalty = -metrics.vocabulary_size * self.objectives.vocabulary_weight
        truss_penalty = metrics.unique_truss_count * self.objectives.truss_complexity_weight
        
        # Secondary objectives
        manufacturability_bonus = -metrics.manufacturability_score * self.objectives.manufacturability_weight
        robustness_bonus = -metrics.robustness_score * self.objectives.robustness_weight
        
        # Constraint penalties
        constraint_penalty = 0.0
        if self.objectives.max_unique_trusses and metrics.unique_truss_count > self.objectives.max_unique_trusses:
            constraint_penalty += 1000.0 * (metrics.unique_truss_count - self.objectives.max_unique_trusses)
        
        if self.objectives.min_vocabulary_size and metrics.vocabulary_size < self.objectives.min_vocabulary_size:
            constraint_penalty += 1000.0 * (self.objectives.min_vocabulary_size - metrics.vocabulary_size)
        
        return vocab_penalty + truss_penalty + manufacturability_bonus + robustness_bonus + constraint_penalty
    
    def _calculate_manufacturability_score(self, geometries: List[Geometry]) -> float:
        """Calculate manufacturability score based on hole spacing regularity.
        
        Args:
            geometries: List of geometries
            
        Returns:
            Manufacturability score (higher is better)
        """
        if not geometries:
            return 0.0
        
        # Prefer hole spacings that are multiples of standard distances
        standard_spacings = [1.0, 1.5, 2.0, 2.5, 3.0]  # cm
        
        # Extract all hole spacings used in geometries
        used_holes = set()
        for geom in geometries:
            used_holes.add(geom.t1_vertical)
            used_holes.add(geom.t1_handle)
            used_holes.add(geom.t2_vertical)
            used_holes.add(geom.t2_handle)
            used_holes.add(geom.t3_handle)
            used_holes.add(geom.t3_forearm)
        
        # Calculate score based on alignment with standard spacings
        alignment_score = 0.0
        for hole_pos in used_holes:
            best_alignment = min(abs(hole_pos % spacing) for spacing in standard_spacings)
            alignment_score += 1.0 / (1.0 + best_alignment)
        
        return alignment_score / len(used_holes) if used_holes else 0.0
    
    def _log_iteration(self, objective: float, parameters: jnp.ndarray) -> None:
        """Log iteration data.
        
        Args:
            objective: Current objective value
            parameters: Current parameter values
        """
        self.history['objectives'].append(float(objective))
        self.history['parameters'].append(parameters.copy())
        
        if objective < self.best_objective:
            self.best_objective = objective
    
    def _check_convergence(self, window_size: int = 10, tolerance: float = 1e-6) -> bool:
        """Check if optimization has converged.
        
        Args:
            window_size: Number of recent iterations to consider
            tolerance: Convergence tolerance
            
        Returns:
            True if converged
        """
        if len(self.history['objectives']) < window_size:
            return False
        
        recent_objectives = self.history['objectives'][-window_size:]
        return (max(recent_objectives) - min(recent_objectives)) < tolerance
    
    def get_pareto_front(
        self, 
        layouts: List[HoleLayout], 
        max_solutions: int = 20
    ) -> List[Tuple[HoleLayout, OptimizationMetrics]]:
        """Calculate Pareto front from multiple solutions.
        
        Args:
            layouts: List of hole layouts to evaluate
            max_solutions: Maximum number of Pareto solutions to return
            
        Returns:
            List of (layout, metrics) tuples on Pareto front
        """
        # Evaluate all layouts
        evaluated = []
        for layout in layouts:
            _, metrics = self.evaluate_layout(layout)
            evaluated.append((layout, metrics))
        
        # Find Pareto front
        pareto_solutions = []
        for i, (layout_i, metrics_i) in enumerate(evaluated):
            is_dominated = False
            
            for j, (layout_j, metrics_j) in enumerate(evaluated):
                if i == j:
                    continue
                
                # Check if j dominates i
                if (metrics_j.vocabulary_size >= metrics_i.vocabulary_size and
                    metrics_j.unique_truss_count <= metrics_i.unique_truss_count and
                    (metrics_j.vocabulary_size > metrics_i.vocabulary_size or
                     metrics_j.unique_truss_count < metrics_i.unique_truss_count)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append((layout_i, metrics_i))
        
        # Sort by vocabulary size and limit
        pareto_solutions.sort(key=lambda x: x[1].vocabulary_size, reverse=True)
        return pareto_solutions[:max_solutions]
    
    def validate_layout(self, hole_layout: HoleLayout) -> bool:
        """Validate that hole layout satisfies physical constraints.
        
        Args:
            hole_layout: Hole layout to validate
            
        Returns:
            True if layout is valid
        """
        # Check hole positions are within rod lengths
        if jnp.any(hole_layout.handle < 0) or jnp.any(hole_layout.handle > self.constraints.handle_length):
            return False
        if jnp.any(hole_layout.vertical < 0) or jnp.any(hole_layout.vertical > self.constraints.vertical_length):
            return False
        if jnp.any(hole_layout.forearm < 0) or jnp.any(hole_layout.forearm > self.constraints.forearm_length):
            return False
        
        # Check minimum hole distances
        for holes in [hole_layout.handle, hole_layout.vertical, hole_layout.forearm]:
            if len(holes) > 1:
                distances = jnp.diff(jnp.sort(holes))
                if jnp.any(distances < self.constraints.min_hole_distance - 1e-9):
                    return False
        
        # Check margins
        margin = self.constraints.hole_margin
        if (jnp.any(hole_layout.handle < margin) or 
            jnp.any(hole_layout.handle > self.constraints.handle_length - margin)):
            return False
        if (jnp.any(hole_layout.vertical < margin) or 
            jnp.any(hole_layout.vertical > self.constraints.vertical_length - margin)):
            return False
        if (jnp.any(hole_layout.forearm < margin) or 
            jnp.any(hole_layout.forearm > self.constraints.forearm_length - margin)):
            return False
        
        return True
