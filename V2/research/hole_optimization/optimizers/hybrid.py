"""Hybrid optimizer combining differentiable pre-training with discrete refinement."""

from __future__ import annotations
from typing import Optional, List

from .base import BaseOptimizer, OptimizationResult
from .differentiable import DifferentiableOptimizer
from .factory import create_optimizer
from ..config import CrutchConstraints, OptimizationObjectives, HybridConfig
from ..geometry import HoleLayout


class HybridOptimizer(BaseOptimizer):
    """Hybrid optimizer combining multiple approaches."""
    
    def __init__(
        self,
        constraints: CrutchConstraints,
        objectives: OptimizationObjectives,
        config: HybridConfig,
        random_seed: int = 42
    ):
        """Initialize hybrid optimizer.
        
        Args:
            constraints: Physical constraints
            objectives: Optimization objectives  
            config: Hybrid-specific configuration
            random_seed: Random seed
        """
        super().__init__(constraints, objectives, random_seed)
        self.config = config
    
    def optimize(
        self,
        initial_layout: Optional[HoleLayout] = None,
        **kwargs
    ) -> OptimizationResult:
        """Run hybrid optimization: differentiable pre-training + discrete refinement.
        
        Args:
            initial_layout: Optional initial hole layout
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best solutions
        """
        # Stage 1: Differentiable pre-training
        diff_optimizer = DifferentiableOptimizer(
            self.constraints,
            self.objectives,
            self.config.differentiable_config,
            self.random_seed
        )
        
        # Limit iterations for pre-training
        original_iterations = self.config.differentiable_config.max_iterations
        self.config.differentiable_config.max_iterations = self.config.differentiable_iterations
        
        diff_result = diff_optimizer.optimize(initial_layout=initial_layout)
        
        # Restore original iterations
        self.config.differentiable_config.max_iterations = original_iterations
        
        # Stage 2: Discrete refinement
        if self.config.discrete_optimizer and self.config.discrete_config:
            # Get top-k solutions from differentiable stage
            candidate_layouts = self._extract_candidate_layouts(diff_result)
            
            # Create discrete optimizer
            discrete_optimizer = create_optimizer(
                self.config.discrete_optimizer,
                self.constraints,
                self.objectives,
                self.config.discrete_config,
                self.random_seed + 1000  # Different seed
            )
            
            # Refine each candidate
            best_result = diff_result
            for layout in candidate_layouts:
                refined_result = discrete_optimizer.optimize(initial_layout=layout)
                
                # Keep best result
                if (refined_result.best_metrics.vocabulary_size > best_result.best_metrics.vocabulary_size or
                    (refined_result.best_metrics.vocabulary_size == best_result.best_metrics.vocabulary_size and
                     refined_result.best_metrics.unique_truss_count < best_result.best_metrics.unique_truss_count)):
                    best_result = refined_result
            
            return best_result
        
        return diff_result
    
    def _extract_candidate_layouts(self, result: OptimizationResult) -> List[HoleLayout]:
        """Extract candidate layouts from differentiable optimization result.
        
        Args:
            result: Differentiable optimization result
            
        Returns:
            List of candidate layouts for refinement
        """
        candidates = [result.best_hole_layout]
        
        # Add layouts from parameter history (if available)
        if result.parameter_history and len(result.parameter_history) > 1:
            # Take samples from different stages of optimization
            n_samples = min(self.config.transfer_top_k - 1, len(result.parameter_history) - 1)
            if n_samples > 0:
                indices = [int(i * len(result.parameter_history) / (n_samples + 1)) 
                          for i in range(1, n_samples + 1)]
                
                for idx in indices:
                    # Convert parameter vector back to layout
                    # This is a simplified conversion - would need proper implementation
                    candidates.append(result.best_hole_layout)  # Placeholder
        
        return candidates[:self.config.transfer_top_k]
