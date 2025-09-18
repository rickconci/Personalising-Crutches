"""Differentiable optimization using JAX for hole placement."""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import jax
import jax.numpy as jnp
import optax
from jax import random
import numpy as np
import logging

from .base import BaseOptimizer, OptimizationResult, OptimizationMetrics
from ..config import CrutchConstraints, OptimizationObjectives, DifferentiableConfig
from ..geometry import HoleLayout, Geometry
from ..logging_system import OptimizationLogger, LoggingMode, LoggingInterest, create_logger_for_optimizer


class DifferentiableOptimizer(BaseOptimizer):
    """Differentiable optimizer using continuous relaxation and neural networks."""
    
    def __init__(
        self,
        constraints: CrutchConstraints,
        objectives: OptimizationObjectives,
        config: DifferentiableConfig,
        random_seed: int = 42,
        logger: Optional[OptimizationLogger] = None
    ):
        """Initialize differentiable optimizer.
        
        Args:
            constraints: Physical constraints
            objectives: Optimization objectives  
            config: Differentiable-specific configuration
            random_seed: Random seed
            logger: Optional optimization logger for detailed tracking
        """
        super().__init__(constraints, objectives, random_seed)
        self.config = config
        self.key = random.PRNGKey(random_seed)
        
        # Setup logging
        if logger is None:
            # Create default logger with comprehensive tracking
            self.logger = create_logger_for_optimizer(
                optimizer_name="differentiable",
                mode=LoggingMode.DETAILED,
                interests=[
                    LoggingInterest.CONVERGENCE,
                    LoggingInterest.GEOMETRY,
                    LoggingInterest.PARAMETERS,
                    LoggingInterest.FORWARD_PASS
                ]
            )
        else:
            self.logger = logger
        
        # Initialize optimizer
        self.optimizer = optax.adam(config.learning_rate)
        
        # Problem dimensions (rod-specific)
        self.max_handle_holes = constraints.max_handle_holes
        self.max_vertical_holes = constraints.max_vertical_holes  
        self.max_forearm_holes = constraints.max_forearm_holes
        self.param_dim = (self.max_handle_holes + self.max_vertical_holes + 
                         self.max_forearm_holes)  # Total parameter vector size
        
        # Setup standard logger
        self.std_logger = logging.getLogger(f"{__name__}.DifferentiableOptimizer")
        
    def optimize(
        self,
        initial_layout: Optional[HoleLayout] = None,
        **kwargs
    ) -> OptimizationResult:
        """Run differentiable optimization.
        
        Args:
            initial_layout: Optional initial hole layout
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best solutions
        """
        self.start_time = jax.time.time()
        
        # Initialize parameters
        if initial_layout is not None:
            params = self._layout_to_params(initial_layout)
        else:
            params = self._initialize_params()
        
        # Initialize optimizer state
        opt_state = self.optimizer.init(params)
        
        # Optimization loop
        best_params = params
        best_objective = float('inf')
        prev_loss = float('inf')
        
        self.std_logger.info(f"🚀 Starting optimization with {self.param_dim} parameters")
        self.std_logger.info(f"📊 Max holes: Handle={self.max_handle_holes}, Vertical={self.max_vertical_holes}, Forearm={self.max_forearm_holes}")
        self.std_logger.info(f"🔧 best params: {best_params}")

        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            
            # Compute gradients and update with detailed forward pass tracking
            loss, grads, forward_pass_info = self._objective_function_with_logging(params)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Project to feasible region
            params = self._project_to_feasible(params)
            
            # Track best solution
            improvement = prev_loss - loss if prev_loss != float('inf') else 0.0
            if loss < best_objective:
                best_objective = loss
                best_params = params
            
            # Comprehensive logging with our logging system
            metrics = {
                'objective': float(loss),
                'improvement': float(improvement),
                'best_objective': float(best_objective),
                'gradient_norm': float(jnp.linalg.norm(grads)),
                'parameter_norm': float(jnp.linalg.norm(params)),
                'temperature': self.config.temperature,
                **forward_pass_info.get('metrics', {})
            }
            
            # Log to our comprehensive system
            self.logger.log_iteration(
                iteration=iteration,
                metrics=metrics,
                parameters=params,
                gradients=grads,
                forward_pass_info=forward_pass_info
            )
            
            # Legacy logging for base class
            self._log_iteration(loss, params)
            
            # Check convergence
            if self._check_convergence():
                self.std_logger.info(f"✅ Converged at iteration {iteration}")
                break
            
            # Update temperature schedule
            if self.config.temperature_schedule != "constant":
                self._update_temperature(iteration)
            
            prev_loss = loss
        
        # Convert best parameters to hole layout
        best_layout = self._params_to_layout(best_params)
        
        # Evaluate final solution
        final_objective, final_metrics = self.evaluate_layout(best_layout)
        final_metrics.convergence_rate = self._calculate_convergence_rate()
        
        # Generate geometries
        best_geometries = self.geometry_calculator.enumerate_geometries(best_layout)
        
        # Create result
        result = OptimizationResult(
            best_hole_layout=best_layout,
            best_geometries=best_geometries,
            best_metrics=final_metrics,
            pareto_solutions=[(best_layout, final_metrics)],  # Single solution for now
            objective_history=self.history['objectives'],
            parameter_history=self.history['parameters'],
            total_time=jax.time.time() - self.start_time,
            iterations=self.iteration + 1,
            converged=self._check_convergence(),
            metadata={
                'config': self.config.__dict__,
                'final_loss': float(final_objective),
                'convergence_iteration': self._find_convergence_iteration()
            }
        )
        
        # Log final results
        self.logger.log_final_results(result)
        
        return result
    
    def _objective_function_with_logging(self, params: jnp.ndarray) -> Tuple[float, jnp.ndarray, Dict[str, Any]]:
        """Objective function with detailed logging information.
        
        Args:
            params: Parameter vector
            
        Returns:
            Tuple of (loss, gradients, forward_pass_info)
        """
        # Use JAX's value_and_grad with has_aux=True to return extra info
        def objective_with_info(params):
            # Convert to hole layout (differentiably)
            layout = self._params_to_layout_differentiable(params)
            
            # Calculate differentiable metrics with detailed tracking
            vocab_score, vocab_info = self._differentiable_vocabulary_score_with_info(layout)
            truss_complexity, truss_info = self._differentiable_truss_complexity_with_info(layout)
            
            # Weighted objective
            objective = (-vocab_score * self.objectives.vocabulary_weight + 
                        truss_complexity * self.objectives.truss_complexity_weight)
            
            # Add regularization
            l2_penalty = self.config.l2_reg * jnp.sum(params ** 2)
            total_loss = objective + l2_penalty
            
            # Detailed forward pass information
            forward_pass_info = {
                'vocabulary_score': float(vocab_score),
                'truss_complexity': float(truss_complexity),
                'l2_penalty': float(l2_penalty),
                'raw_objective': float(objective),
                'total_loss': float(total_loss),
                'hole_counts': {
                    'handle': len(layout['handle']),
                    'vertical': len(layout['vertical']),
                    'forearm': len(layout['forearm'])
                },
                'metrics': {
                    'vocabulary_score': float(vocab_score),
                    'truss_complexity': float(truss_complexity),
                    'l2_penalty': float(l2_penalty)
                },
                **vocab_info,
                **truss_info
            }
            
            return total_loss, forward_pass_info
        
        # Get loss, gradients, and auxiliary info
        (loss, forward_pass_info), grads = jax.value_and_grad(objective_with_info, has_aux=True)(params)
        
        return loss, grads, forward_pass_info
    
    def _initialize_params(self) -> jnp.ndarray:
        """Initialize optimization parameters.
        
        Returns:
            Initial parameter vector
        """
        self.key, subkey = random.split(self.key)
        
        # Initialize hole positions with some randomness around uniform spacing
        params = []
        
        # Handle holes
        handle_spacing = self.constraints.handle_length / (self.max_handle_holes + 1)
        handle_positions = jnp.linspace(handle_spacing, 
                                      self.constraints.handle_length - handle_spacing, 
                                      self.max_handle_holes)
        handle_noise = random.normal(subkey, (self.max_handle_holes,)) * 0.5
        params.extend(handle_positions + handle_noise)
        
        # Vertical holes
        self.key, subkey = random.split(self.key)
        vertical_spacing = self.constraints.vertical_length / (self.max_vertical_holes + 1)
        vertical_positions = jnp.linspace(vertical_spacing,
                                        self.constraints.vertical_length - vertical_spacing,
                                        self.max_vertical_holes)
        vertical_noise = random.normal(subkey, (self.max_vertical_holes,)) * 0.5
        params.extend(vertical_positions + vertical_noise)
        
        # Forearm holes
        self.key, subkey = random.split(self.key)
        forearm_spacing = self.constraints.forearm_length / (self.max_forearm_holes + 1)
        forearm_positions = jnp.linspace(forearm_spacing,
                                       self.constraints.forearm_length - forearm_spacing,
                                       self.max_forearm_holes)
        forearm_noise = random.normal(subkey, (self.max_forearm_holes,)) * 0.5
        params.extend(forearm_positions + forearm_noise)
        
        return jnp.array(params)
    
    def _layout_to_params(self, layout: HoleLayout) -> jnp.ndarray:
        """Convert hole layout to parameter vector.
        
        Args:
            layout: HoleLayout object
            
        Returns:
            Parameter vector
        """
        # Pad or truncate to rod-specific sizes
        def pad_or_truncate(arr: jnp.ndarray, target_size: int) -> jnp.ndarray:
            if len(arr) >= target_size:
                return arr[:target_size]
            else:
                # Pad with extrapolated values
                if len(arr) > 1:
                    spacing = arr[1] - arr[0]
                    padding = jnp.arange(len(arr), target_size) * spacing + arr[-1] + spacing
                else:
                    padding = jnp.full(target_size - len(arr), arr[0] if len(arr) > 0 else 1.0)
                return jnp.concatenate([arr, padding])
        
        handle_params = pad_or_truncate(layout.handle, self.max_handle_holes)
        vertical_params = pad_or_truncate(layout.vertical, self.max_vertical_holes)
        forearm_params = pad_or_truncate(layout.forearm, self.max_forearm_holes)
        
        return jnp.concatenate([handle_params, vertical_params, forearm_params])
    
    def _params_to_layout(self, params: jnp.ndarray) -> HoleLayout:
        """Convert parameter vector to hole layout.
        
        Args:
            params: Parameter vector
            
        Returns:
            HoleLayout object
        """
        # Split parameters using rod-specific indices
        idx = 0
        handle_params = params[idx:idx + self.max_handle_holes]
        idx += self.max_handle_holes
        
        vertical_params = params[idx:idx + self.max_vertical_holes]
        idx += self.max_vertical_holes
        
        forearm_params = params[idx:idx + self.max_forearm_holes]
        
        # Filter and sort valid holes
        def extract_valid_holes(hole_params: jnp.ndarray, max_length: float, margin: float) -> jnp.ndarray:
            # Filter holes within bounds
            valid_mask = (hole_params >= margin) & (hole_params <= max_length - margin)
            valid_holes = hole_params[valid_mask]
            
            # Sort and ensure minimum spacing
            if len(valid_holes) == 0:
                return jnp.array([margin + 0.1])  # At least one hole
            
            sorted_holes = jnp.sort(valid_holes)
            
            # Enforce minimum spacing
            filtered_holes = [sorted_holes[0]]
            for hole in sorted_holes[1:]:
                if hole - filtered_holes[-1] >= self.constraints.min_hole_distance:
                    filtered_holes.append(hole)
            
            return jnp.array(filtered_holes)
        
        handle_holes = extract_valid_holes(handle_params, 
                                         self.constraints.handle_length, 
                                         self.constraints.hole_margin)
        vertical_holes = extract_valid_holes(vertical_params,
                                           self.constraints.vertical_length,
                                           self.constraints.hole_margin)
        forearm_holes = extract_valid_holes(forearm_params,
                                          self.constraints.forearm_length,
                                          self.constraints.hole_margin)
        
        return HoleLayout(handle=handle_holes, vertical=vertical_holes, forearm=forearm_holes)
    
    def _project_to_feasible(self, params: jnp.ndarray) -> jnp.ndarray:
        """Project parameters to feasible region.
        
        Args:
            params: Parameter vector
            
        Returns:
            Projected parameters
        """
        # Split parameters
        handle_params = params[:self.max_holes]
        vertical_params = params[self.max_holes:2*self.max_holes]
        forearm_params = params[2*self.max_holes:]
        
        # Clamp to bounds
        margin = self.constraints.hole_margin
        handle_params = jnp.clip(handle_params, margin, self.constraints.handle_length - margin)
        vertical_params = jnp.clip(vertical_params, margin, self.constraints.vertical_length - margin)
        forearm_params = jnp.clip(forearm_params, margin, self.constraints.forearm_length - margin)
        
        return jnp.concatenate([handle_params, vertical_params, forearm_params])
    
    @jax.jit
    def _objective_function(self, params: jnp.ndarray) -> float:
        """Differentiable objective function.
        
        Args:
            params: Parameter vector
            
        Returns:
            Objective value (to minimize)
        """
        # Convert to hole layout (differentiably)
        layout = self._params_to_layout_differentiable(params)
        
        # Calculate differentiable metrics
        vocabulary_score = self._differentiable_vocabulary_score(layout)
        truss_complexity_score = self._differentiable_truss_complexity(layout)
        
        # Weighted objective
        objective = (-vocabulary_score * self.objectives.vocabulary_weight + 
                    truss_complexity_score * self.objectives.truss_complexity_weight)
        
        # Add regularization
        l2_penalty = self.config.l2_reg * jnp.sum(params ** 2)
        
        return objective + l2_penalty
    
    def _params_to_layout_differentiable(self, params: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Differentiable conversion to hole layout representation.
        
        Args:
            params: Parameter vector
            
        Returns:
            Dictionary with hole arrays
        """
        # Split parameters using rod-specific indices
        idx = 0
        handle_params = params[idx:idx + self.max_handle_holes]
        idx += self.max_handle_holes
        
        vertical_params = params[idx:idx + self.max_vertical_holes]
        idx += self.max_vertical_holes
        
        forearm_params = params[idx:idx + self.max_forearm_holes]
        
        return {
            'handle': handle_params,
            'vertical': vertical_params,
            'forearm': forearm_params
        }
    
    def _differentiable_vocabulary_score(self, layout_dict: Dict[str, jnp.ndarray]) -> float:
        """Differentiable approximation of vocabulary size.
        
        Args:
            layout_dict: Dictionary with hole positions
            
        Returns:
            Vocabulary score (higher is better)
        """
        # Sample geometry space differentiably using configurable parameters (fast version)
        n_alpha_samples = self.config.vocab_angle_samples_fast
        n_beta_samples = self.config.vocab_angle_samples_fast
        alpha_samples = jnp.linspace(self.constraints.alpha_min, self.constraints.alpha_max, n_alpha_samples)
        beta_samples = jnp.linspace(self.constraints.beta_min, self.constraints.beta_max, n_beta_samples)
        
        # For each (alpha, beta) pair, check if it's achievable with soft selection
        achievable_score = 0.0
        
        for alpha in alpha_samples:
            for beta in beta_samples:
                # Check if this geometry is achievable (differentiably)
                if self.constraints.require_alpha_beta_sum_ge_180 and alpha + beta < 180.0:
                    continue
                
                # Soft check for truss realizability
                realizability = self._soft_geometry_realizability(layout_dict, alpha, beta, self.config.vocab_hole_samples_fast)
                achievable_score += realizability
        
        return achievable_score
    
    def _differentiable_vocabulary_score_with_info(self, layout_dict: Dict[str, jnp.ndarray]) -> Tuple[float, Dict[str, Any]]:
        """Differentiable vocabulary score with detailed logging info.
        
        Args:
            layout_dict: Dictionary with hole positions
            
        Returns:
            Tuple of (vocabulary_score, info_dict)
        """
        # Sample the (α, β) space using configurable parameters (full sampling for detailed logging)
        n_alpha_samples = self.config.vocab_angle_samples
        n_beta_samples = self.config.vocab_angle_samples
        alpha_samples = jnp.linspace(self.constraints.alpha_min, self.constraints.alpha_max, n_alpha_samples)
        beta_samples = jnp.linspace(self.constraints.beta_min, self.constraints.beta_max, n_beta_samples)
        
        achievable_score = 0.0
        total_samples = 0
        realizability_scores = []
        
        for alpha in alpha_samples:
            for beta in beta_samples:
                # Check usability constraint
                if self.constraints.require_alpha_beta_sum_ge_180 and alpha + beta < 180.0:
                    continue
                
                total_samples += 1
                # How well can we realize this (α, β)?
                realizability = self._soft_geometry_realizability(layout_dict, alpha, beta, self.config.vocab_hole_samples)
                realizability_scores.append(float(realizability))
                achievable_score += realizability
        
        # Additional info for logging
        info = {
            'vocab_info': {
                'total_angle_combinations_tested': total_samples,
                'mean_realizability': float(jnp.mean(jnp.array(realizability_scores))) if realizability_scores else 0.0,
                'max_realizability': float(jnp.max(jnp.array(realizability_scores))) if realizability_scores else 0.0,
                'min_realizability': float(jnp.min(jnp.array(realizability_scores))) if realizability_scores else 0.0,
                'alpha_range': [float(self.constraints.alpha_min), float(self.constraints.alpha_max)],
                'beta_range': [float(self.constraints.beta_min), float(self.constraints.beta_max)]
            }
        }
        
        return achievable_score, info
    
    def _soft_geometry_realizability(
        self, 
        layout_dict: Dict[str, jnp.ndarray], 
        alpha: float, 
        beta: float,
        n_samples: int = 50
    ) -> float:
        """Soft check if geometry is realizable with given holes.
        
        Args:
            layout_dict: Dictionary with hole positions
            alpha: Alpha angle
            beta: Beta angle
            
        Returns:
            Realizability score [0, 1]
        """
        handle_holes = layout_dict['handle']
        vertical_holes = layout_dict['vertical']
        forearm_holes = layout_dict['forearm']
        
        # Uniform sampling strategy with more samples for better coverage
        hole_combinations = self._sample_hole_combinations_uniform(
            handle_holes, vertical_holes, forearm_holes, n_samples=n_samples
        )
        
        max_realizability = 0.0
        
        for h_pos, v_pos, f_pos in hole_combinations:
            # Calculate truss lengths for this combination
            realizability = self._evaluate_truss_combination(h_pos, v_pos, f_pos, alpha, beta)
            max_realizability = jnp.maximum(max_realizability, realizability)
        
        return max_realizability
    
    def _sample_hole_combinations_uniform(
        self, 
        handle_holes: jnp.ndarray, 
        vertical_holes: jnp.ndarray, 
        forearm_holes: jnp.ndarray,
        n_samples: int = 50
    ) -> List[Tuple[float, float, float]]:
        """Uniformly sample hole combinations for unbiased truss evaluation.
        
        This approach separates sampling from optimization:
        - Sampling: Uniform coverage of hole combination space
        - Optimization: Loss function guides towards optimal truss lengths
        
        Args:
            handle_holes: Handle hole positions
            vertical_holes: Vertical hole positions  
            forearm_holes: Forearm hole positions
            n_samples: Number of combinations to sample
            
        Returns:
            List of (handle_pos, vertical_pos, forearm_pos) tuples
        """
        combinations = []
        
        # Pure uniform random sampling for unbiased exploration
        for _ in range(n_samples):
            h_idx = int(jnp.random.uniform() * len(handle_holes))
            v_idx = int(jnp.random.uniform() * len(vertical_holes))
            f_idx = int(jnp.random.uniform() * len(forearm_holes))
            
            combinations.append((
                float(handle_holes[h_idx]),
                float(vertical_holes[v_idx]),
                float(forearm_holes[f_idx])
            ))
        
        return combinations
    
    def _evaluate_truss_combination(
        self, 
        h_pos: float, 
        v_pos: float, 
        f_pos: float, 
        alpha: float, 
        beta: float
    ) -> float:
        """Evaluate realizability of a specific hole combination with structural optimization.
        
        This function now includes multiple criteria:
        1. Geometric feasibility (can we build this?)
        2. Truss length optimality (are lengths in good range?)
        3. Structural efficiency (do trusses have good leverage?)
        
        Args:
            h_pos: Handle hole position
            v_pos: Vertical hole position
            f_pos: Forearm hole position
            alpha: Alpha angle
            beta: Beta angle
            
        Returns:
            Combined realizability and structural quality score [0, 1]
        """
        truss_evaluations = []
        
        # Truss 1: Vertical to Handle (behind vertical pivot)
        if h_pos < self.constraints.vertical_pivot_length:
            r1 = v_pos  # Moment arm from vertical pivot
            r2 = abs(self.constraints.vertical_pivot_length - h_pos)  # Moment arm on handle
            t1 = self._law_of_cosines_jax(r1, r2, 180.0 - alpha)
            
            # Multi-criteria evaluation
            length_score = self._evaluate_truss_length(t1)
            leverage_score = self._evaluate_structural_leverage(r1, r2)
            
            truss_evaluations.append(length_score * leverage_score)
        
        # Truss 2: Vertical to Handle (ahead of vertical pivot)  
        if h_pos > self.constraints.vertical_pivot_length:
            r1 = v_pos  # Moment arm from vertical pivot
            r2 = abs(h_pos - self.constraints.vertical_pivot_length)  # Moment arm on handle
            t2 = self._law_of_cosines_jax(r1, r2, alpha)
            
            length_score = self._evaluate_truss_length(t2)
            leverage_score = self._evaluate_structural_leverage(r1, r2)
            
            truss_evaluations.append(length_score * leverage_score)
        
        # Truss 3: Handle to Forearm (behind forearm pivot)
        if h_pos < self.constraints.forearm_pivot_length:
            r1_f = abs(self.constraints.forearm_pivot_length - h_pos)  # Moment arm on handle
            r2_f = f_pos  # Moment arm from forearm pivot
            t3 = self._law_of_cosines_jax(r1_f, r2_f, 180.0 - beta)
            
            length_score = self._evaluate_truss_length(t3)
            leverage_score = self._evaluate_structural_leverage(r1_f, r2_f)
            
            truss_evaluations.append(length_score * leverage_score)
        
        # Return average quality across valid trusses
        if truss_evaluations:
            return jnp.mean(jnp.array(truss_evaluations))
        else:
            return 0.0  # No valid truss configurations
    
    def _evaluate_truss_length(self, truss_length: float) -> float:
        """Evaluate quality of truss length.
        
        Preferences:
        - Avoid very short trusses (< 8cm): Poor structural integrity
        - Prefer medium trusses (10-20cm): Good balance of strength and weight
        - Avoid very long trusses (> 25cm): Heavy, unwieldy
        
        Args:
            truss_length: Length of truss in cm
            
        Returns:
            Quality score [0, 1]
        """
        # Multi-modal preference function
        optimal_range_score = jax.nn.sigmoid(-((truss_length - 15.0)**2) / 20.0)  # Peak at 15cm
        minimum_length_penalty = jax.nn.sigmoid((truss_length - 6.0) * 2.0)  # Penalty below 6cm
        maximum_length_penalty = jax.nn.sigmoid(-(truss_length - 30.0) * 0.5)  # Penalty above 30cm
        
        return optimal_range_score * minimum_length_penalty * maximum_length_penalty
    
    def _evaluate_structural_leverage(self, moment_arm1: float, moment_arm2: float) -> float:
        """Evaluate structural leverage quality of truss configuration.
        
        Better leverage = longer moment arms = better force resistance.
        
        Args:
            moment_arm1: First moment arm length
            moment_arm2: Second moment arm length
            
        Returns:
            Leverage quality score [0, 1]
        """
        # Geometric mean of moment arms (favors balanced, long arms)
        geometric_mean = jnp.sqrt(moment_arm1 * moment_arm2)
        
        # Sigmoid to normalize to [0, 1] range
        leverage_score = jax.nn.sigmoid((geometric_mean - 3.0) * 0.5)  # Inflection at 3cm
        
        return leverage_score
    
    def _law_of_cosines_jax(self, a: float, b: float, angle_deg: float) -> float:
        """JAX implementation of law of cosines.
        
        Args:
            a: First side length
            b: Second side length
            angle_deg: Included angle in degrees
            
        Returns:
            Third side length
        """
        angle_rad = jnp.deg2rad(angle_deg)
        c_squared = a**2 + b**2 - 2*a*b*jnp.cos(angle_rad)
        return jnp.sqrt(jnp.maximum(0.0, c_squared))
    
    def _differentiable_truss_complexity(self, layout_dict: Dict[str, jnp.ndarray]) -> float:
        """Differentiable approximation of truss complexity.
        
        Args:
            layout_dict: Dictionary with hole positions
            
        Returns:
            Truss complexity score (lower is better)
        """
        handle_holes = layout_dict['handle']
        vertical_holes = layout_dict['vertical']
        forearm_holes = layout_dict['forearm']
        
        # Use uniform sampling approach for consistency
        sample_trusses = []
        n_angle_samples = self.config.truss_angle_samples_fast
        n_hole_samples = self.config.truss_hole_samples_fast
        
        for alpha in jnp.linspace(self.constraints.alpha_min, self.constraints.alpha_max, n_angle_samples):
            # Uniformly sample hole combinations
            hole_combinations = self._sample_hole_combinations_uniform(
                handle_holes, vertical_holes, forearm_holes, n_hole_samples
            )
            
            for h_pos, v_pos, f_pos in hole_combinations:
                # Truss 1 and 2 calculations (handle-vertical connection)
                if h_pos < self.constraints.vertical_pivot_length:
                    r1 = v_pos
                    r2 = abs(self.constraints.vertical_pivot_length - h_pos)
                    t1 = self._law_of_cosines_jax(r1, r2, 180.0 - alpha)
                    sample_trusses.append(t1)
                
                if h_pos > self.constraints.vertical_pivot_length:
                    r1 = v_pos
                    r2 = abs(h_pos - self.constraints.vertical_pivot_length)
                    t2 = self._law_of_cosines_jax(r1, r2, alpha)
                    sample_trusses.append(t2)
        
        if not sample_trusses:
            return 100.0  # High penalty for no valid trusses
        
        sample_trusses = jnp.array(sample_trusses)
        
        # Estimate unique count using differentiable clustering
        unique_count = self._differentiable_unique_count(sample_trusses)
        return unique_count
    
    def _differentiable_truss_complexity_with_info(self, layout_dict: Dict[str, jnp.ndarray]) -> Tuple[float, Dict[str, Any]]:
        """Differentiable truss complexity with detailed logging info.
        
        Args:
            layout_dict: Dictionary with hole positions
            
        Returns:
            Tuple of (truss_complexity, info_dict)
        """
        handle_holes = layout_dict['handle']
        vertical_holes = layout_dict['vertical'] 
        forearm_holes = layout_dict['forearm']
        
        # Use uniform sampling approach like in geometry realizability
        sample_trusses = []
        truss_length_details = []
        
        # Sample uniformly across angle space and hole combinations
        n_angle_samples = self.config.truss_angle_samples  # More comprehensive sampling for detailed logging
        n_hole_samples = self.config.truss_hole_samples    # Same as geometry sampling
        
        for alpha in jnp.linspace(self.constraints.alpha_min, self.constraints.alpha_max, n_angle_samples):
            # Uniformly sample hole combinations
            hole_combinations = self._sample_hole_combinations_uniform(
                handle_holes, vertical_holes, forearm_holes, n_hole_samples
            )
            
            for h_pos, v_pos, f_pos in hole_combinations:
                # Truss 1 and 2 calculations (handle-vertical connection)
                if h_pos < self.constraints.vertical_pivot_length:
                    r1 = v_pos
                    r2 = abs(self.constraints.vertical_pivot_length - h_pos)
                    t1 = self._law_of_cosines_jax(r1, r2, 180.0 - alpha)
                    sample_trusses.append(t1)
                    truss_length_details.append({
                        'type': 'truss_1',
                        'length': float(t1),
                        'alpha': float(alpha),
                        'moment_arms': [float(r1), float(r2)]
                    })
                
                if h_pos > self.constraints.vertical_pivot_length:
                    r1 = v_pos
                    r2 = abs(h_pos - self.constraints.vertical_pivot_length)
                    t2 = self._law_of_cosines_jax(r1, r2, alpha)
                    sample_trusses.append(t2)
                    truss_length_details.append({
                        'type': 'truss_2',
                        'length': float(t2),
                        'alpha': float(alpha),
                        'moment_arms': [float(r1), float(r2)]
                    })
        
        if not sample_trusses:
            return 100.0, {'truss_info': {'error': 'No valid trusses found'}}
        
        sample_trusses = jnp.array(sample_trusses)
        
        # Estimate unique count using differentiable clustering
        unique_count = self._differentiable_unique_count(sample_trusses)
        
        # Additional info for logging
        info = {
            'truss_info': {
                'total_truss_samples': len(sample_trusses),
                'angle_samples': n_angle_samples,
                'hole_combination_samples': n_hole_samples,
                'mean_truss_length': float(jnp.mean(sample_trusses)),
                'std_truss_length': float(jnp.std(sample_trusses)),
                'min_truss_length': float(jnp.min(sample_trusses)),
                'max_truss_length': float(jnp.max(sample_trusses)),
                'estimated_unique_count': float(unique_count),
                'sample_truss_lengths': [float(t) for t in sample_trusses[:10]]  # First 10 for logging
            }
        }
        
        return unique_count, info
    
    def _differentiable_unique_count(self, values: jnp.ndarray, bandwidth: float = 1.0) -> float:
        """Differentiable approximation of unique value count.
        
        Uses the participation ratio method from quantum physics to estimate
        the effective number of unique values in a continuous, differentiable way.
        
        Args:
            values: Array of values to count uniqueness
            bandwidth: Bandwidth for kernel density estimation (smaller = more sensitive)
            
        Returns:
            Approximate unique count (higher = more unique values)
        """
        if len(values) == 0:
            return 0.0
        
        n = len(values)
        if n == 1:
            return 1.0
        
        # Pairwise distances between all values
        distances = jnp.abs(values[:, None] - values[None, :])
        
        # Gaussian similarity kernel (high similarity for close values)
        similarities = jnp.exp(-distances**2 / (2 * bandwidth**2))
        
        # Method 1: Participation Ratio (recommended)
        # Eigenvalues of normalized similarity matrix represent "participation" of each value
        normalized_similarities = similarities / n
        eigenvalues = jnp.linalg.eigvals(normalized_similarities)
        eigenvalues = jnp.maximum(eigenvalues.real, 1e-12)  # Take real part, avoid zeros
        
        # Participation ratio: 1 / sum(p_i^2) where p_i are normalized eigenvalues
        eigenvalue_probs = eigenvalues / jnp.sum(eigenvalues)
        participation_ratio = 1.0 / jnp.sum(eigenvalue_probs**2)
        
        return jnp.minimum(participation_ratio, float(n))  # Cap at total number of values
        
        # Alternative Method 2: Effective Rank (simpler but less accurate)
        # similarity_sum = jnp.sum(similarities)
        # diagonal_sum = jnp.trace(similarities)  # Always = n (diagonal elements = 1)
        # 
        # # Ranges from 1 (all identical) to n (all different)
        # off_diagonal_similarity = similarity_sum - diagonal_sum
        # max_off_diagonal = n * (n - 1)  # Maximum possible off-diagonal sum
        # 
        # # Higher off-diagonal similarity = fewer unique values
        # uniqueness_ratio = 1.0 - (off_diagonal_similarity / max_off_diagonal)
        # effective_unique_count = 1.0 + uniqueness_ratio * (n - 1)
        # 
        # return effective_unique_count
    
    def _update_temperature(self, iteration: int) -> None:
        """Update temperature for annealing schedule.
        
        Args:
            iteration: Current iteration
        """
        if self.config.temperature_schedule == "linear":
            progress = iteration / self.config.max_iterations
            self.config.temperature = self.config.temperature * (1.0 - progress)
        elif self.config.temperature_schedule == "exponential":
            self.config.temperature = self.config.temperature * 0.995
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate from optimization history.
        
        Returns:
            Convergence rate
        """
        if len(self.history['objectives']) < 10:
            return 0.0
        
        # Fit exponential decay to recent objectives
        recent_obj = np.array(self.history['objectives'][-50:])
        if len(recent_obj) < 2:
            return 0.0
        
        # Simple convergence rate: relative improvement per iteration
        initial_obj = recent_obj[0]
        final_obj = recent_obj[-1]
        
        if initial_obj == final_obj:
            return 1.0  # Already converged
        
        rate = abs(final_obj - initial_obj) / (initial_obj * len(recent_obj))
        return min(1.0, rate)
    
    def _find_convergence_iteration(self) -> int:
        """Find iteration where convergence was achieved.
        
        Returns:
            Convergence iteration (-1 if not converged)
        """
        if not self._check_convergence():
            return -1
        
        # Find where objective stopped improving significantly
        objectives = np.array(self.history['objectives'])
        
        for i in range(10, len(objectives)):
            recent_window = objectives[i-10:i]
            if np.std(recent_window) < 1e-6:
                return i - 10
        
        return len(objectives) - 1
