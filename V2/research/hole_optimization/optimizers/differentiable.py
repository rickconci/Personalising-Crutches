"""
Differentiable hole optimization using JAX - CLEAN VERSION.

This optimizer uses continuous relaxation and gradient-based optimization
to find optimal hole positions for crutch customization.
"""

from typing import Optional, Dict, Any, List, Tuple
import logging
import time
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import numpy as np
from functools import partial

from .base import BaseOptimizer, OptimizationResult, OptimizationMetrics
from ..config import CrutchConstraints, OptimizationObjectives, DifferentiableConfig
from ..geometry import HoleLayout, Geometry
from ..logging_system import OptimizationLogger, LoggingMode, LoggingInterest, create_logger_for_optimizer


class DifferentiableOptimizer(BaseOptimizer):
    """Differentiable optimizer using continuous relaxation and gradient-based methods."""
    
    def __init__(
        self,
        constraints: CrutchConstraints,
        objectives: OptimizationObjectives,
        config: DifferentiableConfig,
        random_seed: int = 42,
        logger: Optional[OptimizationLogger] = None
    ):
        """Initialize differentiable optimizer."""
        super().__init__(constraints, objectives, random_seed)
        self.config = config
        self.key = random.PRNGKey(random_seed)
        
        # Setup logging
        if logger is None:
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
                         self.max_forearm_holes)
        
        # Setup standard logger
        self.std_logger = logging.getLogger(f"{__name__}.DifferentiableOptimizer")
        
    def optimize(
        self,
        initial_layout: Optional[HoleLayout] = None,
        **kwargs
    ) -> OptimizationResult:
        """Run differentiable optimization."""
        self.start_time = time.time()
        
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
        self.std_logger.info(f"best params raw: {best_params}")
        
        # Create progress bar for optimization iterations
        pbar = tqdm(range(self.config.max_iterations), desc="Optimizing hole layout", leave=True)
        
        for iteration in pbar:
            self.iteration = iteration
            
            self.std_logger.info(f"🔄 Iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Compute gradients and update with detailed tracking
            self.std_logger.info("📊 Computing objective function...")
            loss, grads, forward_info = self._objective_function_with_info(params)
            
            self.std_logger.info("⚡ Applying gradient updates...")
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Project to feasible region (simple clipping)
            params = jnp.clip(params, 0.0, 1.0)
            
            # Track best solution
            improvement = prev_loss - loss if prev_loss != float('inf') else 0.0
            if loss < best_objective:
                best_objective = loss
                best_params = params

            # Update progress bar with current metrics
            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Best': f'{best_objective:.4f}',
                'Vocab': f'{forward_info["vocabulary_score"]:.2f}',
                'Truss': f'{forward_info["truss_complexity"]:.2f}',
                'Grad': f'{float(jnp.linalg.norm(grads)):.3f}'
            })
            
            # Detailed, mechanics-focused logging every N iterations
            if iteration % 10 == 0:
                self.std_logger.info(f"--- Iteration {iteration} ---")
                self.std_logger.info(f"  Loss: {loss:.4f} (Improvement: {improvement:.6f}) | Best: {best_objective:.4f}")
                self.std_logger.info(f"  -> Vocab Score: {forward_info['vocabulary_score']:.2f} | Truss Complexity: {forward_info['truss_complexity']:.2f} (L2: {forward_info['l2_penalty']:.4f})")
                
                vocab_info = forward_info.get('vocab_info', {})
                self.std_logger.info(f"  -> Geometry Realizability: Mean={vocab_info.get('mean_realizability', 0):.3f}, Max={vocab_info.get('max_realizability', 0):.3f}")

                truss_info = forward_info.get('truss_info', {})
                self.std_logger.info(f"  -> Truss Stats: Estimated Unique Trusses={truss_info.get('estimated_unique_count', 0):.2f}")

                grad_norm = float(jnp.linalg.norm(grads))
                param_norm = float(jnp.linalg.norm(params))
                self.std_logger.info(f"  -> Gradients Norm: {grad_norm:.4f} | Params Norm: {param_norm:.4f}")

            # Comprehensive logging
            metrics = {
                'objective': float(loss),
                'improvement': float(improvement),
                'best_objective': float(best_objective),
                'gradient_norm': float(jnp.linalg.norm(grads)),
                'parameter_norm': float(jnp.linalg.norm(params)),
                **forward_info.get('metrics', {})
            }
            
            # Log to our comprehensive system
            self.logger.log_iteration(
                iteration=iteration,
                metrics=metrics,
                parameters=params,
                gradients=grads,
                forward_pass_info=forward_info
            )
            
            # Legacy logging for base class
            self._log_iteration(loss, params)
            
            # Simple convergence check
            if improvement < self.config.convergence_threshold:
                self.std_logger.info(f"✅ Converged at iteration {iteration}")
                break
            
            prev_loss = loss
        
        pbar.close()
        
        # Convert best parameters to hole layout
        best_layout = self._params_to_layout(best_params)
        
        # Evaluate final solution
        final_objective = self._objective_function(best_params)
        final_metrics = OptimizationMetrics(
            vocabulary_size=len(self.geometry_calculator.enumerate_geometries(best_layout)),
            num_unique_trusses=self._estimate_unique_trusses(best_layout),
            objective_value=float(final_objective)
        )
        
        # Generate geometries
        best_geometries = self.geometry_calculator.enumerate_geometries(best_layout)
        
        # Create result
        result = OptimizationResult(
            best_hole_layout=best_layout,
            best_geometries=best_geometries,
            best_metrics=final_metrics,
            pareto_solutions=[(best_layout, final_metrics)],
            objective_history=self.history['objectives'],
            parameter_history=self.history['parameters'],
            total_time=time.time() - self.start_time,
            iterations=self.iteration + 1,
            converged=improvement < self.config.convergence_threshold,
            metadata={
                'config': self.config.__dict__,
                'final_loss': float(final_objective),
                'best_hole_layout': {
                    'handle': result.best_hole_layout.handle_holes,
                    'vertical': result.best_hole_layout.vertical_holes,
                    'forearm': result.best_hole_layout.forearm_holes
                }
            }
        )
        
        # Log final results to console
        self.std_logger.info("--- ✅ Final Results ---")
        self.std_logger.info(f"Best Objective: {result.best_metrics.objective_value:.4f}")
        self.std_logger.info(f"Vocabulary Size: {result.best_metrics.vocabulary_size}")
        self.std_logger.info(f"Unique Trusses: {self._estimate_unique_trusses(result.best_hole_layout)}")
        self.std_logger.info("Final Hole Layout:")
        self.std_logger.info(f"  Handle: {[round(h, 2) for h in result.best_hole_layout.handle_holes]}")
        self.std_logger.info(f"  Vertical: {[round(v, 2) for v in result.best_hole_layout.vertical_holes]}")
        self.std_logger.info(f"  Forearm: {[round(f, 2) for f in result.best_hole_layout.forearm_holes]}")
        
        # Log final results
        self.logger.log_final_results(result)
        return result
    
    def _get_realizability_checker(self):
        """Returns a jitted and vmapped version of the realizability checker."""
        # Use a simple cache to avoid re-compiling the function on every call
        if not hasattr(self, '_vmapped_realizability_checker'):
            # Partial function to fix arguments that don't change per-sample
            realizability_func = partial(
                self._soft_geometry_realizability
            )
            # vmap over the (layout, alpha, beta, gamma) arguments
            self._vmapped_realizability_checker = jax.jit(
                jax.vmap(realizability_func, in_axes=(None, 0, 0, 0, None, None)),
                static_argnames=['constraints', 'config']
            )
        return self._vmapped_realizability_checker

    def _objective_function_with_info(self, params: jnp.ndarray) -> Tuple[float, jnp.ndarray, Dict[str, Any]]:
        """Objective function with detailed logging information."""
        self.std_logger.info("🎯 Starting objective function computation...")
        
        # Use JAX's value_and_grad with has_aux=True to return extra info
        def objective_with_info(params):
            self.std_logger.info("🔄 Converting parameters to hole layout...")
            # Convert to hole layout (differentiably)
            layout = self._params_to_layout_differentiable(params)
            
            self.std_logger.info("📈 Computing vocabulary score...")
            # Calculate differentiable metrics with detailed tracking
            vocab_score, vocab_info = self._differentiable_vocabulary_score(layout, return_info=True)
            
            self.std_logger.info("🔧 Computing truss complexity...")
            truss_complexity, truss_info = self._differentiable_truss_complexity(layout, return_info=True)
            
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
    
    def _objective_function(self, params: jnp.ndarray) -> float:
        """Simple objective function without logging overhead."""
        layout = self._params_to_layout_differentiable(params)
        vocab_score = self._differentiable_vocabulary_score(layout)
        truss_complexity = self._differentiable_truss_complexity(layout)
        
        objective = (-vocab_score * self.objectives.vocabulary_weight + 
                    truss_complexity * self.objectives.truss_complexity_weight)
        l2_penalty = self.config.l2_reg * jnp.sum(params ** 2)
        
        return objective + l2_penalty
    
    def _initialize_params(self) -> jnp.ndarray:
        """Initialize optimization parameters."""
        # Initialize with reasonable hole distributions
        handle_init = jnp.linspace(0.1, 0.9, self.max_handle_holes)
        vertical_init = jnp.linspace(0.1, 0.9, self.max_vertical_holes)
        forearm_init = jnp.linspace(0.1, 0.9, self.max_forearm_holes)
        
        # Add small random perturbations
        self.key, *subkeys = random.split(self.key, 4)
        handle_noise = random.normal(subkeys[0], (self.max_handle_holes,)) * 0.05
        vertical_noise = random.normal(subkeys[1], (self.max_vertical_holes,)) * 0.05
        forearm_noise = random.normal(subkeys[2], (self.max_forearm_holes,)) * 0.05
        
        params = jnp.concatenate([
            jnp.clip(handle_init + handle_noise, 0.0, 1.0),
            jnp.clip(vertical_init + vertical_noise, 0.0, 1.0),
            jnp.clip(forearm_init + forearm_noise, 0.0, 1.0)
        ])
        
        return params
    
    def _layout_to_params(self, layout: HoleLayout) -> jnp.ndarray:
        """Convert hole layout to parameter vector."""
        def normalize_positions(positions: List[float], max_length: float) -> jnp.ndarray:
            """Normalize positions to [0, 1] range."""
            if not positions:
                return jnp.zeros(0)
            return jnp.array(positions) / max_length
        
        def pad_or_truncate(arr: jnp.ndarray, target_size: int) -> jnp.ndarray:
            """Pad with zeros or truncate to target size."""
            if len(arr) >= target_size:
                return arr[:target_size]
            else:
                return jnp.pad(arr, (0, target_size - len(arr)), constant_values=0.0)
        
        # Normalize and pad each rod type
        handle_params = pad_or_truncate(
            normalize_positions(layout.handle_holes, self.constraints.handle_length), 
            self.max_handle_holes
        )
        vertical_params = pad_or_truncate(
            normalize_positions(layout.vertical_holes, self.constraints.vertical_length),
            self.max_vertical_holes
        )
        forearm_params = pad_or_truncate(
            normalize_positions(layout.forearm_holes, self.constraints.forearm_length),
            self.max_forearm_holes
        )
        
        return jnp.concatenate([handle_params, vertical_params, forearm_params])
    
    def _params_to_layout(self, params: jnp.ndarray) -> HoleLayout:
        """Convert parameters to hole layout (non-differentiable, for final output)."""
        # Split parameter vector by rod type
        handle_params = params[:self.max_handle_holes]
        vertical_params = params[self.max_handle_holes:self.max_handle_holes + self.max_vertical_holes]
        forearm_params = params[self.max_handle_holes + self.max_vertical_holes:]
        
        def extract_valid_holes(hole_params: jnp.ndarray, max_length: float, margin: float) -> List[float]:
            """Extract valid hole positions from parameters."""
            # Convert to actual positions
            positions = hole_params * max_length
            # Filter out positions that are too close to edges or zeros
            valid_positions = []
            for pos in positions:
                if margin <= pos <= max_length - margin:
                    valid_positions.append(float(pos))
            return sorted(valid_positions)
        
        return HoleLayout(
            handle_holes=extract_valid_holes(handle_params, self.constraints.handle_length, self.constraints.hole_margin),
            vertical_holes=extract_valid_holes(vertical_params, self.constraints.vertical_length, self.constraints.hole_margin),
            forearm_holes=extract_valid_holes(forearm_params, self.constraints.forearm_length, self.constraints.hole_margin)
        )
    
    def _params_to_layout_differentiable(self, params: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Convert parameters to hole layout (differentiable, for optimization)."""
        # Split parameter vector by rod type
        handle_params = params[:self.max_handle_holes]
        vertical_params = params[self.max_handle_holes:self.max_handle_holes + self.max_vertical_holes]
        forearm_params = params[self.max_handle_holes + self.max_vertical_holes:]
        
        # Convert to actual positions (differentiably)
        handle_holes = handle_params * self.constraints.handle_length
        vertical_holes = vertical_params * self.constraints.vertical_length
        forearm_holes = forearm_params * self.constraints.forearm_length
        
        return {
            'handle': handle_holes,
            'vertical': vertical_holes,
            'forearm': forearm_holes
        }
    
    def _differentiable_vocabulary_score(self, layout_dict: Dict[str, jnp.ndarray], return_info: bool = False):
        """Differentiable approximation of vocabulary size using vmap."""
        self.std_logger.info("🔍 Computing vocabulary score...")
        
        alpha_samples = jnp.linspace(self.constraints.alpha_min, self.constraints.alpha_max, self.config.vocab_angle_samples)
        beta_samples = jnp.linspace(self.constraints.beta_min, self.constraints.beta_max, self.config.vocab_angle_samples)
        gamma_samples = jnp.linspace(self.constraints.gamma_min, self.constraints.gamma_max, self.config.vocab_angle_samples // 2)

        alphas, betas, gammas = jnp.meshgrid(alpha_samples, beta_samples, gamma_samples, indexing='ij')
        
        # Create a mask for valid combinations
        valid_mask = (alphas + betas) >= 180.0 if self.constraints.require_alpha_beta_sum_ge_180 else jnp.ones_like(alphas, dtype=bool)
        
        # Get the vectorized realizability checker
        vmapped_checker = self._get_realizability_checker()

        # Run the checker on all combinations at once
        all_realizabilities = vmapped_checker(layout_dict, alphas, betas, gammas, self.constraints, self.config)
        
        # Apply the mask to ignore invalid combinations
        achievable_score = jnp.sum(all_realizabilities * valid_mask)
        
        if return_info:
            valid_realizabilities = all_realizabilities[valid_mask]
            info = {
                'vocab_info': {
                    'total_angle_combinations_tested': jnp.sum(valid_mask),
                    'mean_realizability': jnp.mean(valid_realizabilities),
                    'max_realizability': jnp.max(valid_realizabilities),
                    'alpha_range': [self.constraints.alpha_min, self.constraints.alpha_max],
                    'beta_range': [self.constraints.beta_min, self.constraints.beta_max],
                    'gamma_range': [self.constraints.gamma_min, self.constraints.gamma_max]
                }
            }
            return achievable_score, info
            
        return achievable_score
    
    @staticmethod
    @jax.jit
    def _soft_geometry_realizability(layout_dict: Dict[str, jnp.ndarray], alpha: float, beta: float, gamma: float, constraints: CrutchConstraints, config: DifferentiableConfig) -> float:
        """JIT-compatible, static method to check realizability for a single geometry."""
        handle_holes = layout_dict['handle']
        vertical_holes = layout_dict['vertical']
        forearm_holes = layout_dict['forearm']

        vertical_pivot = constraints.vertical_pivot_length
        forearm_pivot = vertical_pivot + gamma
        
        # Log the geometry being tested (only occasionally to avoid spam)
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
            
        if self._log_counter % 100 == 0:  # Log every 100th geometry
            self.std_logger.info(f"🔍 Testing geometry: α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}cm")

        # --- T1 Feasibility ---
        handle_for_t1 = handle_holes[handle_holes < vertical_pivot]
        # To avoid nested loops in JAX, we can create mesh grids
        v_grid_t1, h_grid_t1 = jnp.meshgrid(vertical_holes, handle_for_t1)
        r1_t1, r2_t1 = v_grid_t1, jnp.abs(vertical_pivot - h_grid_t1)
        t1_lengths = jnp.sqrt(r1_t1**2 + r2_t1**2 - 2 * r1_t1 * r2_t1 * jnp.cos(jnp.deg2rad(180.0 - alpha)))
        t1_feasibilities = self._calculate_truss_feasibility(t1_lengths, r1_t1, r2_t1, config)
        max_f1 = jnp.max(t1_feasibilities) if t1_feasibilities.size > 0 else 0.0

        # --- T2 Feasibility ---
        handle_for_t2 = handle_holes[handle_holes > vertical_pivot]
        v_grid_t2, h_grid_t2 = jnp.meshgrid(vertical_holes, handle_for_t2)
        r1_t2, r2_t2 = v_grid_t2, jnp.abs(h_grid_t2 - vertical_pivot)
        t2_lengths = jnp.sqrt(r1_t2**2 + r2_t2**2 - 2 * r1_t2 * r2_t2 * jnp.cos(jnp.deg2rad(alpha)))
        t2_feasibilities = self._calculate_truss_feasibility(t2_lengths, r1_t2, r2_t2, config)
        max_f2 = jnp.max(t2_feasibilities) if t2_feasibilities.size > 0 else 0.0
        
        # --- T3 Feasibility ---
        handle_for_t3 = handle_holes[handle_holes < forearm_pivot]
        f_grid_t3, h_grid_t3 = jnp.meshgrid(forearm_holes, handle_for_t3)
        r1_t3, r2_t3 = f_grid_t3, jnp.abs(forearm_pivot - h_grid_t3)
        t3_lengths = jnp.sqrt(r1_t3**2 + r2_t3**2 - 2 * r1_t3 * r2_t3 * jnp.cos(jnp.deg2rad(180.0 - beta)))
        t3_feasibilities = self._calculate_truss_feasibility(t3_lengths, r1_t3, r2_t3, config)
        max_f3 = jnp.max(t3_feasibilities) if t3_feasibilities.size > 0 else 0.0

        # A geometry is realizable only if we can find good candidates for T1, T2, AND T3.
        return max_f1 * max_f2 * max_f3
    
    @staticmethod
    def _calculate_truss_feasibility(truss_length: jnp.ndarray, r1: jnp.ndarray, r2: jnp.ndarray, config: DifferentiableConfig) -> jnp.ndarray:
        """Calculates a feasibility score for a single truss based on its length and the internal angles it forms."""
        # 1. Length feasibility with preference for longer trusses
        is_in_range = (truss_length >= config.truss_length_min) & (truss_length <= config.truss_length_max)
        
        # Normalize length to [0, 1] within the feasible range to score longer trusses higher
        normalized_length = (truss_length - config.truss_length_min) / (config.truss_length_max - config.truss_length_min + 1e-6)
        length_feasibility = is_in_range * normalized_length

        # 2. Angle feasibility (45-degree preference)
        # Law of cosines to find internal angles
        cos_angle1 = (truss_length**2 + r1**2 - r2**2) / (2 * truss_length * r1 + 1e-6)
        cos_angle2 = (truss_length**2 + r2**2 - r1**2) / (2 * truss_length * r2 + 1e-6)
        angle1 = jnp.rad2deg(jnp.arccos(jnp.clip(cos_angle1, -1.0, 1.0)))
        angle2 = jnp.rad2deg(jnp.arccos(jnp.clip(cos_angle2, -1.0, 1.0)))

        # Soft penalty for deviating from 45 degrees
        angle_error = jnp.abs(angle1 - 45.0) + jnp.abs(angle2 - 45.0)
        angle_feasibility = jax.nn.sigmoid(-(angle_error - 10.0) * 0.1) # Penalize if total error > 10 degrees

        return length_feasibility * angle_feasibility
    
    def _sample_hole_combinations(self, handle_holes: jnp.ndarray, vertical_holes: jnp.ndarray, 
                                 forearm_holes: jnp.ndarray, n_samples: int) -> List[Tuple[float, float, float]]:
        """Sample hole combinations uniformly."""
        combinations = []
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
    
    def _differentiable_truss_complexity(self, layout_dict: Dict[str, jnp.ndarray], return_info: bool = False):
        """Estimate the number of unique truss lengths required."""
        self.std_logger.info("🔧 Computing truss complexity...")
        
        handle_holes = layout_dict['handle']
        vertical_holes = layout_dict['vertical']
        forearm_holes = layout_dict['forearm']
        
        self.std_logger.info(f"📊 Hole counts: Handle={len(handle_holes)}, Vertical={len(vertical_holes)}, Forearm={len(forearm_holes)}")
        
        # Sample truss lengths
        sample_trusses = []
        
        alpha_samples = jnp.linspace(self.constraints.alpha_min, self.constraints.alpha_max, self.config.truss_angle_samples)
        gamma_samples = jnp.linspace(self.constraints.gamma_min, self.constraints.gamma_max, self.config.truss_angle_samples // 2)
        
        total_combinations = len(alpha_samples) * len(gamma_samples) * self.config.truss_hole_samples
        self.std_logger.info(f"🎯 Sampling {len(alpha_samples)} α × {len(gamma_samples)} γ × {self.config.truss_hole_samples} hole combinations = {total_combinations} total")
        
        for alpha in alpha_samples:
            for gamma in gamma_samples:
                # Uniformly sample hole combinations
                combinations = self._sample_hole_combinations(handle_holes, vertical_holes, forearm_holes, self.config.truss_hole_samples)
                
                vertical_pivot = self.constraints.vertical_pivot_length
                
                for h_pos, v_pos, f_pos in combinations:
                    # Truss calculations (handle-vertical connection)
                    if h_pos < vertical_pivot:
                        r1, r2 = v_pos, abs(vertical_pivot - h_pos)
                        t1 = jnp.sqrt(r1**2 + r2**2 - 2*r1*r2*jnp.cos(jnp.deg2rad(180.0 - alpha)))
                        sample_trusses.append(t1)
                    
                    if h_pos > vertical_pivot:
                        r1, r2 = v_pos, abs(h_pos - vertical_pivot)
                        t2 = jnp.sqrt(r1**2 + r2**2 - 2*r1*r2*jnp.cos(jnp.deg2rad(alpha)))
                        sample_trusses.append(t2)
        
        if not sample_trusses:
            unique_count = 100.0  # High penalty for no valid trusses
        else:
            sample_trusses = jnp.array(sample_trusses)
            unique_count = self._differentiable_unique_count(sample_trusses)
        
        if return_info:
            info = {
                'truss_info': {
                    'total_truss_samples': len(sample_trusses) if sample_trusses else 0,
                    'mean_truss_length': float(jnp.mean(sample_trusses)) if sample_trusses else 0.0,
                    'estimated_unique_count': float(unique_count),
                    'sample_truss_lengths': [float(t) for t in sample_trusses[:10]] if sample_trusses else []
                }
            }
            return unique_count, info
        
        return unique_count
    
    def _differentiable_unique_count(self, values: jnp.ndarray, bandwidth: float = 1.0) -> float:
        """Differentiable approximation of unique value count using participation ratio."""
        if len(values) == 0: 
            return 0.0
        n = len(values)
        if n == 1: 
            return 1.0
        
        # Compute pairwise distances and similarities
        distances = jnp.abs(values[:, None] - values[None, :])
        similarities = jnp.exp(-distances**2 / (2 * bandwidth**2))
        normalized_similarities = similarities / n
        
        # Compute participation ratio
        eigenvalues = jnp.linalg.eigvals(normalized_similarities)
        eigenvalues = jnp.maximum(eigenvalues.real, 1e-12)
        eigenvalue_probs = eigenvalues / jnp.sum(eigenvalues)
        participation_ratio = 1.0 / jnp.sum(eigenvalue_probs**2)
        
        return jnp.minimum(participation_ratio, float(n))
    
    def _estimate_unique_trusses(self, layout: HoleLayout) -> int:
        """Estimate unique truss count for final metrics (non-differentiable)."""
        # Simple enumeration for final result, now including gamma
        trusses = set()
        for alpha in np.linspace(self.constraints.alpha_min, self.constraints.alpha_max, 10):
            for gamma in np.linspace(self.constraints.gamma_min, self.constraints.gamma_max, 5):
                vertical_pivot = self.constraints.vertical_pivot_length
                forearm_pivot = vertical_pivot + gamma
                
                # Sample a few hole combinations
                for h_pos in layout.handle_holes[:3]:
                    for v_pos in layout.vertical_holes[:3]:
                        # Truss 1/2
                        if h_pos < vertical_pivot:
                            r1, r2 = v_pos, abs(vertical_pivot - h_pos)
                            t1 = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(np.deg2rad(180.0 - alpha)))
                            trusses.add(round(t1, 1))
                        else:
                            r1, r2 = v_pos, abs(h_pos - vertical_pivot)
                            t2 = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(np.deg2rad(alpha)))
                            trusses.add(round(t2, 1))
                            
                        # Truss 3 (we need a beta sample for this)
                        for beta in np.linspace(self.constraints.beta_min, self.constraints.beta_max, 10):
                             if h_pos < forearm_pivot:
                                for f_pos in layout.forearm_holes[:3]:
                                    r1_t3, r2_t3 = f_pos, abs(forearm_pivot - h_pos)
                                    t3 = np.sqrt(r1_t3**2 + r2_t3**2 - 2 * r1_t3 * r2_t3 * np.cos(np.deg2rad(180.0 - beta)))
                                    trusses.add(round(t3, 1))
        return len(trusses)
