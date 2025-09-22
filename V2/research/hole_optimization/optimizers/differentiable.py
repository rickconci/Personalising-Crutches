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
from ..gpu_config import GPUConfig, setup_gpu_for_optimization


class DifferentiableOptimizer(BaseOptimizer):
    """Differentiable optimizer using continuous relaxation and gradient-based methods."""
    
    def __init__(
        self,
        constraints: CrutchConstraints,
        objectives: OptimizationObjectives,
        config: DifferentiableConfig,
        random_seed: int = 42,
        logger: Optional[OptimizationLogger] = None,
        use_gpu: bool = False,
        gpu_memory_fraction: float = 0.8
    ):
        """Initialize differentiable optimizer."""
        super().__init__(constraints, objectives, random_seed)
        self.config = config
        self.key = random.PRNGKey(random_seed)
        
        # Setup GPU configuration
        self.gpu_config = setup_gpu_for_optimization(
            use_gpu=use_gpu,
            gpu_memory_fraction=gpu_memory_fraction,
            verbose=True
        )
        
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
                'Grad': f'{jnp.linalg.norm(grads):.3f}'
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

                grad_norm = jnp.linalg.norm(grads)
                param_norm = jnp.linalg.norm(params)
                self.std_logger.info(f"  -> Gradients Norm: {grad_norm:.4f} | Params Norm: {param_norm:.4f}")

            # Comprehensive logging
            metrics = {
                'objective': loss,
                'improvement': improvement,
                'best_objective': best_objective,
                'gradient_norm': jnp.linalg.norm(grads),
                'parameter_norm': jnp.linalg.norm(params),
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
            objective_value=final_objective
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
                'final_loss': final_objective,
                'best_hole_layout': {
                    'handle': result.best_hole_layout.handle,
                    'vertical': result.best_hole_layout.vertical,
                    'forearm': result.best_hole_layout.forearm
                }
            }
        )
        
        # Log final results to console
        self.std_logger.info("--- ✅ Final Results ---")
        self.std_logger.info(f"Best Objective: {result.best_metrics.objective_value:.4f}")
        self.std_logger.info(f"Vocabulary Size: {result.best_metrics.vocabulary_size}")
        self.std_logger.info(f"Unique Trusses: {self._estimate_unique_trusses(result.best_hole_layout)}")
        self.std_logger.info("Final Hole Layout:")
        self.std_logger.info(f"  Handle: {[round(h, 2) for h in result.best_hole_layout.handle]}")
        self.std_logger.info(f"  Vertical: {[round(v, 2) for v in result.best_hole_layout.vertical]}")
        self.std_logger.info(f"  Forearm: {[round(f, 2) for f in result.best_hole_layout.forearm]}")
        
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
                'vocabulary_score': vocab_score,
                'truss_complexity': truss_complexity,
                'l2_penalty': l2_penalty,
                'raw_objective': objective,
                'total_loss': total_loss,
                'hole_counts': {
                    'handle': len(layout['handle']),
                    'vertical': len(layout['vertical']),
                    'forearm': len(layout['forearm'])
                },
                'metrics': {
                    'vocabulary_score': vocab_score,
                    'truss_complexity': truss_complexity,
                    'l2_penalty': l2_penalty
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
        """Initialize optimization parameters with evenly spaced holes."""
        if self.config.use_uniform_initialization:
            # Create uniform hole layout first
            from ..geometry import create_uniform_holes
            uniform_layout = create_uniform_holes(self.constraints)
            
            # Convert to parameters (this ensures proper spacing)
            params = self._layout_to_params(uniform_layout)
            
            # Add small random perturbations if configured
            if self.config.initialization_noise > 0:
                self.key, *subkeys = random.split(self.key, 4)
                handle_noise = random.normal(subkeys[0], (self.max_handle_holes,)) * self.config.initialization_noise
                vertical_noise = random.normal(subkeys[1], (self.max_vertical_holes,)) * self.config.initialization_noise
                forearm_noise = random.normal(subkeys[2], (self.max_forearm_holes,)) * self.config.initialization_noise
                
                # Apply perturbations while maintaining constraints
                handle_params = jnp.clip(params[:self.max_handle_holes] + handle_noise, 0.0, 1.0)
                vertical_params = jnp.clip(params[self.max_handle_holes:self.max_handle_holes + self.max_vertical_holes] + vertical_noise, 0.0, 1.0)
                forearm_params = jnp.clip(params[self.max_handle_holes + self.max_vertical_holes:] + forearm_noise, 0.0, 1.0)
                params = jnp.concatenate([handle_params, vertical_params, forearm_params])
            
            # Log the uniform parameters for verification
            self.std_logger.info(f"📍 Uniform hole layout created:")
            self.std_logger.info(f"  Handle holes (cm): {uniform_layout.handle}")
            self.std_logger.info(f"  Vertical holes (cm): {uniform_layout.vertical}")
            self.std_logger.info(f"  Forearm holes (cm): {uniform_layout.forearm}")
            self.std_logger.info(f"📍 Normalized parameters [0,1]: {params}")
            
            # Verify spacing is uniform
            if len(uniform_layout.handle) > 1:
                handle_spacing = [round(float(uniform_layout.handle[i+1] - uniform_layout.handle[i]), 1) for i in range(len(uniform_layout.handle)-1)]
                self.std_logger.info(f"  Handle spacing (cm): {handle_spacing}")
            if len(uniform_layout.vertical) > 1:
                vertical_spacing = [round(float(uniform_layout.vertical[i+1] - uniform_layout.vertical[i]), 1) for i in range(len(uniform_layout.vertical)-1)]
                self.std_logger.info(f"  Vertical spacing (cm): {vertical_spacing}")
            if len(uniform_layout.forearm) > 1:
                forearm_spacing = [round(float(uniform_layout.forearm[i+1] - uniform_layout.forearm[i]), 1) for i in range(len(uniform_layout.forearm)-1)]
                self.std_logger.info(f"  Forearm spacing (cm): {forearm_spacing}")
            
            return params
        else:
            # Fallback to random initialization
            self.std_logger.info("📍 Using random initialization (not uniform)")
            return self._random_initialize_params()
    
    def _random_initialize_params(self) -> jnp.ndarray:
        """Random initialization fallback."""
        # Initialize with reasonable hole distributions
        handle_init = jnp.linspace(0.1, 0.9, self.max_handle_holes)
        vertical_init = jnp.linspace(0.1, 0.9, self.max_vertical_holes)
        forearm_init = jnp.linspace(0.1, 0.9, self.max_forearm_holes)
        
        # Add random perturbations
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
        def normalize_positions(positions: jnp.ndarray, max_length: float) -> jnp.ndarray:
            """Normalize positions to [0, 1] range."""
            if positions.size == 0:
                return jnp.zeros(0)
            return positions / max_length
        
        def pad_or_truncate(arr: jnp.ndarray, target_size: int) -> jnp.ndarray:
            """Pad with zeros or truncate to target size."""
            if len(arr) >= target_size:
                return arr[:target_size]
            else:
                return jnp.pad(arr, (0, target_size - len(arr)), constant_values=0.0)
        
        # Normalize and pad each rod type
        handle_params = pad_or_truncate(
            normalize_positions(layout.handle, self.constraints.handle_length), 
            self.max_handle_holes
        )
        vertical_params = pad_or_truncate(
            normalize_positions(layout.vertical, self.constraints.vertical_length),
            self.max_vertical_holes
        )
        forearm_params = pad_or_truncate(
            normalize_positions(layout.forearm, self.constraints.forearm_length),
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
            handle=extract_valid_holes(handle_params, self.constraints.handle_length, self.constraints.hole_margin),
            vertical=extract_valid_holes(vertical_params, self.constraints.vertical_length, self.constraints.hole_margin),
            forearm=extract_valid_holes(forearm_params, self.constraints.forearm_length, self.constraints.hole_margin)
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
        gamma_samples = jnp.linspace(self.constraints.gamma_min, self.constraints.gamma_max, self.config.vocab_gamma_samples)

        alphas, betas, gammas = jnp.meshgrid(alpha_samples, beta_samples, gamma_samples, indexing='ij')
        
        # Flatten the arrays for vmap
        alphas_flat = alphas.flatten()
        betas_flat = betas.flatten()
        gammas_flat = gammas.flatten()
        
        # Create a mask for valid combinations
        valid_mask = (alphas_flat + betas_flat) >= 180.0 if self.constraints.require_alpha_beta_sum_ge_180 else jnp.ones_like(alphas_flat, dtype=bool)
        
        # Get the vectorized realizability checker
        vmapped_checker = self._get_realizability_checker()

        # Run the checker on all combinations at once
        all_realizabilities = vmapped_checker(layout_dict, alphas_flat, betas_flat, gammas_flat, self.constraints, self.config)
        
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
    def _soft_geometry_realizability(layout_dict: Dict[str, jnp.ndarray], alpha: float, beta: float, gamma: float, constraints: CrutchConstraints, config: DifferentiableConfig) -> float:
        """JIT-compatible, static method to check realizability for a single geometry."""
        handle_holes = layout_dict['handle']
        vertical_holes = layout_dict['vertical']
        forearm_holes = layout_dict['forearm']

        vertical_pivot = constraints.vertical_pivot_length
        forearm_pivot = vertical_pivot + gamma

        # Simplified approach: sample a few combinations instead of all combinations
        # This avoids the complex meshgrid shape issues
        
        # Sample a few handle-vertical combinations for T1 and T2
        max_combinations = min(5, len(handle_holes), len(vertical_holes))
        t1_feasibilities = []
        t2_feasibilities = []
        
        for i in range(max_combinations):
            h_pos = handle_holes[i % len(handle_holes)]
            v_pos = vertical_holes[i % len(vertical_holes)]
            
            # T1: handle < vertical_pivot
            t1_mask = h_pos < vertical_pivot
            r1_t1, r2_t1 = v_pos, jnp.abs(vertical_pivot - h_pos)
            t1_length = jnp.sqrt(r1_t1**2 + r2_t1**2 - 2*r1_t1*r2_t1*jnp.cos(jnp.deg2rad(180.0 - alpha)))
            t1_feas = DifferentiableOptimizer._calculate_truss_feasibility_single(t1_length, r1_t1, r2_t1, config)
            t1_feasibilities.append(t1_feas * t1_mask)
            
            # T2: handle > vertical_pivot
            t2_mask = h_pos > vertical_pivot
            r1_t2, r2_t2 = v_pos, jnp.abs(h_pos - vertical_pivot)
            t2_length = jnp.sqrt(r1_t2**2 + r2_t2**2 - 2*r1_t2*r2_t2*jnp.cos(jnp.deg2rad(alpha)))
            t2_feas = DifferentiableOptimizer._calculate_truss_feasibility_single(t2_length, r1_t2, r2_t2, config)
            t2_feasibilities.append(t2_feas * t2_mask)
        
        # Sample a few handle-forearm combinations for T3
        t3_feasibilities = []
        max_combinations_t3 = min(5, len(handle_holes), len(forearm_holes))
        
        for i in range(max_combinations_t3):
            h_pos = handle_holes[i % len(handle_holes)]
            f_pos = forearm_holes[i % len(forearm_holes)]
            
            # T3: handle < forearm_pivot
            t3_mask = h_pos < forearm_pivot
            r1_t3, r2_t3 = f_pos, jnp.abs(forearm_pivot - h_pos)
            t3_length = jnp.sqrt(r1_t3**2 + r2_t3**2 - 2*r1_t3*r2_t3*jnp.cos(jnp.deg2rad(180.0 - beta)))
            t3_feas = DifferentiableOptimizer._calculate_truss_feasibility_single(t3_length, r1_t3, r2_t3, config)
            t3_feasibilities.append(t3_feas * t3_mask)
        
        # Get maximum feasibilities
        max_f1 = jnp.max(jnp.array(t1_feasibilities)) if t1_feasibilities else 0.0
        max_f2 = jnp.max(jnp.array(t2_feasibilities)) if t2_feasibilities else 0.0
        max_f3 = jnp.max(jnp.array(t3_feasibilities)) if t3_feasibilities else 0.0

        # A geometry is realizable only if we can find good candidates for T1, T2, AND T3.
        return max_f1 * max_f2 * max_f3
    
    @staticmethod
    def _calculate_truss_feasibility_single(truss_length: float, r1: float, r2: float, config: DifferentiableConfig) -> float:
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
        # Generate random indices using JAX
        key = self.key
        h_indices = jax.random.randint(key, (n_samples,), 0, len(handle_holes))
        key, _ = jax.random.split(key)
        v_indices = jax.random.randint(key, (n_samples,), 0, len(vertical_holes))
        key, _ = jax.random.split(key)
        f_indices = jax.random.randint(key, (n_samples,), 0, len(forearm_holes))
        
        combinations = []
        for i in range(n_samples):
            combinations.append((
                handle_holes[h_indices[i]],
                vertical_holes[v_indices[i]],
                forearm_holes[f_indices[i]]
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
        beta_samples = jnp.linspace(self.constraints.beta_min, self.constraints.beta_max, self.config.truss_angle_samples)
        gamma_samples = jnp.linspace(self.constraints.gamma_min, self.constraints.gamma_max, self.config.truss_gamma_samples)
        
        total_combinations = len(alpha_samples) * len(beta_samples) * len(gamma_samples) * self.config.truss_hole_samples
        self.std_logger.info(f"🎯 Sampling {len(alpha_samples)} α × {len(beta_samples)} β × {len(gamma_samples)} γ × {self.config.truss_hole_samples} hole combinations = {total_combinations} total")
        
        for alpha in alpha_samples:
            for beta in beta_samples:
                for gamma in gamma_samples:
                    # Uniformly sample hole combinations
                    combinations = self._sample_hole_combinations(handle_holes, vertical_holes, forearm_holes, self.config.truss_hole_samples)
                    
                    vertical_pivot = self.constraints.vertical_pivot_length
                    forearm_pivot = vertical_pivot + gamma
                    
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
                        
                        # Truss 3 (handle-forearm connection)
                        if h_pos < forearm_pivot:
                            r1_t3, r2_t3 = f_pos, abs(forearm_pivot - h_pos)
                            t3 = jnp.sqrt(r1_t3**2 + r2_t3**2 - 2*r1_t3*r2_t3*jnp.cos(jnp.deg2rad(180.0 - beta)))
                            sample_trusses.append(t3)
        
        sample_trusses = jnp.array(sample_trusses)
        if sample_trusses.size == 0:
            unique_count = 100.0  # High penalty for no valid trusses
        else:
            unique_count = self._differentiable_unique_count(sample_trusses)
        
        if return_info:
            info = {
                'truss_info': {
                    'total_truss_samples': sample_trusses.size,
                    'mean_truss_length': jnp.mean(sample_trusses) if sample_trusses.size > 0 else 0.0,
                    'estimated_unique_count': unique_count,
                    'sample_truss_lengths': sample_trusses[:10] if sample_trusses.size > 0 else jnp.array([]),
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
                for h_pos in layout.handle[:3]:
                    for v_pos in layout.vertical[:3]:
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
                                for f_pos in layout.forearm[:3]:
                                    r1_t3, r2_t3 = f_pos, abs(forearm_pivot - h_pos)
                                    t3 = np.sqrt(r1_t3**2 + r2_t3**2 - 2 * r1_t3 * r2_t3 * np.cos(np.deg2rad(180.0 - beta)))
                                    trusses.add(round(t3, 1))
        return len(trusses)
