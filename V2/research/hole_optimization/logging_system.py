"""Comprehensive logging and monitoring system for optimization experiments."""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pathlib import Path
import logging
import time
import json
from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp

# Optional dependencies
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class LoggingMode(Enum):
    """Logging detail levels."""
    MINIMAL = "minimal"      # Only final results
    STANDARD = "standard"    # Key metrics every N iterations
    DETAILED = "detailed"    # All metrics every iteration
    DEBUG = "debug"         # Everything including intermediate calculations


class LoggingInterest(Enum):
    """What aspects to focus logging on."""
    CONVERGENCE = "convergence"     # Objective values, gradients
    GEOMETRY = "geometry"          # Vocabulary, truss complexity
    PARAMETERS = "parameters"      # Hole positions, parameter evolution
    PERFORMANCE = "performance"    # Runtime, memory usage
    FORWARD_PASS = "forward_pass"  # Forward pass details
    ALL = "all"                   # Everything


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    mode: LoggingMode = LoggingMode.STANDARD
    interests: List[LoggingInterest] = field(default_factory=lambda: [LoggingInterest.CONVERGENCE])
    
    # Frequency settings
    log_every_n_iterations: int = 10
    plot_every_n_iterations: int = 100
    save_checkpoint_every_n: int = 500
    
    # Output settings
    output_dir: Path = Path("logs")
    use_wandb: bool = True
    wandb_project: str = "hole-optimization"
    wandb_entity: Optional[str] = None
    
    # Plotting settings
    create_live_plots: bool = True
    save_parameter_evolution: bool = True
    save_geometry_evolution: bool = True
    
    # Performance monitoring
    track_memory: bool = False
    track_gradients: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.use_wandb and not HAS_WANDB:
            logging.warning("wandb not available, disabling wandb logging")
            self.use_wandb = False
        
        if self.create_live_plots and not HAS_MATPLOTLIB:
            logging.warning("matplotlib not available, disabling live plots")
            self.create_live_plots = False


class OptimizationLogger:
    """Comprehensive logger for optimization experiments."""
    
    def __init__(
        self, 
        config: LoggingConfig,
        experiment_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Initialize logger.
        
        Args:
            config: Logging configuration
            experiment_name: Name for this experiment
            tags: Tags for experiment organization
        """
        self.config = config
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.tags = tags or []
        
        # Setup directories
        self.output_dir = config.output_dir / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = self._setup_logger()
        
        # Initialize wandb if requested
        self.wandb_run = None
        if config.use_wandb:
            self._setup_wandb()
        
        # Tracking data
        self.metrics_history: Dict[str, List] = {}
        self.start_time = time.time()
        self.iteration_times: List[float] = []
        
        # Live plotting setup
        if config.create_live_plots:
            self._setup_live_plots()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup file and console logging."""
        logger = logging.getLogger(f"optimizer.{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_dir / "optimization.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if not HAS_WANDB:
            return
        
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.experiment_name,
                tags=self.tags,
                dir=str(self.output_dir)
            )
            self.logger.info("✅ W&B logging initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize W&B: {e}")
            self.wandb_run = None
    
    def _setup_live_plots(self):
        """Setup live plotting infrastructure."""
        if not HAS_MATPLOTLIB:
            return
        
        plt.ion()  # Interactive mode
        self.live_fig, self.live_axes = plt.subplots(2, 2, figsize=(15, 10))
        self.live_fig.suptitle(f'Live Optimization: {self.experiment_name}')
        
        # Configure subplots
        self.live_axes[0, 0].set_title('Objective Value')
        self.live_axes[0, 1].set_title('Vocabulary vs Truss Complexity')
        self.live_axes[1, 0].set_title('Parameter Evolution')
        self.live_axes[1, 1].set_title('Gradient Norms')
        
        plt.tight_layout()
    
    def log_iteration(
        self,
        iteration: int,
        metrics: Dict[str, Any],
        parameters: Optional[jnp.ndarray] = None,
        gradients: Optional[jnp.ndarray] = None,
        forward_pass_info: Optional[Dict[str, Any]] = None
    ):
        """Log information for a single iteration.
        
        Args:
            iteration: Current iteration number
            metrics: Dictionary of metrics to log
            parameters: Current parameter values
            gradients: Current gradient values
            forward_pass_info: Detailed forward pass information
        """
        iteration_start = time.time()
        
        # Store metrics
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(float(value))
        
        # Check if we should log this iteration
        should_log = (
            self.config.mode == LoggingMode.DEBUG or
            iteration % self.config.log_every_n_iterations == 0 or
            iteration == 0
        )
        
        if should_log:
            self._log_metrics(iteration, metrics)
        
        # Log parameters if requested
        if (parameters is not None and 
            LoggingInterest.PARAMETERS in self.config.interests):
            self._log_parameters(iteration, parameters)
        
        # Log gradients if requested
        if (gradients is not None and 
            LoggingInterest.CONVERGENCE in self.config.interests and
            self.config.track_gradients):
            self._log_gradients(iteration, gradients)
        
        # Log forward pass details if requested
        if (forward_pass_info is not None and
            LoggingInterest.FORWARD_PASS in self.config.interests):
            self._log_forward_pass(iteration, forward_pass_info)
        
        # Update live plots
        if (self.config.create_live_plots and 
            iteration % self.config.plot_every_n_iterations == 0):
            self._update_live_plots(iteration)
        
        # Log to W&B
        if self.wandb_run:
            wandb_metrics = {"iteration": iteration, **metrics}
            if gradients is not None:
                wandb_metrics["gradient_norm"] = float(jnp.linalg.norm(gradients))
            self.wandb_run.log(wandb_metrics)
        
        # Save checkpoint
        if iteration % self.config.save_checkpoint_every_n == 0:
            self._save_checkpoint(iteration, parameters)
        
        # Track timing
        iteration_time = time.time() - iteration_start
        self.iteration_times.append(iteration_time)
    
    def _log_metrics(self, iteration: int, metrics: Dict[str, Any]):
        """Log metrics based on interests."""
        log_parts = [f"Iter {iteration:4d}"]
        
        if LoggingInterest.CONVERGENCE in self.config.interests:
            if 'objective' in metrics:
                log_parts.append(f"Obj: {metrics['objective']:.6f}")
            if 'improvement' in metrics:
                log_parts.append(f"Δ: {metrics['improvement']:.2e}")
        
        if LoggingInterest.GEOMETRY in self.config.interests:
            if 'vocabulary_score' in metrics:
                log_parts.append(f"Vocab: {metrics['vocabulary_score']:.2f}")
            if 'truss_complexity' in metrics:
                log_parts.append(f"Truss: {metrics['truss_complexity']:.2f}")
        
        if LoggingInterest.PERFORMANCE in self.config.interests:
            if len(self.iteration_times) > 0:
                avg_time = np.mean(self.iteration_times[-10:])
                log_parts.append(f"Time: {avg_time:.3f}s")
        
        self.logger.info(" | ".join(log_parts))
    
    def _log_parameters(self, iteration: int, parameters: jnp.ndarray):
        """Log parameter evolution."""
        if self.config.mode == LoggingMode.DEBUG:
            param_stats = {
                'param_mean': float(jnp.mean(parameters)),
                'param_std': float(jnp.std(parameters)),
                'param_min': float(jnp.min(parameters)),
                'param_max': float(jnp.max(parameters))
            }
            self.logger.debug(f"Iter {iteration} params: {param_stats}")
        
        # Save parameter snapshots
        if self.config.save_parameter_evolution:
            param_file = self.output_dir / f"params_iter_{iteration:06d}.npy"
            np.save(param_file, np.array(parameters))
    
    def _log_gradients(self, iteration: int, gradients: jnp.ndarray):
        """Log gradient information."""
        grad_norm = float(jnp.linalg.norm(gradients))
        grad_max = float(jnp.max(jnp.abs(gradients)))
        
        if self.config.mode in [LoggingMode.DETAILED, LoggingMode.DEBUG]:
            self.logger.info(f"Iter {iteration} | Grad norm: {grad_norm:.6f}, Max: {grad_max:.6f}")
        
        # Store for plotting
        if 'gradient_norm' not in self.metrics_history:
            self.metrics_history['gradient_norm'] = []
        self.metrics_history['gradient_norm'].append(grad_norm)
    
    def _log_forward_pass(self, iteration: int, forward_info: Dict[str, Any]):
        """Log detailed forward pass information."""
        if self.config.mode == LoggingMode.DEBUG:
            self.logger.debug(f"Iter {iteration} forward pass: {forward_info}")
        
        # Save forward pass details
        forward_file = self.output_dir / f"forward_pass_iter_{iteration:06d}.json"
        with open(forward_file, 'w') as f:
            # Convert JAX arrays to lists for JSON serialization
            def make_serializable(obj):
                """Recursively convert JAX arrays and other non-serializable objects to JSON-serializable types."""
                if isinstance(obj, jnp.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                elif hasattr(obj, 'item'):  # Handle scalar JAX arrays
                    return obj.item()
                else:
                    return obj
            
            serializable_info = make_serializable(forward_info)
            json.dump(serializable_info, f, indent=2)
    
    def _update_live_plots(self, iteration: int):
        """Update live plots."""
        if not HAS_MATPLOTLIB or not hasattr(self, 'live_fig'):
            return
        
        try:
            # Clear axes
            for ax in self.live_axes.flat:
                ax.clear()
            
            # Plot objective evolution
            if 'objective' in self.metrics_history:
                self.live_axes[0, 0].plot(self.metrics_history['objective'])
                self.live_axes[0, 0].set_title('Objective Value')
                self.live_axes[0, 0].set_xlabel('Iteration')
                self.live_axes[0, 0].grid(True)
            
            # Plot vocabulary vs truss complexity
            if ('vocabulary_score' in self.metrics_history and 
                'truss_complexity' in self.metrics_history):
                vocab = self.metrics_history['vocabulary_score']
                truss = self.metrics_history['truss_complexity']
                self.live_axes[0, 1].scatter(vocab, truss, alpha=0.6)
                self.live_axes[0, 1].set_title('Vocabulary vs Truss Complexity')
                self.live_axes[0, 1].set_xlabel('Vocabulary Score')
                self.live_axes[0, 1].set_ylabel('Truss Complexity')
                self.live_axes[0, 1].grid(True)
            
            # Plot gradient norms
            if 'gradient_norm' in self.metrics_history:
                self.live_axes[1, 1].semilogy(self.metrics_history['gradient_norm'])
                self.live_axes[1, 1].set_title('Gradient Norms')
                self.live_axes[1, 1].set_xlabel('Iteration')
                self.live_axes[1, 1].grid(True)
            
            # Plot iteration times
            if len(self.iteration_times) > 10:
                self.live_axes[1, 0].plot(self.iteration_times)
                self.live_axes[1, 0].set_title('Iteration Times')
                self.live_axes[1, 0].set_xlabel('Iteration')
                self.live_axes[1, 0].set_ylabel('Time (s)')
                self.live_axes[1, 0].grid(True)
            
            plt.tight_layout()
            plt.pause(0.01)  # Brief pause to update display
            
            # Save plot
            plot_file = self.output_dir / f"live_plot_iter_{iteration:06d}.png"
            self.live_fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            
        except Exception as e:
            self.logger.warning(f"Failed to update live plots: {e}")
    
    def _save_checkpoint(self, iteration: int, parameters: Optional[jnp.ndarray]):
        """Save optimization checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'metrics_history': self.metrics_history,
            'total_time': time.time() - self.start_time,
            'config': {
                'mode': self.config.mode.value,
                'interests': [i.value for i in self.config.interests]
            }
        }
        
        if parameters is not None:
            checkpoint['parameters'] = parameters.tolist()
        
        checkpoint_file = self.output_dir / f"checkpoint_iter_{iteration:06d}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
    
    def log_final_results(self, result: Any):
        """Log final optimization results."""
        total_time = time.time() - self.start_time
        
        self.logger.info("🎉 OPTIMIZATION COMPLETE")
        self.logger.info(f"⏱️  Total time: {total_time:.2f}s")
        self.logger.info(f"🔄 Total iterations: {len(self.metrics_history.get('objective', []))}")
        
        if hasattr(result, 'best_metrics'):
            self.logger.info(f"🎯 Best vocabulary: {result.best_metrics.vocabulary_size}")
            self.logger.info(f"🔧 Unique trusses: {result.best_metrics.unique_truss_count}")
            self.logger.info(f"⚖️  Pareto efficiency: {result.best_metrics.pareto_efficiency:.3f}")
        
        # Save final summary
        summary = {
            'experiment_name': self.experiment_name,
            'total_time': total_time,
            'final_metrics': self.metrics_history,
            'config': {
                'mode': self.config.mode.value,
                'interests': [i.value for i in self.config.interests]
            }
        }
        
        summary_file = self.output_dir / "final_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Final W&B summary
        if self.wandb_run:
            wandb.summary.update({
                'total_time': total_time,
                'final_objective': self.metrics_history.get('objective', [0])[-1],
                'convergence_iterations': len(self.metrics_history.get('objective', []))
            })
            
            # Upload final plots
            if self.config.create_live_plots and hasattr(self, 'live_fig'):
                wandb.log({"final_plots": wandb.Image(self.live_fig)})
            
            self.wandb_run.finish()
    
    def close(self):
        """Clean up logger resources."""
        if hasattr(self, 'live_fig'):
            plt.close(self.live_fig)
        
        if self.wandb_run:
            self.wandb_run.finish()
        
        self.logger.info("🔚 Logger closed")


def create_logger_for_optimizer(
    optimizer_name: str,
    mode: LoggingMode = LoggingMode.STANDARD,
    interests: List[LoggingInterest] = None,
    **kwargs
) -> OptimizationLogger:
    """Factory function to create logger for specific optimizer.
    
    Args:
        optimizer_name: Name of the optimizer
        mode: Logging detail level
        interests: What aspects to focus on
        **kwargs: Additional logging config parameters
    
    Returns:
        Configured OptimizationLogger
    """
    if interests is None:
        interests = [LoggingInterest.CONVERGENCE, LoggingInterest.GEOMETRY]
    
    config = LoggingConfig(
        mode=mode,
        interests=interests,
        wandb_project=f"hole-optimization-{optimizer_name}",
        **kwargs
    )
    
    return OptimizationLogger(
        config=config,
        experiment_name=f"{optimizer_name}_{int(time.time())}",
        tags=[optimizer_name, "hole-optimization"]
    )
