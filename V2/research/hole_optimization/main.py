#!/usr/bin/env python3
"""Hydra-based main entry point for hole optimization experiments."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import time

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import ExperimentConfig, OptimizerType
from .optimizers.factory import create_optimizer, get_available_optimizers
from .geometry import create_uniform_holes
from .visualization import plot_optimization_results, save_results_summary
from .benchmarking import run_benchmark, compare_optimizers, generate_benchmark_report

# Setup logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="hydra_configs", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point with Hydra configuration management.
    
    Supports two modes:
    1. Single optimization: Run one optimizer with given config
    2. Benchmark mode: Compare multiple optimizers
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Primary metric for hyperparameter optimization (single mode)
        or None (benchmark mode)
    """
    log.info("🚀 Starting Hydra-powered hole optimization")
    log.info(f"📁 Working directory: {Path.cwd()}")
    log.info(f"⚙️ Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Check if this is benchmark mode
    if cfg.get("mode") == "benchmark":
        return run_benchmark_mode(cfg)
    else:
        return run_single_optimization_mode(cfg)


def run_single_optimization_mode(cfg: DictConfig) -> Optional[float]:
    """Run single optimization mode.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Primary metric for hyperparameter optimization
    """
    try:
        # Convert Hydra config to our Pydantic config
        # This preserves validation while using Hydra's power
        experiment_config = hydra_to_pydantic_config(cfg)
        
        log.info(f"🔧 Using optimizer: {experiment_config.optimizer_type.value}")
        
        # Get optimizer-specific config
        optimizer_configs = {
            OptimizerType.DIFFERENTIABLE: experiment_config.differentiable,
            OptimizerType.GENETIC_ALGORITHM: experiment_config.genetic_algorithm,
            OptimizerType.SIMULATED_ANNEALING: experiment_config.simulated_annealing,
            OptimizerType.INTEGER_PROGRAMMING: experiment_config.integer_programming,
            OptimizerType.HYBRID: experiment_config.hybrid,
        }
        
        optimizer_config = optimizer_configs[experiment_config.optimizer_type]
        
        # Create optimizer
        optimizer = create_optimizer(
            experiment_config.optimizer_type,
            experiment_config.constraints,
            experiment_config.objectives,
            optimizer_config,
            experiment_config.random_seed
        )
        
        # Create initial layout if needed
        initial_layout = None
        if cfg.get("use_initial_layout", False):
            initial_layout = create_uniform_holes(experiment_config.constraints)
            log.info(f"📍 Created initial layout with {initial_layout.total_holes} holes")
        
        # Run optimization
        log.info("🎯 Running optimization...")
        start_time = time.time()
        result = optimizer.optimize(initial_layout=initial_layout)
        optimization_time = time.time() - start_time
        
        # Log results
        log.info("✅ Optimization completed!")
        log.info(f"⏱️ Runtime: {optimization_time:.2f}s")
        log.info(f"🔄 Iterations: {result.iterations}")
        log.info(f"📈 Converged: {'Yes' if result.converged else 'No'}")
        log.info(f"🎨 Vocabulary size: {result.best_metrics.vocabulary_size}")
        log.info(f"🔧 Unique trusses: {result.best_metrics.unique_truss_count}")
        log.info(f"⚖️ Pareto efficiency: {result.best_metrics.pareto_efficiency:.3f}")
        
        # Save results to Hydra's output directory
        output_dir = Path.cwd()  # Hydra automatically sets this
        
        # Create visualizations if requested
        if cfg.get("create_plots", True):
            try:
                plot_optimization_results(result, output_dir)
                save_results_summary(result, output_dir / "summary.png")
                log.info("📊 Visualizations saved")
            except Exception as e:
                log.warning(f"Failed to create plots: {e}")
        
        # Save detailed results
        results_data = {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "metrics": {
                "vocabulary_size": result.best_metrics.vocabulary_size,
                "unique_truss_count": result.best_metrics.unique_truss_count,
                "pareto_efficiency": result.best_metrics.pareto_efficiency,
                "convergence_rate": result.best_metrics.convergence_rate,
                "robustness_score": result.best_metrics.robustness_score,
                "manufacturability_score": result.best_metrics.manufacturability_score,
            },
            "optimization": {
                "runtime": optimization_time,
                "iterations": result.iterations,
                "converged": result.converged,
                "final_objective": result.objective_history[-1] if result.objective_history else None,
            },
            "hole_layout": {
                "handle": result.best_hole_layout.handle.tolist(),
                "vertical": result.best_hole_layout.vertical.tolist(),
                "forearm": result.best_hole_layout.forearm.tolist(),
                "total_holes": result.best_hole_layout.total_holes,
            }
        }
        
        # Save to Hydra's output directory
        results_file = output_dir / "results.yaml"
        OmegaConf.save(results_data, results_file)
        log.info(f"💾 Results saved to {results_file}")
        
        # Return primary metric for hyperparameter optimization
        primary_metric = result.best_metrics.vocabulary_size - 0.1 * result.best_metrics.unique_truss_count
        log.info(f"📊 Primary metric (vocabulary - 0.1*trusses): {primary_metric:.2f}")
        
        return primary_metric
        
    except Exception as e:
        log.error(f"❌ Optimization failed: {e}")
        raise


def run_benchmark_mode(cfg: DictConfig) -> None:
    """Run benchmark comparison mode.
    
    Args:
        cfg: Hydra configuration with benchmark settings
    """
    log.info("🏁 Starting benchmark mode")
    
    try:
        # Get benchmark configuration
        benchmark_config = cfg.get("benchmark", {})
        
        # Determine which optimizers to benchmark
        optimizer_names = benchmark_config.get("optimizers", ["differentiable"])
        optimizers = []
        
        available_optimizers = get_available_optimizers()
        
        for opt_name in optimizer_names:
            try:
                opt_type = OptimizerType(opt_name)
                if available_optimizers.get(opt_type, False):
                    optimizers.append(opt_type)
                    log.info(f"✅ Added {opt_name} to benchmark")
                else:
                    log.warning(f"⚠️ Optimizer {opt_name} not available (missing dependencies)")
            except ValueError:
                log.error(f"❌ Unknown optimizer: {opt_name}")
        
        if not optimizers:
            log.error("❌ No valid optimizers found for benchmark")
            return
        
        # Convert base config
        base_config = hydra_to_pydantic_config(cfg)
        
        # Benchmark settings
        num_runs = benchmark_config.get("num_runs", 3)
        timeout = benchmark_config.get("timeout", 1800)  # 30 minutes default
        parallel = benchmark_config.get("parallel", True)
        
        log.info(f"🎯 Benchmarking {len(optimizers)} optimizers with {num_runs} runs each")
        log.info(f"⏱️ Timeout: {timeout}s per run")
        log.info(f"🔄 Parallel: {parallel}")
        
        # Run benchmark
        output_dir = Path.cwd()  # Hydra sets this
        start_time = time.time()
        
        benchmark_results = run_benchmark(
            optimizers=optimizers,
            base_config=base_config,
            output_dir=output_dir,
            num_runs=num_runs,
            timeout=timeout,
            parallel=parallel
        )
        
        benchmark_time = time.time() - start_time
        log.info(f"🏁 Benchmark completed in {benchmark_time:.1f}s")
        
        # Analyze results
        comparison = compare_optimizers(benchmark_results)
        
        # Generate report
        report_file = output_dir / "benchmark_report.md"
        generate_benchmark_report(benchmark_results, comparison, report_file)
        
        # Save results
        results_file = output_dir / "benchmark_results.yaml"
        benchmark_data = {
            "benchmark_config": OmegaConf.to_container(benchmark_config, resolve=True),
            "results": benchmark_results,
            "comparison": comparison,
            "total_time": benchmark_time,
            "timestamp": time.time()
        }
        
        OmegaConf.save(benchmark_data, results_file)
        
        # Log summary
        log.info("📊 Benchmark Summary:")
        if comparison.get("rankings", {}).get("combined"):
            for i, optimizer in enumerate(comparison["rankings"]["combined"], 1):
                log.info(f"  {i}. {optimizer}")
        
        log.info(f"📄 Full report: {report_file}")
        log.info(f"💾 Results: {results_file}")
        
        # Create benchmark plots if requested
        if cfg.get("create_plots", True):
            try:
                from .benchmarking import plot_benchmark_comparison
                plot_benchmark_comparison(benchmark_results, output_dir)
                log.info("📊 Benchmark plots saved")
            except Exception as e:
                log.warning(f"Failed to create benchmark plots: {e}")
        
    except Exception as e:
        log.error(f"❌ Benchmark failed: {e}")
        raise


def hydra_to_pydantic_config(cfg: DictConfig) -> ExperimentConfig:
    """Convert Hydra config to Pydantic config for validation.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Validated ExperimentConfig
    """
    # Convert to dictionary and create Pydantic config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Pydantic handles all type conversions automatically!
    return ExperimentConfig(**config_dict)


# Register configs with Hydra
cs = ConfigStore.instance()

# For now, let's skip schema validation to avoid Hydra conflicts
# We'll handle validation in the hydra_to_pydantic_config function instead


if __name__ == "__main__":
    main()
