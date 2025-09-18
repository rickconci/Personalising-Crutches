"""Main entry point for hole optimization experiments."""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import json
from typing import Optional, Dict, Any
import time

from .config import (
    ConfigManager, 
    ExperimentConfig, 
    OptimizerType,
    validate_constraints,
    validate_objectives
)
from .optimizers.factory import create_optimizer, get_available_optimizers
from .geometry import create_uniform_holes
from .visualization import plot_optimization_results, save_results_summary
from .benchmarking import run_benchmark, compare_optimizers


def setup_logging(verbose: bool = True) -> None:
    """Setup logging configuration.
    
    Args:
        verbose: Whether to use verbose logging
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('hole_optimization.log')
        ]
    )


def run_single_optimization(
    config: ExperimentConfig,
    output_dir: Path,
    save_plots: bool = True
) -> Dict[str, Any]:
    """Run a single optimization experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Directory to save results
        save_plots: Whether to save visualization plots
        
    Returns:
        Dictionary with experiment results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting optimization with {config.optimizer_type.value}")
    
    # Validate configuration
    validate_constraints(config.constraints)
    validate_objectives(config.objectives)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get optimizer-specific config
    optimizer_configs = {
        OptimizerType.DIFFERENTIABLE: config.differentiable,
        OptimizerType.GENETIC_ALGORITHM: config.genetic_algorithm,
        OptimizerType.SIMULATED_ANNEALING: config.simulated_annealing,
        OptimizerType.INTEGER_PROGRAMMING: config.integer_programming,
        OptimizerType.HYBRID: config.hybrid,
    }
    
    optimizer_config = optimizer_configs[config.optimizer_type]
    
    # Create optimizer
    try:
        optimizer = create_optimizer(
            config.optimizer_type,
            config.constraints,
            config.objectives,
            optimizer_config,
            config.random_seed
        )
    except ImportError as e:
        logger.error(f"Failed to create optimizer: {e}")
        return {"error": str(e), "status": "failed"}
    
    # Create initial hole layout if needed
    initial_layout = None
    if hasattr(optimizer_config, 'use_initial_layout') and optimizer_config.use_initial_layout:
        initial_layout = create_uniform_holes(config.constraints)
        logger.info(f"Created initial layout with {initial_layout.total_holes} holes")
    
    # Run optimization
    start_time = time.time()
    try:
        result = optimizer.optimize(initial_layout=initial_layout)
        optimization_time = time.time() - start_time
        
        logger.info(f"Optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best vocabulary size: {result.best_metrics.vocabulary_size}")
        logger.info(f"Unique truss count: {result.best_metrics.unique_truss_count}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return {"error": str(e), "status": "failed"}
    
    # Save results
    results_data = {
        "config": config.__dict__,
        "optimizer_type": config.optimizer_type.value,
        "optimization_time": optimization_time,
        "converged": result.converged,
        "iterations": result.iterations,
        "best_metrics": {
            "vocabulary_size": result.best_metrics.vocabulary_size,
            "unique_truss_count": result.best_metrics.unique_truss_count,
            "pareto_efficiency": result.best_metrics.pareto_efficiency,
            "convergence_rate": result.best_metrics.convergence_rate,
            "robustness_score": result.best_metrics.robustness_score,
            "manufacturability_score": result.best_metrics.manufacturability_score,
        },
        "hole_layout": {
            "handle": result.best_hole_layout.handle.tolist(),
            "vertical": result.best_hole_layout.vertical.tolist(),
            "forearm": result.best_hole_layout.forearm.tolist(),
        },
        "num_geometries": len(result.best_geometries),
        "objective_history": result.objective_history,
        "status": "success"
    }
    
    # Save JSON results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save detailed results
    if config.save_intermediate:
        detailed_file = output_dir / "detailed_results.json"
        detailed_data = {
            **results_data,
            "geometries": [
                {
                    "alpha": g.alpha,
                    "beta": g.beta,
                    "gamma": g.gamma,
                    "truss_1": g.truss_1,
                    "truss_2": g.truss_2,
                    "truss_3": g.truss_3,
                    "score_alpha": g.score_alpha,
                    "score_beta": g.score_beta,
                }
                for g in result.best_geometries[:100]  # Limit to first 100
            ],
            "parameter_history": [p.tolist() for p in result.parameter_history[::10]],  # Every 10th
        }
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
    
    # Create visualizations
    if save_plots:
        try:
            plot_optimization_results(result, output_dir)
            save_results_summary(result, output_dir / "summary.png")
            logger.info("Saved visualization plots")
        except Exception as e:
            logger.warning(f"Failed to create plots: {e}")
    
    return results_data


def run_benchmark_experiment(
    config_dir: Path,
    output_dir: Path,
    optimizers: Optional[list[OptimizerType]] = None
) -> Dict[str, Any]:
    """Run benchmark comparing multiple optimizers.
    
    Args:
        config_dir: Directory containing optimizer configs
        output_dir: Directory to save benchmark results
        optimizers: List of optimizers to compare (None for all available)
        
    Returns:
        Benchmark results
    """
    logger = logging.getLogger(__name__)
    
    if optimizers is None:
        available = get_available_optimizers()
        optimizers = [opt_type for opt_type, available in available.items() if available]
    
    logger.info(f"Running benchmark with optimizers: {[opt.value for opt in optimizers]}")
    
    # Load base configuration
    base_config_file = config_dir / "base_config.yaml"
    if base_config_file.exists():
        base_config = ConfigManager.load_config(base_config_file)
    else:
        base_config = ExperimentConfig()
        logger.warning("Using default configuration for benchmark")
    
    # Run benchmark
    benchmark_results = run_benchmark(
        optimizers,
        base_config,
        output_dir,
        num_runs=3,  # Multiple runs for statistical significance
        timeout=3600  # 1 hour timeout per optimizer
    )
    
    # Compare results
    comparison = compare_optimizers(benchmark_results)
    
    # Save benchmark results
    benchmark_file = output_dir / "benchmark_results.json"
    with open(benchmark_file, 'w') as f:
        json.dump({
            "results": benchmark_results,
            "comparison": comparison,
            "timestamp": time.time(),
        }, f, indent=2)
    
    logger.info("Benchmark completed successfully")
    return benchmark_results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hole Optimization for Crutch Design")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single optimization
    opt_parser = subparsers.add_parser('optimize', help='Run single optimization')
    opt_parser.add_argument('--config', type=Path, required=True,
                           help='Path to configuration file')
    opt_parser.add_argument('--output', type=Path, default=Path('results'),
                           help='Output directory')
    opt_parser.add_argument('--no-plots', action='store_true',
                           help='Skip generating plots')
    
    # Benchmark
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmark comparison')
    bench_parser.add_argument('--config-dir', type=Path, default=Path('configs'),
                             help='Directory with optimizer configs')
    bench_parser.add_argument('--output', type=Path, default=Path('benchmark_results'),
                             help='Output directory')
    bench_parser.add_argument('--optimizers', nargs='+', 
                             choices=[opt.value for opt in OptimizerType],
                             help='Optimizers to benchmark')
    
    # Create default configs
    config_parser = subparsers.add_parser('create-configs', 
                                         help='Create default configuration files')
    config_parser.add_argument('--output-dir', type=Path, default=Path('configs'),
                              help='Directory to save configs')
    
    # List available optimizers
    list_parser = subparsers.add_parser('list', help='List available optimizers')
    
    # General arguments
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    if args.command == 'optimize':
        # Load configuration
        try:
            config = ConfigManager.load_config(args.config)
            config.random_seed = args.seed
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return
        
        # Run optimization
        results = run_single_optimization(
            config,
            args.output,
            save_plots=not args.no_plots
        )
        
        if results.get("status") == "success":
            logger.info("Optimization completed successfully!")
            print(f"Results saved to: {args.output}")
            print(f"Vocabulary size: {results['best_metrics']['vocabulary_size']}")
            print(f"Unique trusses: {results['best_metrics']['unique_truss_count']}")
        else:
            logger.error("Optimization failed!")
    
    elif args.command == 'benchmark':
        # Parse optimizer list
        optimizers = None
        if args.optimizers:
            optimizers = [OptimizerType(opt) for opt in args.optimizers]
        
        # Run benchmark
        try:
            results = run_benchmark_experiment(
                args.config_dir,
                args.output,
                optimizers
            )
            logger.info("Benchmark completed successfully!")
            print(f"Benchmark results saved to: {args.output}")
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
    
    elif args.command == 'create-configs':
        # Create default configurations
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        configs = ConfigManager.create_default_configs()
        for name, config in configs.items():
            config_file = args.output_dir / f"{name}_config.yaml"
            ConfigManager.save_config(config, config_file)
            logger.info(f"Created config: {config_file}")
        
        print(f"Default configs created in: {args.output_dir}")
    
    elif args.command == 'list':
        # List available optimizers
        available = get_available_optimizers()
        
        print("Available optimizers:")
        for opt_type, is_available in available.items():
            status = "✓" if is_available else "✗ (missing dependencies)"
            print(f"  {opt_type.value}: {status}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
