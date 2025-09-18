"""Benchmarking utilities for comparing optimization algorithms."""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import time
import logging
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing as mp

from .config import OptimizerType, ExperimentConfig
from .optimizers.factory import create_optimizer
from .optimizers.base import OptimizationResult, OptimizationMetrics
from .geometry import create_uniform_holes


logger = logging.getLogger(__name__)


def run_single_benchmark(
    optimizer_type: OptimizerType,
    config: ExperimentConfig,
    run_id: int,
    timeout: float = 3600
) -> Dict[str, Any]:
    """Run a single benchmark experiment.
    
    Args:
        optimizer_type: Type of optimizer to test
        config: Experiment configuration
        run_id: Unique run identifier
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Starting benchmark run {run_id} with {optimizer_type.value}")
    
    try:
        # Update config for this optimizer
        config.optimizer_type = optimizer_type
        config.random_seed = config.random_seed + run_id  # Different seed per run
        
        # Get optimizer-specific config
        optimizer_configs = {
            OptimizerType.DIFFERENTIABLE: config.differentiable,
            OptimizerType.GENETIC_ALGORITHM: config.genetic_algorithm,
            OptimizerType.SIMULATED_ANNEALING: config.simulated_annealing,
            OptimizerType.INTEGER_PROGRAMMING: config.integer_programming,
            OptimizerType.HYBRID: config.hybrid,
        }
        
        optimizer_config = optimizer_configs[optimizer_type]
        
        # Create optimizer
        optimizer = create_optimizer(
            optimizer_type,
            config.constraints,
            config.objectives,
            optimizer_config,
            config.random_seed
        )
        
        # Create initial layout
        initial_layout = create_uniform_holes(config.constraints)
        
        # Run optimization with timeout
        start_time = time.time()
        result = optimizer.optimize(initial_layout=initial_layout)
        end_time = time.time()
        
        # Extract key metrics
        return {
            "optimizer_type": optimizer_type.value,
            "run_id": run_id,
            "status": "success",
            "runtime": end_time - start_time,
            "iterations": result.iterations,
            "converged": result.converged,
            "vocabulary_size": result.best_metrics.vocabulary_size,
            "unique_truss_count": result.best_metrics.unique_truss_count,
            "pareto_efficiency": result.best_metrics.pareto_efficiency,
            "convergence_rate": result.best_metrics.convergence_rate,
            "robustness_score": result.best_metrics.robustness_score,
            "manufacturability_score": result.best_metrics.manufacturability_score,
            "total_holes": result.best_hole_layout.total_holes,
            "final_objective": result.objective_history[-1] if result.objective_history else None,
            "convergence_iteration": result.metadata.get('convergence_iteration', -1),
        }
        
    except TimeoutError:
        logger.warning(f"Benchmark run {run_id} with {optimizer_type.value} timed out")
        return {
            "optimizer_type": optimizer_type.value,
            "run_id": run_id,
            "status": "timeout",
            "runtime": timeout,
            "error": "Optimization timed out"
        }
    except Exception as e:
        logger.error(f"Benchmark run {run_id} with {optimizer_type.value} failed: {e}")
        return {
            "optimizer_type": optimizer_type.value,
            "run_id": run_id,
            "status": "error",
            "runtime": 0.0,
            "error": str(e)
        }


def run_benchmark(
    optimizers: List[OptimizerType],
    base_config: ExperimentConfig,
    output_dir: Path,
    num_runs: int = 3,
    timeout: float = 3600,
    parallel: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """Run benchmark comparing multiple optimizers.
    
    Args:
        optimizers: List of optimizers to benchmark
        base_config: Base configuration for experiments
        output_dir: Directory to save results
        num_runs: Number of runs per optimizer
        timeout: Timeout per run in seconds
        parallel: Whether to run in parallel
        
    Returns:
        Dictionary mapping optimizer names to list of run results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark_results = {}
    
    if parallel:
        # Run benchmarks in parallel
        with ProcessPoolExecutor(max_workers=min(len(optimizers) * num_runs, mp.cpu_count())) as executor:
            # Submit all jobs
            futures = []
            for optimizer_type in optimizers:
                for run_id in range(num_runs):
                    future = executor.submit(
                        run_single_benchmark,
                        optimizer_type,
                        base_config,
                        run_id,
                        timeout
                    )
                    futures.append((optimizer_type, future))
            
            # Collect results
            for optimizer_type, future in futures:
                try:
                    result = future.result(timeout=timeout + 60)  # Extra buffer
                    if optimizer_type.value not in benchmark_results:
                        benchmark_results[optimizer_type.value] = []
                    benchmark_results[optimizer_type.value].append(result)
                except Exception as e:
                    logger.error(f"Failed to get result for {optimizer_type.value}: {e}")
    else:
        # Run benchmarks sequentially
        for optimizer_type in optimizers:
            benchmark_results[optimizer_type.value] = []
            
            for run_id in range(num_runs):
                result = run_single_benchmark(optimizer_type, base_config, run_id, timeout)
                benchmark_results[optimizer_type.value].append(result)
                
                # Save intermediate results
                intermediate_file = output_dir / f"{optimizer_type.value}_run_{run_id}.json"
                with open(intermediate_file, 'w') as f:
                    json.dump(result, f, indent=2)
    
    return benchmark_results


def compare_optimizers(benchmark_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compare optimizer performance from benchmark results.
    
    Args:
        benchmark_results: Results from run_benchmark
        
    Returns:
        Comparison statistics and rankings
    """
    comparison = {
        "summary_stats": {},
        "rankings": {},
        "statistical_tests": {},
        "recommendations": []
    }
    
    # Calculate summary statistics for each optimizer
    for optimizer_name, runs in benchmark_results.items():
        successful_runs = [run for run in runs if run["status"] == "success"]
        
        if not successful_runs:
            comparison["summary_stats"][optimizer_name] = {
                "success_rate": 0.0,
                "mean_runtime": float('inf'),
                "mean_vocabulary_size": 0.0,
                "mean_unique_trusses": float('inf'),
                "convergence_rate": 0.0
            }
            continue
        
        # Extract metrics
        runtimes = [run["runtime"] for run in successful_runs]
        vocab_sizes = [run["vocabulary_size"] for run in successful_runs]
        truss_counts = [run["unique_truss_count"] for run in successful_runs]
        convergence_rates = [run["convergence_rate"] for run in successful_runs]
        
        comparison["summary_stats"][optimizer_name] = {
            "success_rate": len(successful_runs) / len(runs),
            "mean_runtime": np.mean(runtimes),
            "std_runtime": np.std(runtimes),
            "mean_vocabulary_size": np.mean(vocab_sizes),
            "std_vocabulary_size": np.std(vocab_sizes),
            "mean_unique_trusses": np.mean(truss_counts),
            "std_unique_trusses": np.std(truss_counts),
            "mean_convergence_rate": np.mean(convergence_rates),
            "best_vocabulary_size": max(vocab_sizes) if vocab_sizes else 0,
            "best_truss_count": min(truss_counts) if truss_counts else float('inf'),
        }
    
    # Create rankings
    valid_optimizers = [name for name, stats in comparison["summary_stats"].items() 
                       if stats["success_rate"] > 0]
    
    if valid_optimizers:
        # Rank by vocabulary size (higher is better)
        vocab_ranking = sorted(valid_optimizers, 
                              key=lambda x: comparison["summary_stats"][x]["mean_vocabulary_size"],
                              reverse=True)
        
        # Rank by truss count (lower is better)
        truss_ranking = sorted(valid_optimizers,
                              key=lambda x: comparison["summary_stats"][x]["mean_unique_trusses"])
        
        # Rank by runtime (lower is better)
        runtime_ranking = sorted(valid_optimizers,
                                key=lambda x: comparison["summary_stats"][x]["mean_runtime"])
        
        # Combined ranking (simple scoring)
        def combined_score(optimizer_name: str) -> float:
            stats = comparison["summary_stats"][optimizer_name]
            vocab_score = stats["mean_vocabulary_size"] / 100.0  # Normalize
            truss_penalty = stats["mean_unique_trusses"] / 20.0  # Normalize
            runtime_penalty = stats["mean_runtime"] / 1000.0  # Normalize
            return vocab_score - truss_penalty - runtime_penalty
        
        combined_ranking = sorted(valid_optimizers, key=combined_score, reverse=True)
        
        comparison["rankings"] = {
            "vocabulary_size": vocab_ranking,
            "truss_count": truss_ranking,
            "runtime": runtime_ranking,
            "combined": combined_ranking
        }
    
    # Generate recommendations
    if valid_optimizers:
        best_overall = comparison["rankings"]["combined"][0]
        best_vocab = comparison["rankings"]["vocabulary_size"][0]
        fastest = comparison["rankings"]["runtime"][0]
        
        comparison["recommendations"] = [
            f"Best overall performance: {best_overall}",
            f"Best vocabulary coverage: {best_vocab}",
            f"Fastest execution: {fastest}",
        ]
        
        # Add specific recommendations based on use case
        if best_overall == "differentiable":
            comparison["recommendations"].append(
                "Differentiable optimizer recommended for: gradient-based optimization, "
                "continuous parameter spaces, GPU acceleration"
            )
        elif best_overall == "genetic_algorithm":
            comparison["recommendations"].append(
                "Genetic algorithm recommended for: multi-objective optimization, "
                "discrete parameter spaces, parallel evaluation"
            )
        elif best_overall == "hybrid":
            comparison["recommendations"].append(
                "Hybrid optimizer recommended for: best of both worlds, "
                "when computational budget allows"
            )
    
    return comparison


def generate_benchmark_report(
    benchmark_results: Dict[str, List[Dict[str, Any]]],
    comparison: Dict[str, Any],
    output_file: Path
) -> None:
    """Generate a comprehensive benchmark report.
    
    Args:
        benchmark_results: Raw benchmark results
        comparison: Comparison statistics
        output_file: File to save report
    """
    report_lines = [
        "# Hole Optimization Benchmark Report",
        "=" * 50,
        "",
        "## Executive Summary",
        ""
    ]
    
    # Add recommendations
    if comparison.get("recommendations"):
        report_lines.extend([
            "### Key Recommendations:",
            ""
        ])
        for rec in comparison["recommendations"]:
            report_lines.append(f"- {rec}")
        report_lines.append("")
    
    # Summary statistics table
    report_lines.extend([
        "## Performance Summary",
        "",
        "| Optimizer | Success Rate | Avg Runtime (s) | Avg Vocabulary | Avg Trusses | Best Vocab | Best Trusses |",
        "|-----------|--------------|-----------------|----------------|-------------|------------|--------------|"
    ])
    
    for optimizer_name, stats in comparison["summary_stats"].items():
        report_lines.append(
            f"| {optimizer_name} | {stats['success_rate']:.2f} | "
            f"{stats['mean_runtime']:.1f} | {stats['mean_vocabulary_size']:.1f} | "
            f"{stats['mean_unique_trusses']:.1f} | {stats['best_vocabulary_size']} | "
            f"{stats['best_truss_count']:.1f} |"
        )
    
    report_lines.extend([
        "",
        "## Rankings",
        ""
    ])
    
    if "rankings" in comparison:
        rankings = comparison["rankings"]
        
        report_lines.extend([
            "### By Vocabulary Size (Best to Worst):",
            ""
        ])
        for i, optimizer in enumerate(rankings.get("vocabulary_size", []), 1):
            vocab_size = comparison["summary_stats"][optimizer]["mean_vocabulary_size"]
            report_lines.append(f"{i}. {optimizer} ({vocab_size:.1f} geometries)")
        
        report_lines.extend([
            "",
            "### By Truss Count (Best to Worst):",
            ""
        ])
        for i, optimizer in enumerate(rankings.get("truss_count", []), 1):
            truss_count = comparison["summary_stats"][optimizer]["mean_unique_trusses"]
            report_lines.append(f"{i}. {optimizer} ({truss_count:.1f} unique trusses)")
        
        report_lines.extend([
            "",
            "### By Runtime (Fastest to Slowest):",
            ""
        ])
        for i, optimizer in enumerate(rankings.get("runtime", []), 1):
            runtime = comparison["summary_stats"][optimizer]["mean_runtime"]
            report_lines.append(f"{i}. {optimizer} ({runtime:.1f}s)")
        
        report_lines.extend([
            "",
            "### Overall Combined Ranking:",
            ""
        ])
        for i, optimizer in enumerate(rankings.get("combined", []), 1):
            report_lines.append(f"{i}. {optimizer}")
    
    # Detailed results
    report_lines.extend([
        "",
        "## Detailed Results",
        ""
    ])
    
    for optimizer_name, runs in benchmark_results.items():
        report_lines.extend([
            f"### {optimizer_name}",
            ""
        ])
        
        successful_runs = [run for run in runs if run["status"] == "success"]
        failed_runs = [run for run in runs if run["status"] != "success"]
        
        report_lines.append(f"**Successful runs:** {len(successful_runs)}/{len(runs)}")
        
        if successful_runs:
            runtimes = [run["runtime"] for run in successful_runs]
            vocab_sizes = [run["vocabulary_size"] for run in successful_runs]
            
            report_lines.extend([
                f"**Runtime:** {np.mean(runtimes):.1f} ± {np.std(runtimes):.1f}s",
                f"**Vocabulary Size:** {np.mean(vocab_sizes):.1f} ± {np.std(vocab_sizes):.1f}",
                ""
            ])
        
        if failed_runs:
            report_lines.append("**Failed runs:**")
            for run in failed_runs:
                report_lines.append(f"- Run {run['run_id']}: {run.get('error', 'Unknown error')}")
            report_lines.append("")
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Benchmark report saved to {output_file}")


def plot_benchmark_comparison(
    benchmark_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Path
) -> None:
    """Create visualization plots for benchmark comparison.
    
    Args:
        benchmark_results: Benchmark results
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Prepare data
        plot_data = []
        for optimizer_name, runs in benchmark_results.items():
            successful_runs = [run for run in runs if run["status"] == "success"]
            for run in successful_runs:
                plot_data.append({
                    'optimizer': optimizer_name,
                    'runtime': run['runtime'],
                    'vocabulary_size': run['vocabulary_size'],
                    'unique_trusses': run['unique_truss_count'],
                    'convergence_rate': run['convergence_rate']
                })
        
        if not plot_data:
            logger.warning("No successful runs to plot")
            return
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Runtime comparison
        runtimes_by_optimizer = {}
        for data in plot_data:
            opt = data['optimizer']
            if opt not in runtimes_by_optimizer:
                runtimes_by_optimizer[opt] = []
            runtimes_by_optimizer[opt].append(data['runtime'])
        
        ax1.boxplot(runtimes_by_optimizer.values(), labels=runtimes_by_optimizer.keys())
        ax1.set_title('Runtime Comparison')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Vocabulary size comparison
        vocab_by_optimizer = {}
        for data in plot_data:
            opt = data['optimizer']
            if opt not in vocab_by_optimizer:
                vocab_by_optimizer[opt] = []
            vocab_by_optimizer[opt].append(data['vocabulary_size'])
        
        ax2.boxplot(vocab_by_optimizer.values(), labels=vocab_by_optimizer.keys())
        ax2.set_title('Vocabulary Size Comparison')
        ax2.set_ylabel('Number of Geometries')
        ax2.tick_params(axis='x', rotation=45)
        
        # Truss count comparison
        truss_by_optimizer = {}
        for data in plot_data:
            opt = data['optimizer']
            if opt not in truss_by_optimizer:
                truss_by_optimizer[opt] = []
            truss_by_optimizer[opt].append(data['unique_trusses'])
        
        ax3.boxplot(truss_by_optimizer.values(), labels=truss_by_optimizer.keys())
        ax3.set_title('Unique Truss Count Comparison')
        ax3.set_ylabel('Number of Unique Trusses')
        ax3.tick_params(axis='x', rotation=45)
        
        # Scatter plot: vocabulary vs trusses
        colors = plt.cm.tab10(np.linspace(0, 1, len(runtimes_by_optimizer)))
        for i, (opt, color) in enumerate(zip(runtimes_by_optimizer.keys(), colors)):
            opt_data = [d for d in plot_data if d['optimizer'] == opt]
            vocab_sizes = [d['vocabulary_size'] for d in opt_data]
            truss_counts = [d['unique_trusses'] for d in opt_data]
            ax4.scatter(vocab_sizes, truss_counts, c=[color], label=opt, alpha=0.7, s=50)
        
        ax4.set_xlabel('Vocabulary Size')
        ax4.set_ylabel('Unique Truss Count')
        ax4.set_title('Trade-off: Vocabulary vs Truss Complexity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "benchmark_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Benchmark comparison plots saved")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plots")
    except Exception as e:
        logger.error(f"Failed to create benchmark plots: {e}")
