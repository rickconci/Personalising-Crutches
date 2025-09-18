"""Visualization utilities for hole optimization results."""

from __future__ import annotations
from typing import List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .optimizers.base import OptimizationResult
from .geometry import HoleLayout, Geometry


def plot_optimization_results(result: OptimizationResult, output_dir: Path) -> None:
    """Create comprehensive visualization of optimization results.
    
    Args:
        result: Optimization result to visualize
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Convergence plot
    plot_convergence(result, output_dir / "convergence.png")
    
    # 2. Hole layout visualization
    plot_hole_layout(result.best_hole_layout, output_dir / "hole_layout.png")
    
    # 3. Geometry space coverage
    plot_geometry_space(result.best_geometries, output_dir / "geometry_space.png")
    
    # 4. Truss length distribution
    plot_truss_distribution(result.best_geometries, output_dir / "truss_distribution.png")
    
    # 5. Interactive 3D crutch visualization
    plot_interactive_crutch(result.best_hole_layout, result.best_geometries[:5], 
                           output_dir / "interactive_crutch.html")
    
    # 6. Pareto front (if multi-objective)
    if len(result.pareto_solutions) > 1:
        plot_pareto_front(result.pareto_solutions, output_dir / "pareto_front.png")


def plot_convergence(result: OptimizationResult, output_file: Path) -> None:
    """Plot optimization convergence history.
    
    Args:
        result: Optimization result
        output_file: File to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Objective value over time
    iterations = range(len(result.objective_history))
    ax1.plot(iterations, result.objective_history, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optimization Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Add convergence point if available
    if result.converged:
        conv_iter = result.metadata.get('convergence_iteration', -1)
        if conv_iter >= 0:
            ax1.axvline(conv_iter, color='red', linestyle='--', alpha=0.7, 
                       label=f'Converged at iteration {conv_iter}')
            ax1.legend()
    
    # Moving average of improvement rate
    if len(result.objective_history) > 10:
        window = min(20, len(result.objective_history) // 4)
        moving_avg = np.convolve(result.objective_history, 
                               np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(result.objective_history)), moving_avg, 
                'r-', linewidth=2, label=f'Moving Average (window={window})')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value (smoothed)')
        ax2.set_title('Convergence Trend')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_hole_layout(layout: HoleLayout, output_file: Path) -> None:
    """Visualize hole positions on crutch rods.
    
    Args:
        layout: Hole layout to visualize
        output_file: File to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Rod dimensions (approximate for visualization)
    handle_length = 38.0
    vertical_length = 20.0
    forearm_length = 17.0
    
    # Rod positions
    handle_y = 0
    vertical_x = 19.0  # Vertical pivot position
    forearm_start_x = 19.0  # Forearm pivot position
    forearm_angle = 30  # degrees for visualization
    
    # Draw handle rod (horizontal)
    handle_rect = Rectangle((0, handle_y - 0.5), handle_length, 1.0, 
                           facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(handle_rect)
    
    # Draw vertical rod
    vertical_rect = Rectangle((vertical_x - 0.5, handle_y), 1.0, -vertical_length,
                             facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(vertical_rect)
    
    # Draw forearm rod (angled)
    forearm_end_x = forearm_start_x + forearm_length * np.cos(np.radians(forearm_angle))
    forearm_end_y = handle_y + forearm_length * np.sin(np.radians(forearm_angle))
    
    # Simplified forearm as line for now
    ax.plot([forearm_start_x, forearm_end_x], [handle_y, forearm_end_y], 
           'k-', linewidth=8, color='lightgray')
    
    # Plot holes
    hole_size = 100
    
    # Handle holes
    for hole_pos in layout.handle:
        ax.scatter(hole_pos, handle_y, s=hole_size, c='red', marker='o', 
                  edgecolors='darkred', linewidth=2, zorder=10)
    
    # Vertical holes
    for hole_pos in layout.vertical:
        ax.scatter(vertical_x, handle_y - hole_pos, s=hole_size, c='blue', marker='s',
                  edgecolors='darkblue', linewidth=2, zorder=10)
    
    # Forearm holes (approximate positions along angled rod)
    for hole_pos in layout.forearm:
        ratio = hole_pos / forearm_length
        hole_x = forearm_start_x + ratio * (forearm_end_x - forearm_start_x)
        hole_y = handle_y + ratio * (forearm_end_y - handle_y)
        ax.scatter(hole_x, hole_y, s=hole_size, c='green', marker='^',
                  edgecolors='darkgreen', linewidth=2, zorder=10)
    
    # Add pivot points
    ax.scatter(vertical_x, handle_y, s=200, c='black', marker='X', zorder=15,
              label='Vertical Pivot')
    ax.scatter(forearm_start_x, handle_y, s=200, c='purple', marker='X', zorder=15,
              label='Forearm Pivot')
    
    # Labels and formatting
    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Position (cm)', fontsize=12)
    ax.set_title('Crutch Hole Layout', fontsize=14, fontweight='bold')
    
    # Create custom legend
    legend_elements = [
        plt.scatter([], [], s=hole_size, c='red', marker='o', label='Handle Holes'),
        plt.scatter([], [], s=hole_size, c='blue', marker='s', label='Vertical Holes'),
        plt.scatter([], [], s=hole_size, c='green', marker='^', label='Forearm Holes'),
        plt.scatter([], [], s=200, c='black', marker='X', label='Vertical Pivot'),
        plt.scatter([], [], s=200, c='purple', marker='X', label='Forearm Pivot'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Set limits with some padding
    ax.set_xlim(-2, handle_length + 5)
    ax.set_ylim(-vertical_length - 2, forearm_end_y + 2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_geometry_space(geometries: List[Geometry], output_file: Path) -> None:
    """Plot coverage of geometry space (alpha vs beta).
    
    Args:
        geometries: List of geometries to plot
        output_file: File to save plot
    """
    if not geometries:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract angles
    alphas = [g.alpha for g in geometries]
    betas = [g.beta for g in geometries]
    
    # 2D scatter plot
    scatter = ax1.scatter(alphas, betas, c=[g.score_alpha + g.score_beta for g in geometries],
                         cmap='viridis', alpha=0.7, s=50)
    ax1.set_xlabel('Alpha (degrees)', fontsize=12)
    ax1.set_ylabel('Beta (degrees)', fontsize=12)
    ax1.set_title('Geometry Space Coverage', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add constraint boundaries
    ax1.axhline(y=180, color='red', linestyle='--', alpha=0.5, label='α + β = 180°')
    ax1.plot([85, 115], [180-115, 180-85], 'r--', alpha=0.5)
    
    plt.colorbar(scatter, ax=ax1, label='Angle Score (lower is better)')
    ax1.legend()
    
    # Marginal distributions
    ax2.hist(alphas, bins=20, alpha=0.7, label='Alpha', density=True)
    ax2.hist(betas, bins=20, alpha=0.7, label='Beta', density=True)
    ax2.set_xlabel('Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Angle Distributions', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_truss_distribution(geometries: List[Geometry], output_file: Path) -> None:
    """Plot distribution of truss lengths.
    
    Args:
        geometries: List of geometries
        output_file: File to save plot
    """
    if not geometries:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract truss lengths
    truss1_lengths = [g.truss_1 for g in geometries]
    truss2_lengths = [g.truss_2 for g in geometries]
    truss3_lengths = [g.truss_3 for g in geometries]
    all_trusses = truss1_lengths + truss2_lengths + truss3_lengths
    
    # Individual truss distributions
    ax1.hist(truss1_lengths, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax1.set_title('Truss 1 Length Distribution')
    ax1.set_xlabel('Length (cm)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(truss2_lengths, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title('Truss 2 Length Distribution')
    ax2.set_xlabel('Length (cm)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    ax3.hist(truss3_lengths, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('Truss 3 Length Distribution')
    ax3.set_xlabel('Length (cm)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Combined distribution with unique count
    ax4.hist(all_trusses, bins=30, alpha=0.7, color='purple', edgecolor='black')
    
    # Count unique lengths (with tolerance)
    unique_lengths = set()
    tolerance = 0.25
    for length in all_trusses:
        rounded_length = round(length / tolerance) * tolerance
        unique_lengths.add(rounded_length)
    
    ax4.set_title(f'All Truss Lengths (Unique: {len(unique_lengths)})')
    ax4.set_xlabel('Length (cm)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Add vertical lines for unique lengths
    for unique_length in sorted(unique_lengths):
        ax4.axvline(unique_length, color='red', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_interactive_crutch(
    layout: HoleLayout, 
    sample_geometries: List[Geometry],
    output_file: Path
) -> None:
    """Create interactive 3D visualization of crutch configurations.
    
    Args:
        layout: Hole layout
        sample_geometries: Sample geometries to show
        output_file: HTML file to save interactive plot
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter"}]],
        subplot_titles=["3D Crutch Visualization", "Geometry Space"]
    )
    
    # 3D visualization (simplified)
    # Add crutch rods as lines
    handle_x = [0, 38]
    handle_y = [0, 0]
    handle_z = [0, 0]
    
    fig.add_trace(
        go.Scatter3d(x=handle_x, y=handle_y, z=handle_z,
                    mode='lines+markers',
                    line=dict(color='gray', width=10),
                    name='Handle Rod'),
        row=1, col=1
    )
    
    # Add holes
    fig.add_trace(
        go.Scatter3d(x=layout.handle.tolist(), 
                    y=[0] * len(layout.handle),
                    z=[0] * len(layout.handle),
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name='Handle Holes'),
        row=1, col=1
    )
    
    # Geometry space plot
    if sample_geometries:
        alphas = [g.alpha for g in sample_geometries]
        betas = [g.beta for g in sample_geometries]
        
        fig.add_trace(
            go.Scatter(x=alphas, y=betas,
                      mode='markers',
                      marker=dict(size=8, color='blue'),
                      name='Sample Geometries'),
            row=1, col=2
        )
    
    fig.update_layout(
        title="Interactive Crutch Optimization Results",
        height=600
    )
    
    fig.write_html(output_file)


def plot_pareto_front(
    pareto_solutions: List[Tuple], 
    output_file: Path
) -> None:
    """Plot Pareto front of solutions.
    
    Args:
        pareto_solutions: List of (layout, metrics) tuples
        output_file: File to save plot
    """
    if len(pareto_solutions) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract objectives
    vocab_sizes = [metrics.vocabulary_size for _, metrics in pareto_solutions]
    truss_counts = [metrics.unique_truss_count for _, metrics in pareto_solutions]
    
    # Plot Pareto front
    ax.scatter(vocab_sizes, truss_counts, s=100, c='red', alpha=0.7, edgecolors='black')
    
    # Connect points to show front
    sorted_solutions = sorted(zip(vocab_sizes, truss_counts))
    front_x, front_y = zip(*sorted_solutions)
    ax.plot(front_x, front_y, 'r--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Vocabulary Size (maximize)', fontsize=12)
    ax.set_ylabel('Unique Truss Count (minimize)', fontsize=12)
    ax.set_title('Pareto Front: Vocabulary vs Truss Complexity', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Annotate points
    for i, (vocab, truss) in enumerate(zip(vocab_sizes, truss_counts)):
        ax.annotate(f'Sol {i+1}', (vocab, truss), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def save_results_summary(result: OptimizationResult, output_file: Path) -> None:
    """Create a summary visualization of optimization results.
    
    Args:
        result: Optimization result
        output_file: File to save summary
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Metrics summary
    metrics = result.best_metrics
    metric_names = ['Vocabulary\nSize', 'Unique\nTrusses', 'Pareto\nEfficiency', 
                   'Convergence\nRate', 'Robustness\nScore', 'Manufacturability\nScore']
    metric_values = [metrics.vocabulary_size, metrics.unique_truss_count, 
                    metrics.pareto_efficiency, metrics.convergence_rate,
                    metrics.robustness_score, metrics.manufacturability_score]
    
    bars = ax1.bar(metric_names, metric_values, color=['red', 'orange', 'green', 
                                                      'blue', 'purple', 'brown'])
    ax1.set_title('Optimization Metrics Summary', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Convergence plot
    if result.objective_history:
        ax2.plot(result.objective_history, 'b-', linewidth=2)
        ax2.set_title('Optimization Convergence', fontsize=14)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.grid(True, alpha=0.3)
    
    # Hole count summary
    hole_counts = [len(result.best_hole_layout.handle),
                  len(result.best_hole_layout.vertical),
                  len(result.best_hole_layout.forearm)]
    rod_names = ['Handle', 'Vertical', 'Forearm']
    
    ax3.pie(hole_counts, labels=rod_names, autopct='%1.0f%%', startangle=90,
           colors=['red', 'blue', 'green'])
    ax3.set_title('Hole Distribution by Rod', fontsize=14)
    
    # Runtime information
    info_text = f"""
    Optimizer: {result.metadata.get('optimizer_type', 'Unknown')}
    Runtime: {result.total_time:.2f}s
    Iterations: {result.iterations}
    Converged: {'Yes' if result.converged else 'No'}
    
    Best Solution:
    • Vocabulary Size: {metrics.vocabulary_size}
    • Unique Trusses: {metrics.unique_truss_count}
    • Total Holes: {result.best_hole_layout.total_holes}
    """
    
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Optimization Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
