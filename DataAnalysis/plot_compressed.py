#!/usr/bin/env python3
"""
Compressed plotting functions for alpha, beta, gamma parameters colored by scores.
Only includes the essential functions for 3D plotting and temporal analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict, List
from scipy import stats
import GPy
import colorsys
import re
from matplotlib.colors import LinearSegmentedColormap


def handle_duplicates_and_add_index(df: pd.DataFrame, score_columns: list = None) -> pd.DataFrame:
    """
    Handle duplicate alpha, beta, gamma combinations by taking the mean of scores,
    and add testing order index.
    
    Args:
        df: DataFrame with alpha, beta, gamma, and score columns
        score_columns: List of score columns to aggregate (if None, auto-detect)
        
    Returns:
        DataFrame with duplicates handled and testing order index added
    """
    # Add testing order index
    df_with_index = df.copy()
    df_with_index['testing_order'] = range(1, len(df_with_index) + 1)
    
    # Auto-detect score columns if not provided
    if score_columns is None:
        # Exclude non-numeric and identifier columns, include all other numeric columns
        exclude_cols = ['alpha', 'beta', 'gamma', 'timestamp', 'participant_id', 'testing_order']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        score_columns = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create aggregation dictionary
    agg_dict = {col: 'mean' for col in score_columns}
    agg_dict['testing_order'] = ['min', 'max', 'count']
    
    # Group by alpha, beta, gamma and take the mean of scores
    grouped = df_with_index.groupby(['alpha', 'beta', 'gamma']).agg(agg_dict).reset_index()
    
    # Flatten column names
    new_columns = ['alpha', 'beta', 'gamma'] + score_columns + ['first_test_order', 'last_test_order', 'num_trials']
    grouped.columns = new_columns
    
    # Add average testing order
    grouped['avg_test_order'] = (grouped['first_test_order'] + grouped['last_test_order']) / 2
    
    print(f"Handled duplicates: {len(df)} original rows -> {len(grouped)} unique configurations")
    return grouped


def create_3d_score_plot(
    df: pd.DataFrame, 
    score_column: str, 
    title: str,
    color_scale: str = 'viridis',
    reverse_scale: bool = False
) -> go.Figure:
    """
    Create a 3D scatter plot of alpha, beta, gamma colored by a score column.
    Handles duplicates by taking the mean.
    
    Args:
        df: DataFrame containing alpha, beta, gamma, and score columns
        score_column: Name of the column to use for coloring
        title: Title for the plot
        color_scale: Plotly color scale name
        reverse_scale: Whether to reverse the color scale
    
    Returns:
        Plotly Figure object
    """
    # Check if score column exists
    if score_column not in df.columns:
        print(f"Error: Column '{score_column}' not found in DataFrame")
        print(f"Available columns: {list(df.columns)}")
        return go.Figure()
    
    # Handle duplicates first
    if 'num_trials' not in df.columns:
        df = handle_duplicates_and_add_index(df)
    
    # Filter out NaN values
    plot_df = df.dropna(subset=[score_column])
    
    if len(plot_df) == 0:
        print(f"Warning: No valid data for {score_column}")
        return go.Figure()
    
    # Determine color scale
    colorscale = f"{color_scale}_r" if reverse_scale else color_scale
    
    fig = go.Figure(data=go.Scatter3d(
        x=plot_df['alpha'],
        y=plot_df['beta'], 
        z=plot_df['gamma'],
        mode='markers',
        marker=dict(
            size=10,
            color=plot_df[score_column],
            colorscale=colorscale,
            colorbar=dict(title=f"{score_column.replace('_', ' ').title()}"),
            opacity=0.8,
            line=dict(width=2, color='black')
        ),
        text=[f"α={a:.1f}, β={b:.1f}, γ={g:.1f}<br>{score_column}={s:.2f}" 
              for a, b, g, s in zip(plot_df['alpha'], plot_df['beta'], 
                                   plot_df['gamma'], plot_df[score_column])],
        hovertemplate="%{text}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Alpha (α)',
            yaxis_title='Beta (β)', 
            zaxis_title='Gamma (γ)',
            xaxis_range=[plot_df['alpha'].min()-5, plot_df['alpha'].max()+5],
            yaxis_range=[plot_df['beta'].min()-5, plot_df['beta'].max()+5],
            zaxis_range=[plot_df['gamma'].min()-2, plot_df['gamma'].max()+2]
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        width=900,
        height=700
    )
    
    return fig


def plot_order_vs_score(
    df: pd.DataFrame,
    score_column: str,
    title: str = None,
    use_existing_testing_order: bool = True
) -> go.Figure:
    """
    Scatter the score vs the test order (1..N) and overlay a linear regression line.

    Args:
        df: DataFrame with at least `score_column` and optionally `testing_order`.
        score_column: Column name of the score to plot.
        title: Plot title (optional).
        use_existing_testing_order: If True and 'testing_order' exists, use it;
                                    otherwise use the row order (1..N).

    Returns:
        Plotly Figure
    """
    # Choose x = testing order
    if use_existing_testing_order and 'testing_order' in df.columns:
        x = df['testing_order'].to_numpy()
    else:
        # Preserve current row order
        x = np.arange(1, len(df) + 1)

    # y = the score column
    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' not found in DataFrame.")
    y = df[score_column].to_numpy()

    # Keep only finite pairs
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    fig = go.Figure()

    if x.size == 0:
        # Nothing to show
        fig.update_layout(title=title or f"{score_column} vs Order")
        return fig

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name=score_column.replace('_', ' ').title(),
        marker=dict(size=9, opacity=0.9, line=dict(width=1, color='black')),
        hovertemplate="Order %{x}<br>Score %{y:.3f}<extra></extra>"
    ))

    # Fit simple linear regression
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=f"Fit: y = {slope:.3f}x + {intercept:.3f} (R²={r**2:.3f}, p={p:.3g})"
    ))

    # Layout
    fig.update_layout(
        title=title or f"{score_column} vs Testing Order",
        xaxis_title="Testing Order (1..N)",
        yaxis_title=score_column.replace('_', ' ').title(),
        template="plotly_white",
        width=800,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    return fig


def create_temporal_plot(df: pd.DataFrame, score_columns: list = None) -> go.Figure:
    """
    Create a multi-panel plot showing multiple scores vs testing order.
    Uses the simplified plot_order_vs_score function for each score.
    
    Args:
        df: DataFrame containing testing order and score columns
        score_columns: List of score columns to plot (if None, auto-detect)
    
    Returns:
        Plotly Figure object with subplots
    """
    # Handle duplicates first
    if 'num_trials' not in df.columns:
        df = handle_duplicates_and_add_index(df, score_columns)
    
    # Auto-detect score columns if not provided
    if score_columns is None:
        exclude_cols = ['alpha', 'beta', 'gamma', 'timestamp', 'participant_id', 'testing_order', 
                       'first_test_order', 'last_test_order', 'num_trials', 'avg_test_order']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        score_columns = [col for col in numeric_cols if col not in exclude_cols]
    
    # Limit to first 4 score columns for subplot layout
    score_columns = score_columns[:4]
    
    # Create subplots based on number of score columns
    n_scores = len(score_columns)
    if n_scores == 0:
        print("No score columns found for temporal analysis")
        return go.Figure()
    
    # Determine subplot layout
    if n_scores == 1:
        rows, cols = 1, 1
    elif n_scores == 2:
        rows, cols = 1, 2
    elif n_scores == 3:
        rows, cols = 2, 2
    else:  # 4 or more
        rows, cols = 2, 2
    
    # Create subplots
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'{col.replace("_", " ").title()} vs Testing Order' for col in score_columns],
        specs=[[{'type': 'scatter'} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Define colors for different scores
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot each score column
    for i, score_col in enumerate(score_columns):
        if score_col in df.columns:
            row = (i // cols) + 1
            col = (i % cols) + 1
            color = colors[i % len(colors)]
            
            # Get data for this score
            plot_df = df.dropna(subset=[score_col, 'avg_test_order']).copy()
            
            if len(plot_df) > 0:
                # Use the simplified plotting logic
                x = plot_df['avg_test_order'].to_numpy()
                y = plot_df[score_col].to_numpy()
                
                # Keep only finite pairs
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                
                if x.size > 0:
                    # Scatter points
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        name=score_col.replace('_', ' ').title(),
                        marker=dict(size=8, color=color, opacity=0.8, line=dict(width=1, color='black')),
                        hovertemplate=f"Order %{{x}}<br>{score_col} %{{y:.3f}}<extra></extra>",
                        showlegend=False
                    ), row=row, col=col)
                    
                    # Fit linear regression
                    slope, intercept, r, p, _ = stats.linregress(x, y)
                    x_line = np.array([x.min(), x.max()])
                    y_line = slope * x_line + intercept
                    
                    # Add trend line
                    fig.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f"Fit: y = {slope:.3f}x + {intercept:.3f} (R²={r**2:.3f}, p={p:.3g})",
                        showlegend=False
                    ), row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title="Scores vs Testing Order - Linear Regression Analysis",
        height=400 * rows,
        width=600 * cols,
        template="plotly_white"
    )
    
    # Update axis labels
    for i, score_col in enumerate(score_columns):
        row = (i // cols) + 1
        col = (i % cols) + 1
        fig.update_xaxes(title_text="Testing Order", row=row, col=col)
        fig.update_yaxes(title_text=score_col.replace('_', ' ').title(), row=row, col=col)
    
    return fig


def analyze_and_plot_gp(
    X: np.ndarray,
    Y: np.ndarray,
    model: GPy.models.GPRegression,
    loss_name: str,
    lengthscale: float,
    optimization_mode: str = 'min',
    param_ranges: Dict = None
) -> Tuple[Optional[go.Figure], Optional[np.ndarray], Optional[float]]:
    """
    Analyzes a GP model, finds the optimum, and creates a 3D plot with robust checks.
    
    Args:
        X: Input parameters (alpha, beta, gamma)
        Y: Target values
        model: Trained GPy model
        loss_name: Name of the metric being optimized
        lengthscale: Lengthscale used for the model
        optimization_mode: 'min' or 'max'
        param_ranges: Dictionary with parameter ranges for prediction grid
        
    Returns:
        Tuple of (figure, optimal_point, optimal_value) or (None, None, None) if failed
    """
    if param_ranges is None:
        param_ranges = {
            'alpha': (75, 115, 2), 'beta': (100, 140, 2), 'gamma': (0, 10, 0.5)
        }

    # 1. Generate prediction grid
    alpha_range = np.arange(param_ranges['alpha'][0], param_ranges['alpha'][1] + 1, param_ranges['alpha'][2])
    beta_range = np.arange(param_ranges['beta'][0], param_ranges['beta'][1] + 1, param_ranges['beta'][2])
    gamma_range = np.arange(param_ranges['gamma'][0], param_ranges['gamma'][1] + 1, param_ranges['gamma'][2])
    grid_points = np.vstack(np.meshgrid(alpha_range, beta_range, gamma_range)).reshape(3, -1).T
    valid_grid_points = grid_points[(grid_points[:, 0] + grid_points[:, 1]) >= 190]

    # 2. Predict over the grid
    predicted_mean, _ = model.predict(valid_grid_points)    

    # Robustness check
    num_nan = np.isnan(predicted_mean).sum()
    num_inf = np.isinf(predicted_mean).sum()
    if num_nan > 0 or num_inf > 0:
        print(f"[ERROR] Prediction for '{loss_name}' failed.")
        print(f"        Found {num_nan} NaN(s) and {num_inf} Inf(s) in the output.")
        return None, None, None

    # 3. Find the optimum based on the mode
    if optimization_mode == 'max':
        optimal_index = np.argmax(predicted_mean)
        optimum_label = "Max"
    else:
        optimal_index = np.argmin(predicted_mean)
        optimum_label = "Min"
        
    optimal_point = valid_grid_points[optimal_index]
    optimal_value = predicted_mean[optimal_index][0]

    print(f"  Predicted Optimum ({optimum_label}) Geometry (α, β, γ): {optimal_point}")
    print(f"  Predicted {optimum_label} Value: {optimal_value:.4f}")

    # Create custom colormap
    loss_colors = ["#FFFF00", "#800080"]
    loss_cmap = []
    for i in range(100):
        frac = i / 99
        r, g, b, _ = LinearSegmentedColormap.from_list("", loss_colors)(frac)
        loss_cmap.append([frac, f'rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})'])

    # Normalize values for coloring
    spread = np.ptp(predicted_mean)
    if spread < 1e-12:
        values_normalized = np.zeros_like(predicted_mean)
    else:
        values_normalized = (predicted_mean - predicted_mean.min()) / spread

    # Create 3D plot
    fig = go.Figure()
    
    # Add prediction points (semi-transparent)
    fig.add_trace(go.Scatter3d(
        x=valid_grid_points[:, 0], y=valid_grid_points[:, 1], z=valid_grid_points[:, 2],
        mode='markers', 
        marker=dict(
            size=20, 
            color=values_normalized.flatten(), 
            colorscale=loss_cmap,
            colorbar=dict(title=f'{loss_name.replace("_", " ").title()}'), 
            opacity=0.1
        ), 
        showlegend=False,
    ))
    
    # Add original data points
    fig.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers', 
        marker=dict(size=10, color=Y.flatten(), colorscale=loss_cmap, opacity=1),
        showlegend=False,
    ))
    
    # Add optimal point
    fig.add_trace(go.Scatter3d(
        x=[optimal_point[0]], y=[optimal_point[1]], z=[optimal_point[2]],
        mode='markers', 
        marker=dict(symbol='diamond', color='blue', size=15, line=dict(color='black', width=2)),
        name=f'{optimum_label} ({optimal_value:.2f})'
    ))
    
    fig.update_layout(
        title=f'"{loss_name.replace("_", " ").title()}" ({optimum_label} | Lengthscale: {lengthscale})',
        scene=dict(
            xaxis_title='Alpha', yaxis_title='Beta', zaxis_title='Gamma',
            xaxis_range=[param_ranges['alpha'][0], param_ranges['alpha'][1]],
            yaxis_range=[param_ranges['beta'][0], param_ranges['beta'][1]],
            zaxis_range=[param_ranges['gamma'][0], param_ranges['gamma'][1] + 2],
        ), 
        margin=dict(l=0, r=0, b=0, t=50), 
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig, optimal_point, optimal_value


def get_shaded_color(base_rgb_str: str, value: float, min_val: float, max_val: float, 
                    lightness_range: Tuple[float, float] = (0.8, 0.3)) -> str:
    """
    Calculates a shaded color by adjusting the lightness of a base RGB color.
    
    Args:
        base_rgb_str: The base color in 'rgb(r, g, b)' format
        value: The value (e.g., lengthscale) to map to a shade
        min_val: The minimum of the value range
        max_val: The maximum of the value range
        lightness_range: A tuple (max_lightness, min_lightness) for the output
        
    Returns:
        A new 'rgb(r, g, b)' string with adjusted lightness
    """
    try:
        r, g, b = map(int, re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', base_rgb_str).groups())
    except AttributeError:
        return 'rgb(0,0,0)'

    if max_val == min_val:
        norm_val = 0.5
    else:
        norm_val = (value - min_val) / (max_val - min_val)

    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    new_lightness = lightness_range[0] - (norm_val * (lightness_range[0] - lightness_range[1]))
    new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_lightness, s)

    return f'rgb({int(new_r * 255)}, {int(new_g * 255)}, {int(new_b * 255)})'


def comprehensive_gp_analysis(
    df: pd.DataFrame,
    target_columns: List[str],
    optimization_goals: Dict[str, str] = None,
    lengthscales: List[float] = None,
    param_ranges: Dict = None,
    output_dir: str = "gp_analysis_output"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform comprehensive GP analysis on multiple metrics with different lengthscales.
    
    Args:
        df: DataFrame with alpha, beta, gamma and target columns
        target_columns: List of column names to analyze
        optimization_goals: Dict mapping column names to 'min' or 'max'
        lengthscales: List of lengthscales to test
        param_ranges: Parameter ranges for prediction grid
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (results_df, summary_dict)
    """
    import os
    from pathlib import Path
    
    # Default parameters
    if optimization_goals is None:
        optimization_goals = {
            'cost_of_transport': 'min',
            'tlx_score': 'min', 
            'nrs_score': 'min',
            'sus_score': 'max',
            'step_variance': 'min'
        }
    
    if lengthscales is None:
        lengthscales = [15, 30, 50, 100]
        
    if param_ranges is None:
        param_ranges = {
            'alpha': (75, 115, 2), 'beta': (100, 140, 2), 'gamma': (-10, 10, 0.5)
        }
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    param_cols = ['alpha', 'beta', 'gamma']
    results = []
    
    print("="*80)
    print("COMPREHENSIVE GP ANALYSIS")
    print("="*80)
    
    for target_col in target_columns:
        if target_col not in df.columns:
            print(f"[WARNING] Column '{target_col}' not found in DataFrame")
            continue
            
        optimization_mode = optimization_goals.get(target_col, 'min')
        
        for lengthscale in lengthscales:
            print(f"\n--- Processing '{target_col.upper()}' with Lengthscale: {lengthscale} ---")
            
            try:
                # Prepare data - keep ALL data points, let GP handle duplicates naturally
                temp_df = df[param_cols + [target_col]].dropna()
                X = temp_df[param_cols].values
                Y = temp_df[[target_col]].values

                if len(X) < 2:
                    print(f"[INFO] SKIPPING: Not enough valid data points ({len(X)}) for '{target_col}'.")
                    continue

                Y_var = np.var(Y)
                if Y_var < 1e-9:
                    print(f"[INFO] SKIPPING: Variance of '{target_col}' is negligible.")
                    continue

                # Create and optimize GP model
                kernel = GPy.kern.Matern52(input_dim=X.shape[1], variance=Y_var, lengthscale=float(lengthscale))
                model = GPy.models.GPRegression(X, Y, kernel, noise_var=Y_var * 0.01)
                model.kern.lengthscale.fix()
                model.optimize(messages=False, max_iters=1000)

                # Analyze and plot
                fig, optimal_point, optimal_value = analyze_and_plot_gp(
                    X, Y, model, target_col, lengthscale, optimization_mode, param_ranges
                )

                if fig:
                    # Save plot
                    filename = f"{target_col}_lengthscale_{lengthscale}.html"
                    filepath = os.path.join(output_dir, filename)
                    fig.write_html(filepath)
                    print(f"[INFO] Saved plot to: {filepath}")

                # Get uncertainty at optimal point
                optimal_uncertainty = None
                if optimal_point is not None:
                    try:
                        _, optimal_variance = model.predict(optimal_point.reshape(1, -1))
                        optimal_uncertainty = np.sqrt(optimal_variance[0, 0])  # Standard deviation
                    except:
                        optimal_uncertainty = None

                # Store results
                results.append({
                    'metric': target_col,
                    'lengthscale': lengthscale,
                    'optimization_mode': optimization_mode,
                    'optimal_alpha': optimal_point[0] if optimal_point is not None else None,
                    'optimal_beta': optimal_point[1] if optimal_point is not None else None,
                    'optimal_gamma': optimal_point[2] if optimal_point is not None else None,
                    'predicted_value': optimal_value if optimal_value is not None else None,
                    'predicted_uncertainty': optimal_uncertainty,
                    'num_data_points': len(X)
                })

            except Exception as e:
                print(f"[ERROR] Failed to process '{target_col}' with lengthscale {lengthscale}: {e}")
                continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Create summary
    summary = {
        'total_combinations': len(target_columns) * len(lengthscales),
        'successful_combinations': len(results_df.dropna(subset=['optimal_alpha'])),
        'failed_combinations': len(results_df) - len(results_df.dropna(subset=['optimal_alpha'])),
        'metrics_analyzed': list(set(results_df['metric'].tolist())),
        'output_directory': output_dir
    }
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total combinations attempted: {summary['total_combinations']}")
    print(f"Successful: {summary['successful_combinations']}")
    print(f"Failed: {summary['failed_combinations']}")
    print(f"Output directory: {output_dir}")
    
    return results_df, summary


def create_gp_summary_plot(results_df: pd.DataFrame, title: str = "GP Optimization Results Summary", 
                          color_by_uncertainty: bool = False, size_by_value: bool = True) -> go.Figure:
    """
    Create a summary plot showing all optimal points from GP analysis.
    This is the "wavelength" summary plot that shows optimal geometries
    with color by metric and shading by lengthscale.
    
    Args:
        results_df: DataFrame from comprehensive_gp_analysis
        title: Title for the plot
        color_by_uncertainty: If True, color points by uncertainty density instead of metric
        size_by_value: If True, make point size proportional to predicted value
        
    Returns:
        Plotly Figure object
    """
    # Filter out failed results
    plot_df = results_df.dropna(subset=['optimal_alpha', 'optimal_beta', 'optimal_gamma'])
    
    if len(plot_df) == 0:
        print("No valid results to plot")
        return go.Figure()
    
    # Define mappings for symbols and base colors for each metric
    # Using the same color scheme as the original GP plots notebook
    symbol_map = {
        'cost_of_transport': 'circle', 
        'sus_score': 'diamond', 
        'metabolic_cost': 'square',
        'tlx_score': 'circle-open', 
        'nrs_score': 'cross', 
        'step_variance': 'x',
        'y_average': 'square',  # For metabolics
        'metabolics': 'square'  # Alternative name
    }
    
    color_map = {
        'cost_of_transport': 'rgb(31, 119, 180)',  # Blue
        'sus_score': 'rgb(255, 127, 14)',   # Orange
        'metabolic_cost': 'rgb(44, 160, 44)',   # Green
        'y_average': 'rgb(44, 160, 44)',   # Green (metabolics)
        'metabolics': 'rgb(44, 160, 44)',   # Green (metabolics)
        'tlx_score': 'rgb(214, 39, 40)',   # Red
        'nrs_score': 'rgb(148, 103, 189)',  # Purple
        'step_variance': 'rgb(140, 86, 75)'    # Brown
    }
    
    # Create the figure
    fig = go.Figure()
    
    if color_by_uncertainty:
        # Color by uncertainty density
        if 'predicted_uncertainty' not in plot_df.columns:
            print("Warning: 'predicted_uncertainty' column not found. Falling back to metric-based coloring.")
            color_by_uncertainty = False
        else:
            # Filter out NaN uncertainties
            plot_df = plot_df.dropna(subset=['predicted_uncertainty'])
            
            if len(plot_df) == 0:
                print("No valid uncertainty data found. Falling back to metric-based coloring.")
                color_by_uncertainty = False
    
    if color_by_uncertainty:
        # Color all points by uncertainty using a continuous color scale
        uncertainty_values = plot_df['predicted_uncertainty'].values
        
        # Calculate size based on predicted value and optimization mode
        if size_by_value:
            sizes = []
            for _, row in plot_df.iterrows():
                value = row['predicted_value']
                mode = row['optimization_mode']
                if mode == 'min':
                    # For minimization: smaller values = bigger points
                    # Invert the value for size calculation
                    size = 20 - (value - plot_df['predicted_value'].min()) / (plot_df['predicted_value'].max() - plot_df['predicted_value'].min()) * 15
                else:
                    # For maximization: larger values = bigger points
                    size = 5 + (value - plot_df['predicted_value'].min()) / (plot_df['predicted_value'].max() - plot_df['predicted_value'].min()) * 15
                sizes.append(max(5, min(20, size)))  # Clamp between 5 and 20
        else:
            sizes = 12
        
        fig.add_trace(go.Scatter3d(
            x=plot_df['optimal_alpha'],
            y=plot_df['optimal_beta'],
            z=plot_df['optimal_gamma'],
            mode='markers',
            hovertemplate=(
                "<b>Metric:</b> %{customdata[0]}<br>"
                "<b>Lengthscale:</b> %{customdata[1]}<br>"
                "<b>Alpha:</b> %{x}<br>"
                "<b>Beta:</b> %{y}<br>"
                "<b>Gamma:</b> %{z}<br>"
                "<b>Predicted Value:</b> %{customdata[2]:.3f}<br>"
                "<b>Uncertainty:</b> %{marker.color:.3f}<br>"
                "<b>Size:</b> %{marker.size:.1f}<extra></extra>"
            ),
            customdata=plot_df[['metric', 'lengthscale', 'predicted_value']],
            marker=dict(
                color=uncertainty_values,
                colorscale='viridis_r',  # Reverse viridis: low uncertainty = bright, high uncertainty = dark
                colorbar=dict(title="Predicted Uncertainty"),
                size=sizes,
                opacity=0.8,
                line=dict(color='black', width=1),
                symbol=[symbol_map.get(metric, 'circle') for metric in plot_df['metric']]
            ),
            name="All Metrics (colored by uncertainty)"
        ))
    else:
        # Original metric-based coloring
        # Get the global min and max lengthscale for consistent shading
        min_ls = plot_df['lengthscale'].min()
        max_ls = plot_df['lengthscale'].max()
        
        # Iterate through each metric to plot it as a separate trace
        for metric_name, group_df in plot_df.groupby('metric'):
            # Convert 'y_average' to 'metabolics' for display purposes
            display_name = 'metabolics' if metric_name == 'y_average' else metric_name
            
            # Calculate the specific color for each point in the group based on its lengthscale
            colors_for_trace = [
                get_shaded_color(color_map.get(display_name, 'rgb(128, 128, 128)'), ls, min_ls, max_ls)
                for ls in group_df['lengthscale']
            ]
            
            # Calculate size based on predicted value and optimization mode
            if size_by_value:
                sizes = []
                for _, row in group_df.iterrows():
                    value = row['predicted_value']
                    mode = row['optimization_mode']
                    if mode == 'min':
                        # For minimization: smaller values = bigger points
                        size = 20 - (value - group_df['predicted_value'].min()) / (group_df['predicted_value'].max() - group_df['predicted_value'].min()) * 15
                    else:
                        # For maximization: larger values = bigger points
                        size = 5 + (value - group_df['predicted_value'].min()) / (group_df['predicted_value'].max() - group_df['predicted_value'].min()) * 15
                    sizes.append(max(5, min(20, size)))  # Clamp between 5 and 20
            else:
                sizes = 10

            fig.add_trace(go.Scatter3d(
                x=group_df['optimal_alpha'],
                y=group_df['optimal_beta'],
                z=group_df['optimal_gamma'],
                mode='markers',
                # Use hover text to show details
                hovertemplate=(
                    "<b>Metric:</b> " + display_name.replace("_", " ").title() + "<br>"
                    "<b>Lengthscale:</b> %{customdata[0]}<br>"
                    "<b>Alpha:</b> %{x}<br>"
                    "<b>Beta:</b> %{y}<br>"
                    "<b>Gamma:</b> %{z}<br>"
                    "<b>Predicted Value:</b> %{customdata[1]:.3f}<br>"
                    "<b>Size:</b> %{marker.size:.1f}<extra></extra>"
                ),
                customdata=group_df[['lengthscale', 'predicted_value']],
                marker=dict(
                    color=colors_for_trace,  # Apply the manually calculated colors
                    symbol=symbol_map.get(display_name, 'circle'),
                    size=sizes,
                    opacity=0.9,
                    line=dict(color='black', width=1) # Add a border for better visibility
                ),
                name=display_name.replace("_", " ").title()
            ))

    # Customize layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Optimal Alpha',
            yaxis_title='Optimal Beta',
            zaxis_title='Optimal Gamma',
            xaxis_range=[75, 115],
            yaxis_range=[100, 140],
            zaxis_range=[-11, 11]
        ),
        legend_title_text='Metric',
        margin=dict(l=0, r=0, b=0, t=50),
        width=1000,
        height=800
    )

    return fig


def create_uncertainty_density_plot(results_df: pd.DataFrame, title: str = "GP Optimization Uncertainty Density") -> go.Figure:
    """
    Create a 3D plot showing the density of optimal points colored by their uncertainty.
    This helps identify regions of parameter space where predictions are most/least certain.
    
    Args:
        results_df: DataFrame from comprehensive_gp_analysis with uncertainty data
        title: Title for the plot
        
    Returns:
        Plotly Figure object
    """
    # Filter out failed results and missing uncertainty data
    plot_df = results_df.dropna(subset=['optimal_alpha', 'optimal_beta', 'optimal_gamma', 'predicted_uncertainty'])
    
    if len(plot_df) == 0:
        print("No valid uncertainty data found")
        return go.Figure()
    
    # Create density-based coloring
    # Use a continuous color scale based on uncertainty values
    uncertainty_values = plot_df['predicted_uncertainty'].values
    
    # Create the figure
    fig = go.Figure()
    
    # Add all points colored by uncertainty
    fig.add_trace(go.Scatter3d(
        x=plot_df['optimal_alpha'],
        y=plot_df['optimal_beta'],
        z=plot_df['optimal_gamma'],
        mode='markers',
        hovertemplate=(
            "<b>Metric:</b> %{customdata[0]}<br>"
            "<b>Lengthscale:</b> %{customdata[1]}<br>"
            "<b>Alpha:</b> %{x}<br>"
            "<b>Beta:</b> %{y}<br>"
            "<b>Gamma:</b> %{z}<br>"
            "<b>Predicted Value:</b> %{customdata[2]:.3f}<br>"
            "<b>Uncertainty:</b> %{marker.color:.3f}<extra></extra>"
        ),
        customdata=plot_df[['metric', 'lengthscale', 'predicted_value']],
        marker=dict(
            color=uncertainty_values,
            colorscale='plasma',  # Good for uncertainty: bright = low uncertainty, dark = high uncertainty
            colorbar=dict(title="Predicted Uncertainty (σ)"),
            size=15,
            opacity=0.8,
            line=dict(color='black', width=1)
        ),
        name="Optimal Points (colored by uncertainty)"
    ))
    
    # Add a density surface if we have enough points
    if len(plot_df) > 10:
        # Create a grid for density visualization
        alpha_range = np.linspace(plot_df['optimal_alpha'].min(), plot_df['optimal_alpha'].max(), 20)
        beta_range = np.linspace(plot_df['optimal_beta'].min(), plot_df['optimal_beta'].max(), 20)
        gamma_range = np.linspace(plot_df['optimal_gamma'].min(), plot_df['optimal_gamma'].max(), 20)
        
        # Create meshgrid
        A, B, G = np.meshgrid(alpha_range, beta_range, gamma_range)
        
        # Calculate density using a simple kernel density estimation
        from scipy.stats import gaussian_kde
        points = plot_df[['optimal_alpha', 'optimal_beta', 'optimal_gamma']].values.T
        kde = gaussian_kde(points)
        
        # Evaluate density on the grid
        grid_points = np.vstack([A.ravel(), B.ravel(), G.ravel()])
        density = kde(grid_points).reshape(A.shape)
        
        # Add isosurface for density
        fig.add_trace(go.Isosurface(
            x=A.flatten(),
            y=B.flatten(), 
            z=G.flatten(),
            value=density.flatten(),
            isomin=density.min() + 0.1 * (density.max() - density.min()),
            isomax=density.max(),
            surface_count=3,
            opacity=0.3,
            colorscale='viridis',
            name="Point Density"
        ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Optimal Alpha',
            yaxis_title='Optimal Beta',
            zaxis_title='Optimal Gamma',
            xaxis_range=[plot_df['optimal_alpha'].min()-5, plot_df['optimal_alpha'].max()+5],
            yaxis_range=[plot_df['optimal_beta'].min()-5, plot_df['optimal_beta'].max()+5],
            zaxis_range=[plot_df['optimal_gamma'].min()-2, plot_df['optimal_gamma'].max()+2]
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        width=1000,
        height=800
    )
    
    return fig


# Example usage:
if __name__ == "__main__":
    # Example with your MIH_27_full_merged data
    # MIH_27_full_merged = your_data_here  # Your DataFrame with alpha, beta, gamma, and any numeric columns
    
    # === BASIC 3D PLOTTING ===
    # Create 3D plots for any numeric column
    # cost_of_transport_fig = create_3d_score_plot(MIH_27_full_merged, 'cost_of_transport', 'Cost of Transport (Lower is Better)', reverse_scale=True)
    # tlx_fig = create_3d_score_plot(MIH_27_full_merged, 'tlx_score', 'TLX Score (Lower is Better)', reverse_scale=True)
    # sus_fig = create_3d_score_plot(MIH_27_full_merged, 'sus_score', 'SUS Score (Higher is Better)', reverse_scale=False)
    # nrs_fig = create_3d_score_plot(MIH_27_full_merged, 'nrs_score', 'NRS Score (Lower is Better)', reverse_scale=True)
    
    # === TEMPORAL ANALYSIS ===
    # Create single score temporal plots (clean, focused)
    # cost_temporal = plot_order_vs_score(MIH_27_full_merged, 'cost_of_transport', 'Cost of Transport vs Testing Order')
    # tlx_temporal = plot_order_vs_score(MIH_27_full_merged, 'tlx_score', 'TLX Score vs Testing Order')
    
    # Create multi-panel temporal plot (auto-detects all numeric columns)
    # temporal_fig = create_temporal_plot(MIH_27_full_merged)
    
    # === GP WAVELENGTH ANALYSIS ===
    # Define target columns for GP analysis
    # target_columns = ['cost_of_transport', 'tlx_score', 'nrs_score', 'sus_score', 'metabolic_cost', 'step_variance']
    
    # Define optimization goals (minimize most, maximize SUS)
    # optimization_goals = {
    #     'cost_of_transport': 'min',
    #     'tlx_score': 'min', 
    #     'nrs_score': 'min',
    #     'sus_score': 'max',
    #     'metabolic_cost': 'min',
    #     'step_variance': 'min'
    # }
    
    # Run comprehensive GP analysis
    # results_df, summary = comprehensive_gp_analysis(
    #     df=MIH_27_full_merged,
    #     target_columns=target_columns,
    #     optimization_goals=optimization_goals,
    #     lengthscales=[2, 5, 10, 15, 30, 50, 100],  # Test different "wavelengths"
    #     output_dir="gp_analysis_MIH27"
    # )
    
    # Create summary plot of all optimal points
    # summary_fig = create_gp_summary_plot(results_df, "MIH27 GP Optimization Results")
    # summary_fig.show()
    
    # Create uncertainty-based density plot
    # uncertainty_fig = create_uncertainty_density_plot(results_df, "MIH27 Uncertainty Density")
    # uncertainty_fig.show()
    
    # Create summary plot colored by uncertainty instead of metric
    # uncertainty_summary_fig = create_gp_summary_plot(results_df, "MIH27 GP Results (Uncertainty Colored)", color_by_uncertainty=True)
    # uncertainty_summary_fig.show()
    
    # === USAGE EXAMPLES ===
    print("="*80)
    print("COMPRESSED PLOTTING FUNCTIONS READY!")
    print("="*80)
    print("\n📊 BASIC 3D PLOTTING:")
    print("  create_3d_score_plot(df, 'column_name', 'Title')")
    print("  - Auto-detects numeric columns")
    print("  - Handles duplicates by taking mean")
    print("  - Interactive 3D plots with hover info")
    
    print("\n📈 TEMPORAL ANALYSIS:")
    print("  plot_order_vs_score(df, 'column_name') - Single score vs testing order")
    print("  create_temporal_plot(df) - Multi-panel temporal analysis")
    print("  - Auto-detects all numeric columns")
    print("  - Linear regression with R² and p-values")
    
    print("\n🧠 GP WAVELENGTH ANALYSIS:")
    print("  comprehensive_gp_analysis(df, target_columns, optimization_goals)")
    print("  - Tests multiple lengthscales (2, 5, 10, 15, 30, 50, 100)")
    print("  - Creates 'wavelength' predictions across parameter space")
    print("  - Finds optimal geometries for each metric")
    print("  - Saves interactive plots for each combination")
    print("  - create_gp_summary_plot(results_df) - Summary of all optima")
    
    print("\n🎯 FOR MIH_27_full_merged:")
    print("  target_columns = ['cost_of_transport', 'tlx_score', 'nrs_score', 'sus_score']")
    print("  results_df, summary = comprehensive_gp_analysis(MIH_27_full_merged, target_columns)")
    print("  summary_fig = create_gp_summary_plot(results_df)")
    print("  summary_fig.show()")