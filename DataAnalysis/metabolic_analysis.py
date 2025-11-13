#!/usr/bin/env python3
"""
Metabolic Analysis with Real Exponential Fitting
===============================================

Clean, efficient metabolic analysis using exponential fitting for accurate 
steady-state estimation. Based on the gold standard approach for short-duration 
exercise protocols.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
from pathlib import Path
import re
from io import StringIO
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import json
from datetime import datetime


def exponential_rise(t: np.ndarray, y_ss: float, tau_fit: float) -> np.ndarray:
    """Exponential rise model: y_ss * (1 - exp(-t/tau_fit))"""
    return y_ss * (1 - np.exp(-t / tau_fit))


def metabolic_rate_estimation(time: np.ndarray, y_meas: np.ndarray, tau: float = 42.0) -> Tuple[float, np.ndarray, Dict]:
    """
    Estimate steady-state metabolic cost using exponential rise model.
    
    Args:
        time: Time array in seconds
        y_meas: Measured metabolic cost array (W/kg)
        tau: Time constant for exponential fit (default: 42s)
    
    Returns:
        y_estimate: Estimated steady-state metabolic cost (W/kg)
        y_bar: Fitted exponential curve
        fit_params: Dictionary with fit parameters
    """
    if len(time) < 10 or len(y_meas) < 10:
        y_bar = np.full_like(y_meas, np.mean(y_meas))
        return np.mean(y_meas), y_bar, {'method': 'simple_average', 'reason': 'insufficient_data'}
    
    try:
        # Initial guess: steady state from last 30%, tau from parameter
        y_ss_guess = np.mean(y_meas[-int(0.3 * len(y_meas)):])
        tau_guess = tau
        
        # Fit with reasonable bounds
        bounds = ([0.5, 10.0], [50.0, 200.0])
        popt, pcov = curve_fit(exponential_rise, time, y_meas, 
                              p0=[y_ss_guess, tau_guess], bounds=bounds, maxfev=10000)
        
        y_ss_fitted, tau_fitted = popt
        y_bar = exponential_rise(time, y_ss_fitted, tau_fitted)
        
        r_squared = 1 - np.sum((y_meas - y_bar)**2) / np.sum((y_meas - np.mean(y_meas))**2)
        
        return y_ss_fitted, y_bar, {
            'method': 'exponential_fit',
            'y_ss': y_ss_fitted,
            'tau': tau_fitted,
            'r_squared': r_squared
        }
        
    except (RuntimeError, ValueError) as e:
        print(f"Exponential fitting failed: {e}")
        y_bar = np.full_like(y_meas, np.mean(y_meas))
        return np.mean(y_meas), y_bar, {'method': 'simple_average', 'reason': 'fitting_failed'}


def parse_mih_csv(csv_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse MIH metabolics CSV file.
    
    Returns:
        data_df: DataFrame with BxB data
        metadata: Dictionary with metadata
    """
    csv_path = Path(csv_path)
    with csv_path.open('r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find the IDS header
    header_idx = next((i for i, line in enumerate(lines) if line.startswith(('IDS,', 'IDs,'))), None)
    if header_idx is None:
        raise RuntimeError("Couldn't find the 'IDS,' header line.")

    # Parse metadata
    meta = {}
    kv_pattern = re.compile(r'^\s*([^,]+)\s*,\s*(.*)\s*$')
    for line in lines[:header_idx]:
        line = line.strip()
        if line and not line.startswith('#') and not line.lower().startswith('filename'):
            match = kv_pattern.match(line)
            if match:
                meta[match.group(1)] = match.group(2)

    # Read the table
    data_text = ''.join(lines[header_idx:])
    df = pd.read_csv(StringIO(data_text), header=0, engine='python', on_bad_lines='warn')
    
    # Clean headers and filter BxB data
    df.columns = [c.strip() for c in df.columns]
    if len(df.columns) > 1 and re.fullmatch(r'\d+', str(df.columns[1])):
        df = df.rename(columns={df.columns[0]: 'Type', df.columns[1]: 'SampleRate'})
    else:
        df = df.rename(columns={df.columns[0]: 'Type'})

    # Parse time and convert to numeric
    data_df = df[df['Type'].str.upper() == 'BXB'].reset_index(drop=True)
    if 'hh:mm:ss' in data_df.columns:
        data_df['hh:mm:ss'] = pd.to_timedelta(data_df['hh:mm:ss'], errors='coerce')
        data_df['time_seconds'] = data_df['hh:mm:ss'].dt.total_seconds()
    else:
        data_df['time_seconds'] = data_df['Item'] - data_df['Item'].iloc[0]

    # Convert numeric columns
    non_numeric = {'Type', 'hh:mm:ss', 'time_seconds'}
    numeric_cols = [c for c in data_df.columns if c not in non_numeric]
    data_df[numeric_cols] = data_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return data_df, meta


def calculate_metabolic_cost(df: pd.DataFrame, subject_weight_kg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate metabolic cost from VO2 and VCO2 data."""
    # Remove NaN values
    valid_mask = ~(df[['K5_VO2', 'K5_VO2']].isna().any(axis=1))
    time = df.loc[valid_mask, 'time_seconds'].values
    vo2 = df.loc[valid_mask, 'K5_VO2'].values
    vco2 = df.loc[valid_mask, 'K5_VO2'].values
    
    # Normalize time to start at 0
    time = time - time[0]
    
    # Calculate metabolic cost (W/kg)
    metabolic_cost = (0.278 * vo2 + 0.075 * vco2) / subject_weight_kg
    
    return time, metabolic_cost, valid_mask


def create_plots(df: pd.DataFrame, metadata: Dict, selected_markers: Optional[List[int]] = None, 
                filename_prefix: str = "metabolic_analysis") -> Dict[str, str]:
    """
    Create and save all plots (interactive HTML, PNG, and static matplotlib).
    
    Returns:
        Dictionary with filenames of created plots
    """
    subject_weight_kg = float(metadata['Weight'])
    time, metabolic_cost, _ = calculate_metabolic_cost(df, subject_weight_kg)
    
    # Get marker times
    marker_data = df[df['Marker'] == 1]
    marker_times = marker_data['time_seconds'].values - time[0]  # Normalize to start at 0
    
    # Create interactive Plotly plot
    fig = go.Figure()
    
    # Calculate exponential fit for the entire dataset
    y_estimate, y_bar, _ = metabolic_rate_estimation(time, metabolic_cost)
    
    # Show metabolic cost as a line (smoothed) instead of raw points
    fig.add_trace(go.Scatter(
        x=time, y=metabolic_cost,
        mode='lines', name='Metabolic Cost',
        line=dict(color='lightblue', width=1),
        opacity=0.7,
        hovertemplate='Time: %{x:.1f}s<br>Metabolic Cost: %{y:.3f} W/kg<extra></extra>'
    ))
    
    # Exponential fit overlay
    fig.add_trace(go.Scatter(
        x=time, y=y_bar, mode='lines', name='Exponential Fit',
        line=dict(color='blue', width=2)
    ))
    
    # Steady state line
    fig.add_hline(y=y_estimate, line_dash="dash", line_color="green",
                  annotation_text=f"Steady State: {y_estimate:.3f} W/kg")
    
    # Markers
    colors = px.colors.qualitative.Set3
    for i, marker_time in enumerate(marker_times):
        color = colors[i % len(colors)]
        fig.add_vline(x=marker_time, line_dash="dash", line_color=color, opacity=0.8,
                      annotation_text=f"M{i}")
        
        # Highlight selected markers
        if selected_markers and i in selected_markers:
            fig.add_vrect(x0=marker_time-10, x1=marker_time+10, fillcolor=color, 
                         opacity=0.2, layer="below", line_width=0)
    
    fig.update_layout(
        title=f"Metabolic Analysis - {metadata.get('First name', 'Unknown')} {metadata.get('Last name', '')}",
        xaxis_title="Time (seconds)", yaxis_title="Metabolic Cost (W/kg)",
        hovermode='x unified', width=1200, height=600
    )
    
    # Save plots
    filenames = {}
    html_file = f"{filename_prefix}.html"
    fig.write_html(html_file)
    filenames['html'] = html_file
    
    png_file = f"{filename_prefix}.png"
    fig.write_image(png_file, width=1200, height=600, scale=2)
    filenames['png'] = png_file
    
    # Static matplotlib plot
    plt.figure(figsize=(12, 8))
    plt.plot(time, metabolic_cost, 'b-', alpha=0.6, linewidth=1, label='Metabolic Cost')
    plt.plot(time, y_bar, 'r-', linewidth=2, label=f'Exponential Fit (SS: {y_estimate:.3f} W/kg)')
    plt.axhline(y=y_estimate, color='green', linestyle=':', alpha=0.8, 
               label=f'Steady State: {y_estimate:.3f} W/kg')
    
    for i, marker_time in enumerate(marker_times):
        plt.axvline(x=marker_time, color='red', linestyle='--', alpha=0.7)
        plt.text(marker_time, plt.ylim()[1]*0.95, f'M{i}', rotation=90, 
                verticalalignment='top', fontsize=10, color='red')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Metabolic Cost (W/kg)')
    plt.title(f'Metabolic Analysis - {metadata.get("First name", "Unknown")} {metadata.get("Last name", "")}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    static_file = f"{filename_prefix}_static.png"
    plt.savefig(static_file, dpi=300, bbox_inches='tight')
    plt.close()
    filenames['static'] = static_file
    
    return filenames


def analyze_marker_advanced(df: pd.DataFrame, metadata: Dict, marker_indices: List[int], 
                          tau: float = 42, max_experiment_duration: float = 60*2.5) -> Dict:
    """
    Advanced marker analysis: find highest cost between markers, then fit exponential.
    
    Args:
        df: DataFrame with metabolic data
        metadata: Subject metadata
        marker_indices: List of selected marker indices
        tau: Time constant for exponential fitting
    
    Returns:
        Dictionary with results for each marker
    """
    subject_weight_kg = float(metadata['Weight'])
    time, metabolic_cost, valid_mask = calculate_metabolic_cost(df, subject_weight_kg)
    
    # Get marker times
    marker_data = df[df['Marker'] == 1]
    all_marker_times = marker_data['time_seconds'].values
    all_marker_indices = np.where(df['Marker'] == 1)[0]
    
    selected_marker_times = all_marker_times[marker_indices]
    selected_marker_indices = all_marker_indices[marker_indices]
    
    results = {}
    
    for i, (marker_time, marker_idx) in enumerate(zip(selected_marker_times, selected_marker_indices)):
        print(f"\n=== Processing Marker {i+1} at {marker_time:.1f}s ===")
        
        # Find data between this marker and the next
        if i < len(selected_marker_times) - 1:
            window_end = selected_marker_times[i + 1]
        else:
            window_end = df['time_seconds'].max()
        
        # Get data in this window
        window_mask = (df['time_seconds'] >= marker_time) & (df['time_seconds'] <= window_end)
        window_data = df[window_mask]
        
        if len(window_data) == 0:
            print(f"  No data found in window")
            continue
        
        # Calculate metabolic cost for this window
        time_window = window_data['time_seconds'].values - marker_time  # Start at 0
        vo2_window = window_data['K5_VO2'].values
        vco2_window = window_data['K5_VO2'].values
        metabolic_cost_window = (0.278 * vo2_window + 0.075 * vco2_window) / subject_weight_kg
        
        # Remove NaN values
        valid_window_mask = ~(np.isnan(vo2_window) | np.isnan(vco2_window))
        time_clean = time_window[valid_window_mask]
        metabolic_cost_clean = metabolic_cost_window[valid_window_mask]
        
        if len(time_clean) < 60:
            print(f"  Insufficient data for fitting")
            continue
        
        # Cut data at specified duration before finding max
        cutoff_mask = time_clean <= max_experiment_duration
        time_clean = time_clean[cutoff_mask]
        metabolic_cost_clean = metabolic_cost_clean[cutoff_mask]
        
        # Find highest metabolic cost in the 2.5min window
        max_cost_idx = np.argmax(metabolic_cost_clean)
        max_metabolic_cost = metabolic_cost_clean[max_cost_idx]
        max_cost_time = time_clean[max_cost_idx]
        
        print(f"  Highest metabolic cost: {max_metabolic_cost:.3f} W/kg at {max_cost_time:.1f}s")
        
        # Cut data from start to highest cost point
        cleaned_mask = time_clean <= max_cost_time
        time_cleaned = time_clean[cleaned_mask]
        metabolic_cost_cleaned = metabolic_cost_clean[cleaned_mask]
        
        print(f"  Cleaned data: {len(time_cleaned)} points")
        
        # Fit exponential to cleaned data
        y_estimate, y_bar, fit_params = metabolic_rate_estimation(time_cleaned, metabolic_cost_cleaned, tau)
        
        print(f"  Exponential fit R²: {fit_params.get('r_squared', 'N/A'):.3f}")
        print(f"  5-minute extrapolated cost: {y_estimate:.3f} W/kg")
        
        results[f'marker_{i+1}'] = {
            'marker_index': i + 1,
            'marker_time': marker_time,
            'window_start': marker_time,
            'window_end': window_end,
            'max_cost_time': marker_time + max_cost_time,
            'max_metabolic_cost': max_metabolic_cost,
            'cleaned_data_points': len(time_cleaned),
            'cleaned_duration': time_cleaned[-1] - time_cleaned[0],
            'extrapolated_5min': y_estimate,
            'average_cleaned': np.mean(metabolic_cost_cleaned),
            'fit_params': fit_params,
            'time_cleaned': time_cleaned,
            'metabolic_cost_cleaned': metabolic_cost_cleaned,
            'y_bar_fitted': y_bar
        }
    
    return results


def create_summary_table(results: Dict) -> pd.DataFrame:
    """Create a summary table of marker analysis results."""
    summary_data = []
    
    for marker_key, marker_data in results.items():
        summary_data.append({
            'Marker': marker_data['marker_index'],
            'Marker Time (s)': f"{marker_data['marker_time']:.1f}",
            'Window (s)': f"{marker_data['window_start']:.1f} - {marker_data['window_end']:.1f}",
            'Max Cost (W/kg)': f"{marker_data['max_metabolic_cost']:.3f}",
            'Max Time (s)': f"{marker_data['max_cost_time']:.1f}",
            'Cleaned Points': marker_data['cleaned_data_points'],
            'Cleaned Duration (s)': f"{marker_data['cleaned_duration']:.1f}",
            'Avg Cleaned (W/kg)': f"{marker_data['average_cleaned']:.3f}",
            'Extrapolated 5min (W/kg)': f"{marker_data['extrapolated_5min']:.3f}",
            'Fit R²': f"{marker_data['fit_params'].get('r_squared', 'N/A'):.3f}"
        })
    
    return pd.DataFrame(summary_data)


def plot_marker_intervals(selected_markers: List[int], marker_times: np.ndarray, 
                         filename_prefix: str = "metabolic_analysis") -> str:
    """
    Create a plot showing the distribution of times between selected markers.
    
    Args:
        selected_markers: List of selected marker indices
        marker_times: Array of all marker times
        filename_prefix: Base filename for the plot
    
    Returns:
        Path to the saved plot file
    """
    if len(selected_markers) < 2:
        print("Need at least 2 markers to calculate intervals")
        return None
    
    # Get selected marker times
    selected_times = marker_times[selected_markers]
    
    # Calculate intervals between consecutive markers
    intervals = np.diff(selected_times)
    
    # Create the plot
    fig = go.Figure()
    
    # Histogram of intervals
    fig.add_trace(go.Histogram(
        x=intervals,
        nbinsx=20,
        name='Marker Intervals',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add statistics
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    median_interval = np.median(intervals)
    
    # Add vertical lines for statistics
    fig.add_vline(x=mean_interval, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_interval:.1f}s")
    fig.add_vline(x=median_interval, line_dash="dot", line_color="green", 
                  annotation_text=f"Median: {median_interval:.1f}s")
    
    # Update layout
    fig.update_layout(
        title=f"Distribution of Marker Intervals<br>"
              f"Mean: {mean_interval:.1f}s, Std: {std_interval:.1f}s, "
              f"Range: {np.min(intervals):.1f}s - {np.max(intervals):.1f}s",
        xaxis_title="Time Between Markers (seconds)",
        yaxis_title="Frequency",
        showlegend=True,
        width=800, height=500
    )
    
    # Save the plot
    html_filename = f"{filename_prefix}_marker_intervals.html"
    fig.write_html(html_filename)
    
    png_filename = f"{filename_prefix}_marker_intervals.png"
    fig.write_image(png_filename, width=800, height=500, scale=2)
    
    print(f"Marker intervals plot saved: {html_filename} & {png_filename}")
    return html_filename


def plot_individual_markers(results: Dict, metadata: Dict, filename_prefix: str = "metabolic_analysis") -> str:
    """
    Create individual plots for each marker's cleaned data with exponential fit.
    
    Args:
        results: Dictionary with marker analysis results
        metadata: Subject metadata
        filename_prefix: Base filename for plots
    
    Returns:
        Path to the plotting directory
    """
    # Create plotting directory
    plotting_dir = Path("plotting_dir")
    plotting_dir.mkdir(exist_ok=True)
    
    print(f"\nCreating individual marker plots in: {plotting_dir}")
    
    for marker_key, marker_data in results.items():
        marker_num = marker_data['marker_index']
        time_cleaned = marker_data['time_cleaned']
        metabolic_cost_cleaned = marker_data['metabolic_cost_cleaned']
        y_bar_fitted = marker_data['y_bar_fitted']
        extrapolated_5min = marker_data['extrapolated_5min']
        fit_params = marker_data['fit_params']
        
        # Create the plot
        fig = go.Figure()
        
        # Raw data points
        fig.add_trace(go.Scatter(
            x=time_cleaned, y=metabolic_cost_cleaned,
            mode='markers', name='Raw Data',
            marker=dict(size=6, opacity=0.7, color='blue'),
            hovertemplate='Time: %{x:.1f}s<br>Metabolic Cost: %{y:.3f} W/kg<extra></extra>'
        ))
        
        # Exponential fit line
        fig.add_trace(go.Scatter(
            x=time_cleaned, y=y_bar_fitted,
            mode='lines', name='Exponential Fit',
            line=dict(color='red', width=3),
            hovertemplate='Time: %{x:.1f}s<br>Fitted: %{y:.3f} W/kg<extra></extra>'
        ))
        
        # Steady state line
        fig.add_hline(
            y=extrapolated_5min, line_dash="dash", line_color="green", line_width=2,
            annotation_text=f"5-min Extrapolated: {extrapolated_5min:.3f} W/kg",
            annotation_position="top right"
        )
        
        # Add R² annotation
        r_squared = fit_params.get('r_squared', 'N/A')
        r_squared_text = f"R² = {r_squared:.3f}" if isinstance(r_squared, (int, float)) else f"R² = {r_squared}"
        
        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=r_squared_text, showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Marker {marker_num} - {metadata.get('First name', 'Unknown')} {metadata.get('Last name', '')}<br>"
                  f"Cleaned Data: {len(time_cleaned)} points, Duration: {time_cleaned[-1] - time_cleaned[0]:.1f}s",
            xaxis_title="Time (seconds)",
            yaxis_title="Metabolic Cost (W/kg)",
            hovermode='x unified',
            width=800, height=500,
            showlegend=True
        )
        
        # Save as HTML
        html_filename = plotting_dir / f"{filename_prefix}_marker_{marker_num:02d}.html"
        fig.write_html(html_filename)
        
        # Save as PNG
        png_filename = plotting_dir / f"{filename_prefix}_marker_{marker_num:02d}.png"
        fig.write_image(png_filename, width=800, height=500, scale=2)
        
        print(f"  Marker {marker_num}: {html_filename.name} & {png_filename.name}")
    
    print(f"Individual marker plots saved in: {plotting_dir.absolute()}")
    return str(plotting_dir)


def export_results(results: Dict, filename_prefix: str = "metabolic_analysis") -> Tuple[str, str]:
    """Export results to CSV and JSON files."""
    if not results:
        print("No results to export")
        return None, None
    
    # Export summary table
    summary_df = create_summary_table(results)
    csv_filename = f"{filename_prefix}_summary.csv"
    summary_df.to_csv(csv_filename, index=False)
    print(f"Exported summary table to: {csv_filename}")
    
    # Export detailed results as JSON
    json_data = {}
    for marker_key, marker_data in results.items():
        json_data[marker_key] = {
            'marker_index': int(marker_data['marker_index']),
            'marker_time': float(marker_data['marker_time']),
            'window_start': float(marker_data['window_start']),
            'window_end': float(marker_data['window_end']),
            'max_cost_time': float(marker_data['max_cost_time']),
            'max_metabolic_cost': float(marker_data['max_metabolic_cost']),
            'cleaned_data_points': int(marker_data['cleaned_data_points']),
            'cleaned_duration': float(marker_data['cleaned_duration']),
            'extrapolated_5min': float(marker_data['extrapolated_5min']),
            'average_cleaned': float(marker_data['average_cleaned']),
            'fit_params': marker_data['fit_params'],
            'time_cleaned': marker_data['time_cleaned'].tolist(),
            'metabolic_cost_cleaned': marker_data['metabolic_cost_cleaned'].tolist()
        }
    
    json_filename = f"{filename_prefix}_detailed.json"
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Exported detailed results to: {json_filename}")
    
    return csv_filename, json_filename


def interactive_metabolic_analysis(csv_path: Union[str, Path], tau: float = 42, save_dir: Optional[Path] = None) -> Dict:
    """
    Complete interactive metabolic analysis workflow.
    
    Args:
        csv_path: Path to MIH metabolics CSV file
        tau: Time constant for exponential fitting
    
    Returns:
        Dictionary with complete analysis results
    """
    print("=== Interactive Metabolic Analysis with Real Exponential Fitting ===")
    
    # Parse CSV file
    print("1. Parsing CSV file...")
    df, metadata = parse_mih_csv(csv_path)
    print(f"   Loaded {len(df)} data points")
    print(f"   Subject weight: {metadata.get('Weight', 'Unknown')} kg")
    
    # Create plots with all markers
    print("\n2. Creating plots with all markers...")
    if save_dir is None:
        filename_prefix = f"metabolic_markers_{Path(csv_path).stem}"
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        filename_prefix = str(save_dir / f"metabolic_markers_{Path(csv_path).stem}")
    
    plot_files = create_plots(df, metadata, filename_prefix=filename_prefix)
    
    print(f"   Plots saved:")
    for plot_type, filename in plot_files.items():
        print(f"   - {plot_type.upper()}: {filename}")
    
    # Interactive marker selection
    print("\n3. Interactive marker selection...")
    marker_data = df[df['Marker'] == 1]
    all_marker_times = marker_data['time_seconds'].values
    
    print(f"Found {len(all_marker_times)} total markers:")
    for i, marker_time in enumerate(all_marker_times):
        print(f"  Marker {i}: {marker_time:.1f}s")
    
    print(f"\nPlease select which markers to analyze:")
    print("Enter marker indices separated by commas (e.g., 0,2,4) or 'all' for all markers:")
    
    while True:
        try:
            user_input = input("Marker indices: ").strip()
            
            if user_input.lower() == 'all':
                selected_markers = list(range(len(all_marker_times)))
                break
            else:
                selected_markers = [int(x.strip()) for x in user_input.split(',')]
                
                if all(0 <= idx < len(all_marker_times) for idx in selected_markers):
                    break
                else:
                    print(f"Invalid indices. Please use numbers between 0 and {len(all_marker_times)-1}")
                    
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas or 'all'")
        except KeyboardInterrupt:
            print("\nExiting...")
            return {}
    
    print(f"Selected markers: {selected_markers}")
    
    # Process selected markers
    print(f"\n4. Processing {len(selected_markers)} selected markers...")
    marker_results = analyze_marker_advanced(df, metadata, selected_markers, tau)
    
    if not marker_results:
        print("No valid marker results generated.")
        return {}
    
    # Create summary and export
    print("\n5. Creating summary and exporting results...")
    summary_df = create_summary_table(marker_results)
    print("\n=== Marker Analysis Summary ===")
    print(summary_df.to_string(index=False))
    
    # Create individual marker plots
    if save_dir is None:
        plotting_prefix = f"interactive_metabolic_analysis_{Path(csv_path).stem}"
    else:
        plotting_prefix = str(save_dir / f"interactive_metabolic_analysis_{Path(csv_path).stem}")
    
    plotting_dir = plot_individual_markers(marker_results, metadata, plotting_prefix)
    
    # Create marker intervals plot
    marker_data = df[df['Marker'] == 1]
    all_marker_times = marker_data['time_seconds'].values
    intervals_plot = plot_marker_intervals(selected_markers, all_marker_times, plotting_prefix)
    
    # Export results
    csv_file, json_file = export_results(marker_results, plotting_prefix)
    
    return {
        'marker_results': marker_results,
        'metadata': metadata,
        'selected_markers': selected_markers,
        'summary_table': summary_df,
        'plot_files': plot_files,
        'plotting_dir': plotting_dir,
        'export_files': {'csv': csv_file, 'json': json_file}
    }


if __name__ == "__main__":
    # Example usage
    core_data_path = Path("/Users/riccardoconci/Local_documents/!CURRENT_RESEARCH/Personalising-Crutches/data_V2/raw/")
    #file_path_MIH27 = core_data_path / 'MIH27' / '2025-10-07' / 'Metabolics'/'Metabolics_MH27.csv'
    #file_path_MIH15 = core_data_path / 'MIH15' / '2025-10-21' / 'Metabolics'/'TEST_NO_1159.csv'
    file_path_MIH01 = core_data_path / 'MIH01' / '2025-10-15' / 'metabolics' / 'TEST_NO_1108.csv'
    
    file_path_list = [file_path_MIH01]
    for file_path in file_path_list:
        # Create save directory based on the data file path
        save_dir = file_path.parent / 'metabolics_analysis'
        save_dir.mkdir(parents=True, exist_ok=True)
        results = interactive_metabolic_analysis(file_path, save_dir=save_dir)
        
        if results:
            print(f"\n=== Analysis Complete ===")
            print(f"Selected markers: {results['selected_markers']}")
            print(f"Results exported to: {results['export_files']['csv']} and {results['export_files']['json']}")
            print(f"Individual marker plots saved in: {results['plotting_dir']}")
        else:
            print("Analysis failed or was cancelled.")