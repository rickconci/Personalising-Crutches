#!/usr/bin/env python3
"""
luke_ble_data_processing.py - Process 3-axis BLE data for instability analysis

This script processes data from Luke_ble.py (force, accX, accY) to calculate:
1. Cycle duration variance (instability metric 1)
2. Y-axis acceleration RMS (instability metric 2)

Adapted from the full data_analysis.py to work with simplified 3-axis data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.signal import find_peaks, lfilter
from scipy.optimize import curve_fit
import re

def load_ble_data(file_path):
    """
    Load and preprocess BLE data from Luke_ble.py output.
    
    Args:
        file_path: Path to the CSV file from Luke_ble.py
        
    Returns:
        DataFrame with processed data
    """
    print(f"Loading BLE data from: {file_path}")
    
    # Load the CSV data
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    required_columns = ['relative_time_ms', 'force', 'accX', 'accY']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}")
    
    # Convert relative_time_ms to seconds
    df['timestamp'] = df['relative_time_ms'] / 1000.0
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'accX': 'acc_x_data',
        'accY': 'acc_y_data'
    })
    
    # Add missing columns that the original analysis expects
    # Set acc_z_data to zeros since we don't have it
    df['acc_z_data'] = 0.0
    
    # Calculate smoothed accelerometer magnitude (using only X and Y)
    alpha = 25.0 / 100.0  # 25% smoothing
    acc_x_smooth = lfilter([alpha], [1, -(1 - alpha)], df['acc_x_data'])
    acc_y_smooth = lfilter([alpha], [1, -(1 - alpha)], df['acc_y_data'])
    df['acc_x_z_smooth'] = acc_x_smooth**2 + acc_y_smooth**2
    
    print(f"Loaded {len(df)} data points")
    print(f"Duration: {df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]:.1f} seconds")
    print(f"Sampling rate: {len(df) / (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]):.1f} Hz")
    print(f"Expected duration at 200Hz: {len(df) / 200.0:.1f} seconds")
    print(f"Data gaps detected: {'Yes' if len(df) / 200.0 > (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) else 'No'}")
    
    return df

def detect_steps_force_gradient(force_signal, time_signal, fs):
    """
    Detect steps using force gradient method (adapted for 3-axis data).
    
    Args:
        force_signal: Force data array
        time_signal: Time data array
        fs: Sampling frequency
        
    Returns:
        Array of detected step times
    """
    if force_signal.size < int(fs):  # Need at least 1s of data
        return np.array([])
    
    # Calculate force gradient
    d_force_dt = np.gradient(force_signal, time_signal)
    
    # Set threshold based on the most negative peak
    threshold = d_force_dt.min() * 0.2
    
    # Filter the signal, keeping only values below the threshold
    d_force_dt_filtered = np.where(d_force_dt < threshold, d_force_dt, 0)
    
    # Find peaks in the negated filtered signal to identify troughs
    min_dist_samples = int(0.3 * fs)  # Minimum 0.3s between steps
    peaks, _ = find_peaks(-d_force_dt_filtered, distance=min_dist_samples)
    
    return time_signal[peaks]

def postprocess_steps(step_times, tolerance_ratio=0.2, isolation_threshold=5.0):
    """
    Apply post-processing filters to detected step times.
    
    Args:
        step_times: Array of detected step times
        tolerance_ratio: Tolerance for step interval regularization
        isolation_threshold: Threshold for removing isolated steps
        
    Returns:
        Array of post-processed step times
    """
    if len(step_times) < 2:
        return step_times
    
    # Determine expected interval from median
    intervals = np.diff(step_times)
    expected_interval = np.median(intervals) if len(intervals) >= 3 else 1.1
    
    # Interval-based regularization
    min_conflict_interval = expected_interval * tolerance_ratio
    regularized_steps = [step_times[0]]
    
    for i in range(1, len(step_times)):
        current_step = step_times[i]
        last_accepted_step = regularized_steps[-1]
        
        if current_step - last_accepted_step < min_conflict_interval:
            prev_accepted_step = regularized_steps[-2] if len(regularized_steps) > 1 else 0.0
            error_if_keep_last = abs((last_accepted_step - prev_accepted_step) - expected_interval)
            error_if_use_current = abs((current_step - prev_accepted_step) - expected_interval)
            if error_if_use_current < error_if_keep_last:
                regularized_steps[-1] = current_step  # Swap
        else:
            regularized_steps.append(current_step)
    
    regularized_steps = np.array(regularized_steps)
    
    # Isolation filter
    if len(regularized_steps) < 2:
        return np.array([])
    
    final_steps = []
    padded_steps = np.concatenate(([-np.inf], regularized_steps, [np.inf]))
    
    for i in range(1, len(padded_steps) - 1):
        dist_to_prev = padded_steps[i] - padded_steps[i-1]
        dist_to_next = padded_steps[i+1] - padded_steps[i]
        if not ((dist_to_prev > isolation_threshold) and (dist_to_next > isolation_threshold)):
            final_steps.append(padded_steps[i])
    
    return np.array(final_steps)

def compute_cycle_duration_variance(step_times):
    """
    Compute the variance of cycle durations (step intervals).
    
    Args:
        step_times: Array of detected step times
        
    Returns:
        Variance of cycle durations (mean squared deviation from mean)
    """
    if len(step_times) < 2:
        return float('nan')
    
    # Calculate time differences between consecutive steps
    durations = np.diff(step_times)
    return np.var(durations)  # Return variance directly

def compute_y_axis_rms_per_cycle(df, step_times):
    """
    Compute the mean RMS of Y-axis acceleration per cycle.
    
    Args:
        df: DataFrame with 'timestamp' and 'acc_y_data' columns
        step_times: Array of detected step times
        
    Returns:
        Mean RMS value across all cycles
    """
    if len(step_times) < 2:
        return float('nan')
    
    times = df['timestamp'].values
    y_accel = df['acc_y_data'].values
    
    # Find array indices corresponding to step times
    step_indices = np.searchsorted(times, step_times)
    
    cycle_rms = []
    
    # Loop through each cycle
    for i in range(len(step_indices) - 1):
        start, end = step_indices[i], step_indices[i+1]
        seg_y = y_accel[start:end]
        
        if seg_y.size > 0:
            # Calculate RMS of Y-axis acceleration for this cycle
            cycle_rms.append(np.sqrt(np.mean(seg_y**2)))
    
    return float(np.mean(cycle_rms)) if cycle_rms else float('nan')

def compute_x_axis_rms_per_cycle(df, step_times):
    """
    Compute the mean RMS of X-axis acceleration per cycle.
    
    Args:
        df: DataFrame with 'timestamp' and 'acc_x_data' columns
        step_times: Array of detected step times
        
    Returns:
        Mean RMS value across all cycles
    """
    if len(step_times) < 2:
        return float('nan')
    
    times = df['timestamp'].values
    x_accel = df['acc_x_data'].values
    
    # Find array indices corresponding to step times
    step_indices = np.searchsorted(times, step_times)
    
    cycle_rms = []
    
    # Loop through each cycle
    for i in range(len(step_indices) - 1):
        start, end = step_indices[i], step_indices[i+1]
        seg_x = x_accel[start:end]
        
        if seg_x.size > 0:
            # Calculate RMS of X-axis acceleration for this cycle
            cycle_rms.append(np.sqrt(np.mean(seg_x**2)))
    
    return float(np.mean(cycle_rms)) if cycle_rms else float('nan')

def create_visualization_with_info(df, step_times, output_path, participant, geometry):
    """
    Create comprehensive visualization of the analysis with participant and geometry info.
    
    Args:
        df: DataFrame with processed data
        step_times: Array of detected step times
        output_path: Path to save the visualization
        participant: Participant name
        geometry: Geometry being tested
    """
    # Extract trial number from output_path or participant+trial pattern
    filename = os.path.basename(output_path)
    trial_number = "Unknown"
    # Try to extract from patterns like Alex3, Luke2, etc.
    match = re.search(r'(\D+)(\d+)', filename)
    if match:
        trial_number = match.group(2)
    else:
        # Fallback: try to extract from _data or _analysis
        match2 = re.search(r'(\d+)', filename)
        if match2:
            trial_number = match2.group(1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{participant} - Trial {trial_number} - {geometry}\nBluetooth Instability Data Collection', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Force signal with detected steps
    ax1.plot(df['timestamp'], df['force'], 'b-', linewidth=1, label='Force')
    if len(step_times) > 0:
        step_forces = np.interp(step_times, df['timestamp'], df['force'])
        ax1.scatter(step_times, step_forces, color='red', s=30, marker='o', label='Detected Steps')
    ax1.set_title('Force Signal with Detected Steps')
    ax1.set_ylabel('Force')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Y-axis acceleration with detected steps
    ax2.plot(df['timestamp'], df['acc_y_data'], 'g-', linewidth=1, label='Y-axis Acceleration')
    if len(step_times) > 0:
        step_y_accel = np.interp(step_times, df['timestamp'], df['acc_y_data'])
        ax2.scatter(step_times, step_y_accel, color='red', s=30, marker='o', label='Detected Steps')
    ax2.set_title('Y-axis Acceleration with Detected Steps')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Y-axis acceleration RMS per cycle
    if len(step_times) >= 2:
        times = df['timestamp'].values
        y_accel = df['acc_y_data'].values
        step_indices = np.searchsorted(times, step_times)
        
        cycle_rms_values = []
        cycle_starts = []
        
        for i in range(len(step_indices) - 1):
            start, end = step_indices[i], step_indices[i+1]
            seg_y = y_accel[start:end]
            if seg_y.size > 0:
                cycle_rms_values.append(np.sqrt(np.mean(seg_y**2)))
                cycle_starts.append(times[start])
        
        if cycle_rms_values:
            ax3.plot(cycle_starts, cycle_rms_values, 'o-', color='purple', linewidth=2)
            ax3.axhline(np.mean(cycle_rms_values), color='red', linestyle='--', 
                       label=f'Mean RMS: {np.mean(cycle_rms_values):.3f} m/s²')
            ax3.set_title('Y-axis RMS per Cycle')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('RMS (m/s²)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No valid cycles found', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Y-axis RMS per Cycle')
    else:
        ax3.text(0.5, 0.5, 'Insufficient steps for cycle analysis', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Y-axis RMS per Cycle')
    
    # Plot 4: X-axis acceleration RMS per cycle
    if len(step_times) >= 2:
        times = df['timestamp'].values
        x_accel = df['acc_x_data'].values
        step_indices = np.searchsorted(times, step_times)
        
        cycle_rms_values = []
        cycle_starts = []
        
        for i in range(len(step_indices) - 1):
            start, end = step_indices[i], step_indices[i+1]
            seg_x = x_accel[start:end]
            if seg_x.size > 0:
                cycle_rms_values.append(np.sqrt(np.mean(seg_x**2)))
                cycle_starts.append(times[start])
        
        if cycle_rms_values:
            ax4.plot(cycle_starts, cycle_rms_values, 'o-', color='orange', linewidth=2)
            ax4.axhline(np.mean(cycle_rms_values), color='red', linestyle='--', 
                       label=f'Mean RMS: {np.mean(cycle_rms_values):.3f} m/s²')
            ax4.set_title('X-axis RMS per Cycle')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('RMS (m/s²)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No valid cycles found', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('X-axis RMS per Cycle')
    else:
        ax4.text(0.5, 0.5, 'Insufficient steps for cycle analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('X-axis RMS per Cycle')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

def create_visualization(df, step_times, output_path):
    """
    Create comprehensive visualization of the analysis.
    
    Args:
        df: DataFrame with processed data
        step_times: Array of detected step times
        output_path: Path to save the visualization
    """
    # Extract trial number from output_path
    filename = os.path.basename(output_path)
    trial_match = re.search(r'recorded_data_(\d+)', filename)
    trial_number = trial_match.group(1) if trial_match else "Unknown"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Trial {trial_number} Bluetooth Instability Data Collection', fontsize=16, fontweight='bold')
    
    # Plot 1: Force signal with detected steps
    ax1.plot(df['timestamp'], df['force'], 'b-', linewidth=1, label='Force')
    if len(step_times) > 0:
        step_forces = np.interp(step_times, df['timestamp'], df['force'])
        ax1.scatter(step_times, step_forces, color='red', s=30, marker='o', label='Detected Steps')
    ax1.set_title('Force Signal with Detected Steps')
    ax1.set_ylabel('Force')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Y-axis acceleration with detected steps
    ax2.plot(df['timestamp'], df['acc_y_data'], 'g-', linewidth=1, label='Y-axis Acceleration')
    if len(step_times) > 0:
        step_y_accel = np.interp(step_times, df['timestamp'], df['acc_y_data'])
        ax2.scatter(step_times, step_y_accel, color='red', s=30, marker='o', label='Detected Steps')
    ax2.set_title('Y-axis Acceleration with Detected Steps')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Y-axis acceleration RMS per cycle
    if len(step_times) >= 2:
        times = df['timestamp'].values
        y_accel = df['acc_y_data'].values
        step_indices = np.searchsorted(times, step_times)
        
        cycle_rms_values = []
        cycle_starts = []
        
        for i in range(len(step_indices) - 1):
            start, end = step_indices[i], step_indices[i+1]
            seg_y = y_accel[start:end]
            if seg_y.size > 0:
                cycle_rms_values.append(np.sqrt(np.mean(seg_y**2)))
                cycle_starts.append(times[start])
        
        if cycle_rms_values:
            ax3.plot(cycle_starts, cycle_rms_values, 'o-', color='purple', linewidth=2)
            ax3.axhline(np.mean(cycle_rms_values), color='red', linestyle='--', 
                       label=f'Mean RMS: {np.mean(cycle_rms_values):.3f} m/s²')
            ax3.set_title('Y-axis RMS per Cycle')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('RMS (m/s²)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No valid cycles found', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Y-axis RMS per Cycle')
    else:
        ax3.text(0.5, 0.5, 'Insufficient steps for cycle analysis', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Y-axis RMS per Cycle')
    
    # Plot 4: X-axis acceleration RMS per cycle
    if len(step_times) >= 2:
        times = df['timestamp'].values
        x_accel = df['acc_x_data'].values
        step_indices = np.searchsorted(times, step_times)
        
        cycle_rms_values = []
        cycle_starts = []
        
        for i in range(len(step_indices) - 1):
            start, end = step_indices[i], step_indices[i+1]
            seg_x = x_accel[start:end]
            if seg_x.size > 0:
                cycle_rms_values.append(np.sqrt(np.mean(seg_x**2)))
                cycle_starts.append(times[start])
        
        if cycle_rms_values:
            ax4.plot(cycle_starts, cycle_rms_values, 'o-', color='orange', linewidth=2)
            ax4.axhline(np.mean(cycle_rms_values), color='red', linestyle='--', 
                       label=f'Mean RMS: {np.mean(cycle_rms_values):.3f} m/s²')
            ax4.set_title('X-axis RMS per Cycle')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('RMS (m/s²)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No valid cycles found', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('X-axis RMS per Cycle')
    else:
        ax4.text(0.5, 0.5, 'Insufficient steps for cycle analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('X-axis RMS per Cycle')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

def main():
    """
    Main function for processing BLE data.
    """
    parser = argparse.ArgumentParser(description="Process BLE data for instability analysis")
    parser.add_argument("input_file", type=str, help="Path to the BLE data CSV file")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for results")
    parser.add_argument("--participant", type=str, help="Participant name")
    parser.add_argument("--geometry", type=str, help="Geometry being tested")
    args = parser.parse_args()
    
    # Valid geometry options
    valid_geometries = ["10.5", "18.2", "14"]
    
    # Prompt for participant name and geometry if not provided
    if args.participant is None:
        args.participant = input("Enter participant name: ").strip()
    
    if args.geometry is None:
        while True:
            print(f"Valid geometries: {', '.join(valid_geometries)}")
            args.geometry = input("Enter geometry being tested: ").strip()
            if args.geometry in valid_geometries:
                break
            else:
                print(f"Invalid geometry. Please choose from: {', '.join(valid_geometries)}")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return 1
    
    try:
        # Load and preprocess data
        df = load_ble_data(args.input_file)
        
        # Calculate sampling frequency
        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        fs = len(df) / duration
        
        print(f"\n=== Step Detection ===")
        
        # Detect steps using force gradient
        raw_steps = detect_steps_force_gradient(df['force'].values, df['timestamp'].values, fs)
        print(f"Raw steps detected: {len(raw_steps)}")
        
        # Apply post-processing
        final_steps = postprocess_steps(raw_steps)
        print(f"Final steps after post-processing: {len(final_steps)}")
        
        # Calculate instability metrics
        print(f"\n=== Instability Metrics ===")
        
        # 1. Cycle duration variance
        cycle_duration_loss = compute_cycle_duration_variance(final_steps)
        print(f"Cycle duration loss (std): {cycle_duration_loss:.6f} s")
        
        # 2. Y-axis acceleration RMS
        y_rms_loss = compute_y_axis_rms_per_cycle(df, final_steps)
        print(f"Y-axis RMS loss (mean per cycle): {y_rms_loss:.6f} m/s²")
        
        # 3. X-axis acceleration RMS
        x_rms_loss = compute_x_axis_rms_per_cycle(df, final_steps)
        print(f"X-axis RMS loss (mean per cycle): {x_rms_loss:.6f} m/s²")
        
        # Save results
        output_base = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save step times
        steps_file = os.path.join(output_dir, f"{output_base}_steps.csv")
        pd.DataFrame({'step_time': final_steps}).to_csv(steps_file, index=False)
        print(f"Step times saved to: {steps_file}")
        
        # Save metrics with participant and geometry info
        metrics_file = os.path.join(output_dir, f"{output_base}_metrics.csv")
        metrics_df = pd.DataFrame({
            'metric': ['participant', 'geometry', 'cycle_duration_loss', 'y_rms_loss', 'x_rms_loss'],
            'value': [args.participant, args.geometry, cycle_duration_loss, y_rms_loss, x_rms_loss],
            'unit': ['', '', 's', 'm/s²', 'm/s²']
        })
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Metrics saved to: {metrics_file}")
        
        # Create visualization with participant and geometry info
        viz_file = os.path.join(output_dir, f"{output_base}_analysis.png")
        create_visualization_with_info(df, final_steps, viz_file, args.participant, args.geometry)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Input file: {args.input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Participant: {args.participant}")
        print(f"Geometry: {args.geometry}")
        print(f"Steps detected: {len(final_steps)}")
        print(f"Trial duration: {duration:.1f} seconds")
        
        return 0
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 