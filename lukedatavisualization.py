"""
Pure visualization script for data analysis results.
Reads the output from data_analysis.py and creates comprehensive plots.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

def load_data_from_analysis(data_file_path):
    """
    Load data and step detection results from data_analysis.py output.
    """
    # Construct expected file paths based on data_analysis.py naming convention
    base_path = os.path.splitext(data_file_path)[0]
    step_file_path = base_path + '_steps.csv'
    
    # Load raw data
    df = pd.read_csv(data_file_path)
    
    # Convert timestamp to seconds
    if 'acc_x_time' in df.columns:
        df['timestamp'] = (df['acc_x_time'] - df['acc_x_time'].iloc[0]) / 1000.0
    elif 'timestamp' not in df.columns:
        raise ValueError("CSV must contain either 'timestamp' or 'acc_x_time' column")
    
    # Load detected steps
    if os.path.exists(step_file_path):
        steps_df = pd.read_csv(step_file_path)
        step_times = steps_df['step_time'].values
    else:
        raise FileNotFoundError(f"Step file not found: {step_file_path}. Run data_analysis.py first.")
    
    return df, step_times

def plot_cycle_times_analysis(step_times):
    """Plot cycle times with mean and standard deviation."""
    if len(step_times) < 2:
        print("Not enough steps to analyze cycle times")
        return
    
    cycle_durations = np.diff(step_times)
    mean_duration = np.mean(cycle_durations)
    std_duration = np.std(cycle_durations)
    
    plt.figure(figsize=(12, 8))
    
    # Plot cycle durations over time
    plt.subplot(2, 1, 1)
    cycle_numbers = np.arange(1, len(cycle_durations) + 1)
    plt.plot(cycle_numbers, cycle_durations, 'bo-', markersize=4, linewidth=1)
    plt.axhline(y=mean_duration, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_duration:.3f}s')
    plt.fill_between(cycle_numbers, mean_duration - std_duration, mean_duration + std_duration, 
                     alpha=0.3, color='red', label=f'±1σ: {std_duration:.3f}s')
    plt.title('Cycle Durations Over Time')
    plt.xlabel('Cycle Number')
    plt.ylabel('Duration (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram of cycle durations
    plt.subplot(2, 1, 2)
    plt.hist(cycle_durations, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=mean_duration, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_duration:.3f}s')
    plt.axvline(x=mean_duration + std_duration, color='orange', linestyle=':', linewidth=2, label=f'+1σ: {mean_duration + std_duration:.3f}s')
    plt.axvline(x=mean_duration - std_duration, color='orange', linestyle=':', linewidth=2, label=f'-1σ: {mean_duration - std_duration:.3f}s')
    plt.title('Distribution of Cycle Durations')
    plt.xlabel('Duration (s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n--- Cycle Duration Statistics ---")
    print(f"Number of cycles: {len(cycle_durations)}")
    print(f"Mean duration: {mean_duration:.3f} ± {std_duration:.3f} s")
    print(f"Min duration: {np.min(cycle_durations):.3f} s")
    print(f"Max duration: {np.max(cycle_durations):.3f} s")
    print(f"Coefficient of variation: {std_duration/mean_duration:.3f}")

def plot_y_acceleration_analysis(df, step_times):
    """Plot Y-axis acceleration analysis per cycle."""
    if len(step_times) < 2:
        print("Not enough steps to analyze Y acceleration")
        return
    
    # Calculate RMS per cycle
    cycle_rms = []
    cycle_starts = []
    
    for i in range(len(step_times) - 1):
        start, end = np.searchsorted(df['timestamp'], [step_times[i], step_times[i+1]])
        seg_y = df['acc_y_data'].iloc[start:end]
        if seg_y.size > 0:
            cycle_rms.append(np.sqrt(np.mean(seg_y**2)))
            cycle_starts.append(df['timestamp'].iloc[start])
    
    if not cycle_rms:
        print("No valid Y acceleration data found")
        return
    
    mean_rms = np.mean(cycle_rms)
    std_rms = np.std(cycle_rms)
    
    plt.figure(figsize=(12, 8))
    
    # Plot RMS per cycle
    plt.subplot(2, 1, 1)
    cycle_numbers = np.arange(1, len(cycle_rms) + 1)
    plt.plot(cycle_numbers, cycle_rms, 'go-', markersize=4, linewidth=1)
    plt.axhline(y=mean_rms, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rms:.3f} m/s²')
    plt.fill_between(cycle_numbers, mean_rms - std_rms, mean_rms + std_rms, 
                     alpha=0.3, color='red', label=f'±1σ: {std_rms:.3f} m/s²')
    plt.title('Y-axis Acceleration RMS per Cycle')
    plt.xlabel('Cycle Number')
    plt.ylabel('RMS (m/s²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram of RMS values
    plt.subplot(2, 1, 2)
    plt.hist(cycle_rms, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(x=mean_rms, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rms:.3f} m/s²')
    plt.axvline(x=mean_rms + std_rms, color='orange', linestyle=':', linewidth=2, label=f'+1σ: {mean_rms + std_rms:.3f} m/s²')
    plt.axvline(x=mean_rms - std_rms, color='orange', linestyle=':', linewidth=2, label=f'-1σ: {mean_rms - std_rms:.3f} m/s²')
    plt.title('Distribution of Y-axis Acceleration RMS')
    plt.xlabel('RMS (m/s²)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n--- Y-axis Acceleration RMS Statistics ---")
    print(f"Number of cycles: {len(cycle_rms)}")
    print(f"Mean RMS: {mean_rms:.3f} ± {std_rms:.3f} m/s²")
    print(f"Min RMS: {np.min(cycle_rms):.3f} m/s²")
    print(f"Max RMS: {np.max(cycle_rms):.3f} m/s²")
    print(f"Coefficient of variation: {std_rms/mean_rms:.3f}")

def plot_comprehensive_analysis(df, step_times):
    """Plot comprehensive analysis with all signals and detected steps."""
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Force signal with detected steps
    plt.subplot(4, 1, 1)
    plt.plot(df['timestamp'], df['force'], 'b-', linewidth=1, label='Force')
    plt.scatter(step_times, np.interp(step_times, df['timestamp'], df['force']), 
                color='red', s=30, marker='o', label='Detected Steps')
    plt.title('Force Signal with Detected Steps')
    plt.ylabel('Force')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Force gradient with detected steps
    plt.subplot(4, 1, 2)
    force_gradient = np.gradient(df['force'], df['timestamp'])
    plt.plot(df['timestamp'], force_gradient, 'g-', linewidth=1, label='Force Gradient')
    plt.scatter(step_times, np.interp(step_times, df['timestamp'], force_gradient), 
                color='red', s=30, marker='o', label='Detected Steps')
    plt.title('Force Gradient with Detected Steps')
    plt.ylabel('d(Force)/dt')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Y-axis acceleration
    plt.subplot(4, 1, 3)
    plt.plot(df['timestamp'], df['acc_y_data'], 'purple', linewidth=1, label='Y-axis Acceleration')
    plt.scatter(step_times, np.interp(step_times, df['timestamp'], df['acc_y_data']), 
                color='red', s=30, marker='o', label='Detected Steps')
    plt.title('Y-axis Acceleration with Detected Steps')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: All accelerometer axes
    plt.subplot(4, 1, 4)
    plt.plot(df['timestamp'], df['acc_x_data'], 'b-', linewidth=1, label='X-axis')
    plt.plot(df['timestamp'], df['acc_y_data'], 'g-', linewidth=1, label='Y-axis')
    plt.plot(df['timestamp'], df['acc_z_data'], 'orange', linewidth=1, label='Z-axis')
    plt.scatter(step_times, np.interp(step_times, df['timestamp'], df['acc_y_data']), 
                color='red', s=30, marker='o', label='Detected Steps')
    plt.title('All Accelerometer Axes with Detected Steps')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compute_loss_values(df, step_times):
    """Compute the two instability loss values."""
    # LUKE: Check if we have at least 2 steps to compute intervals between them
    if len(step_times) < 2:
        # LUKE: Return NaN for both losses if insufficient data
        return float('nan'), float('nan')
    
    # LUKE: Added loss computation for visualization - matches data_analysis.py implementation
    # LUKE: Calculate the time differences between consecutive steps (cycle durations)
    cycle_durations = np.diff(step_times)
    # LUKE: Compute the standard deviation of cycle durations as the first loss
    cycle_duration_loss = np.std(cycle_durations)
    
    # LUKE: Initialize list to store RMS values for each cycle
    cycle_rms = []
    # LUKE: Loop through each cycle (from one step to the next)
    for i in range(len(step_times) - 1):
        # LUKE: Find the array indices corresponding to step times using binary search
        start, end = np.searchsorted(df['timestamp'], [step_times[i], step_times[i+1]])
        # LUKE: Extract Y-axis acceleration data for this cycle
        seg_y = df['acc_y_data'].iloc[start:end]
        # LUKE: Only process if we have data in this cycle
        if seg_y.size > 0:
            # LUKE: Calculate RMS (Root Mean Square) of Y-axis acceleration for this cycle
            cycle_rms.append(np.sqrt(np.mean(seg_y**2)))
    
    # LUKE: Compute the mean RMS across all cycles as the second loss, or NaN if no cycles
    y_rms_loss = np.mean(cycle_rms) if cycle_rms else float('nan')
    
    # LUKE: Return both loss values for display in visualization
    return cycle_duration_loss, y_rms_loss

def create_metabolic_visualization(base_path: str, body_weight_kg: float = 77.0):
    from src.data_analysis import process_metabolic_data_complete
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    metabolic_data = process_metabolic_data_complete(base_path, body_weight_kg)
    if metabolic_data is None:
        print("LUKE: No metabolic data available for visualization")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Metabolic Analysis: {os.path.basename(base_path)}', fontsize=14, fontweight='bold')

    # Top plot: Brockway metabolic cost as a function of time (raw calculation)
    time = metabolic_data['time']
    y_meas = metabolic_data['y_meas']
    ax1.plot(time/60, y_meas, 'k.-', markersize=3, alpha=0.8)
    ax1.set_ylabel('Metabolic Cost (W/kg)')
    ax1.set_title('Brockway Metabolic Cost (Raw)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Metabolic cost and convergence
    time_bar = metabolic_data['time_bar']
    y_bar = metabolic_data['y_bar']
    y_average = metabolic_data['y_average']
    y_estimate = metabolic_data['y_estimate']

    ax2.plot(time/60, y_meas, 'ko', markersize=3, alpha=0.7, label='Measured Data')
    ax2.plot(time_bar/60, y_bar, 'r-', linewidth=2, label='Exponential Fit')
    ax2.axhline(y=y_average, color='g', linestyle='--', linewidth=1.5, label=f'Average: {y_average:.3f} W/kg')
    if y_estimate is not None:
        ax2.axhline(y=y_estimate, color='orange', linestyle=':', linewidth=1.5, label=f'Estimate: {y_estimate:.3f} W/kg')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Metabolic Cost (W/kg)')
    ax2.set_title('Metabolic Cost Convergence Analysis', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize data analysis results from data_analysis.py")
    parser.add_argument("data_file", type=str, help="Path to the raw data CSV file")
    args = parser.parse_args()
    
    try:
        # Load data and step detection results
        print(f"Loading data from: {args.data_file}")
        df, step_times = load_data_from_analysis(args.data_file)
        
        print(f"Loaded {len(df)} data points and {len(step_times)} detected steps")
        
        # LUKE: Compute the two instability loss values for display
        cycle_duration_loss, y_rms_loss = compute_loss_values(df, step_times)
        
        # LUKE: Compute the third loss: metabolic cost using Brockway equation (gold standard)
        # LUKE: Extract base path from the data file path to find associated metabolic data
        base_path = os.path.splitext(args.data_file)[0]
        # Use the new Excel-based pipeline
        from src.data_analysis import process_metabolic_data_complete
        metabolic_data = process_metabolic_data_complete(base_path)
        if metabolic_data is not None:
            metabolic_cost_loss = metabolic_data['y_average']
        else:
            metabolic_cost_loss = float('nan')
        
        # LUKE: Print the computed instability metrics to the terminal for user feedback
        print(f"\n--- Instability Metrics ---")
        # LUKE: Display cycle duration loss with 6 decimal places and units
        print(f"Cycle duration loss (std): {cycle_duration_loss:.6f} s")
        # LUKE: Display Y-axis RMS loss with 6 decimal places and units
        print(f"Y-axis RMS loss (mean per cycle): {y_rms_loss:.6f} m/s²")
        # LUKE: Display metabolic cost loss with 6 decimal places and units
        print(f"Metabolic cost loss (Brockway): {metabolic_cost_loss:.6f} W/kg")
        
        # Create comprehensive visualizations
        print("\n--- Creating Visualizations ---")
        
        # 1. Comprehensive signal analysis
        plot_comprehensive_analysis(df, step_times)
        
        # 2. Cycle times analysis with statistics
        plot_cycle_times_analysis(step_times)
        
        # 3. Y-axis acceleration analysis
        plot_y_acceleration_analysis(df, step_times)
        
        # 4. Metabolic visualization
        metabolic_fig = create_metabolic_visualization(base_path)
        if metabolic_fig is not None:
            metabolic_fig.savefig(f"{base_path}_metabolic_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Saved metabolic analysis to {base_path}_metabolic_analysis.png")
            plt.show()  # Display the metabolic visualization on screen
        
        print("\n--- Visualization Complete ---")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Make sure to run data_analysis.py first to generate the step detection results.")

if __name__ == "__main__":
    main()