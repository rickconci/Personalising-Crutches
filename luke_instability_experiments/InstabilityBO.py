#!/usr/bin/env python3
"""
Single ObjectiveBayesian Optimization for Crutch Instability Optimization
Includes data collection, processing, BO, visualization

1. Ask for participant name and demographics
2. Ask for number of trials to run
3. For each trial:
   - Collect IMU data using BLE
   - Process data to get instability loss from cycle duration std
   - Use BO to suggest next geometry
4. Generate visualization of optimization process
5. Save results to CSV file
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path to import processing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import GPy and GPyOpt
try:
    import GPy
    from GPyOpt.methods import BayesianOptimization
except ImportError:
    print("GPy and GPyOpt are required. Install with:\n"
          "    python3 -m pip install GPy GPyOpt\n", file=sys.stderr)
    sys.exit(1)

# Import our data processing modules
from luke_ble_data_processing import load_ble_data, detect_steps_force_gradient, postprocess_steps, compute_cycle_duration_variance

# -------------- HELPER FUNCTIONS ------------------------------------------ #

def ask_float(prompt):
    """Force user to enter a valid number."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a number")

def ask_int(prompt):
    """Force user to enter a valid integer."""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a whole number")

def round_float(value, bounds):
    """Round to nearest allowed value."""
    return min(bounds, key=lambda x: abs(x - value))

# -------------- BO SETUP --------------------------------------- #

# Define the parameter ranges (same as luke.py)
alpha_range = list(range(70, 125, 5))   
beta_range  = list(range(90, 145, 5))   
gamma_range = list(range(-15, 25, 3))    

# Define search space for GPyOpt (3D optimization)
SEARCH_SPACE = [
    {'name': 'alpha', 'type': 'discrete', 'domain': alpha_range},
    {'name': 'beta',  'type': 'discrete', 'domain': beta_range},
    {'name': 'gamma', 'type': 'discrete', 'domain': gamma_range}
]

class InstabilityBO:
    """Bayesian optimization for crutch parameters (alpha, beta, gamma) based on instability metrics."""
    
    def __init__(self):
        self._X = []  # List of [alpha, beta, gamma] values
        self._Y = []  # List of instability losses
        self._kernel = GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=1.0)
    
    def record_trial(self, alpha, beta, gamma, loss):
        """Record a trial with parameters and instability loss."""
        self._X.append([alpha, beta, gamma])
        self._Y.append([loss])
    
    def suggest_next(self) -> tuple[int, int, int]:
        """Suggest next alpha, beta, gamma to test."""
        if not self._X:
            # If no trials yet, suggest random parameters
            return (np.random.choice(alpha_range),
                    np.random.choice(beta_range),
                    np.random.choice(gamma_range))
        
        X = np.array(self._X)
        Y = np.array(self._Y)
        
        # Dummy objective always returns 0 (real data provided via X, Y)
        def objective(x):
            return np.array([[0]])
        
        bo = BayesianOptimization(
            f=objective,
            domain=SEARCH_SPACE,
            model_type='GP',
            acquisition_type='EI',
            exact_feval=True,
            initial_design_numdata=0,
            X=X,
            Y=Y,
            kernel=self._kernel
        )
        
        bo.run_optimization(max_iter=1)
        a, b, g = bo.x_opt
        
        # Round to nearest allowed values
        a = round_float(a, alpha_range)
        b = round_float(b, beta_range)
        g = round_float(g, gamma_range)
        return a, b, g

# -------------- DATA COLLECTION AND PROCESSING ------------------------- #

def collect_ble_data(participant_name: str, trial_num: int) -> str:
    """Collect BLE data for a trial."""
    filename = f"{participant_name}{trial_num}_data.csv"
    
    print(f"\nCollecting BLE data for trial {trial_num}...")
    print(f"Data will be saved to: {filename}")
    print("Please connect the BLE device and perform the trial.")
    print("Press SPACEBAR when ready to start data collection...")
    
    # Wait for spacebar
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            ch = sys.stdin.read(1)
            if ch == ' ':
                print("\nSpacebar pressed - starting data collection...")
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    # Run BLE data collection using Luke_ble.py with real-time output
    cmd = ["python", "Luke_ble.py", "--filename", filename]
    try:
        # Run without capturing output so we can see real-time data
        result = subprocess.run(cmd, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"Error collecting data: return code {result.returncode}")
            return None
        
        # Verify the file was actually created
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✓ Data collected and saved to {filename} ({file_size} bytes)")
            return filename
        else:
            print(f"✗ Error: File {filename} was not created")
            return None
            
    except Exception as e:
        print(f"Error running BLE collection: {e}")
        return None

def process_trial_data(data_file: str, participant_name: str, alpha: int, beta: int, gamma: int) -> dict:
    """Process trial data to extract instability metrics."""
    print(f"Processing data for parameters: α={alpha}°, β={beta}°, γ={gamma}°...")
    
    try:
        # Load and process the data
        df = load_ble_data(data_file)
        
        # Detect steps
        force_signal = df['force'].values
        time_signal = df['timestamp'].values
        fs = 1 / np.mean(np.diff(time_signal))  # Sampling frequency
        
        step_times = detect_steps_force_gradient(force_signal, time_signal, fs)
        final_steps = postprocess_steps(step_times)
        
        # Calculate instability metric - only cycle duration variance
        cycle_duration_loss = compute_cycle_duration_variance(final_steps)
        
        # Scale the loss to make differences more apparent for BO
        # Multiply by 1000 to convert from seconds to milliseconds, making differences more visible
        total_loss = cycle_duration_loss * 1000
        
        metrics = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'cycle_duration_loss': cycle_duration_loss,
            'total_loss': total_loss,
            'n_steps': len(final_steps)
        }
        
        print(f"  Cycle duration variance: {cycle_duration_loss:.4f} s")
        print(f"  Total instability loss: {total_loss:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def save_metrics_to_file(metrics: dict, participant_name: str, trial_num: int):
    """Save metrics to file for later analysis."""
    filename = f"{participant_name}{trial_num}_data_metrics.csv"
    
    metrics_df = pd.DataFrame({
        'metric': ['participant', 'alpha', 'beta', 'gamma', 'cycle_duration_loss', 'total_loss', 'n_steps'],
        'value': [participant_name, metrics['alpha'], metrics['beta'], metrics['gamma'],
                 metrics['cycle_duration_loss'], metrics['total_loss'], metrics['n_steps']],
        'unit': ['', '°', '°', '°', 's', '', '']
    })
    
    metrics_df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")

# -------------- VISUALIZATION -------------------------------------------- #

def create_optimization_plots(df: pd.DataFrame, participant_name: str):
    """Create visualization plots for the optimization process."""
    
    # Color map normalization (red = high loss, green = low)
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mpl.colors.Normalize(vmin=df["total_loss"].min(), vmax=df["total_loss"].max())
    
    # Create 3 subplots for each parameter vs loss
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Alpha vs loss
    scatter1 = axes[0].scatter(df["alpha"], df["total_loss"], c=df["total_loss"], 
                              cmap=cmap, norm=norm, s=100, edgecolors="k")
    for idx, (alpha, loss) in df[["alpha", "total_loss"]].iterrows():
        axes[0].text(alpha, loss, str(idx + 1), fontsize=12, ha="center", va="center", 
                    color="white", fontweight='bold')
    axes[0].set_xlabel("α (degrees)")
    axes[0].set_ylabel("Total Instability Loss")
    axes[0].set_title("α vs Loss")
    axes[0].grid(True, alpha=0.3)
    
    # Beta vs loss
    scatter2 = axes[1].scatter(df["beta"], df["total_loss"], c=df["total_loss"], 
                              cmap=cmap, norm=norm, s=100, edgecolors="k")
    for idx, (beta, loss) in df[["beta", "total_loss"]].iterrows():
        axes[1].text(beta, loss, str(idx + 1), fontsize=12, ha="center", va="center", 
                    color="white", fontweight='bold')
    axes[1].set_xlabel("β (degrees)")
    axes[1].set_ylabel("Total Instability Loss")
    axes[1].set_title("β vs Loss")
    axes[1].grid(True, alpha=0.3)
    
    # Gamma vs loss
    scatter3 = axes[2].scatter(df["gamma"], df["total_loss"], c=df["total_loss"], 
                              cmap=cmap, norm=norm, s=100, edgecolors="k")
    for idx, (gamma, loss) in df[["gamma", "total_loss"]].iterrows():
        axes[2].text(gamma, loss, str(idx + 1), fontsize=12, ha="center", va="center", 
                    color="white", fontweight='bold')
    axes[2].set_xlabel("γ (degrees)")
    axes[2].set_ylabel("Total Instability Loss")
    axes[2].set_title("γ vs Loss")
    axes[2].grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter1, ax=axes, label="Instability Loss (lower = more stable)")
    
    plt.suptitle(f"{participant_name}: Instability Optimization Progress", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{participant_name}_instability_optimization.png", dpi=300, bbox_inches='tight')
    print(f"Optimization plot saved to {participant_name}_instability_optimization.png")
    
    # 3D scatter plot
    if len(df) >= 3:
        create_3d_optimization_plot(df, participant_name)
    
    plt.show()

def create_3d_optimization_plot(df: pd.DataFrame, participant_name: str):
    """Create 3D scatter plot showing optimization progress in parameter space."""
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for loss values
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mpl.colors.Normalize(vmin=df["total_loss"].min(), vmax=df["total_loss"].max())
    
    # 3D scatter plot
    scatter = ax.scatter(df["alpha"], df["beta"], df["gamma"], 
                        c=df["total_loss"], cmap=cmap, norm=norm, s=100, edgecolors="k")
    
    # Label each point with trial number
    for idx, (alpha, beta, gamma, loss) in df[["alpha", "beta", "gamma", "total_loss"]].iterrows():
        ax.text(alpha, beta, gamma, str(idx + 1), fontsize=12, ha="center", va="center", 
               color="white", fontweight='bold')
    
    # Draw lines between successive trials
    for i in range(len(df) - 1):
        ax.plot([df["alpha"][i], df["alpha"][i + 1]],
                [df["beta"][i], df["beta"][i + 1]],
                [df["gamma"][i], df["gamma"][i + 1]],
                color="gray", alpha=0.6, linewidth=2)
    
    ax.set_xlabel("α (degrees)")
    ax.set_ylabel("β (degrees)")
    ax.set_zlabel("γ (degrees)")
    ax.set_title(f"{participant_name}: 3D Parameter Space Optimization")
    
    # Add colorbar
    plt.colorbar(scatter, label="Instability Loss (lower = more stable)")
    
    plt.tight_layout()
    plt.savefig(f"{participant_name}_3d_optimization.png", dpi=300, bbox_inches='tight')
    print(f"3D optimization plot saved to {participant_name}_3d_optimization.png")

# -------------- MAIN EXECUTION ------------------------------------------- #

def main():
    """Main execution function."""
    print("\n=== Instability Bayesian Optimization ===")
    print("This script will automatically collect data and optimize crutch parameters (α, β, γ) for stability.")
    
    # Get participant information
    participant_name = input("\nParticipant name: ").strip()
    print("\nEnter participant demographics:")
    height = ask_float("  Height (cm):  ")
    weight = ask_float("  Weight (kg):  ")
    age = ask_float("  Age (years):  ")
    
    # Get number of trials
    while True:
        n_trials = ask_int("\nHow many trials do you want to run?: ")
        if n_trials > 0:
            break
        print("  → please enter a positive whole number.")
    
    # Create DataFrame to store results
    cols = ["trial", "alpha", "beta", "gamma", "height", "weight", "age", "cycle_duration_loss", 
            "total_loss", "n_steps"]
    df = pd.DataFrame(columns=cols)
    
    # Initialize BO
    bo = InstabilityBO()
    
    print(f"\nStarting optimization with {n_trials} trials...")
    print(f"Parameter ranges: α={alpha_range[0]}-{alpha_range[-1]}°, β={beta_range[0]}-{beta_range[-1]}°, γ={gamma_range[0]}-{gamma_range[-1]}°")
    
    for trial in range(1, n_trials + 1):
        print(f"\n=== Trial {trial} / {n_trials} ===")
        
        # Get parameters (manual input for first trial, BO suggestion for subsequent)
        if trial == 1:
            print("Enter the first geometry to test:")
            
            # Validate alpha input
            while True:
                alpha = int(ask_float(f"  Alpha ({alpha_range[0]}-{alpha_range[-1]}):  "))
                if alpha in alpha_range:
                    break
                print(f"  → Alpha must be one of: {alpha_range}")
            
            # Validate beta input
            while True:
                beta = int(ask_float(f"  Beta  ({beta_range[0]}-{beta_range[-1]}):  "))
                if beta in beta_range:
                    break
                print(f"  → Beta must be one of: {beta_range}")
            
            # Validate gamma input (including negative values)
            while True:
                gamma = int(ask_float(f"  Gamma ({gamma_range[0]}-{gamma_range[-1]}):  "))
                if gamma in gamma_range:
                    break
                print(f"  → Gamma must be one of: {gamma_range}")
            
            print(f"First trial parameters: α={alpha}°, β={beta}°, γ={gamma}°")
        else:
            alpha, beta, gamma = bo.suggest_next()
            print(f"BO suggests parameters: α={alpha}°, β={beta}°, γ={gamma}°")
        
        # Check if geometry is realistic
        print(f"\nCan you physically build and test this geometry?")
        while True:
            response = input("Enter 'y' for yes, 'n' for no: ").strip().lower()
            if response in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")
        
        if response == 'n':
            # Assign high penalty loss for unrealistic geometry
            penalty_loss = 500  # 500ms (0.5s * 1000) is very high for cycle duration variance
            print(f"✗ Geometry rejected - assigning penalty loss: {penalty_loss} s")
            
            # Record penalty trial in BO
            bo.record_trial(alpha, beta, gamma, penalty_loss)
            
            # Add penalty trial to DataFrame
            row = {
                "trial": trial,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "height": height,
                "weight": weight,
                "age": age,
                "cycle_duration_loss": penalty_loss,
                "total_loss": penalty_loss,
                "n_steps": 0
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            
            print(f"Trial {trial} completed with penalty loss")
            continue
        
        # Geometry is realistic, proceed with data collection
        print("✓ Geometry accepted - proceeding with data collection")
        
        # Collect data
        data_file = collect_ble_data(participant_name, trial)
        if data_file is None:
            print("Failed to collect data. Skipping trial.")
            continue
        
        # Process data
        metrics = process_trial_data(data_file, participant_name, alpha, beta, gamma)
        if metrics is None:
            print("Failed to process data. Skipping trial.")
            continue
        
        # Save metrics
        save_metrics_to_file(metrics, participant_name, trial)
        
        # Record trial in BO
        bo.record_trial(alpha, beta, gamma, metrics['total_loss'])
        
        # Add to DataFrame
        row = {
            "trial": trial,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "height": height,
            "weight": weight,
            "age": age,
            "cycle_duration_loss": metrics['cycle_duration_loss'],
            "total_loss": metrics['total_loss'],
            "n_steps": metrics['n_steps']
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        # Show progress
        print(f"\nTrial {trial} completed:")
        print(f"  Parameters: α={alpha}°, β={beta}°, γ={gamma}°")
        print(f"  Total loss: {metrics['total_loss']:.4f}")
    
    # End of experiment
    print("\n=== End of Experiment – Summary ===")
    print(df[["trial", "alpha", "beta", "gamma", "total_loss", "n_steps"]].to_string(index=False))
    
    # Save results
    out_path = Path(f"{participant_name}_instability_trials.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} trials to {out_path.resolve()}")
    
    # Create visualizations
    print("\nGenerating optimization visualizations...")
    create_optimization_plots(df, participant_name)
    
    # Find best parameters
    best_idx = df["total_loss"].idxmin()
    best_alpha = df.loc[best_idx, "alpha"]
    best_beta = df.loc[best_idx, "beta"]
    best_gamma = df.loc[best_idx, "gamma"]
    best_loss = df.loc[best_idx, "total_loss"]
    print(f"\nBest parameters found: α={best_alpha}°, β={best_beta}°, γ={best_gamma}° (loss: {best_loss:.4f})")
    
    # List all files created
    print(f"\n=== Files Created ===")
    print(f"Raw IMU data files:")
    for i in range(1, n_trials + 1):
        raw_file = f"{participant_name}{i}_data.csv"
        if os.path.exists(raw_file):
            file_size = os.path.getsize(raw_file)
            print(f"  ✓ {raw_file} ({file_size} bytes)")
        else:
            print(f"  ✗ {raw_file} (missing)")
    
    print(f"\nAnalysis files:")
    analysis_files = [
        f"{participant_name}_instability_trials.csv",
        f"{participant_name}_instability_optimization.png",
        f"{participant_name}_3d_optimization.png"
    ]
    for file in analysis_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file)
            print(f"  ✓ {file} ({file_size} bytes)")
        else:
            print(f"  ✗ {file} (missing)")
    
    print(f"\nOptimization complete!")

if __name__ == "__main__":
    main() 