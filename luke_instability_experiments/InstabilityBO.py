#!/usr/bin/env python3
"""
Single ObjectiveBayesian Optimization for Crutch Instability Optimization
Includes data collection, processing, BO, visualization.
Full BO optimization across alpha, beta, and gamma. 

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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# Import our data processing modules for instability loss analysis.
from luke_instability_data_processing import load_ble_data, detect_steps_force_gradient, postprocess_steps, compute_cycle_duration_variance

# Import config for kernel parameters for the BO model.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from config import kernel_params

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

# Define the parameter ranges for crutch geometry optimization
alpha_range = list(range(70, 125, 5))   # α: handle angle from vertical (70-120°)
beta_range  = list(range(90, 145, 5))   # β: angle between forearm and hand grip (90-140°)
gamma_range = list(range(-12, 13, 3))   # γ: distance between forearm and vertical strut (-12 to +12°)    

# Define search space for GPyOpt (3D optimization)
SEARCH_SPACE = [
    {'name': 'alpha', 'type': 'discrete', 'domain': alpha_range},
    {'name': 'beta',  'type': 'discrete', 'domain': beta_range},
    {'name': 'gamma', 'type': 'discrete', 'domain': gamma_range}
]

class InstabilityBO:
    """Bayesian optimization for crutch parameters (alpha, beta, gamma) based on instability metrics."""
    
    def __init__(self, acquisition_type='EI', exact_feval=True):
        self.acquisition_type = acquisition_type
        self.exact_feval = exact_feval
        self._X = []  # List of [alpha, beta, gamma] values
        self._Y = []  # List of instability losses
        self._tested_geometries = set()  # Track tested geometries to avoid duplicates
    
    def _get_kernel(self):
        """Creates and returns a GPy kernel (same logic as GP_BO)."""
        return GPy.kern.Matern52(
            input_dim=3,  # 3 crutch parameters (alpha, beta, gamma)
            variance=kernel_params['variance'],
            lengthscale=kernel_params['lengthscale']
        )
    
    def record_trial(self, alpha, beta, gamma, loss):
        """Record a trial with parameters and instability loss."""
        self._X.append([alpha, beta, gamma])
        self._Y.append([loss])
        self._tested_geometries.add((alpha, beta, gamma))
    
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
        
        # Create BO optimizer (same approach as GP_BO)
        bo = BayesianOptimization(
            f=objective,
            domain=SEARCH_SPACE,
            model_type='GP',
            kernel=self._get_kernel(),
            acquisition_type=self.acquisition_type,
            exact_feval=self.exact_feval,
            X=X,
            Y=Y
        )
        
        # Use suggest_next_locations() method (same as GP_BO)
        next_params_array = bo.suggest_next_locations()
        a, b, g = next_params_array[0]
        
        # Round to nearest allowed values
        a = round_float(a, alpha_range)
        b = round_float(b, beta_range)
        g = round_float(g, gamma_range)
        
        # Check if this geometry has already been tested
        if (a, b, g) in self._tested_geometries:
            print(f"⚠️  Geometry α={a}°, β={b}°, γ={g}° already tested. Finding alternative...")
            return self._suggest_alternative(a, b, g)
        
        return a, b, g
    
    def _suggest_alternative(self, alpha: int, beta: int, gamma: int) -> tuple[int, int, int]:
        """Suggest an alternative geometry close to the duplicate."""
        # Try small variations
        variations = [
            (alpha + 5, beta, gamma),
            (alpha - 5, beta, gamma),
            (alpha, beta + 5, gamma),
            (alpha, beta - 5, gamma),
            (alpha, beta, gamma + 3),
            (alpha, beta, gamma - 3),
            (alpha + 5, beta + 5, gamma),
            (alpha - 5, beta - 5, gamma),
        ]
        
        for a, b, g in variations:
            if (a in alpha_range and b in beta_range and g in gamma_range and 
                (a, b, g) not in self._tested_geometries):
                print(f"→ Alternative: α={a}°, β={b}°, γ={g}°")
                return a, b, g
        
        # If no close alternatives, suggest random untested geometry
        available_geometries = []
        for a in alpha_range:
            for b in beta_range:
                for g in gamma_range:
                    if (a, b, g) not in self._tested_geometries:
                        available_geometries.append((a, b, g))
        
        if available_geometries:
            choice = np.random.choice(len(available_geometries))
            a, b, g = available_geometries[choice]
            print(f"→ Random alternative: α={a}°, β={b}°, γ={g}°")
            return a, b, g
        
        # If all geometries tested, return the original (shouldn't happen in practice)
        print("⚠️  All geometries tested! Returning original suggestion.")
        return alpha, beta, gamma

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
        
        # Calculate instability metric - cycle duration variance
        cycle_duration_variance = compute_cycle_duration_variance(final_steps)
        cycle_duration_loss = cycle_duration_variance  
        
        # Scale the loss to make differences more apparent for BO
        # Multiply by 1000 to convert from seconds² to milliseconds², making differences more visible
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

def plot_step_detection_results(data_file: str, participant_name: str, trial_num: int, alpha: int, beta: int, gamma: int):
    """Plot step detection results with accelerometer and force data."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Load data
        df = load_ble_data(data_file)
        
        # Detect steps
        force_signal = df['force'].values
        time_signal = df['timestamp'].values
        fs = 1 / np.mean(np.diff(time_signal))
        
        step_times = detect_steps_force_gradient(force_signal, time_signal, fs)
        final_steps = postprocess_steps(step_times)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot force data with detected steps
        ax1.plot(df['timestamp'], df['force'], 'b-', alpha=0.7, label='Force')
        ax1.scatter(final_steps, [df['force'].max() * 0.9] * len(final_steps), 
                   color='red', s=100, marker='v', label=f'Detected Steps ({len(final_steps)})')
        ax1.set_ylabel('Force')
        ax1.set_title(f'{participant_name} Trial {trial_num}: Force Data & Step Detection\nα={alpha}°, β={beta}°, γ={gamma}°')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accelerometer data
        ax2.plot(df['timestamp'], df['acc_x_data'], 'g-', alpha=0.7, label='AccX')
        ax2.plot(df['timestamp'], df['acc_y_data'], 'r-', alpha=0.7, label='AccY')
        ax2.scatter(final_steps, [df['acc_x_data'].max() * 0.9] * len(final_steps), 
                   color='red', s=100, marker='v', label='Step Times')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Acceleration (m/s²)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{participant_name}{trial_num}_step_detection.png', dpi=300, bbox_inches='tight')
        print(f'Step detection plot saved to {participant_name}{trial_num}_step_detection.png')
        plt.show()
        
    except Exception as e:
        print(f"Error creating step detection plot: {e}")
        print("Continuing without plot...")

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
    
    # Check if we have enough data to plot
    if len(df) == 0:
        print("No data to plot - skipping visualization")
        return
    
    # Ensure n_steps column exists (for penalty trials)
    if 'n_steps' not in df.columns:
        df['n_steps'] = 0
    
    # Remove any rows with NaN values
    df_clean = df.dropna(subset=["alpha", "beta", "gamma", "total_loss"])
    
    if len(df_clean) == 0:
        print("No valid data points after cleaning - skipping visualization")
        return
    
    # Color map normalization (red = high loss, green = low)
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mpl.colors.Normalize(vmin=df_clean["total_loss"].min(), vmax=df_clean["total_loss"].max())
    
    # Create 3 subplots for each parameter vs loss
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Alpha vs loss
    scatter1 = axes[0].scatter(df_clean["alpha"], df_clean["total_loss"], c=df_clean["total_loss"], 
                              cmap=cmap, norm=norm, s=100, edgecolors="k")
    for idx, (alpha, loss) in df_clean[["alpha", "total_loss"]].iterrows():
        axes[0].text(alpha, loss, str(idx + 1), fontsize=12, ha="center", va="center", 
                    color="white", fontweight='bold')
    axes[0].set_xlabel("α (degrees)")
    axes[0].set_ylabel("Total Instability Loss")
    axes[0].set_title("α vs Loss")
    axes[0].grid(True, alpha=0.3)
    
    # Beta vs loss
    scatter2 = axes[1].scatter(df_clean["beta"], df_clean["total_loss"], c=df_clean["total_loss"], 
                              cmap=cmap, norm=norm, s=100, edgecolors="k")
    for idx, (beta, loss) in df_clean[["beta", "total_loss"]].iterrows():
        axes[1].text(beta, loss, str(idx + 1), fontsize=12, ha="center", va="center", 
                    color="white", fontweight='bold')
    axes[1].set_xlabel("β (degrees)")
    axes[1].set_ylabel("Total Instability Loss")
    axes[1].set_title("β vs Loss")
    axes[1].grid(True, alpha=0.3)
    
    # Gamma vs loss
    scatter3 = axes[2].scatter(df_clean["gamma"], df_clean["total_loss"], c=df_clean["total_loss"], 
                              cmap=cmap, norm=norm, s=100, edgecolors="k")
    for idx, (gamma, loss) in df_clean[["gamma", "total_loss"]].iterrows():
        axes[2].text(gamma, loss, str(idx + 1), fontsize=12, ha="center", va="center", 
                    color="white", fontweight='bold')
    axes[2].set_xlabel("γ (degrees)")
    axes[2].set_ylabel("Total Instability Loss")
    axes[2].set_title("γ vs Loss")
    axes[2].grid(True, alpha=0.3)
    
    # Add colorbar to the right of the last subplot
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(scatter1, cax=cax, label="Instability Loss (lower = more stable)")
    
    plt.suptitle(f"{participant_name}: Instability Optimization Progress", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{participant_name}_instability_optimization.png", dpi=300, bbox_inches='tight')
    print(f"Optimization plot saved to {participant_name}_instability_optimization.png")
    
    # 3D scatter plot
    if len(df_clean) >= 3:
        create_3d_optimization_plot(df_clean, participant_name)
    
    plt.show()

def create_3d_optimization_plot(df: pd.DataFrame, participant_name: str):
    """Create 3D scatter plot showing optimization progress in parameter space."""
    
    try:
        # Data should already be cleaned, but double-check
        if len(df) < 2:
            print("Not enough valid data points for 3D plot (need at least 2)")
            return
        
        # Ensure all required columns exist and are numeric
        required_cols = ["alpha", "beta", "gamma", "total_loss", "is_penalty"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Missing required column: {col}")
                return
        
        # Convert to numeric and handle any non-numeric values
        df_plot = df.copy()
        for col in ["alpha", "beta", "gamma", "total_loss"]:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        
        # Remove any rows with NaN values
        df_plot = df_plot.dropna(subset=["alpha", "beta", "gamma", "total_loss"])
        
        if len(df_plot) < 2:
            print("Not enough valid numeric data points for 3D plot")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map for loss values
        cmap = plt.get_cmap("RdYlGn_r")
        norm = mpl.colors.Normalize(vmin=df_plot["total_loss"].min(), vmax=df_plot["total_loss"].max())
        
        # 3D scatter plot
        scatter = ax.scatter(df_plot["alpha"], df_plot["beta"], df_plot["gamma"], 
                            c=df_plot["total_loss"], cmap=cmap, norm=norm, s=100, edgecolors="k")
        
        # Label each point with trial number and mark penalty trials
        for idx, (alpha, beta, gamma, loss, is_penalty) in df_plot[["alpha", "beta", "gamma", "total_loss", "is_penalty"]].iterrows():
            label = f"{idx + 1}P" if is_penalty else str(idx + 1)
            color = "red" if is_penalty else "white"
            ax.text(alpha, beta, gamma, label, fontsize=12, ha="center", va="center", 
                   color=color, fontweight='bold')
        
        # Draw lines between successive trials
        for i in range(len(df_plot) - 1):
            ax.plot([df_plot["alpha"].iloc[i], df_plot["alpha"].iloc[i + 1]],
                    [df_plot["beta"].iloc[i], df_plot["beta"].iloc[i + 1]],
                    [df_plot["gamma"].iloc[i], df_plot["gamma"].iloc[i + 1]],
                    color="gray", alpha=0.6, linewidth=2)
        
        ax.set_xlabel("α (degrees)")
        ax.set_ylabel("β (degrees)")
        ax.set_zlabel("γ (degrees)")
        ax.set_title(f"{participant_name}: 3D Parameter Space Optimization")
        
        # Add colorbar
        plt.colorbar(scatter, label="Instability Loss (lower = more stable)")
        
        # Add legend for penalty trials
        if any(df_plot["is_penalty"] == True):
            ax.text2D(0.02, 0.98, "P = Penalty trial (geometry rejected)", 
                     transform=ax.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"{participant_name}_3d_optimization.png", dpi=300, bbox_inches='tight')
        print(f"3D optimization plot saved to {participant_name}_3d_optimization.png")
        
    except Exception as e:
        print(f"Error creating 3D plot: {e}")
        print("Continuing without 3D plot...")

# -------------- MAIN EXECUTION ------------------------------------------- #

def main():
    """Main execution function."""
    print("\n=== Instability Bayesian Optimization ===")
    print("We will collect data and optimize crutch parameters (α, β, γ) for stability.")
    
    # Get participant information
    participant_name = input("\nParticipant name: ").strip()
    
    # Check if previous data exists for this participant
    previous_data_file = f"{participant_name}_instability_trials.csv"
    if os.path.exists(previous_data_file):
        print(f"\n📁 Found previous data for {participant_name}")
        while True:
            continue_choice = input("Continue from previous data? (y/n): ").strip().lower()
            if continue_choice in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")
        
        if continue_choice == 'y':
            # Load previous data
            df = pd.read_csv(previous_data_file)
            # Ensure is_penalty column exists and is clean
            if 'is_penalty' not in df.columns:
                df['is_penalty'] = False
            else:
                df['is_penalty'] = df['is_penalty'].fillna(False)
            print(f"Loaded {len(df)} previous trials")
            
            # Initialize BO with previous data
            bo = InstabilityBO(acquisition_type='EI', exact_feval=True)
            for _, row in df.iterrows():
                bo.record_trial(row['alpha'], row['beta'], row['gamma'], row['total_loss'])
            
            # Get demographics from previous data
            height = df['height'].iloc[0]
            weight = df['weight'].iloc[0]
            age = df['age'].iloc[0]
            print(f"Using previous demographics: Height={height}cm, Weight={weight}kg, Age={age} years")
            
            # Get starting trial number
            start_trial = len(df) + 1
        else:
            # Start fresh
            df = pd.DataFrame(columns=["trial", "alpha", "beta", "gamma", "height", "weight", "age", "cycle_duration_loss", "total_loss", "n_steps", "is_penalty"])
            bo = InstabilityBO(acquisition_type='EI', exact_feval=True)
            print("\nEnter participant demographics:")
            height = ask_float("  Height (cm):  ")
            weight = ask_float("  Weight (kg):  ")
            age = ask_float("  Age (years):  ")
            start_trial = 1
    else:
        # No previous data, start fresh
        df = pd.DataFrame(columns=["trial", "alpha", "beta", "gamma", "height", "weight", "age", "cycle_duration_loss", "total_loss", "n_steps", "is_penalty"])
        bo = InstabilityBO(acquisition_type='EI', exact_feval=True)
        print("\nEnter participant demographics:")
        height = ask_float("  Height (cm):  ")
        weight = ask_float("  Weight (kg):  ")
        age = ask_float("  Age (years):  ")
        start_trial = 1
    
    # Get number of trials
    while True:
        n_trials = ask_int("\nHow many trials do you want to run?: ")
        if n_trials > 0:
            break
        print("  → please enter a positive whole number.")
    
    # Calculate total trials (previous + new)
    total_trials = start_trial + n_trials - 1
    print(f"\nWill run trials {start_trial} to {total_trials} (total: {n_trials} new trials)")
    
    print(f"\nStarting optimization with {n_trials} trials...")
    print(f"Parameter ranges: α={alpha_range[0]}-{alpha_range[-1]}°, β={beta_range[0]}-{beta_range[-1]}°, γ={gamma_range[0]}-{gamma_range[-1]}°")
    
    for trial in range(start_trial, total_trials + 1):
        print(f"\n=== Trial {trial} / {total_trials} ===")
        
        # Get parameters (manual input for first trial of session, BO suggestion for subsequent)
        if trial == start_trial:
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
            response = input("Enter 'y' for yes, 'n' for no, 'a' for alternative: ").strip().lower()
            if response in ['y', 'n', 'a']:
                break
            print("Please enter 'y', 'n', or 'a'")
        
        if response == 'n':
            is_penalty = True  # Always mark as penalty when assigning manual loss
            # Show previous losses for reference
            if len(df) > 0:
                print("\nPrevious losses in this experiment:")
                print(df[["trial", "alpha", "beta", "gamma", "total_loss"]].to_string(index=False))
            else:
                print("No previous losses to display.")
            # Prompt user for manual loss entry
            while True:
                try:
                    manual_loss = float(input("Enter a loss value to assign for this rejected geometry: "))
                    break
                except ValueError:
                    print("Please enter a valid number.")
            print(f"✗ Geometry rejected - assigning manual loss: {manual_loss}")
            # Record penalty trial in BO
            bo.record_trial(alpha, beta, gamma, manual_loss)
            # Add penalty trial to DataFrame
            row = {
                "trial": trial,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "height": height,
                "weight": weight,
                "age": age,
                "cycle_duration_loss": manual_loss,
                "total_loss": manual_loss,
                "n_steps": 0,
                "is_penalty": is_penalty
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            print(f"Trial {trial} completed with manual loss")
            continue
        
        elif response == 'a':
            # Let user input alternative geometry manually
            print("Enter your preferred geometry:")
            
            # Validate alpha input
            while True:
                alt_alpha = int(ask_float(f"  Alpha ({alpha_range[0]}-{alpha_range[-1]}):  "))
                if alt_alpha in alpha_range:
                    break
                print(f"  → Alpha must be one of: {alpha_range}")
            
            # Validate beta input
            while True:
                alt_beta = int(ask_float(f"  Beta  ({beta_range[0]}-{beta_range[-1]}):  "))
                if alt_beta in beta_range:
                    break
                print(f"  → Beta must be one of: {beta_range}")
            
            # Validate gamma input (including negative values)
            while True:
                alt_gamma = int(ask_float(f"  Gamma ({gamma_range[0]}-{gamma_range[-1]}):  "))
                if alt_gamma in gamma_range:
                    break
                print(f"  → Gamma must be one of: {gamma_range}")
            
            # Check if this geometry has already been tested
            if (alt_alpha, alt_beta, alt_gamma) in bo._tested_geometries:
                print(f"⚠️  Geometry α={alt_alpha}°, β={alt_beta}°, γ={alt_gamma}° already tested.")
                while True:
                    proceed_choice = input("Proceed anyway? (y/n): ").strip().lower()
                    if proceed_choice in ['y', 'n']:
                        break
                    print("Please enter 'y' or 'n'")
                
                if proceed_choice == 'n':
                    # Assign penalty loss for original geometry
                    penalty_loss = 100.0
                    print(f"✗ Alternative rejected - assigning penalty loss: {penalty_loss}")
                    
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
                        "n_steps": 0,
                        "is_penalty": True
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    
                    print(f"Trial {trial} completed with penalty loss")
                    continue
            
            # Use the manually entered geometry
            alpha, beta, gamma = alt_alpha, alt_beta, alt_gamma
            print(f"✓ Using manually entered geometry: α={alpha}°, β={beta}°, γ={gamma}°")
        
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
        
        # Plot step detection results
        try:
            plot_step_detection_results(data_file, participant_name, trial, alpha, beta, gamma)
            print("✓ Step detection plot created")
        except Exception as e:
            print(f"✗ Error creating step detection plot: {e}")
            print("Continuing without plot...")
        
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
            "n_steps": metrics['n_steps'],
            "is_penalty": False
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
    try:
        create_optimization_plots(df, participant_name)
        print("✓ Optimization visualizations created successfully")
    except Exception as e:
        print(f"✗ Error creating optimization visualizations: {e}")
        print("Continuing without visualizations...")
    
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