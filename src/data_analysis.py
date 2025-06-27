import os
import sys
sys.path.append(os.path.dirname(__file__))  # Add src directory to path
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, correlate
from gait_analysis import calculate_gait_metrics_from_steps
import config
from ble_MCU_2 import record_trial_data
import asyncio
import plotly.graph_objs as go
import shutil
from scipy.signal import correlate, lfilter
import argparse
import json
from scipy.optimize import curve_fit
from data_processing_metabolic import _parse_time_value, _parse_number
def _postprocess_steps(preds: np.ndarray, tolerance_ratio: float = 0.2, isolation_threshold: float = 5.0) -> np.ndarray:
    """
    Applies post-processing filters to a series of detected step times.
    1. Regularizes step intervals based on a data-driven expected interval.
    2. Removes isolated steps that are too far from any other step.
    """
    if len(preds) < 2:
        return preds
    # 1. Determine the data-driven expected interval from the median of diffs
    intervals = np.diff(preds)
    expected_interval = np.median(intervals) if len(intervals) >= 3 else 1.1
    # --- Interval-based regularization ---
    min_conflict_interval = expected_interval * tolerance_ratio
    
    regularized_preds = [preds[0]]
    for i in range(1, len(preds)):
        current_pred = preds[i]
        last_accepted_pred = regularized_preds[-1]
        
        if current_pred - last_accepted_pred < min_conflict_interval:
            prev_accepted_pred = regularized_preds[-2] if len(regularized_preds) > 1 else 0.0
            error_if_keep_last = abs((last_accepted_pred - prev_accepted_pred) - expected_interval)
            error_if_use_current = abs((current_pred - prev_accepted_pred) - expected_interval)
            if error_if_use_current < error_if_keep_last:
                regularized_preds[-1] = current_pred  # Swap
        else:
            regularized_preds.append(current_pred)
    
    regularized_preds = np.array(regularized_preds)
    # --- Isolation filter ---
    if len(regularized_preds) < 2:
        return np.array([])
        
    final_preds = []
    padded_preds = np.concatenate(([np.NINF], regularized_preds, [np.PINF]))
    
    for i in range(1, len(padded_preds) - 1):
        dist_to_prev = padded_preds[i] - padded_preds[i-1]
        dist_to_next = padded_preds[i+1] - padded_preds[i]
        if not ((dist_to_prev > isolation_threshold) and (dist_to_next > isolation_threshold)):
            final_preds.append(padded_preds[i])
            
    return np.array(final_preds)
def detect_steps_unsupervised(force_signal: np.ndarray, time_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Detects steps using a dynamic threshold of the force gradient. The threshold is
    set relative to the most negative point in the gradient signal.
    """
    if force_signal.size < int(fs): # Need at least 1s of data
        return np.array([])
    # 1. Calculate force gradient
    d_force_dt = np.gradient(force_signal, time_signal)
   
    
    # 3. Set a threshold based on the most negative peak
    threshold = d_force_dt.min() * 0.2
    # 4. Filter the signal, keeping only values *below* the threshold.
    d_force_dt_filtered = np.where(d_force_dt < threshold, d_force_dt, 0)
    # 5. Find peaks in the *negated* filtered signal to identify troughs.
    #    Use a refractory period to avoid multiple detections on the same step.
    min_dist_samples = int(0.3 * fs)
    peaks, _ = find_peaks(-d_force_dt_filtered, distance=min_dist_samples)
    
    return time_signal[peaks]
# LUKE: Added first instability loss function for cycle duration standard deviation.
# note that we have the data frame as an input but don't use it in this function. 
def compute_cycle_std(df: pd.DataFrame, step_times: np.ndarray) -> float:
    """
    Compute the standard deviation of cycle durations (step intervals).
    This is the first loss function for instability metrics.
    """
    # LUKE: Check if we have at least 2 steps to compute intervals between them
    if len(step_times) < 2:
        return float('nan')  # LUKE: Return NaN if insufficient data
    
    # LUKE: Calculate the time differences between consecutive steps (cycle durations)
    durations = np.diff(step_times)
    # LUKE: Return the standard deviation of cycle durations as the instability metric
    return np.std(durations)
# LUKE: Added second instability loss function for Y-axis acceleration RMS
def compute_mean_y_rms_per_cycle(df: pd.DataFrame, step_times: np.ndarray, return_all=False):
    """
    Compute the mean RMS of Y-axis acceleration per cycle.
    This is the second loss function for instability metrics.
    
    Args:
        df: DataFrame with 'timestamp' and 'acc_y_data' columns
        step_times: Array of detected step times in seconds
        return_all: If True, return all per-cycle RMS values and cycle start times
    
    Returns:
        If return_all=False: mean RMS value (float)
        If return_all=True: (cycle_rms_list, cycle_starts_list)
    """
    # LUKE: Check if we have at least 2 steps to define cycles between them
    if len(step_times) < 2:
        # LUKE: Return appropriate empty result based on return_all flag
        return float('nan') if not return_all else ([], [])
    
    # LUKE: Extract time and Y-axis acceleration data from the dataframe
    times = df['timestamp'].values
    y = df['acc_y_data'].values
    
    # LUKE: Find the array indices corresponding to each step time using binary search
    # This is vital to find the indexes of the step times in the times array to convert to per cycle.
    step_indices = np.searchsorted(times, step_times)
    
    # LUKE: Initialize lists to store RMS values and cycle start times
    cycle_rms = []
    cycle_starts = []
    
    # LUKE: Loop through each cycle (from one step to the next)
    for i in range(len(step_indices) - 1):
        # LUKE: Define start and end indices for this cycle
        start, end = step_indices[i], step_indices[i+1]
        # LUKE: Extract Y-axis acceleration data for this cycle
        seg_y = y[start:end]
        
        # LUKE: Only process if we have data in this cycle
        if seg_y.size > 0:
            # LUKE: Calculate RMS of Y-axis acceleration for this cycle
            cycle_rms.append(np.sqrt(np.mean(seg_y**2)))
            # LUKE: Store the start time of this cycle
            cycle_starts.append(times[start])
    
    if return_all:
        return cycle_rms, cycle_starts
    # LUKE: Otherwise, return the mean RMS across all cycles (the instability metric)
    return float(np.mean(cycle_rms)) if cycle_rms else float('nan')

# LUKE: Added proper metabolic rate estimation function for 2-minute protocols
def metabolic_rate_estimation(time: np.ndarray, y_meas: np.ndarray, tau: float = 42.0):
    """
    Estimate steady-state metabolic cost using exponential rise model.
    This is the gold standard approach for short-duration exercise protocols.
    
    Args:
        time: Time array in seconds
        y_meas: Measured metabolic cost array (W/kg)
        tau: Time constant for exponential fit (default: 42s, typical for moderate exercise)
    
    Returns:
        y_estimate: Estimated steady-state metabolic cost (W/kg)
        y_bar: Fitted exponential curve
        fit_params: Dictionary with fit parameters for debugging
    """
    # LUKE: Check if we have enough data points for fitting (need at least 10)
    if len(time) < 10 or len(y_meas) < 10:
        # LUKE: If insufficient data, return simple average as fallback
        y_bar = np.full_like(y_meas, np.mean(y_meas))
        return np.mean(y_meas), y_bar, {'method': 'simple_average', 'reason': 'insufficient_data'}
    
    # LUKE: Define the exponential rise model: y = y_ss * (1 - exp(-t/tau)). y_xx is steady state
    # LUKE: This models the rise in metabolic cost toward steady state
    def exponential_rise(t, y_ss, tau_fit):
        return y_ss * (1 - np.exp(-t / tau_fit))
    
    try:
        # LUKE: Initial guess for steady-state value (y_ss) and time constant (tau_fit)
        # LUKE: Use the last 30% of data to estimate steady state
        last_30_percent = int(0.3 * len(y_meas))
        y_ss_guess = np.mean(y_meas[-last_30_percent:])
        tau_guess = tau  # Use provided tau as initial guess
        
        # LUKE: Set bounds for the fit parameters
        # LUKE: y_ss should be positive and reasonable (0.5 to 50 W/kg)
        # LUKE: tau should be positive and reasonable (10 to 200 seconds)
        bounds = ([0.5, 10.0], [50.0, 200.0])
        
        # LUKE: Fit the exponential model to the data
        popt, pcov = curve_fit(exponential_rise, time, y_meas, 
                              p0=[y_ss_guess, tau_guess], 
                              bounds=bounds, 
                              maxfev=10000)
        
        # LUKE: Extract fitted parameters
        y_ss_fitted, tau_fitted = popt
        
        # LUKE: Generate the fitted curve
        y_bar = exponential_rise(time, y_ss_fitted, tau_fitted)
        
        # LUKE: Return the estimated steady-state value (this is what we want for the loss function)
        fit_params = {
            'method': 'exponential_fit',
            'y_ss': y_ss_fitted,
            'tau': tau_fitted,
            'r_squared': 1 - np.sum((y_meas - y_bar)**2) / np.sum((y_meas - np.mean(y_meas))**2)
        }
        
        return y_ss_fitted, y_bar, fit_params
        
    except (RuntimeError, ValueError) as e:
        # LUKE: If fitting fails, fall back to simple average of last 30% of data
        print(f"LUKE: Warning - Exponential fit failed: {e}. Using fallback method.")
        last_30_percent = int(0.3 * len(y_meas))
        y_estimate = np.mean(y_meas[-last_30_percent:])
        y_bar = np.full_like(y_meas, y_estimate)
        
        fit_params = {
            'method': 'fallback_average',
            'reason': str(e),
            'last_30_percent_avg': y_estimate
        }
        
        return y_estimate, y_bar, fit_params
# LUKE: Added enhanced metabolic cost loss function for 2-minute protocols
# It uses exponential fitting to the data to estimate steady state metabolic cost 
# If the trial is long enough, it just uses the average of the last 2 minutes and assumes a steady state.
def compute_metabolic_cost_loss_2min(vo2_data: np.ndarray, vco2_data: np.ndarray, 
                                    time_data: np.ndarray, body_weight_kg: float = 77.0,
                                    use_estimation: bool = True) -> float:
    """
    Compute metabolic cost using Brockway equation with estimation for 2-minute protocols.
    This is the enhanced loss function for effort/energy expenditure metrics.
    
    Args:
        vo2_data: Oxygen consumption data (mL/min)
        vco2_data: Carbon dioxide production data (mL/min)
        time_data: Time array in seconds
        body_weight_kg: Subject body weight in kg (default: 77.0 kg)
        use_estimation: Whether to use exponential estimation for short protocols
    
    Returns:
        Metabolic cost in W/kg (float) - either measured average or estimated steady state
    """
    # LUKE: Check if we have valid data
    if vo2_data.size == 0 or vco2_data.size == 0 or time_data.size == 0:
        return float('nan')
    
    # LUKE: Check if body weight is valid (must be positive)
    if body_weight_kg <= 0:
        return float('nan')
    
    # LUKE: Calculate metabolic cost using Brockway equation for each time point
    # LUKE: Metabolic Cost (W/kg) = (0.278 × VO2 + 0.075 × VCO2) / body_weight
    y_meas = (0.278 * vo2_data + 0.075 * vco2_data) / body_weight_kg
    
    # LUKE: Determine protocol duration in minutes
    protocol_duration_min = time_data[-1] / 60.0
    
    # LUKE: For protocols shorter than 5 minutes, use exponential estimation
    # LUKE: This is the gold standard approach for short-duration exercise
    if use_estimation and protocol_duration_min < 5:
        # LUKE: Use exponential estimation to project where metabolic cost would converge
        y_estimate, y_bar, fit_params = metabolic_rate_estimation(time_data, y_meas)
        
        # LUKE: Print estimation details for debugging
        print(f"LUKE: 2-minute protocol detected ({protocol_duration_min:.1f} min)")
        print(f"LUKE: Using exponential estimation - steady state: {y_estimate:.4f} W/kg")
        print(f"LUKE: Fit method: {fit_params['method']}")
        if 'r_squared' in fit_params:
            print(f"LUKE: Fit quality (R²): {fit_params['r_squared']:.3f}")
        
        return y_estimate
    
    else:
        # LUKE: For longer trials, use the gold standard: average of last 2 minutes
        # LUKE: This assumes steady state has been reached
        if protocol_duration_min >= 5:
            # LUKE: Find the last 2 minutes of data
            last_2min_start = time_data[-1] - 120  # 2 minutes = 120 seconds
            last_2min_mask = time_data >= last_2min_start
            y_last_2min = y_meas[last_2min_mask]
            
            # LUKE: Calculate average of last 2 minutes (gold standard)
            y_average = np.mean(y_last_2min)
            print(f"LUKE: Long protocol detected ({protocol_duration_min:.1f} min)")
            print(f"LUKE: Using gold standard - average of last 2 min: {y_average:.4f} W/kg")
            return y_average
        
        else:
            # LUKE: For very short protocols without estimation, use simple average
            y_average = np.mean(y_meas)
            print(f"LUKE: Short protocol without estimation ({protocol_duration_min:.1f} min)")
            print(f"LUKE: Using simple average: {y_average:.4f} W/kg")
            return y_average
# LUKE: Added function to load raw metabolic data from Excel files for enhanced estimation
def load_raw_metabolic_data(base_path: str, body_weight_kg: float = 77.0) -> tuple:
    """
    Load raw metabolic data (VO2, VCO2, time) from Excel files for enhanced estimation.
    
    Args:
        base_path: Base path for the data file (e.g., '2025.06.18/Luke_test2')
        body_weight_kg: Subject body weight in kg (default: 77.0 kg)
    
    Returns:
        Tuple of (time_data, vo2_data, vco2_data) or (None, None, None) if file not found
    """
    import pandas as pd
    
    # LUKE: Try multiple possible file paths for the metabolic Excel file
    # LUKE: The metabolic processing pipeline reads from: {base_path}_COSMED.xlsx
    possible_paths = [
        f"{base_path}_COSMED.xlsx",
        f"{base_path}_COSMED.xls",
        # LUKE: Also try the pattern from the existing JSON file
        f"{os.path.dirname(base_path)}/20250618_LukeChung_COSMED.xlsx",
        f"{os.path.dirname(base_path)}/20250618_LukeChung_COSMED.xls",
        # LUKE: Try the actual naming convention you're using
        f"{os.path.dirname(base_path)}/20250618 Luke Chung (CPET Breath by Breath) Trial Last 5 min (1).xlsx",
        # LUKE: Try variations for future trials
        f"{os.path.dirname(base_path)}/20250618_LukeChung_CPET.xlsx",
        f"{os.path.dirname(base_path)}/20250618_LukeChung_CPET.xls"
    ]
    
    metabolic_excel_path = None
    for path in possible_paths:
        if os.path.exists(path):
            metabolic_excel_path = path
            break
    
    # LUKE: Check if the metabolic Excel file exists
    if metabolic_excel_path is None:
        print(f"LUKE: Warning - Raw metabolic data file not found. Tried paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return None, None, None
    
    try:
        # LUKE: Load the Excel file using pandas (matching MATLAB readcell behavior)
        df = pd.read_excel(metabolic_excel_path, header=None)
        
        # LUKE: Extract experiment time from row 2, column 5 (matching MATLAB)
        exp_time_raw = df.iloc[1, 4]  # MATLAB: T{2,5}
        
        # LUKE: Parse experiment time (matching MATLAB logic)
        if isinstance(exp_time_raw, str):
            exp_time = exp_time_raw
        else:
            # LUKE: Convert datetime to string format
            exp_time = str(exp_time_raw)
        
        # LUKE: Extract the data section (columns 10 onwards, starting from row 4) - matching MATLAB
        data_raw = df.iloc[:, 9:].copy()  # MATLAB: T(:,10:end)
        data_raw.columns = data_raw.iloc[0]
        data = data_raw.iloc[3:].copy()  # MATLAB: T(4:end, :)
        
        # LUKE: Find the index of the columns containing t, VO2, and VCO2 (matching MATLAB)
        time_index = data_raw.columns.get_loc('t') if 't' in data_raw.columns else None
        VO2_index = data_raw.columns.get_loc('VO2') if 'VO2' in data_raw.columns else None
        VCO2_index = data_raw.columns.get_loc('VCO2') if 'VCO2' in data_raw.columns else None
        
        if time_index is None or VO2_index is None or VCO2_index is None:
            print(f"LUKE: Error - Required columns (t, VO2, VCO2) not found in {metabolic_excel_path}")
            return None, None, None
        
        # LUKE: Parse time, VO2, and VCO2 columns using the same functions as the metabolic pipeline
        # LUKE: Import the parsing functions from the metabolic processing module
        
        # LUKE: Convert relevant columns to numeric, ignore non-numeric entries
        data['t'] = data['t'].apply(_parse_time_value)
        data['VO2'] = data['VO2'].apply(_parse_number)
        data['VCO2'] = data['VCO2'].apply(_parse_number)
        
        # LUKE: Drop any rows where required numeric values are missing
        data = data.dropna(subset=['t', 'VO2', 'VCO2']).reset_index(drop=True)
        if data.empty:
            print("LUKE: Warning - No valid numeric rows found for t, VO2, and VCO2")
            return None, None, None
        
        # LUKE: Convert to numpy arrays (matching MATLAB cell2mat behavior)
        time_data = data['t'].to_numpy(dtype=float)
        vo2_data = data['VO2'].to_numpy(dtype=float)
        vco2_data = data['VCO2'].to_numpy(dtype=float)
        
        # LUKE: Normalize time to start at 0 (matching MATLAB: time = time - time(1))
        time_data -= time_data[0]
        
        # LUKE: Cut off at 5 minutes (matching MATLAB: [~, cut_idx] = min(abs(time-60*5)))
        cutoff_idx = np.argmin(np.abs(time_data - 5*60))
        time_data = time_data[:cutoff_idx+1]
        vo2_data = vo2_data[:cutoff_idx+1]
        vco2_data = vco2_data[:cutoff_idx+1]
        
        print(f"LUKE: Successfully loaded raw metabolic data from {metabolic_excel_path}")
        print(f"LUKE: Experiment time: {exp_time}")
        print(f"LUKE: Data points: {len(time_data)}, Duration: {time_data[-1]/60:.1f} minutes")
        return time_data, vo2_data, vco2_data
        
    except Exception as e:
        # LUKE: Handle any errors in loading or parsing the Excel file
        print(f"LUKE: Error loading raw metabolic data: {e}")
        return None, None, None
# LUKE: Added metabolic processing function matching the MATLAB code. IDK IF THIS IS NECESSARY
def process_metabolic_data_complete(base_path: str, body_weight_kg: float = 77.0, 
                                   estimate_threshold_min: float = 4.8, 
                                   avg_window_min: float = 2, tau: float = 42.0):
    """
    Args:
        base_path: Base path for the data file
        body_weight_kg: Subject body weight in kg
        estimate_threshold_min: Threshold for using estimation vs average (default: 4.8 min)
        avg_window_min: Window for averaging in long protocols (default: 2 min)
        tau: Time constant for exponential estimation (default: 42s)
    
    Returns:
        Dictionary with processed metabolic data and visualization
    """
    # LUKE: Load raw metabolic data
    time_data, vo2_data, vco2_data = load_raw_metabolic_data(base_path, body_weight_kg)
    
    if time_data is None:
        return None
    
    # LUKE: Calculate metabolic cost using Brockway equation (matching MATLAB exactly)
    # LUKE: MATLAB: y_meas = (0.278*VO2 + 0.075*VCO2)/weight
    y_meas = (0.278 * vo2_data + 0.075 * vco2_data) / body_weight_kg
    
    # LUKE: Determine protocol duration
    protocol_duration_min = time_data[-1] / 60.0
    
    # LUKE: Apply the same logic as MATLAB for short vs long protocols
    if protocol_duration_min < estimate_threshold_min:
        # LUKE: Short protocol - use exponential estimation
        y_average, y_bar, fit_params = metabolic_rate_estimation(time_data, y_meas, tau)
        time_bar = time_data
        y_estimate = None
        print(f"LUKE: Short protocol ({protocol_duration_min:.1f} min) - using exponential estimation")
    else:
        # LUKE: Long protocol - use average of last 2 minutes (gold standard)
        start_time = time_data[-1] - avg_window_min * 60
        start_idx = np.argmin(np.abs(time_data - start_time))
        y_average = np.mean(y_meas[start_idx:])
        
        # LUKE: Also compute estimation for comparison
        end_idx = np.argmin(np.abs(time_data - 180))  # 3 minutes
        time_estimate = time_data[:end_idx+1]
        y_estimate, y_bar, fit_params = metabolic_rate_estimation(
            time_estimate, y_meas[:end_idx+1], tau
        )
        time_bar = time_estimate
        print(f"LUKE: Long protocol ({protocol_duration_min:.1f} min) - using gold standard (last {avg_window_min} min)")
    
    # LUKE: Print results (matching MATLAB output format)
    print(f"LUKE: Average metabolic cost: {y_average:.4f} W/kg")
    if y_estimate is not None:
        print(f"LUKE: Estimated metabolic cost: {y_estimate:.4f} W/kg")
    
    # LUKE: Create visualization data
    viz_data = {
        'time': time_data,
        'y_meas': y_meas,
        'time_bar': time_bar,
        'y_bar': y_bar,
        'y_average': y_average,
        'y_estimate': y_estimate,
        'protocol_duration_min': protocol_duration_min,
        'fit_params': fit_params,
        'body_weight_kg': body_weight_kg
    }
    
    return viz_data
class DataProcessor:
    """
    Orchestrates the data handling pipeline for each trial, including
    recording, initial analysis, and final processing.
    """
    def __init__(self, user_id, trial_num, smooth_fraction=25, visualize_steps=False, data_path=None):
        self.user_id = user_id
        self.trial_num = trial_num
        self.visualize_steps = visualize_steps
        self.smooth_fraction = smooth_fraction
        
        # Configure paths based on whether this is a live trial or a standalone run
        if data_path:
            # Standalone mode: paths are relative to the input data file
            self.raw_data_path = data_path
            base_path = os.path.splitext(data_path)[0]
            self.step_file_path = base_path + '_steps.csv'
            self.visualization_path = base_path + '_visualization.html'
        else:
            # Live mode: paths are based on user/trial in the data directory
            self.user_data_path = os.path.join(config.DATA_DIRECTORY, self.user_id)
            os.makedirs(self.user_data_path, exist_ok=True)
            base_filename = f"{self.user_id}_trial_{self.trial_num}"
            self.raw_data_path = os.path.join(self.user_data_path, base_filename + config.RAW_DATA_SUFFIX)
            self.step_file_path = os.path.join(self.user_data_path, base_filename + config.STEP_FILE_SUFFIX)
            self.visualization_path = os.path.join(self.user_data_path, base_filename + '_steps_visualization.html')
    def get_trial_data_path(self):
        return self.raw_data_path
    def _create_and_save_plot(self, df, steps):
        """
        Generates and saves an interactive Plotly plot of the step detection results.
        """
        fig = go.Figure()
        # 1. Smoothed Accelerometer Signal
        if 'acc_x_z_smooth' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['acc_x_z_smooth'],
                                     mode='lines', name='Smoothed Accel Magnitude', yaxis='y1'))
        # 2. Force Signal and its Processed Gradient on secondary axis
        if 'force' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['force'],
                                     mode='lines', name='Force',
                                     line=dict(dash='solid', color='rgba(255,153,51,0.6)'), # Orange
                                     yaxis='y2'))
        if 'force_grad_filtered' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['force_grad_filtered'],
                                     mode='lines', name='Processed Force Grad',
                                     line=dict(dash='dashdot', color='rgba(230,0,230,0.6)'), # Magenta
                                     yaxis='y2'))
        # 3. Detected Steps
        if 'acc_x_z_smooth' in df.columns and len(steps) > 0:
            step_y_values = np.interp(steps, df['timestamp'], df['acc_x_z_smooth'])
            fig.add_trace(go.Scatter(x=steps, y=step_y_values,
                                     mode='markers', marker=dict(color='red', symbol='cross', size=8),
                                     name='Detected Steps'))
        # Configure layout
        title = f'Step Detection Visualization for {os.path.basename(self.raw_data_path)}'
        if self.user_id:
            title = f'Step Detection Visualization for {self.user_id} Trial {self.trial_num}'
        layout_update = {
            'title_text': title,
            'xaxis_title': 'Time (s)',
            'yaxis_title': 'Smoothed Accel Magnitude',
            'legend_title_text': 'Signals'
        }
        if 'force' in df.columns:
            layout_update['yaxis2'] = {
                'title': 'Force',
                'overlaying': 'y',
                'side': 'right'
            }
        
        fig.update_layout(**layout_update)
        fig.write_html(self.visualization_path)
        print(f"Saved step detection visualization to {self.visualization_path}")
    def record_and_detect_peaks(self):
        """
        Manages data recording for live trials. If recording fails, offers a fallback.
        For standalone runs, this method directly uses the provided data path.
        Then performs automated peak detection.
        """
        # --- Live Trial Recording ---
        # This block is skipped if a data_path was provided to __init__
        if not hasattr(self, 'user_data_path'):
            # This is a standalone run, self.raw_data_path is already set.
            pass
        else:
            # This is a live trial run.
            # --- Pre-trial cleanup ---
            if os.path.exists(self.raw_data_path):
                os.remove(self.raw_data_path)
            # 1. Attempt to record raw data
            print("\n--- Preparing for Data Recording ---")
            try:
                asyncio.run(record_trial_data(self.raw_data_path))
            except Exception as e:
                print(f"\nAn error occurred during data recording: {e}")
            # 2. Verify data was recorded. If not, prompt user for existing file.
            if not os.path.exists(self.raw_data_path) or os.path.getsize(self.raw_data_path) == 0:
                print("\nWarning: New data was not recorded (e.g., Bluetooth connection failed).")
                use_existing = input("Would you like to use an existing raw data file for this trial? (yes/no): ").lower()
                
                if use_existing == 'yes':
                    existing_path = input("Please enter the full path to the existing raw data file: ")
                    if os.path.exists(existing_path):
                        try:
                            shutil.copy(existing_path, self.raw_data_path)
                            print(f"Successfully copied existing data to '{self.raw_data_path}'.")
                        except Exception as e:
                            print(f"Error: Failed to copy the file. {e}")
                            return False
                    else:
                        print("Error: The file path you entered does not exist.")
                        return False
                else:
                    print("The trial cannot proceed without data.")
                    return False
        
        # 3. Perform peak detection on the new, copied, or provided file
        try:
            df = pd.read_csv(self.raw_data_path)
            
            # Standardize column names for compatibility.
            df.rename(columns={
                "acc_x_time": "timestamp"
            }, inplace=True, errors='ignore')
            alpha = self.smooth_fraction / 100.0
            acc_x_smooth = lfilter([alpha], [1, -(1 - alpha)], df['acc_x_data'])
            acc_z_smooth = lfilter([alpha], [1, -(1 - alpha)], df['acc_z_data'])
            df['acc_x_z_smooth'] = acc_x_smooth**2 + acc_z_smooth**2
                
            # Convert timestamp to seconds for analysis
            # The raw data, whether from the sensor or a fallback file, uses milliseconds.
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
            # Pre-process signals
            alpha = self.smooth_fraction / 100.0
            acc_x_smooth = lfilter([alpha], [1, -(1 - alpha)], df['acc_x_data'])
            # NOTE: Using accZ for this calculation as per your last edit.
            acc_z_smooth = lfilter([alpha], [1, -(1 - alpha)], df['acc_z_data']) 
            df['acc_x_z_smooth'] = acc_x_smooth**2 + acc_z_smooth**2
            # Calculate force gradient for plotting and analysis
            force_gradient = np.gradient(df['force'].values, df['timestamp'].values)
            threshold = force_gradient.min() * 0.2
            df['force_grad_filtered'] = np.where(force_gradient < threshold, force_gradient, 0)
            
            # Calculate sampling frequency
            fs = 1.0 / np.median(np.diff(df['timestamp'].values))
            # Detect steps using the best algorithm. It now expects the raw force signal.
            raw_steps = detect_steps_unsupervised(df['force'].values, df['timestamp'].values, fs)
            
            # Apply post-processing filters
            final_steps = _postprocess_steps(raw_steps)
            
            # LUKE: Added integration of instability loss functions after step detection and post-processing
            # LUKE: Compute the first instability loss: standard deviation of cycle durations
            cycle_duration_loss = compute_cycle_std(df, final_steps)
            # LUKE: Compute the second instability loss: mean RMS of Y-axis acceleration per cycle
            y_rms_loss = compute_mean_y_rms_per_cycle(df, final_steps)
            
            # LUKE: Compute the third loss: metabolic cost using Brockway equation (gold standard)
            # LUKE: Extract base path from the raw data path to find associated metabolic data
            base_path = os.path.splitext(self.raw_data_path)[0]
            
            # LUKE: Use the complete metabolic processing function (matching MATLAB exactly)
            metabolic_data = process_metabolic_data_complete(base_path, body_weight_kg=77.0)
            
            if metabolic_data is not None:
                # LUKE: Use the processed metabolic cost as the loss function
                metabolic_cost_loss = metabolic_data['y_average']
                print(f"LUKE: Using complete metabolic processing: {metabolic_cost_loss:.4f} W/kg")
            else:
                # LUKE: Fall back to JSON method if raw data not available
                metabolic_cost_loss = load_metabolic_cost_from_json(base_path)
                print(f"LUKE: Raw metabolic data not available, using JSON result: {metabolic_cost_loss:.4f} W/kg")
            
            # LUKE: Print the computed instability metrics to the terminal for user feedback
            print(f"\n--- Instability Metrics ---")
            # LUKE: Display cycle duration loss with 6 decimal places and units
            print(f"Cycle duration loss (std): {cycle_duration_loss:.6f} s")
            # LUKE: Display Y-axis RMS loss with 6 decimal places and units
            print(f"Y-axis RMS loss (mean per cycle): {y_rms_loss:.6f} m/s²")
            # LUKE: Display metabolic cost loss with 6 decimal places and units
            print(f"Metabolic cost loss (Brockway): {metabolic_cost_loss:.6f} W/kg")
            
            # Conditionally visualize the results before saving
            if self.visualize_steps:
                self._create_and_save_plot(df, final_steps)
            # Save the final step times to the step file
            pd.DataFrame({'step_time': final_steps}).to_csv(self.step_file_path, index=False)
            print(f"Detected and saved {len(final_steps)} steps to {self.step_file_path}")
            
        except FileNotFoundError:
            print(f"Error: Raw data file not found at {self.raw_data_path}. Cannot perform peak detection.")
            return False
        except Exception as e:
            print(f"An error occurred during peak detection: {e}")
            return False
            
        return True # Signal that this stage was successful
    def featurize_trial_data(self):
        """
        Performs the final feature extraction using the raw data and the
        (potentially manually corrected) step file.
        """
        try:
            raw_df = pd.read_csv(self.raw_data_path)
            # The raw data file has headers: 'acc_x_time', 'force', 'roll', 'acc_x_data'
            # We need to convert time to seconds and rename columns for the analysis functions.
            raw_df['timestamp'] = (raw_df['acc_x_time'] - raw_df['acc_x_time'].iloc[0]) / 1000.0
            raw_df.rename(columns={"acc_x_data": "accX"}, inplace=True)
            # 'force' and 'roll' are kept as-is, assuming downstream functions use these names.
            # The step file now contains a single column 'step_time' with a header.
            steps_df = pd.read_csv(self.step_file_path)
            
            # Ensure the step_time column is numeric, handling potential read errors
            steps_df['step_time'] = pd.to_numeric(steps_df['step_time'], errors='coerce')
            steps_df.dropna(subset=['step_time'], inplace=True) # Remove any rows that failed conversion
            # To robustly find the step indices, we find the closest timestamp
            # in the raw data for each step time you identified.
            raw_df_sorted = raw_df.sort_values('timestamp').reset_index()
            steps_df_sorted = steps_df.sort_values('step_time')
            print(steps_df_sorted)
            # Convert both timestamp columns to float64 to ensure compatibility
            raw_df_sorted['timestamp'] = raw_df_sorted['timestamp'].astype('float64')
            steps_df_sorted['step_time'] = steps_df_sorted['step_time'].astype('float64')
            # Find the row in raw_df that corresponds to each step via the timestamp
            merged = pd.merge_asof(
                left=steps_df_sorted,
                right=raw_df_sorted,
                left_on='step_time',
                right_on='timestamp',
                direction='nearest'
            )
            
            # Get the original row indices to pass to the gait calculation function
            step_indices = merged['index'].to_numpy()
            gait_metrics = calculate_gait_metrics_from_steps(raw_df, step_indices)
            
            # LUKE: Added computation of instability metrics in featurization for final output
            # LUKE: Extract step times from the sorted step dataframe for loss computation
            step_times = steps_df_sorted['step_time'].values
            # LUKE: Compute cycle duration loss using the same function as in step detection
            cycle_duration_loss = compute_cycle_std(raw_df, step_times)
            # LUKE: Compute Y-axis RMS loss using the same function as in step detection
            y_rms_loss = compute_mean_y_rms_per_cycle(raw_df, step_times)
            # LUKE: Compute metabolic cost loss using the same function as in step detection
            base_path = os.path.splitext(self.raw_data_path)[0]
            
            # LUKE: Use the complete metabolic processing function (matching MATLAB exactly)
            metabolic_data = process_metabolic_data_complete(base_path, body_weight_kg=77.0)
            
            if metabolic_data is not None:
                # LUKE: Use the processed metabolic cost as the loss function
                metabolic_cost_loss = metabolic_data['y_average']
            else:
                # LUKE: Fall back to JSON method if raw data not available
                metabolic_cost_loss = load_metabolic_cost_from_json(base_path)
        except FileNotFoundError:
            print(f"Error: Could not find data files. Ensure both '{self.raw_data_path}' and '{self.step_file_path}' exist.")
            # LUKE: Set default values for gait metrics when files are not found
            gait_metrics = {'step_variance': 0, 'Y_change': 0, 'Y_total': 0}
            # LUKE: Set instability losses to NaN when files are not found
            cycle_duration_loss = y_rms_loss = metabolic_cost_loss = float('nan')
        except Exception as e:
            print(f"An error occurred during data featurization: {e}")
            print("Raw data shape:", raw_df.shape if 'raw_df' in locals() else "Not loaded")
            print("Steps data shape:", steps_df.shape if 'steps_df' in locals() else "Not loaded")
            # LUKE: Set default values for gait metrics when an error occurs
            gait_metrics = {'step_variance': 0, 'Y_change': 0, 'Y_total': 0}
            # LUKE: Set instability losses to NaN when an error occurs
            cycle_duration_loss = y_rms_loss = metabolic_cost_loss = float('nan')
        # Placeholder for other data sources (surveys, metabolic, etc.)
        processed_data = {
            'metabolic_cost': 3.5,
            'effort_survey_answer': 3,
            'stability_survey_answer': 4,
            'pain_survey_answer': 1,
            'RMS_load_cell_force': 120,
            **gait_metrics,
            # LUKE: Added instability metrics to final processed data output
            # LUKE: Include cycle duration loss in the final processed data dictionary
            'cycle_duration_loss': cycle_duration_loss,
            # LUKE: Include Y-axis RMS loss in the final processed data dictionary
            'y_rms_loss': y_rms_loss,
            # LUKE: Include metabolic cost loss in the final processed data dictionary
            'metabolic_cost_loss': metabolic_cost_loss
        }
        return processed_data
# LUKE: Simplified metabolic cost function that always processes from raw Excel data
def get_metabolic_cost_from_excel(base_path: str, body_weight_kg: float = 77.0) -> float:
    """
    Simplified function that always processes metabolic cost from raw Excel data.
    This replaces the JSON loading approach for a direct Excel → Processing → Result workflow.
    
    Args:
        base_path: Base path for the data file (e.g., '2025.06.18/Luke_test2')
        body_weight_kg: Subject body weight in kg (default: 77.0 kg)
    
    Returns:
        Metabolic cost in W/kg (float), or NaN if processing fails
    """
    # Load raw metabolic data from Excel
    time_data, vo2_data, vco2_data = load_raw_metabolic_data(base_path, body_weight_kg)
    
    # Check if data loading was successful
    if time_data is None:
        print(f"LUKE: Failed to load raw metabolic data for {base_path}")
        return float('nan')
    
    # Calculate metabolic cost using exponential fitting
    metabolic_cost = compute_metabolic_cost_loss_2min(vo2_data, vco2_data, time_data, body_weight_kg)
    
    return metabolic_cost
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run step detection and gait analysis on a single data file.")
    parser.add_argument("data_file", nargs='?', type=str, default='2025.06.18/Luke_test2.csv', help="Path to the raw data CSV file.")
    args = parser.parse_args()
    # Instantiate the processor in standalone mode, with visualization enabled.
    processor = DataProcessor(user_id=None, trial_num=None, 
                              data_path=args.data_file, 
                              visualize_steps=True)
    # Run the step detection part.
    # This will load the file, find steps, save step file, and save visualization.
    print("--- Running Step Detection ---")
    success = processor.record_and_detect_peaks()
    if success:
        # Run the featurization part.
        # This reads the step file we just created and calculates metrics.
        print("\n--- Running Gait Featurization ---")
        metrics = processor.featurize_trial_data()
        print("\n--- Calculated Gait Metrics ---")
        if metrics:
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        else:
            print("Featurization did not return any metrics.")
    else:
        print("\nAnalysis failed. Could not proceed to featurization.")