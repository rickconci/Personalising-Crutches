import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

def exponential_smooth(data, factor):
    """
    Applies an exponential smoothing filter. Equivalent to the timeconstantsmooth
    function in the JavaScript.
    """
    return data.ewm(alpha=factor, adjust=False).mean()

def high_pass_filter(data, cutoff=0.5, fs=100):
    """
    Applies a high-pass filter to the data.
    A more robust implementation than direct translation, using scipy.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def differentiate(data, time_interval):
    """
    Calculates the derivative of the data.
    """
    return np.gradient(data, time_interval)

def detect_initial_peaks(df, acc_col='accX'):
    """
    Performs an initial automated detection of gait cycle peaks using a single
    accelerometer axis.
    
    Args:
        df (pd.DataFrame): The raw trial data.
        acc_col (str): The name of the accelerometer column to use.

    Returns:
        np.array: An array of indices corresponding to the detected peaks.
    """
    if acc_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain column: {acc_col}")
        
    fs = 1 / df['timestamp'].diff().mean()
    
    # Use the specified accelerometer column for peak detection
    acc_data = df[acc_col]
    
    # Find peaks, assuming a minimum step distance of 0.4 seconds
    peaks, _ = find_peaks(acc_data, height=np.std(acc_data), distance=fs * 0.4)
    
    return peaks

def calculate_gait_metrics_from_steps(df, step_indices):
    """
    Calculates various gait metrics based on a DataFrame and a list of step indices.
    """
    if len(step_indices) < 2:
        return {
            'step_variance': 0,
            'Y_change': 0,
            'Y_total': 0
        }
        
    step_times = df['timestamp'].iloc[step_indices].values
    step_variance = np.var(np.diff(step_times))
    
    y_values = df['accX'].iloc[step_indices].values # Note: Using 'accX' as per your original file
    y_change = np.mean(np.abs(np.diff(y_values)))
    y_total = np.mean(y_values)
    
    return {
        'step_variance': step_variance,
        'Y_change': y_change,
        'Y_total': y_total
    }

def _postprocess_steps(preds: np.ndarray, tolerance_ratio: float = 0.2, isolation_threshold: float = 5.0) -> np.ndarray:
    """
    Applies post-processing filters to a series of detected step times.
    1. Regularizes step intervals based on a data-driven expected interval.
    2. Removes isolated steps that are too far from any other step.
    """
    if len(preds) < 2:
        return preds

    intervals = np.diff(preds)
    expected_interval = np.median(intervals) if len(intervals) >= 3 else 1.1
    
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
                regularized_preds[-1] = current_pred
        else:
            regularized_preds.append(current_pred)
    
    regularized_preds = np.array(regularized_preds)
    
    if len(regularized_preds) < 2:
        return np.array([])
        
    final_preds = []
    padded_preds = np.concatenate(([-np.inf], regularized_preds, [np.inf]))
    
    for i in range(1, len(padded_preds) - 1):
        dist_to_prev = padded_preds[i] - padded_preds[i-1]
        dist_to_next = padded_preds[i+1] - padded_preds[i]
        if not ((dist_to_prev > isolation_threshold) and (dist_to_next > isolation_threshold)):
            final_preds.append(padded_preds[i])
            
    return np.array(final_preds)

def detect_steps_unsupervised(force_signal: np.ndarray, time_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Detects steps using a dynamic threshold of the force gradient.
    """
    if force_signal.size < int(fs):
        return np.array([])
        
    d_force_dt = np.gradient(force_signal, time_signal)
    threshold = d_force_dt.min() * 0.2
    d_force_dt_filtered = np.where(d_force_dt < threshold, d_force_dt, 0)
    
    min_dist_samples = int(0.3 * fs)
    peaks, _ = find_peaks(-d_force_dt_filtered, distance=min_dist_samples)
    
    return time_signal[peaks]

def compute_cycle_variance(df: pd.DataFrame, step_times: np.ndarray) -> float:
    """
    Compute the variance of cycle durations (step intervals).
    """
    if len(step_times) < 2:
        return float('nan')
    
    durations = np.diff(step_times)
    return np.var(durations)

def compute_mean_y_rms_per_cycle(df: pd.DataFrame, step_times: np.ndarray, return_all=False):
    """
    Compute the mean RMS of Y-axis acceleration per cycle.
    """
    if len(step_times) < 2:
        return float('nan') if not return_all else ([], [])
    
    times = df['timestamp'].values
    y = df['acc_y_data'].values
    
    step_indices = np.searchsorted(times, step_times)
    
    cycle_rms = []
    cycle_starts = []
    
    for i in range(len(step_indices) - 1):
        start, end = step_indices[i], step_indices[i+1]
        seg_y = y[start:end]
        
        if seg_y.size > 0:
            cycle_rms.append(np.sqrt(np.mean(seg_y**2)))
            cycle_starts.append(times[start])
    
    if return_all:
        return cycle_rms, cycle_starts
        
    return float(np.mean(cycle_rms)) if cycle_rms else float('nan') 