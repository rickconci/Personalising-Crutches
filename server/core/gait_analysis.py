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

def detect_steps_from_accel(accel_signal: np.ndarray, fs: float, smooth_factor: float = 0.25) -> np.ndarray:
    """
    Detects steps from a 2D accelerometer vector magnitude signal, ported from the
    logic in Accelerometer_Processing_Program.html.
    
    Args:
        accel_signal (np.ndarray): The 2D vector magnitude of the smoothed accelerometer data.
        fs (float): The sampling frequency of the signal.

    Returns:
        np.ndarray: An array of indices corresponding to the detected steps.
    """
    if accel_signal.size == 0:
        return np.array([])
        
    # 1. Calculate derivative of the signal
    differential_duration = 0.24  # seconds, from JS code
    differential_points = int(round(differential_duration * fs))
    deriv = np.zeros_like(accel_signal)
    if differential_points > 0:
        # Note: The JS implementation used a simple difference, but np.gradient is more robust
        deriv = np.gradient(accel_signal, 1/fs)

    # 2. Adaptive threshold based on derivative signal's median absolute deviation
    # Using 5.0 as a robust multiplier, similar to other algorithms in the project
    threshold = np.median(np.abs(deriv)) * 5.0

    # 3. Find upward crossings of the threshold
    crossings = np.where((deriv[:-1] < threshold) & (deriv[1:] >= threshold))[0] + 1
    if crossings.size == 0:
        return np.array([])

    # 4. Refine step start by finding peak in subsequent window of original signal
    peak_indices = []
    for i in crossings:
        # Look in a ~100 sample window, as in the JS code
        window_end = min(i + 100, len(accel_signal))
        if window_end > i:
            # Find the index of the max value *within the window*
            max_in_window_idx = np.argmax(accel_signal[i:window_end])
            # Add the start index of the window to get the index relative to the whole signal
            peak_indices.append(i + max_in_window_idx)
    
    if not peak_indices:
        return np.array([])
        
    # 5. Filter out peaks that are too close (10 samples in JS)
    unique_peaks = []
    last_peak_idx = -np.inf
    for peak_idx in peak_indices:
        if peak_idx - last_peak_idx > 10:
            unique_peaks.append(peak_idx)
            last_peak_idx = peak_idx
            
    return np.array(unique_peaks)


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