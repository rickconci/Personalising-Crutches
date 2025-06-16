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
    Calculates gait metrics based on pre-defined step cycle peaks.
    If 'accY' is not present, Y-axis metrics will be set to 0.

    Args:
        df (pd.DataFrame): The raw trial data.
        step_indices (np.array): An array of indices where steps occur.

    Returns:
        dict: A dictionary of calculated stability metrics.
    """
    if len(step_indices) < 2:
        print("Warning: Not enough step indices to calculate metrics.")
        return {'step_variance': 0, 'Y_change': 0, 'Y_total': 0}

    step_times = df['timestamp'].iloc[step_indices]
    step_durations = step_times.diff().dropna()
    step_variance = step_durations.var() if len(step_durations) > 0 else 0

    y_change = 0
    y_total = 0
    # Calculate Y-axis (medio-lateral) metrics for each step cycle if data is available
    if 'accY' in df.columns:
        y_acc = df['accY']
        y_changes = []
        y_totals = []
        for i in range(len(step_indices) - 1):
            start_idx, end_idx = step_indices[i], step_indices[i+1]
            step_segment = y_acc.iloc[start_idx:end_idx]
            if not step_segment.empty:
                y_changes.append(step_segment.max() - step_segment.min())
                y_totals.append(step_segment.abs().sum())
        
        y_change = np.mean(y_changes) if y_changes else 0
        y_total = np.mean(y_totals) if y_totals else 0

    return {
        'step_variance': step_variance,
        'Y_change': y_change,
        'Y_total': y_total
    } 