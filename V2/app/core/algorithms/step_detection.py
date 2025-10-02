"""
Unified step detection algorithms for accelerometer data.

This module consolidates all step detection algorithms from the original codebase
into a clean, unified interface with proper type hints and documentation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy.signal import (
    find_peaks, 
    savgol_filter, 
    butter, 
    filtfilt, 
    find_peaks_cwt,
    correlate,
    lfilter
)
from dataclasses import dataclass
from enum import Enum

from ..config import step_detection_config, data_processing_config


class StepDetectionAlgorithm(str, Enum):
    """Available step detection algorithms."""
    PEAKS = "algo1_peaks"
    DERIVATIVE = "algo2_derivative"
    ADAPTIVE = "algo3_adaptive"
    TKEO = "algo4_tkeo"
    MATCHED = "algo5_matched"
    JAVASCRIPT = "algo6_javascript"
    FORCE_DERIVATIVE = "algo7_force_derivative"
    MIN_THRESHOLD = "algo8_min_threshold"
    DEEP_LEARNING = "deep_learning"


@dataclass
class StepDetectionResult:
    """Result of step detection analysis."""
    algorithm: StepDetectionAlgorithm
    step_times: np.ndarray
    step_indices: np.ndarray
    confidence_scores: Optional[np.ndarray] = None
    processing_time: float = 0.0
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class StepDetector:
    """
    Unified step detection interface.
    
    Consolidates all step detection algorithms from the original codebase
    into a clean, maintainable interface.
    """
    
    def __init__(self, sampling_frequency: float = None):
        """
        Initialize the step detector.
        
        Args:
            sampling_frequency: Sampling frequency in Hz. If None, will be estimated from data.
        """
        self.fs = sampling_frequency or data_processing_config.DEFAULT_SAMPLING_FREQUENCY
        self.smoothing_factor = data_processing_config.SMOOTHING_FACTOR
        
    def detect_steps(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        algorithm: StepDetectionAlgorithm,
        use_force_gradient: bool = False,
        **kwargs
    ) -> StepDetectionResult:
        """
        Detect steps using the specified algorithm.
        
        Args:
            data: Input data (DataFrame with accelerometer columns or numpy array)
            algorithm: Step detection algorithm to use
            use_force_gradient: Whether to use force gradient signal
            **kwargs: Algorithm-specific parameters
            
        Returns:
            StepDetectionResult with detected steps and metadata
        """
        import time
        start_time = time.time()
        
        # Preprocess data
        processed_signal, multichannel_signal = self._preprocess_data(data, use_force_gradient)
        
        # Select algorithm
        algorithm_func = self._get_algorithm_function(algorithm)
        
        # Run algorithm
        step_indices = algorithm_func(processed_signal, multichannel_signal, **kwargs)
        
        # Convert indices to times
        step_times = step_indices / self.fs
        
        processing_time = time.time() - start_time
        
        return StepDetectionResult(
            algorithm=algorithm,
            step_times=step_times,
            step_indices=step_indices,
            processing_time=processing_time,
            parameters=kwargs
        )
    
    def _preprocess_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        use_force_gradient: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess accelerometer data for step detection.
        
        Args:
            data: Input data
            use_force_gradient: Whether to include force gradient
            
        Returns:
            Tuple of (processed_signal, multichannel_signal)
        """
        if isinstance(data, pd.DataFrame):
            # Extract accelerometer data
            if 'acc_x_data' in data.columns and 'acc_z_data' in data.columns:
                acc_x = data['acc_x_data'].values
                acc_z = data['acc_z_data'].values
            else:
                raise ValueError("DataFrame must contain 'acc_x_data' and 'acc_z_data' columns")
            
            # Estimate sampling frequency from time column if available
            if 'time' in data.columns:
                times = data['time'].dropna().values
                if len(times) > 1:
                    self.fs = 1.0 / np.median(np.diff(times))
        else:
            # Assume numpy array with shape (n_samples, 2) for acc_x and acc_z
            acc_x = data[:, 0]
            acc_z = data[:, 1]
        
        # Apply exponential smoothing
        alpha = self.smoothing_factor / 100.0
        acc_x_smooth = lfilter([alpha], [1, -(1 - alpha)], acc_x)
        acc_z_smooth = lfilter([alpha], [1, -(1 - alpha)], acc_z)
        
        # Calculate vector magnitude
        processed_signal = np.sqrt(acc_x_smooth**2 + acc_z_smooth**2)
        
        # Prepare multichannel signal
        signals_to_stack = [processed_signal]
        
        if use_force_gradient and isinstance(data, pd.DataFrame) and 'force' in data.columns:
            force = data['force'].values
            time = data['time'].values if 'time' in data.columns else np.arange(len(force)) / self.fs
            force_gradient = np.gradient(force, time)
            signals_to_stack.append(force_gradient)
        
        multichannel_signal = np.stack(signals_to_stack, axis=1)
        
        return processed_signal, multichannel_signal
    
    def _get_algorithm_function(self, algorithm: StepDetectionAlgorithm):
        """Get the function for the specified algorithm."""
        algorithm_map = {
            StepDetectionAlgorithm.PEAKS: self._algorithm_1_peaks,
            StepDetectionAlgorithm.DERIVATIVE: self._algorithm_2_derivative,
            StepDetectionAlgorithm.ADAPTIVE: self._algorithm_3_adaptive,
            StepDetectionAlgorithm.TKEO: self._algorithm_4_tkeo,
            StepDetectionAlgorithm.MATCHED: self._algorithm_5_matched,
            StepDetectionAlgorithm.JAVASCRIPT: self._algorithm_6_javascript,
            StepDetectionAlgorithm.FORCE_DERIVATIVE: self._algorithm_7_force_derivative,
            StepDetectionAlgorithm.MIN_THRESHOLD: self._algorithm_8_min_threshold,
        }
        
        if algorithm not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return algorithm_map[algorithm]
    
    def _algorithm_1_peaks(self, signal: np.ndarray, multichannel: np.ndarray, **kwargs) -> np.ndarray:
        """Algorithm 1: Simple peak detection."""
        params = step_detection_config.ALGORITHM_PARAMS['algo1_peaks']
        params.update(kwargs)
        
        min_distance = int(params['min_distance'] * self.fs)
        height = np.std(signal) * params['height_multiplier']
        
        peaks, _ = find_peaks(signal, height=height, distance=min_distance)
        return peaks
    
    def _algorithm_2_derivative(self, signal: np.ndarray, multichannel: np.ndarray, **kwargs) -> np.ndarray:
        """Algorithm 2: Derivative-based detection."""
        params = step_detection_config.ALGORITHM_PARAMS['algo2_derivative']
        params.update(kwargs)
        
        # Calculate derivative
        differential_duration = params['differential_duration']
        differential_points = int(round(differential_duration * self.fs))
        
        deriv = np.zeros_like(signal)
        if differential_points > 0:
            time_interval = differential_points / self.fs
            deriv[differential_points:] = (signal[differential_points:] - signal[:-differential_points]) / time_interval
        
        # Find peaks in derivative
        threshold = np.median(np.abs(deriv)) * params['threshold_multiplier']
        peaks, _ = find_peaks(deriv, height=threshold)
        
        return peaks
    
    def _algorithm_3_adaptive(self, signal: np.ndarray, multichannel: np.ndarray, **kwargs) -> np.ndarray:
        """Algorithm 3: Adaptive threshold detection."""
        # Apply high-pass filter
        cutoff = data_processing_config.HIGH_PASS_CUTOFF
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(2, normal_cutoff, btype='high', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        
        # Adaptive threshold based on local statistics
        window_size = int(2.0 * self.fs)  # 2 second window
        peaks = []
        
        for i in range(0, len(filtered_signal) - window_size, window_size // 2):
            window = filtered_signal[i:i + window_size]
            local_std = np.std(window)
            local_mean = np.mean(window)
            threshold = local_mean + 2 * local_std
            
            window_peaks, _ = find_peaks(window, height=threshold, distance=int(0.4 * self.fs))
            peaks.extend(window_peaks + i)
        
        return np.array(peaks)
    
    def _algorithm_4_tkeo(self, signal: np.ndarray, multichannel: np.ndarray, **kwargs) -> np.ndarray:
        """Algorithm 4: Teager-Kaiser Energy Operator (TKEO)."""
        params = step_detection_config.ALGORITHM_PARAMS['algo4_tkeo']
        params.update(kwargs)
        
        # Apply TKEO
        tkeo = signal[1:-1]**2 - signal[:-2] * signal[2:]
        
        # Find peaks in TKEO signal
        threshold = np.median(tkeo) * params['threshold_multiplier']
        peaks, _ = find_peaks(tkeo, height=threshold, distance=int(0.3 * self.fs))
        
        # Adjust for TKEO offset
        return peaks + 1
    
    def _algorithm_5_matched(self, signal: np.ndarray, multichannel: np.ndarray, **kwargs) -> np.ndarray:
        """Algorithm 5: Matched filter detection."""
        params = step_detection_config.ALGORITHM_PARAMS['algo5_matched']
        params.update(kwargs)
        
        template_length = int(params['template_length'] * self.fs)
        correlation_threshold = params['correlation_threshold']
        
        # Create a simple step template (this is a simplified version)
        template = np.zeros(template_length)
        template[template_length//4:3*template_length//4] = 1.0
        
        # Cross-correlation
        correlation = correlate(signal, template, mode='valid')
        
        # Find peaks above threshold
        peaks, _ = find_peaks(correlation, height=correlation_threshold, distance=int(0.4 * self.fs))
        
        return peaks
    
    def _algorithm_6_javascript(self, signal: np.ndarray, multichannel: np.ndarray, **kwargs) -> np.ndarray:
        """Algorithm 6: JavaScript port (from Accelerometer_Processing_Program.html)."""
        params = step_detection_config.ALGORITHM_PARAMS['algo2_derivative']
        params.update(kwargs)
        
        # Calculate derivative (same as algorithm 2)
        differential_duration = params['differential_duration']
        differential_points = int(round(differential_duration * self.fs))
        
        deriv = np.zeros_like(signal)
        if differential_points > 0:
            time_interval = differential_points / self.fs
            deriv[differential_points:] = (signal[differential_points:] - signal[:-differential_points]) / time_interval
        
        # Adaptive threshold
        threshold = np.median(np.abs(deriv)) * params['threshold_multiplier']
        
        # Find upward crossings
        crossings = np.where((deriv[:-1] < threshold) & (deriv[1:] >= threshold))[0] + 1
        
        # Refine step start by finding peak in subsequent window
        peak_indices = []
        for i in crossings:
            window_end = min(i + 100, len(signal))
            if window_end > i:
                max_in_window_idx = np.argmax(signal[i:window_end])
                peak_indices.append(i + max_in_window_idx)
        
        # Filter out peaks that are too close
        unique_peaks = []
        last_peak = -np.inf
        for peak_idx in peak_indices:
            if peak_idx - last_peak > 10:
                unique_peaks.append(peak_idx)
                last_peak = peak_idx
        
        return np.array(unique_peaks)
    
    def _algorithm_7_force_derivative(self, signal: np.ndarray, multichannel: np.ndarray, **kwargs) -> np.ndarray:
        """
        Algorithm 7: Improved Force derivative detection.
        
        This algorithm detects steps by finding peaks in the force derivative (gradient).
        Steps typically show up as rapid positive changes (rising edges) in force.
        
        Improvements:
        1. Apply smoothing to reduce noise in derivative
        2. Use absolute value to catch both positive and negative changes
        3. More adaptive threshold based on median absolute deviation (MAD)
        4. Configurable parameters
        """
        if multichannel.shape[1] < 2:
            return np.array([])
        
        force_gradient = multichannel[:, 1]
        
        # Apply Savitzky-Golay filter to smooth the derivative signal
        # This reduces noise while preserving step transitions
        window_length = min(11, len(force_gradient) if len(force_gradient) % 2 == 1 else len(force_gradient) - 1)
        if window_length >= 5:
            force_gradient_smooth = savgol_filter(force_gradient, window_length, polyorder=2)
        else:
            force_gradient_smooth = force_gradient
        
        # Use absolute value to detect both positive and negative changes
        force_gradient_abs = np.abs(force_gradient_smooth)
        
        # More robust threshold using Median Absolute Deviation (MAD)
        # This is less sensitive to outliers than std
        median = np.median(force_gradient_abs)
        mad = np.median(np.abs(force_gradient_abs - median))
        threshold = median + 3.0 * mad  # 3 MAD is typically good for outlier detection
        
        # Alternative: Use a percentile-based threshold (e.g., top 10% of values)
        percentile_threshold = np.percentile(force_gradient_abs, 90)
        threshold = max(threshold, percentile_threshold)
        
        # Find peaks with minimum distance between steps
        min_distance = int(0.3 * self.fs)  # Reduced from 0.4s to catch faster steps
        peaks, properties = find_peaks(
            force_gradient_abs, 
            height=threshold, 
            distance=min_distance,
            prominence=threshold * 0.5  # Require some prominence to avoid noise
        )
        
        return peaks
    
    def _algorithm_8_min_threshold(self, signal: np.ndarray, multichannel: np.ndarray, **kwargs) -> np.ndarray:
        """Algorithm 8: Minimum threshold detection."""
        # Use a very low threshold
        min_height = np.min(signal) + 0.1 * (np.max(signal) - np.min(signal))
        peaks, _ = find_peaks(signal, height=min_height, distance=int(0.3 * self.fs))
        
        return peaks
    
    def compare_algorithms(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        ground_truth: Optional[np.ndarray] = None,
        algorithms: Optional[List[StepDetectionAlgorithm]] = None
    ) -> Dict[StepDetectionAlgorithm, StepDetectionResult]:
        """
        Compare multiple step detection algorithms.
        
        Args:
            data: Input data
            ground_truth: Ground truth step times (optional)
            algorithms: List of algorithms to compare (default: all available)
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        if algorithms is None:
            algorithms = [algo for algo in StepDetectionAlgorithm if algo != StepDetectionAlgorithm.DEEP_LEARNING]
        
        results = {}
        for algorithm in algorithms:
            try:
                result = self.detect_steps(data, algorithm)
                results[algorithm] = result
            except Exception as e:
                print(f"Error running {algorithm}: {e}")
                continue
        
        return results
