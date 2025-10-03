"""
Gait analysis algorithms for crutch movement analysis.

This module provides gait analysis capabilities including step variance,
stability metrics, and other gait-related measurements.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class GaitMetrics:
    """Container for gait analysis metrics."""
    
    step_count: int
    step_variance: Optional[float] = None
    y_change: Optional[float] = None
    y_total: Optional[float] = None
    rms_load_cell_force: Optional[float] = None
    step_frequency: Optional[float] = None
    step_regularity: Optional[float] = None
    step_symmetry: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling NaN values."""
        def clean_value(val):
            """Convert NaN/inf to None for JSON compatibility."""
            if val is None:
                return None
            if isinstance(val, (int, str, bool)):
                return val
            # Handle numpy and Python float NaN/inf
            try:
                if np.isnan(val) or np.isinf(val):
                    return None
            except (TypeError, ValueError):
                pass
            return val
        
        return {
            "step_count": clean_value(self.step_count),
            "step_variance": clean_value(self.step_variance),
            "y_change": clean_value(self.y_change),
            "y_total": clean_value(self.y_total),
            "rms_load_cell_force": clean_value(self.rms_load_cell_force),
            "step_frequency": clean_value(self.step_frequency),
            "step_regularity": clean_value(self.step_regularity),
            "step_symmetry": clean_value(self.step_symmetry),
        }


class GaitAnalyzer:
    """
    Gait analysis for crutch movement data.
    
    This class provides various gait analysis metrics that can be used
    for optimization objectives like stability and effort.
    """
    
    def __init__(self, sampling_frequency: float = 100.0):
        """
        Initialize the gait analyzer.
        
        Args:
            sampling_frequency: Sampling frequency in Hz
        """
        self.fs = sampling_frequency
    
    def analyze_gait(
        self, 
        data: pd.DataFrame, 
        step_times: np.ndarray,
        force_data: Optional[np.ndarray] = None
    ) -> GaitMetrics:
        """
        Analyze gait metrics from accelerometer data and step times.
        
        Args:
            data: Accelerometer data DataFrame
            step_times: Array of step timestamps
            force_data: Optional force sensor data
            
        Returns:
            GaitMetrics object with calculated metrics
        """
        step_count = len(step_times)
        
        if step_count < 2:
            return GaitMetrics(step_count=step_count)
        
        # Calculate step intervals
        step_intervals = np.diff(step_times)
        
        # Step variance (stability metric)
        step_variance = np.var(step_intervals) if len(step_intervals) > 1 else None
        
        # Step frequency
        step_frequency = 1.0 / np.mean(step_intervals) if len(step_intervals) > 0 else None
        
        # Step regularity (coefficient of variation)
        step_regularity = (
            np.std(step_intervals) / np.mean(step_intervals) 
            if len(step_intervals) > 1 and np.mean(step_intervals) > 0 
            else None
        )
        
        # Y-axis metrics (vertical movement)
        y_change, y_total = self._calculate_y_metrics(data, step_times)
        
        # RMS load cell force
        rms_load_cell_force = self._calculate_rms_force(force_data) if force_data is not None else None
        
        # Step symmetry (if we have enough steps)
        step_symmetry = self._calculate_step_symmetry(step_intervals) if len(step_intervals) > 3 else None
        
        return GaitMetrics(
            step_count=step_count,
            step_variance=step_variance,
            y_change=y_change,
            y_total=y_total,
            rms_load_cell_force=rms_load_cell_force,
            step_frequency=step_frequency,
            step_regularity=step_regularity,
            step_symmetry=step_symmetry
        )
    
    def _calculate_y_metrics(self, data: pd.DataFrame, step_times: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate Y-axis (vertical) movement metrics.
        
        Args:
            data: Accelerometer data
            step_times: Step timestamps
            
        Returns:
            Tuple of (y_change, y_total)
        """
        if 'acc_z_data' not in data.columns:
            return None, None
        
        try:
            # Get accelerometer data
            acc_z = data['acc_z_data'].values
            
            # Calculate Y_change: sum of absolute differences in Z acceleration
            y_change = np.sum(np.abs(np.diff(acc_z)))
            
            # Calculate Y_total: total variation in Z acceleration
            y_total = np.sum(np.abs(acc_z - np.mean(acc_z)))
            
            return float(y_change), float(y_total)
            
        except Exception:
            return None, None
    
    def _calculate_rms_force(self, force_data: np.ndarray) -> Optional[float]:
        """
        Calculate RMS of force data.
        
        Args:
            force_data: Force sensor data
            
        Returns:
            RMS force value
        """
        if force_data is None or len(force_data) == 0:
            return None
        
        try:
            rms = np.sqrt(np.mean(force_data**2))
            return float(rms)
        except Exception:
            return None
    
    def _calculate_step_symmetry(self, step_intervals: np.ndarray) -> Optional[float]:
        """
        Calculate step symmetry metric.
        
        Args:
            step_intervals: Array of step intervals
            
        Returns:
            Step symmetry value (0 = perfect symmetry, 1 = maximum asymmetry)
        """
        if len(step_intervals) < 4:
            return None
        
        try:
            # Split intervals into left and right steps (assuming alternating)
            left_steps = step_intervals[::2]
            right_steps = step_intervals[1::2]
            
            if len(left_steps) == 0 or len(right_steps) == 0:
                return None
            
            # Calculate mean intervals for each side
            left_mean = np.mean(left_steps)
            right_mean = np.mean(right_steps)
            
            # Calculate symmetry index
            symmetry = abs(left_mean - right_mean) / (left_mean + right_mean)
            
            return float(symmetry)
            
        except Exception:
            return None
    
    def calculate_cycle_variance(self, step_times: np.ndarray) -> Optional[float]:
        """
        Calculate cycle variance for stability assessment.
        
        Args:
            step_times: Array of step timestamps
            
        Returns:
            Cycle variance value
        """
        if len(step_times) < 3:
            return None
        
        try:
            # Calculate step intervals
            step_intervals = np.diff(step_times)
            
            # Calculate cycle times (time between every other step)
            cycle_times = []
            for i in range(0, len(step_intervals) - 1, 2):
                cycle_time = step_intervals[i] + step_intervals[i + 1]
                cycle_times.append(cycle_time)
            
            if len(cycle_times) < 2:
                return None
            
            # Calculate variance of cycle times
            cycle_variance = np.var(cycle_times)
            
            return float(cycle_variance)
            
        except Exception:
            return None
    
    def detect_gait_events(
        self, 
        data: pd.DataFrame, 
        step_times: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Detect various gait events from step times.
        
        Args:
            data: Accelerometer data
            step_times: Step timestamps
            
        Returns:
            Dictionary of detected events
        """
        events = {}
        
        if len(step_times) < 2:
            return events
        
        # Calculate step intervals
        step_intervals = np.diff(step_times)
        
        # Detect irregular steps (outliers)
        if len(step_intervals) > 2:
            z_scores = np.abs(stats.zscore(step_intervals))
            irregular_steps = step_times[1:][z_scores > 2]  # Steps with z-score > 2
            events['irregular_steps'] = irregular_steps
        
        # Detect very fast steps (potential double steps)
        if len(step_intervals) > 0:
            mean_interval = np.mean(step_intervals)
            fast_steps = step_times[1:][step_intervals < 0.5 * mean_interval]
            events['fast_steps'] = fast_steps
        
        # Detect very slow steps (potential hesitations)
        if len(step_intervals) > 0:
            mean_interval = np.mean(step_intervals)
            slow_steps = step_times[1:][step_intervals > 2.0 * mean_interval]
            events['slow_steps'] = slow_steps
        
        return events
    
    def calculate_stability_score(
        self, 
        step_times: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate overall stability score from gait metrics.
        
        Args:
            step_times: Step timestamps
            weights: Optional weights for different metrics
            
        Returns:
            Stability score (0-1, higher is more stable)
        """
        if len(step_times) < 2:
            return 0.0
        
        # Default weights
        if weights is None:
            weights = {
                'step_variance': 0.4,
                'step_regularity': 0.3,
                'step_symmetry': 0.3
            }
        
        # Calculate metrics
        step_intervals = np.diff(step_times)
        
        # Normalize step variance (lower is better)
        step_variance = np.var(step_intervals) if len(step_intervals) > 1 else 1.0
        variance_score = max(0, 1 - (step_variance / 0.1))  # Normalize to 0.1 max variance
        
        # Normalize step regularity (lower is better)
        if len(step_intervals) > 1 and np.mean(step_intervals) > 0:
            regularity = np.std(step_intervals) / np.mean(step_intervals)
            regularity_score = max(0, 1 - (regularity / 0.5))  # Normalize to 0.5 max regularity
        else:
            regularity_score = 0.0
        
        # Calculate step symmetry
        symmetry = self._calculate_step_symmetry(step_intervals)
        symmetry_score = max(0, 1 - symmetry) if symmetry is not None else 0.5
        
        # Calculate weighted stability score
        stability_score = (
            weights['step_variance'] * variance_score +
            weights['step_regularity'] * regularity_score +
            weights['step_symmetry'] * symmetry_score
        )
        
        return float(stability_score)
