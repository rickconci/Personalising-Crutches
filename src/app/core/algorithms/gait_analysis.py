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
from pydantic import BaseModel, Field


class GaitMetrics(BaseModel):
    """Container for gait analysis metrics."""
    
    step_count: int = Field(..., description="Number of steps taken")
    step_variance: float = Field(..., description="Variance of step intervals")

    # Movement metrics
    x_change: float = Field(..., description="Change in forward/backward position")
    x_total: float = Field(..., description="Total forward/backward movement")
    y_change: float = Field(..., description="Change in lateral position")
    y_total: float = Field(..., description="Total lateral movement")
    z_change: float = Field(..., description="Change in vertical position")
    z_total: float = Field(..., description="Total vertical movement")

    # Force metrics
    rms_load_cell_force: float = Field(..., description="RMS of load cell force")
    step_frequency: float = Field(..., description="Frequency of steps")
    step_regularity: float = Field(..., description="Regularity of steps")
    step_symmetry: float = Field(..., description="Symmetry of steps")
    
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
            "x_change": clean_value(self.x_change),
            "x_total": clean_value(self.x_total),
            "z_change": clean_value(self.z_change),
            "z_total": clean_value(self.z_total),
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
        
        # Movement metrics
        metrics = self._calculate_metrics(data, step_times)
        
        # RMS load cell force
        rms_load_cell_force = self._calculate_rms_force(force_data) if force_data is not None else None
        
        # Step symmetry (if we have enough steps)
        step_symmetry = self._calculate_step_symmetry(step_intervals) if len(step_intervals) > 3 else None
        
        return GaitMetrics(
            step_count=step_count,
            step_variance=step_variance,
            y_change=metrics['acc_y_data']['change'],
            y_total=metrics['acc_y_data']['total'],
            x_change=metrics['acc_x_data']['change'],
            x_total=metrics['acc_x_data']['total'],
            z_change=metrics['acc_z_data']['change'],
            z_total=metrics['acc_z_data']['total'],
            rms_load_cell_force=rms_load_cell_force,
            step_frequency=step_frequency,
            step_regularity=step_regularity,
            step_symmetry=step_symmetry
        )
    
    def _calculate_metrics(self, data: pd.DataFrame, step_times: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate movement metrics.
        
        Args:
            data: Accelerometer data
            step_times: Step timestamps
            
        Returns:
            Dictionary of movement metrics
        """
        metrics = {
            'acc_x_data': [],
            'acc_y_data': [],
            'acc_z_data': []
        }
        for col in ['acc_x_data', 'acc_y_data', 'acc_z_data']:
            if col not in data.columns:
                return None
            acc_data = data[col].values
            acc_change = np.sum(np.abs(np.diff(acc_data)))
            acc_total = np.sum(np.abs(acc_data - np.mean(acc_data)))
            metrics[col] = {
                'change': acc_change,
                'total': acc_total
            }
        return metrics
        
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
    