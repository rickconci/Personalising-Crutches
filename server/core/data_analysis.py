import os
import sys
import pandas as pd
import numpy as np
from scipy.signal import lfilter
import plotly.graph_objs as go
import asyncio
import shutil

# Local application imports
from . import config
from .gait_analysis import (
    detect_steps_unsupervised, 
    _postprocess_steps,
    calculate_gait_metrics_from_steps,
    compute_cycle_variance,
    compute_mean_y_rms_per_cycle
)
from .data_processing_metabolic import get_metabolic_cost_from_excel

class TrialAnalyzer:
    """Handles the analysis of data from a single trial."""

    def __init__(self, raw_data_path: str):
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
        self.raw_data_path = raw_data_path
        self.base_path = os.path.splitext(raw_data_path)[0]
        self.step_file_path = self.base_path + config.STEP_FILE_SUFFIX
        self.visualization_path = self.base_path + '_steps_visualization.html'

    def _create_and_save_plot(self, df: pd.DataFrame, steps: np.ndarray):
        fig = go.Figure()
        # Smoothed Accelerometer Signal
        if 'acc_x_z_smooth' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['acc_x_z_smooth'], mode='lines', name='Smoothed Accel Mag'))
        # Force Signal
        if 'force' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['force'], mode='lines', name='Force', yaxis='y2'))
        # Detected Steps
        if 'acc_x_z_smooth' in df.columns and len(steps) > 0:
            step_y_values = np.interp(steps, df['timestamp'], df['acc_x_z_smooth'])
            fig.add_trace(go.Scatter(x=steps, y=step_y_values, mode='markers', name='Detected Steps'))
        
        fig.update_layout(
            title_text=f'Step Detection: {os.path.basename(self.raw_data_path)}',
            xaxis_title='Time (s)',
            yaxis_title='Smoothed Accel Magnitude',
            yaxis2={'title': 'Force', 'overlaying': 'y', 'side': 'right'}
        )
        fig.write_html(self.visualization_path)
        print(f"Saved step detection visualization to {self.visualization_path}")

    def run_initial_analysis(self, visualize: bool = False) -> str:
        df = pd.read_csv(self.raw_data_path)
        df.rename(columns={"relative_time_ms": "timestamp", "accX": "acc_x_data", "accY": "acc_y_data"}, inplace=True)
        df['timestamp'] = (pd.to_numeric(df['timestamp'], errors='coerce') - df['timestamp'].iloc[0]) / 1000.0
        df.dropna(subset=['timestamp'], inplace=True)
        
        # Signal processing for visualization
        alpha = 25 / 100.0
        acc_x_smooth = lfilter([alpha], [1, -(1 - alpha)], df['acc_x_data'])
        acc_y_smooth = lfilter([alpha], [1, -(1 - alpha)], df['acc_y_data'])
        df['acc_x_z_smooth'] = acc_x_smooth**2 + acc_y_smooth**2
        
        fs = 1.0 / np.median(np.diff(df['timestamp'].values))
        raw_steps = detect_steps_unsupervised(df['force'].values, df['timestamp'].values, fs)
        final_steps = _postprocess_steps(raw_steps)
        
        if visualize:
            self._create_and_save_plot(df, final_steps)
            
        pd.DataFrame({'step_time': final_steps}).to_csv(self.step_file_path, index=False)
        print(f"Detected and saved {len(final_steps)} steps to {self.step_file_path}")
        return self.step_file_path

    def featurize_trial(self, step_file_path: str) -> dict:
        try:
            raw_df = pd.read_csv(self.raw_data_path)
            raw_df.rename(columns={"relative_time_ms": "timestamp", "accX": "acc_x_data", "accY": "acc_y_data"}, inplace=True)
            raw_df['timestamp'] = (raw_df['timestamp'] - raw_df['timestamp'].iloc[0]) / 1000.0

            steps_df = pd.read_csv(step_file_path)
            steps_df['step_time'] = pd.to_numeric(steps_df['step_time'], errors='coerce')
            steps_df.dropna(subset=['step_time'], inplace=True)
            step_times = steps_df['step_time'].values

            # Gait metrics from original `calculate_gait_metrics` function
            raw_df_sorted = raw_df.sort_values('timestamp').reset_index()
            merged = pd.merge_asof(left=steps_df.sort_values('step_time'), right=raw_df_sorted, left_on='step_time', right_on='timestamp', direction='nearest')
            step_indices = merged['index'].to_numpy()
            gait_metrics = calculate_gait_metrics_from_steps(raw_df, step_indices)
            
            # Instability and metabolic metrics
            cycle_duration_variance = compute_cycle_variance(raw_df, step_times)
            y_rms_loss = compute_mean_y_rms_per_cycle(raw_df, step_times)
            metabolic_cost_loss = get_metabolic_cost_from_excel(self.base_path)
            
            return {
                **gait_metrics,
                'cycle_duration_variance': cycle_duration_variance,
                'y_rms_loss': y_rms_loss,
                'metabolic_cost_loss': metabolic_cost_loss
            }
        except Exception as e:
            print(f"An error occurred during data featurization: {e}")
            return {}