"""
EMG Data Visualization Module

Provides clean, modular visualization capabilities for EMG data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Optional, Union
from emg_parser import EMGData


class EMGVisualizer:
    """
    Visualization utilities for EMG data.
    
    Provides both static matplotlib plots and interactive Plotly visualizations
    for EMG analysis results.
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
    
    def plot_muscle_groups(
        self, 
        emg_data: EMGData, 
        trial_name: str, 
        output_dir: Path,
        muscle_groups: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        Create comprehensive EMG visualization for muscle groups.
        
        Args:
            emg_data: Parsed EMG data
            trial_name: Name of the trial
            output_dir: Directory to save the plot
            muscle_groups: Optional dict mapping group names to muscle lists
        """
        if muscle_groups is None:
            muscle_groups = {
                'Upper Body': ['Pec_mV', 'AD_mV', 'LD_mV', 'PD_mV'],
                'Arms': ['Tric_mV', 'forearm_mV'],
                'Core': ['Lat_mV']
            }
        
        # Filter muscle groups to available muscles
        available_muscles = emg_data.get_available_muscles()
        filtered_groups = {}
        for group_name, muscles in muscle_groups.items():
            available_in_group = [m for m in muscles if m in available_muscles]
            if available_in_group:
                filtered_groups[group_name] = available_in_group
        
        if not filtered_groups:
            print(f"No muscle groups found for {trial_name}")
            return
        
        # Create figure with subplots
        n_groups = len(filtered_groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(15, 4 * n_groups))
        
        if n_groups == 1:
            axes = [axes]
        
        # Plot each muscle group
        for i, (group_name, muscles) in enumerate(filtered_groups.items()):
            ax = axes[i]
            
            for muscle in muscles:
                if muscle in emg_data.data.columns:
                    abs_values = np.abs(emg_data.data[muscle].values)
                    ax.plot(emg_data.data['time_s'], abs_values, 
                           label=muscle.replace('_mV', ''), linewidth=1.5, alpha=0.8)
            
            ax.set_title(f'{group_name} - {trial_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('EMG Amplitude (mV)', fontsize=12)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_dir / f'{trial_name}_emg_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_emg_with_steps(
        self,
        emg_data: EMGData,
        step_times: np.ndarray,
        trial_name: str,
        output_dir: Path,
        muscle: str = 'forearm',
        processing_info: Optional[Dict] = None
    ) -> None:
        """
        Create interactive Plotly visualization of EMG signal with step markers.
        
        Args:
            emg_data: Parsed EMG data
            step_times: Detected step times
            trial_name: Name of the trial
            output_dir: Directory to save the plot
            muscle: Muscle to plot (default: 'forearm')
            processing_info: Optional processing information from gait analyzer
        """
        muscle_col = f"{muscle}_mV"
        if muscle_col not in emg_data.data.columns:
            print(f"Muscle {muscle} not found in data")
            return
        
        # Create subplots for robust processing pipeline
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('EMG Signal Processing Pipeline', 'EMG Envelope', 'Normalized Envelope with Peaks'),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Get time signal
        time_signal = emg_data.data['time_s'].values
        
        if processing_info is not None:
            # Use downsampled data for visualization
            ds_time = processing_info['time_s']
            ds_env = processing_info['envelope_ds']
            ds_thr = processing_info['thr_ds']
            ds_height = processing_info.get('height_ds', ds_thr)  # hybrid threshold
            
            # Plot 1: Signal processing pipeline (sample of original data)
            if 'cleaned_emg_sample' in processing_info:
                sample_time = time_signal[::10]  # Match the sampling
                fig.add_trace(go.Scatter(
                    x=sample_time,
                    y=processing_info['cleaned_emg_sample'],
                    mode='lines',
                    name='Cleaned EMG (sample)',
                    line=dict(color='blue', width=1.5)
                ), row=1, col=1)
            
            # Plot 2: Downsampled EMG Envelope
            fig.add_trace(go.Scatter(
                x=ds_time,
                y=ds_env,
                mode='lines',
                name='EMG Envelope (20Hz)',
                line=dict(color='darkgreen', width=2)
            ), row=2, col=1)
            
            # Plot 3: Downsampled envelope with adaptive threshold
            fig.add_trace(go.Scatter(
                x=ds_time,
                y=ds_env,
                mode='lines',
                name='Envelope + Threshold',
                line=dict(color='orange', width=2)
            ), row=3, col=1)
            
            # Add adaptive threshold line
            fig.add_trace(go.Scatter(
                x=ds_time,
                y=ds_thr,
                mode='lines',
                name='Adaptive Threshold',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.7
            ), row=3, col=1)
            
            # Add hybrid threshold line (if different from adaptive)
            if 'height_ds' in processing_info and not np.array_equal(ds_thr, ds_height):
                fig.add_trace(go.Scatter(
                    x=ds_time,
                    y=ds_height,
                    mode='lines',
                    name='Hybrid Threshold',
                    line=dict(color='purple', width=1, dash='dot'),
                    opacity=0.8
                ), row=3, col=1)
            
            # Add step markers on all plots
            if len(step_times) > 0:
                # Get values at step times for downsampled signals
                step_env_values = np.interp(step_times, ds_time, ds_env)
                step_thr_values = np.interp(step_times, ds_time, ds_thr)
                
                # Plot 2 markers (envelope)
                fig.add_trace(go.Scatter(
                    x=step_times,
                    y=step_env_values,
                    mode='markers',
                    name='Detected Steps',
                    marker=dict(
                        symbol='star',
                        size=10,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    showlegend=False
                ), row=2, col=1)
                
                # Plot 3 markers (envelope + threshold)
                fig.add_trace(go.Scatter(
                    x=step_times,
                    y=step_env_values,
                    mode='markers',
                    name='Step Peaks',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    showlegend=False
                ), row=3, col=1)
        else:
            # Fallback to simple plot
            fig.add_trace(go.Scatter(
                x=time_signal,
                y=emg_data.data[muscle_col],
                mode='lines',
                name=f'{muscle.title()} EMG',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            if len(step_times) > 0:
                step_values = np.interp(step_times, time_signal, emg_data.data[muscle_col])
                fig.add_trace(go.Scatter(
                    x=step_times,
                    y=step_values,
                    mode='markers',
                    name='Detected Steps',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    showlegend=False
                ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'Robust EMG Step Detection Pipeline - {trial_name}',
            hovermode='x unified',
            showlegend=True,
            width=1400,
            height=1000,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="EMG Amplitude (mV)", row=1, col=1)
        fig.update_yaxes(title_text="Envelope (mV)", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Envelope (z-score)", row=3, col=1)
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Save as HTML
        output_file = output_dir / f'{trial_name}_emg_steps_interactive.html'
        fig.write_html(str(output_file))
        print(f"Saved interactive plot: {output_file}")
    
    def plot_gait_metrics_summary(
        self,
        metrics_df: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """
        Create summary plots of gait metrics across trials.
        
        Args:
            metrics_df: DataFrame with gait metrics for all trials
            output_dir: Directory to save the plots
        """
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gait Metrics Summary', fontsize=16, fontweight='bold')
        
        # Step count
        axes[0, 0].bar(metrics_df['trial_name'], metrics_df['step_count'])
        axes[0, 0].set_title('Step Count by Trial')
        axes[0, 0].set_ylabel('Number of Steps')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Step frequency
        axes[0, 1].bar(metrics_df['trial_name'], metrics_df['step_frequency_hz'])
        axes[0, 1].set_title('Step Frequency by Trial')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Step regularity
        axes[1, 0].bar(metrics_df['trial_name'], metrics_df['step_regularity_cv'])
        axes[1, 0].set_title('Step Regularity (CV) by Trial')
        axes[1, 0].set_ylabel('Coefficient of Variation')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # EMG amplitude
        axes[1, 1].bar(metrics_df['trial_name'], metrics_df['mean_emg_amplitude_mv'])
        axes[1, 1].set_title('Mean EMG Amplitude by Trial')
        axes[1, 1].set_ylabel('EMG Amplitude (mV)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_dir / 'gait_metrics_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot: {output_file}")
        plt.close()
