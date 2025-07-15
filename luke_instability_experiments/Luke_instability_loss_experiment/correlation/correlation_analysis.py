#!/usr/bin/env python3
"""
Correlation Analysis for Crutch Instability
Analyzes BIN files to find correlations between cycle duration variance and IMU metrics.

This script:
1. Loads 5 BIN files from Luke 7:7
2. Calculates cycle duration variance for each trial
3. Computes various IMU metrics (RMS, variance, etc.) for each axis
4. Analyzes correlations between cycle duration variance and IMU metrics
5. Identifies which axes/combinations best predict instability
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import processing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our data processing modules
from src.data_analysis import compute_cycle_std, detect_steps_unsupervised, _postprocess_steps

class BINDataProcessor:
    """Process CSV files and extract IMU metrics for correlation analysis."""
    
    def __init__(self, data_directory="LukeCorrelation"):
        self.data_directory = Path(data_directory)
        self.results = []
        
    def load_csv_file(self, filepath):
        """Load CSV file and extract IMU data."""
        try:
            # Load CSV file
            df = pd.read_csv(filepath)
            
            # Standardize column names for compatibility
            df.rename(columns={
                "acc_x_time": "timestamp"
            }, inplace=True, errors='ignore')
            
            # Convert timestamp to seconds for analysis
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
            
            return df
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def detect_steps_from_force(self, df):
        """Detect steps using force signal."""
        # Use force signal for step detection
        time_signal = df['timestamp'].values
        fs = 1 / np.mean(np.diff(time_signal))
        
        # Detect steps using existing algorithm
        step_times = detect_steps_unsupervised(df['force'].values, time_signal, fs)
        final_steps = _postprocess_steps(step_times)
        
        return final_steps
    
    def compute_imu_metrics(self, df, step_times):
        """Compute various IMU metrics for correlation analysis."""
        if len(step_times) < 2:
            return None
        
        metrics = {}
        
        # 1. Cycle duration variance (our target metric)
        cycle_durations = np.diff(step_times)
        metrics['cycle_duration_variance'] = np.var(cycle_durations)
        metrics['cycle_duration_std'] = np.std(cycle_durations)
        metrics['cycle_duration_mean'] = np.mean(cycle_durations)
        
        # 2. Per-cycle metrics for each axis (no max/min)
        times = df['timestamp'].values
        axes = ['acc_x_data', 'acc_y_data', 'acc_z_data', 'gyro_x_data', 'gyro_y_data', 'gyro_z_data']
        for axis in axes:
            signal = df[axis].values
            step_indices = np.searchsorted(times, step_times)
            cycle_rms = []
            cycle_variance = []
            cycle_range = []
            for i in range(len(step_indices) - 1):
                start, end = step_indices[i], step_indices[i+1]
                cycle_data = signal[start:end]
                if len(cycle_data) > 0:
                    cycle_rms.append(np.sqrt(np.mean(cycle_data**2)))
                    cycle_variance.append(np.var(cycle_data))
                    cycle_range.append(np.max(cycle_data) - np.min(cycle_data))
            if cycle_rms:
                metrics[f'{axis}_rms_mean'] = np.mean(cycle_rms)
                metrics[f'{axis}_rms_variance'] = np.var(cycle_rms)
                metrics[f'{axis}_variance_mean'] = np.mean(cycle_variance)
                metrics[f'{axis}_variance_variance'] = np.var(cycle_variance)
                metrics[f'{axis}_range_mean'] = np.mean(cycle_range)
        # 3. Overall signal metrics (no max/min)
        for axis in axes:
            signal = df[axis].values
            metrics[f'{axis}_overall_rms'] = np.sqrt(np.mean(signal**2))
            metrics[f'{axis}_overall_variance'] = np.var(signal)
            metrics[f'{axis}_overall_range'] = np.max(signal) - np.min(signal)
        # 4. Combined metrics (magnitudes)
        acc_mag = np.sqrt(df['acc_x_data']**2 + df['acc_y_data']**2 + df['acc_z_data']**2)
        gyro_mag = np.sqrt(df['gyro_x_data']**2 + df['gyro_y_data']**2 + df['gyro_z_data']**2)
        metrics['acc_magnitude_rms'] = np.sqrt(np.mean(acc_mag**2))
        metrics['acc_magnitude_variance'] = np.var(acc_mag)
        metrics['gyro_magnitude_rms'] = np.sqrt(np.mean(gyro_mag**2))
        metrics['gyro_magnitude_variance'] = np.var(gyro_mag)
        # 5. Cross-axis correlations
        metrics['acc_xy_correlation'] = np.corrcoef(df['acc_x_data'], df['acc_y_data'])[0,1]
        metrics['acc_xz_correlation'] = np.corrcoef(df['acc_x_data'], df['acc_z_data'])[0,1]
        metrics['acc_yz_correlation'] = np.corrcoef(df['acc_y_data'], df['acc_z_data'])[0,1]
        metrics['gyro_xy_correlation'] = np.corrcoef(df['gyro_x_data'], df['gyro_y_data'])[0,1]
        metrics['gyro_xz_correlation'] = np.corrcoef(df['gyro_x_data'], df['gyro_z_data'])[0,1]
        metrics['gyro_yz_correlation'] = np.corrcoef(df['gyro_y_data'], df['gyro_z_data'])[0,1]
        return metrics
    
    def process_all_files(self):
        """Process all CSV files in the directory."""
        # Look for both .csv and .CSV files
        csv_files = list(self.data_directory.glob("*.csv")) + list(self.data_directory.glob("*.CSV"))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_directory}")
            return
        
        print(f"Found {len(csv_files)} CSV files")
        
        for i, csv_file in enumerate(csv_files):
            print(f"\nProcessing {csv_file.name} ({i+1}/{len(csv_files)})...")
            
            # Load data
            df = self.load_csv_file(csv_file)
            if df is None:
                continue
            
            # Detect steps
            step_times = self.detect_steps_from_force(df)
            if len(step_times) < 2:
                print(f"  Insufficient steps detected: {len(step_times)}")
                continue
            
            # Compute metrics
            metrics = self.compute_imu_metrics(df, step_times)
            if metrics is None:
                continue
            
            # Add file info
            metrics['filename'] = csv_file.name
            metrics['n_steps'] = len(step_times)
            metrics['trial_duration'] = df['timestamp'].max() - df['timestamp'].min()
            
            self.results.append(metrics)
            print(f"  Processed: {len(step_times)} steps, cycle variance: {metrics['cycle_duration_variance']:.4f}")
        
        print(f"\nSuccessfully processed {len(self.results)} files")
    
    def analyze_correlations(self):
        """Analyze correlations between cycle duration variance and other metrics."""
        if not self.results:
            print("No data to analyze")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Get all metric columns (excluding metadata)
        metric_cols = [col for col in df.columns if col not in ['filename', 'n_steps', 'trial_duration']]
        
        # Calculate correlations with cycle duration variance
        correlations = []
        for col in metric_cols:
            if col != 'cycle_duration_variance':
                corr, p_value = stats.pearsonr(df['cycle_duration_variance'], df[col])
                correlations.append({
                    'metric': col,
                    'correlation': corr,
                    'p_value': p_value,
                    'abs_correlation': abs(corr)
                })
        
        # Sort by absolute correlation
        correlations_df = pd.DataFrame(correlations)
        correlations_df = correlations_df.sort_values('abs_correlation', ascending=False)
        
        return df, correlations_df
    
    def create_visualizations(self, df, correlations_df):
        """Create correlation visualizations."""
        # 1. Correlation heatmap
        plt.figure(figsize=(15, 10))
        
        # Get top 20 correlated metrics
        top_metrics = correlations_df.head(20)['metric'].tolist()
        top_metrics.append('cycle_duration_variance')
        
        # Create correlation matrix
        corr_matrix = df[top_metrics].corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f', annot_kws={'size': 8}, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix: Top 20 Metrics vs Cycle Duration Variance', fontsize=12)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Top correlations scatter plots
        top_n = min(8, len(correlations_df))
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(top_n):
            metric = correlations_df.iloc[i]['metric']
            corr = correlations_df.iloc[i]['correlation']
            
            axes[i].scatter(df[metric], df['cycle_duration_variance'], alpha=0.7)
            axes[i].set_xlabel(metric)
            axes[i].set_ylabel('Cycle Duration Variance')
            axes[i].set_title(f'{metric}\nr = {corr:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('top_correlations_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Bar plot of top correlations
        plt.figure(figsize=(12, 8))
        top_10 = correlations_df.head(10)
        
        colors = ['red' if x < 0 else 'blue' for x in top_10['correlation']]
        bars = plt.barh(range(len(top_10)), top_10['correlation'], color=colors)
        
        plt.yticks(range(len(top_10)), top_10['metric'])
        plt.xlabel('Correlation with Cycle Duration Variance')
        plt.title('Top 10 Correlations with Cycle Duration Variance')
        plt.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, top_10['correlation'])):
            plt.text(bar.get_width() + (0.01 if corr > 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', ha='left' if corr > 0 else 'right', va='center')
        
        plt.tight_layout()
        plt.savefig('top_correlations_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cycle_variance_by_file(self, df):
        """Plot cycle duration variance for each CSV file."""
        # Remove .csv from filenames and sort by numeric part
        df = df.copy()
        df['trial_num'] = df['filename'].str.replace('.csv', '', regex=False).astype(float)
        df = df.sort_values('trial_num')
        labels = df['trial_num'].astype(str)
        
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        bars = plt.bar(range(len(df)), df['cycle_duration_variance'], 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add file names as x-axis labels (no .csv)
        plt.xticks(range(len(df)), labels, rotation=45, ha='right')
        plt.ylabel('Cycle Duration Variance (s²)')
        plt.title('Cycle Duration Variance by Trial')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, variance) in enumerate(zip(bars, df['cycle_duration_variance'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{variance:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('cycle_variance_by_file.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also create a scatter plot of cycle variance vs number of steps
        plt.figure(figsize=(10, 6))
        plt.scatter(df['n_steps'], df['cycle_duration_variance'], s=100, alpha=0.7)
        
        # Add file names as labels
        for i, row in df.iterrows():
            plt.annotate(str(row['trial_num']), (row['n_steps'], row['cycle_duration_variance']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Number of Steps')
        plt.ylabel('Cycle Duration Variance (s²)')
        plt.title('Cycle Duration Variance vs Number of Steps')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cycle_variance_vs_steps.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, df, correlations_df):
        """Save results to CSV files."""
        df.to_csv('trial_metrics.csv', index=False)
        correlations_df.to_csv('correlation_analysis.csv', index=False)
        
        print("\nResults saved:")
        print("  - trial_metrics.csv: All computed metrics for each trial")
        print("  - correlation_analysis.csv: Correlation analysis results")
        print("  - cycle_variance_by_file.png: Cycle variance for each CSV file")
        print("  - cycle_variance_vs_steps.png: Cycle variance vs number of steps")
        if len(correlations_df) > 0:
            print("  - correlation_heatmap.png: Correlation heatmap")
            print("  - top_correlations_scatter.png: Scatter plots of top correlations")
            print("  - top_correlations_bar.png: Bar plot of top correlations")

def main():
    """Main execution function."""
    print("=== Crutch Instability Correlation Analysis ===")
    print("This script analyzes correlations between cycle duration variance and IMU metrics.")
    
    # Initialize processor
    processor = BINDataProcessor()
    
    # Process all CSV files
    processor.process_all_files()
    
    if not processor.results:
        print("No data processed. Exiting.")
        return
    
    # Analyze correlations
    print("\nAnalyzing correlations...")
    df, correlations_df = processor.analyze_correlations()
    
    # Display top correlations
    print("\nTop 10 correlations with cycle duration variance:")
    print(correlations_df.head(10)[['metric', 'correlation', 'p_value']].to_string(index=False))
    
    # Plot cycle variance by file first
    print("\nCreating cycle variance visualizations...")
    processor.plot_cycle_variance_by_file(df)
    
    # Create correlation visualizations if we have multiple files
    if len(df) > 1:
        print("\nCreating correlation visualizations...")
        processor.create_visualizations(df, correlations_df)
    
    # Save results
    processor.save_results(df, correlations_df)
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Processed {len(df)} trials")
    print(f"Cycle duration variance range: {df['cycle_duration_variance'].min():.4f} - {df['cycle_duration_variance'].max():.4f}")
    print(f"Average cycle duration variance: {df['cycle_duration_variance'].mean():.4f}")
    
    if len(correlations_df) > 0:
        print(f"Analyzed {len(correlations_df)} metrics")
        
        # Find best predictors
        significant_correlations = correlations_df[correlations_df['p_value'] < 0.05]
        print(f"Found {len(significant_correlations)} significantly correlated metrics (p < 0.05)")
        
        if len(significant_correlations) > 0:
            best_predictor = significant_correlations.iloc[0]
            print(f"Best predictor: {best_predictor['metric']} (r = {best_predictor['correlation']:.3f})")
    else:
        print("Need at least 2 trials to analyze correlations")

if __name__ == "__main__":
    main() 