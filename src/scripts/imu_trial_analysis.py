#!/usr/bin/env python3
"""
IMU Trial Data Analysis Script

This script analyzes IMU trial data from the MIH15 experiments, specifically:
1. Identifies trial files that are not opencap events
2. Plots accelerometer data (acc_x_time, acc_x_data, acc_z_data) using seaborn
3. Saves the plots to a designated output directory

Author: Research Team
Date: 2025-01-27
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('default')



class IMUTrialAnalyzer:
    """
    Analyzer for IMU trial data from MIH15 experiments.
    
    This class handles the identification of non-opencap trial files,
    data loading, visualization, and plot saving.
    """
    
    def __init__(self, trials_dir: Path, output_dir: Path, sampling_frequency: float = 200.0, geometry_mapping_path: Path = None) -> None:
        """
        Initialize the IMU Trial Analyzer.
        
        Args:
            trials_dir: Path to the trials directory containing CSV files
            output_dir: Path to save generated plots
            sampling_frequency: Sampling frequency in Hz (default: 200.0)
            geometry_mapping_path: Path to geometry ID to angles mapping JSON file
        """
        self.trials_dir = Path(trials_dir)
        self.output_dir = Path(output_dir)
        self.sampling_frequency = sampling_frequency
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load geometry mapping
        self.geometry_mapping = self._load_geometry_mapping(geometry_mapping_path)
        
        # Ensure trials directory exists
        if not self.trials_dir.exists():
            raise FileNotFoundError(f"Trials directory not found: {self.trials_dir}")
        
        logger.info(f"Initialized analyzer with trials_dir: {self.trials_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Sampling frequency: {self.sampling_frequency} Hz")
        logger.info(f"Geometry mapping loaded: {len(self.geometry_mapping)} geometries")
    
    def _load_geometry_mapping(self, geometry_mapping_path: Path) -> Dict[str, Dict[str, float]]:
        """
        Load geometry ID to angles mapping from JSON file.
        
        Args:
            geometry_mapping_path: Path to the geometry mapping JSON file
            
        Returns:
            Dictionary mapping geometry IDs to angle dictionaries
        """
        if geometry_mapping_path is None:
            # Default path
            geometry_mapping_path = Path('/Users/riccardoconci/Local_documents/!CURRENT_RESEARCH/Personalising-Crutches/DataAnalysis/results/geometry_id_to_angles.json')
        
        try:
            if geometry_mapping_path.exists():
                with open(geometry_mapping_path, 'r') as f:
                    mapping = json.load(f)
                logger.info(f"Loaded geometry mapping from {geometry_mapping_path}")
                return mapping
            else:
                logger.warning(f"Geometry mapping file not found: {geometry_mapping_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading geometry mapping: {str(e)}")
            return {}
    
    def _convert_geometry_name(self, file_name: str) -> str:
        """
        Convert file name with G ID to geometry angles format.
        
        Args:
            file_name: Original file name (e.g., "51_G8_111729")
            
        Returns:
            Converted file name with geometry angles (e.g., "51_105_125_0")
        """
        if not self.geometry_mapping:
            return file_name
        
        # Parse the file name to extract G ID or Control
        parts = file_name.split('_')
        g_id = None
        
        if len(parts) >= 2:
            # Look for G pattern in the parts
            for part in parts:
                if part.startswith('G') and part[1:].isdigit():
                    g_id = part
                    break
            
            # Check if it's a control trial
            if g_id is None and 'Control' in file_name:
                g_id = 'Control'
        
        if g_id is None:
            return file_name
        
        # Get geometry angles
        if g_id in self.geometry_mapping:
            angles = self.geometry_mapping[g_id]
            alpha = int(angles['alpha'])
            beta = int(angles['beta'])
            gamma = angles['gamma']
            
            # Format gamma: +9 -> p9, -9 -> m9, 0 -> 0
            if gamma > 0:
                gamma_str = f"p{int(gamma)}"
            elif gamma < 0:
                gamma_str = f"m{int(abs(gamma))}"
            else:
                gamma_str = "0"
            
            # Replace G ID or Control with geometry angles
            if g_id == 'Control':
                # Replace "Control" with geometry angles
                new_name = file_name.replace("Control", f"{alpha}_{beta}_{gamma_str}")
                logger.info(f"Converting control: {file_name} -> {new_name}")
            else:
                # Replace G ID with geometry angles
                new_name = file_name.replace(f"_{g_id}_", f"_{alpha}_{beta}_{gamma_str}_")
                logger.info(f"Converting geometry: {file_name} -> {new_name}")
            return new_name
        
        return file_name
    
    def identify_trial_files(self) -> List[Path]:
        """
        Identify all trial files that are not opencap events.
        
        Returns:
            List of Path objects for non-opencap trial files
        """
        all_files = list(self.trials_dir.glob("*.csv"))
        non_opencap_files = [
            f for f in all_files 
            if not f.name.endswith("_opencap_events.csv")
        ]
        
        logger.info(f"Found {len(all_files)} total CSV files")
        logger.info(f"Identified {len(non_opencap_files)} non-opencap trial files")
        
        return sorted(non_opencap_files)
    
    def load_trial_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load trial data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the trial data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
            pd.errors.ParserError: If the file cannot be parsed
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data from {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {str(e)}")
            raise
    
    def validate_data_columns(self, df: pd.DataFrame, file_name: str) -> bool:
        """
        Validate that the required columns exist in the data.
        
        Args:
            df: DataFrame to validate
            file_name: Name of the file for logging
            
        Returns:
            True if all required columns exist, False otherwise
        """
        required_columns = ['acc_x_time', 'acc_x_data', 'acc_z_data']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in {file_name}: {missing_columns}")
            return False
        
        return True
    
    def create_accelerometer_plot(self, df: pd.DataFrame, file_name: str) -> plt.Figure:
        """
        Create a comprehensive accelerometer data plot using matplotlib.
        
        Args:
            df: DataFrame containing the accelerometer data
            file_name: Name of the file for plot title
            
        Returns:
            matplotlib Figure object
        """
        # Recalculate time based on sampling frequency
        time_seconds = np.arange(len(df)) / self.sampling_frequency
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Accelerometer Data Analysis - {file_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: X-axis acceleration over time
        axes[0].plot(time_seconds, df['acc_x_data'], linewidth=1.5, alpha=0.8, color='blue')
        axes[0].set_title('X-axis Acceleration Over Time', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (seconds)', fontsize=12)
        axes[0].set_ylabel('X-axis Acceleration (g)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics text
        x_mean = df['acc_x_data'].mean()
        x_std = df['acc_x_data'].std()
        axes[0].text(0.02, 0.98, f'Mean: {x_mean:.3f}g\nStd: {x_std:.3f}g', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Z-axis acceleration over time
        axes[1].plot(time_seconds, df['acc_z_data'], linewidth=1.5, alpha=0.8, color='orange')
        axes[1].set_title('Z-axis Acceleration Over Time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time (seconds)', fontsize=12)
        axes[1].set_ylabel('Z-axis Acceleration (g)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        z_mean = df['acc_z_data'].mean()
        z_std = df['acc_z_data'].std()
        axes[1].text(0.02, 0.98, f'Mean: {z_mean:.3f}g\nStd: {z_std:.3f}g', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Calculate duration and actual sampling rate
        duration = time_seconds.max() - time_seconds.min()
        actual_sample_rate = len(df) / duration if duration > 0 else 0
        
        # Add data summary with corrected information
        fig.text(0.02, 0.02, f'Duration: {duration:.1f}s | Expected: {self.sampling_frequency} Hz | Actual: {actual_sample_rate:.1f} Hz | Data Points: {len(df)}', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig: plt.Figure, file_name: str) -> Path:
        """
        Save the plot to the output directory with geometry-based naming.
        
        Args:
            fig: matplotlib Figure object
            file_name: Name of the original file (without extension)
            
        Returns:
            Path to the saved plot file
        """
        # Convert geometry name for plot naming
        geometry_name = self._convert_geometry_name(file_name)
        plot_name = f"{geometry_name}_accelerometer_analysis.png"
        plot_path = self.output_dir / plot_name
        
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved plot: {plot_path} (original: {file_name})")
        
        return plot_path
    
    def analyze_all_trials(self) -> List[Tuple[str, bool, str]]:
        """
        Analyze all non-opencap trial files.
        
        Returns:
            List of tuples containing (file_name, success, message)
        """
        trial_files = self.identify_trial_files()
        results = []
        
        logger.info(f"Starting analysis of {len(trial_files)} trial files...")
        
        for file_path in trial_files:
            file_name = file_path.stem  # Get filename without extension
            try:
                # Load data
                df = self.load_trial_data(file_path)
                
                # Validate columns
                if not self.validate_data_columns(df, file_name):
                    results.append((file_name, False, "Missing required columns"))
                    continue
                
                # Create plot
                fig = self.create_accelerometer_plot(df, file_name)
                
                # Save plot
                plot_path = self.save_plot(fig, file_name)
                
                # Close figure to free memory
                plt.close(fig)
                
                results.append((file_name, True, f"Successfully processed and saved to {plot_path.name}"))
                
            except Exception as e:
                logger.error(f"Error processing {file_name}: {str(e)}")
                results.append((file_name, False, f"Error: {str(e)}"))
        
        return results
    
    def generate_summary_report(self, results: List[Tuple[str, bool, str]]) -> None:
        """
        Generate a summary report of the analysis results.
        
        Args:
            results: List of analysis results
        """
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        
        report_path = self.output_dir / "analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("IMU Trial Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total files processed: {len(results)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 30 + "\n")
            for file_name, success, message in results:
                status = "✓" if success else "✗"
                f.write(f"{status} {file_name}: {message}\n")
        
        logger.info(f"Summary report saved to: {report_path}")


def main(sampling_frequency: float = 100.0) -> None:
    """
    Main function to run the IMU trial analysis.
    """
    # Define paths
    core_data_path = Path('/Users/riccardoconci/Local_documents/!CURRENT_RESEARCH/Personalising-Crutches/data_V2/raw')
    trials_dir = core_data_path / 'MIH15' / '2025-10-21' / 'trials'
    output_dir = core_data_path / 'MIH15' / '2025-10-21' / 'plots'
    
    try:
        # Initialize analyzer with custom sampling frequency and geometry mapping
        geometry_mapping_path = Path('/Users/riccardoconci/Local_documents/!CURRENT_RESEARCH/Personalising-Crutches/DataAnalysis/results/geometry_id_to_angles.json')
        analyzer = IMUTrialAnalyzer(trials_dir, output_dir, sampling_frequency, geometry_mapping_path)
        
        # Run analysis
        results = analyzer.analyze_all_trials()
        
        # Generate summary report
        analyzer.generate_summary_report(results)
        
        # Print summary
        successful = sum(1 for _, success, _ in results if success)
        print(f"\nAnalysis complete!")
        print(f"Successfully processed: {successful}/{len(results)} files")
        print(f"Plots saved to: {output_dir}")
        print(f"Using sampling frequency: {sampling_frequency} Hz")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
