"""
EMG Analysis Main Script

Clean orchestration script for EMG data analysis pipeline.
Processes all trials, performs step detection, and generates visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from emg_parser import EMGParser, EMGData
from emg_visualizer import EMGVisualizer
from emg_gait_analyzer import EMGGaitAnalyzer, GaitMetrics
from emg_interactive_editor import EMGInteractiveEditor


class AnalysisConfig(BaseModel):
    """Configuration for EMG analysis pipeline."""
    
    emg_dir: str = Field(..., description="Path to EMG data directory")
    output_dir: Optional[str] = Field(None, description="Output directory path")
    min_duration: float = Field(60.0, gt=0, description="Minimum trial duration in seconds")
    max_duration: float = Field(180.0, gt=0, description="Maximum trial duration in seconds")
    muscle: str = Field("forearm", description="Muscle to analyze")
    k: int = Field(20, gt=0, description="Points on each side for moving average")
    gradient_threshold_multiplier: float = Field(2.0, gt=0, description="Gradient threshold multiplier")
    min_step_distance_s: float = Field(0.3, gt=0, description="Minimum distance between steps in seconds")
    
    @field_validator('max_duration')
    @classmethod
    def validate_max_duration(cls, v, info):
        """Ensure max_duration is greater than min_duration."""
        if 'min_duration' in info.data and v <= info.data['min_duration']:
            raise ValueError('max_duration must be greater than min_duration')
        return v
    
    def get_output_path(self) -> Path:
        """Get the output directory path."""
        if self.output_dir:
            return Path(self.output_dir)
        return Path(self.emg_dir) / 'EMG_analysis'


class EMGAnalysisPipeline:
    """
    Main pipeline for EMG analysis.
    
    Orchestrates the complete EMG analysis workflow including
    parsing, step detection, gait analysis, and visualization.
    """
    
    def __init__(self, config: AnalysisConfig):
        """
        Initialize the EMG analysis pipeline.
        
        Args:
            config: AnalysisConfig object with all pipeline parameters
        """
        self.config = config
        self.emg_path = Path(config.emg_dir)
        self.output_dir = config.get_output_path()
        
        # Initialize components
        self.parser = EMGParser()
        self.visualizer = EMGVisualizer()
        self.gait_analyzer = EMGGaitAnalyzer()
        self.interactive_editor = EMGInteractiveEditor(Path(self.output_dir) / "interactive")
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'interactive').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
    
    def process_all_trials(self) -> pd.DataFrame:
        """
        Process all EMG trial files.
        
        Returns:
            DataFrame with combined metrics from all trials
        """
        # Find all CSV files
        csv_files = list(self.emg_path.glob('*.csv'))
        csv_files.sort()
        
        print(f"Found {len(csv_files)} CSV files in {self.emg_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Duration filter: {self.config.min_duration}s - {self.config.max_duration}s")
        
        successful_trials = 0
        failed_trials = []
        all_metrics = []
        
        for csv_file in csv_files:
            trial_name = csv_file.stem
            
            try:
                print(f"\nProcessing {trial_name}...")
                
                # Parse the CSV file
                emg_data = self.parser.parse_file(csv_file)
                
                # Filter by duration
                filtered_data = self.gait_analyzer.filter_trial_by_duration(
                    emg_data, self.config.min_duration, self.config.max_duration
                )
                
                if filtered_data is None:
                    print(f"✗ {trial_name}: Trial too short (< {self.config.min_duration}s)")
                    continue
                
                # Detect steps with memory-safe adaptive processing
                step_times, step_indices, processing_info = self.gait_analyzer.detect_steps_from_emg_robust(
                    filtered_data, 
                    muscle=self.config.muscle,
                    notch_freq=50.0,
                    bandpass_range=(20.0, 250.0),
                    envelope_method="hilbert",  # Use Hilbert transform for better envelope
                    envelope_cutoff=1.0,
                    highpass_env_hz=0.3,  # Remove baseline drift
                    min_step_distance_s=1.2,  # Base distance (adaptive will adjust)
                    prominence_z=2.0,  # Not used in adaptive mode
                    min_peak_width_s=0.25
                )
                
                print(f"✓ {trial_name}: Detected {len(step_times)} steps")
                
                # Create visualizations
                self._create_visualizations(filtered_data, step_times, trial_name, processing_info)
                
                # Calculate gait metrics
                metrics = self.gait_analyzer.analyze_gait(
                    filtered_data, step_times, trial_name, muscle=self.config.muscle
                )
                
                # Save individual metrics
                self._save_trial_metrics(metrics, trial_name)
                
                all_metrics.append(metrics.to_dict())
                successful_trials += 1
                
            except Exception as e:
                print(f"✗ Failed to process {trial_name}: {e}")
                failed_trials.append(trial_name)
        
        # Create combined metrics DataFrame
        if all_metrics:
            combined_metrics_df = pd.DataFrame(all_metrics)
            combined_metrics_file = self.output_dir / 'metrics' / 'all_trials_gait_metrics.csv'
            combined_metrics_df.to_csv(combined_metrics_file, index=False)
            print(f"\nSaved combined metrics: {combined_metrics_file}")
            
            # Create summary visualizations
            self.visualizer.plot_gait_metrics_summary(
                combined_metrics_df, 
                self.output_dir / 'plots'
            )
        
        # Print summary
        self._print_summary(successful_trials, failed_trials)
        
        return pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    
    def _create_visualizations(
        self, 
        emg_data, 
        step_times: np.ndarray, 
        trial_name: str,
        processing_info: Optional[Dict] = None
    ) -> None:
        """Create all visualizations for a trial."""
        # Static muscle group plots
        self.visualizer.plot_muscle_groups(
            emg_data, 
            trial_name, 
            self.output_dir / 'plots'
        )
        
        # Interactive step detection plot with processing info
        self.visualizer.plot_emg_with_steps(
            emg_data, 
            step_times, 
            trial_name, 
            self.output_dir / 'interactive',
            muscle=self.config.muscle,
            processing_info=processing_info
        )
        
        # Create interactive step editor
        try:
            interactive_file = self.interactive_editor.create_interactive_plot(
                emg_data, processing_info, trial_name
            )
            print(f"✓ {trial_name}: Interactive editor created at {interactive_file}")
        except Exception as e:
            print(f"✗ Failed to create interactive editor for {trial_name}: {e}")
            # Don't fail the whole process, just continue
    
    def _save_trial_metrics(self, metrics, trial_name: str) -> None:
        """Save metrics for a single trial."""
        metrics_file = self.output_dir / 'metrics' / f'{trial_name}_gait_metrics.csv'
        metrics_df = pd.DataFrame([metrics.to_dict()])
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics: {metrics_file}")
    
    def _print_summary(self, successful: int, failed: List[str]) -> None:
        """Print processing summary."""
        print(f"\n{'='*50}")
        print(f"EMG Analysis Summary:")
        print(f"  Successful trials: {successful}")
        print(f"  Failed trials: {len(failed)}")
        if failed:
            print(f"  Failed trial names: {failed}")
        print(f"  Results saved to: {self.output_dir}")


def main():
    """Main function to run the EMG analysis pipeline."""
    # Create configuration
    config = AnalysisConfig(
        emg_dir='/Users/riccardoconci/Local_documents/!CURRENT_RESEARCH/Personalising-Crutches/data_V2/raw/MIH15/2025-10-21/EMG',
        min_duration=60.0,
        max_duration=180.0,
        muscle='forearm',
        k=200,  # More aggressive smoothing - try 10, 20, 50, 100
        gradient_threshold_multiplier=2.0,
        min_step_distance_s=0.3
    )
    
    # Create and run the pipeline
    pipeline = EMGAnalysisPipeline(config)
    
    # Process all trials
    metrics_df = pipeline.process_all_trials()
    
    print(f"\nAnalysis complete! Processed {len(metrics_df)} trials.")
    
    # Print configuration summary
    print(f"\nConfiguration used:")
    print(f"  EMG Directory: {config.emg_dir}")
    print(f"  Output Directory: {config.get_output_path()}")
    print(f"  Duration Filter: {config.min_duration}s - {config.max_duration}s")
    print(f"  Muscle Analyzed: {config.muscle}")
    print(f"  Moving Average k: {config.k}")
    print(f"  Gradient Threshold: {config.gradient_threshold_multiplier}x")
    print(f"  Min Step Distance: {config.min_step_distance_s}s")


if __name__ == "__main__":
    main()
