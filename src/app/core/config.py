"""
Configuration management for the Personalising Crutches application.

This module centralizes all configuration settings, including:
- Experiment parameters
- Crutch geometry boundaries
- Optimization settings
- Data processing parameters
"""

from typing import Dict, List, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from pathlib import Path
import os
import numpy as np

# Project base = directory containing src (Personalising-Crutches). Config lives at project_base/dot_env.txt.
_PROJECT_BASE = Path(__file__).resolve().parent.parent.parent.parent


def get_config_from_project_file() -> Dict[str, str]:
    """
    Read all configuration from dot_env.txt at the project base (directory containing src/V2).
    
    Supports two formats:
    1. KEY=VALUE (standard env var format)
    2. external db url: VALUE (special format for database URL)
    
    Returns:
        Dictionary of configuration values
    """
    dot_env_file = _PROJECT_BASE / "dot_env.txt"
    
    config = {}
    
    if not dot_env_file.exists():
        return config
    
    try:
        # First pass: Handle special "external db url:" format (takes precedence)
        with open(dot_env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Handle special "external db url:" format (takes precedence over DATABASE_URL=)
                if line.startswith("external db url:"):
                    url = line.split(":", 1)[1].strip()
                    if url:
                        config["DATABASE_URL"] = url
                        break  # Found it, can stop looking
        
        # Second pass: Handle standard KEY=VALUE format
        with open(dot_env_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Skip "external db url:" line (already processed)
                if line.startswith("external db url:"):
                    continue
                
                # Handle standard KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        # Don't overwrite DATABASE_URL if it was set by "external db url:"
                        if key == "DATABASE_URL" and "DATABASE_URL" in config:
                            continue
                        config[key] = value
    except Exception:
        # If file exists but can't be read, silently fail
        pass
    
    return config


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    # Priority: 1) project base dot_env.txt, 2) DATABASE_URL env var, 3) .env file, 4) default SQLite
    database_url: str = Field(default="sqlite:///./experiments.db", env="DATABASE_URL")
    
    @model_validator(mode='before')
    @classmethod
    def load_from_home_file(cls, data: Any) -> Any:
        """
        Load configuration from dot_env.txt at project base (directory containing src).
        
        Priority order:
        1. Environment variables (highest priority)
        2. dot_env.txt at project base (all settings)
        3. .env file in project directory
        4. Default values
        """
        # Convert to dict if needed
        if not isinstance(data, dict):
            data = {}
        
        # Get all config from project base dot_env.txt
        home_config = get_config_from_project_file()
        
        # Apply home file config, but don't override existing values (env vars have priority)
        for key, value in home_config.items():
            # Convert KEY to field name (DATABASE_URL -> database_url)
            field_name = key.lower()
            
            # Only set if not already set (env vars have priority)
            if field_name not in data or (field_name == "database_url" and data[field_name] == "sqlite:///./experiments.db"):
                # Special handling for database_url default
                if field_name == "database_url" and value:
                    data[field_name] = value
                elif field_name == "debug":
                    # Convert string to boolean
                    data[field_name] = value.lower() in ("true", "1", "yes", "on")
                elif field_name == "api_port":
                    # Convert string to int
                    try:
                        data[field_name] = int(value)
                    except (ValueError, TypeError):
                        pass  # Keep default if conversion fails
                elif field_name in ["data_directory", "raw_data_directory", "processed_data_directory", 
                                   "results_directory", "plots_directory", "api_host"]:
                    data[field_name] = value
        
        return data
    
    # Data directories
    data_directory: str = Field(default="data", env="DATA_DIRECTORY")
    raw_data_directory: str = Field(default="data/raw", env="RAW_DATA_DIRECTORY")
    processed_data_directory: str = Field(default="data/processed", env="PROCESSED_DATA_DIRECTORY")
    results_directory: str = Field(default="data/results", env="RESULTS_DIRECTORY")
    plots_directory: str = Field(default="data/plots", env="PLOTS_DIRECTORY")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Experiment Configuration
class ExperimentConfig:
    """Configuration for crutch personalization experiments."""
    
    # Available objectives for optimization
    OBJECTIVES = ['effort', 'stability', 'pain']
    
    # User characteristics collected at experiment start
    USER_CHARACTERISTICS = ['height', 'weight', 'forearm_length', 'fitness_level']
    
    # Initial crutch geometry (default starting point)
    INITIAL_CRUTCH_GEOMETRY = {'alpha': 90, 'beta': 110, 'gamma': 0, 'delta': 0}
    
    # Default geometry sequence for experiments
    DEFAULT_GEOMETRY_SEQUENCE = 'baseline_comparison'
    
    # Available geometry sequences
    AVAILABLE_SEQUENCES = [
        'grid_search_3x3x3',
        'baseline_comparison', 
        'custom_sequence'
    ]
    
    # Crutch parameter boundaries for optimization
    CRUTCH_PARAM_BOUNDARIES = [
        {'name': 'alpha', 'type': 'discrete', 'domain': np.arange(70, 125, 5).tolist()},
        {'name': 'beta', 'type': 'discrete', 'domain': np.arange(90, 145, 5).tolist()},
        {'name': 'gamma', 'type': 'discrete', 'domain': np.arange(-12, 13, 3).tolist()},
    ]
    
    # Step sizes for discrete optimization
    CRUTCH_PARAM_STEPS = {
        'alpha': 5,
        'beta': 5,
        'gamma': 3
    }
    
    # Gaussian Process kernel parameters
    KERNEL_PARAMS = {
        'variance': 1.0,
        'lengthscale': 3.0,
    }
    
    # Mapping from objectives to quantitative measurements
    OBJECTIVE_TO_QUANTITATIVE_METRICS = {
        'effort': ['metabolic_cost'],
        'stability': ['Y_change', 'Y_total', 'step_variance'],
        'pain': []
    }
    
    # Mapping from objectives to survey measurements
    OBJECTIVE_TO_SURVEY_METRICS = {
        'effort': ['effort_survey_answer'],
        'stability': ['stability_survey_answer'],
        'pain': ['pain_survey_answer']
    }
    
    # Metric weighting values for loss calculation
    METRIC_WEIGHTS = {
        'metabolic_cost': 20,
        'Y_change': 2,
        'Y_total': 100,
        'step_variance': 100,
        'RMS_load_cell_force': 3000,
        'effort_survey_answer': 1,
        'stability_survey_answer': 1,
        'pain_survey_answer': 1,
    }


# Data Processing Configuration
class DataProcessingConfig:
    """Configuration for data processing and step detection."""
    
    # Sampling frequency (Hz)
    DEFAULT_SAMPLING_FREQUENCY = 100.0
    
    # Signal processing parameters
    SMOOTHING_FACTOR = 25  # Percentage for exponential smoothing
    HIGH_PASS_CUTOFF = 0.5  # Hz
    
    # Step detection parameters
    MIN_STEP_DISTANCE = 0.4  # seconds
    PEAK_HEIGHT_MULTIPLIER = 1.0  # Standard deviation multiplier
    
    # File naming conventions
    RAW_DATA_SUFFIX = "_raw.csv"
    STEP_FILE_SUFFIX = "_steps.csv"
    MASTER_LOG_FILE = "master_experiment_log.csv"
    
    # Master log columns for consistency
    MASTER_LOG_COLUMNS = [
        'objective', 'user_id', 'height', 'weight', 'forearm_length', 'fitness_level',
        'alpha', 'beta', 'gamma', 'delta',
        'effort_survey_answer', 'pain_survey_answer', 'stability_survey_answer',
        'metabolic_cost', 'Y_change', 'Y_total', 'step_variance', 'RMS_load_cell_force',
        'Total_Combined_Loss'
    ]


# Step Detection Algorithm Configuration
class StepDetectionConfig:
    """Configuration for step detection algorithms."""
    
    # Available algorithms
    ALGORITHMS = [
        'algo1_peaks',
        'algo2_derivative',
        'algo3_adaptive',
        'algo4_tkeo',
        'algo5_matched',
        'algo6_javascript',
        'algo7_force_derivative',
        'algo8_min_threshold',
        'deep_learning'
    ]
    
    # Algorithm-specific parameters
    ALGORITHM_PARAMS = {
        'algo1_peaks': {
            'min_distance': 0.4,
            'height_multiplier': 1.0,
        },
        'algo2_derivative': {
            'differential_duration': 0.24,
            'threshold_multiplier': 5.0,
        },
        'algo4_tkeo': {
            'window_size': 0.6,
            'threshold_multiplier': 2.0,
        },
        'algo5_matched': {
            'template_length': 0.6,
            'correlation_threshold': 0.7,
        },
        'deep_learning': {
            'model_name': 'google/timesfm-2.0-500m-pytorch',
            'window_seconds': 2.0,
            'stride_seconds': 1.0,
        }
    }


# Create global config instances
experiment_config = ExperimentConfig()
data_processing_config = DataProcessingConfig()
step_detection_config = StepDetectionConfig()
