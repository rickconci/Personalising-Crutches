"""
Data processing service for accelerometer data analysis.

This service handles data upload, processing, and step detection.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from sqlalchemy.orm import Session
from fastapi import UploadFile
import json

from database.models import DataFile, Trial
from ..algorithms.step_detection import StepDetector, StepDetectionAlgorithm, StepDetectionResult
from ..algorithms.gait_analysis import GaitAnalyzer, GaitMetrics
from ..config import settings, data_processing_config


class DataProcessingService:
    """Service for processing accelerometer data and detecting steps."""
    
    def __init__(self, db: Session):
        """Initialize the data processing service."""
        self.db = db
        self.step_detector = StepDetector()
        self.gait_analyzer = GaitAnalyzer()
    
    async def upload_data_file(
        self, 
        file: UploadFile, 
        trial_id: Optional[int] = None,
        participant_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Upload and store a data file.
        
        Args:
            file: Uploaded file
            trial_id: Associated trial ID
            participant_id: Associated participant ID
            
        Returns:
            Dictionary with file information
        """
        # Create upload directory
        upload_dir = Path(settings.raw_data_directory)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix
        unique_filename = f"{trial_id}_{file.filename}" if trial_id else file.filename
        file_path = upload_dir / unique_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create database record
        data_file = DataFile(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=len(content),
            file_type=file_extension[1:].lower(),
            trial_id=trial_id
        )
        
        self.db.add(data_file)
        self.db.commit()
        self.db.refresh(data_file)
        
        return {
            "id": data_file.id,
            "filename": data_file.filename,
            "file_path": data_file.file_path,
            "file_size": data_file.file_size,
            "file_type": data_file.file_type,
            "uploaded_at": data_file.uploaded_at.isoformat()
        }
    
    def process_accelerometer_data(
        self, 
        file_path: str, 
        algorithm: StepDetectionAlgorithm = StepDetectionAlgorithm.JAVASCRIPT,
        use_force_gradient: bool = False
    ) -> Dict[str, Any]:
        """
        Process accelerometer data and detect steps.
        
        Args:
            file_path: Path to the data file
            algorithm: Step detection algorithm to use
            use_force_gradient: Whether to use force gradient signal
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Load data
            data = self._load_data_file(file_path)
            
            # Detect steps
            step_result = self.step_detector.detect_steps(
                data, 
                algorithm=algorithm,
                use_force_gradient=use_force_gradient
            )
            
            # Analyze gait metrics
            gait_metrics = self.gait_analyzer.analyze_gait(
                data, 
                step_result.step_times
            )
            
            # Prepare results
            results = {
                "step_detection": {
                    "algorithm": algorithm.value,
                    "step_count": len(step_result.step_times),
                    "step_times": step_result.step_times.tolist(),
                    "step_indices": step_result.step_indices.tolist(),
                    "confidence_scores": step_result.confidence_scores.tolist() if step_result.confidence_scores is not None else None,
                    "processing_time": step_result.processing_time
                },
                "gait_metrics": gait_metrics.to_dict(),
                "data_info": {
                    "sampling_frequency": self.step_detector.fs,
                    "duration": len(data) / self.step_detector.fs if hasattr(data, '__len__') else 0,
                    "data_points": len(data) if hasattr(data, '__len__') else 0
                }
            }
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error processing data: {str(e)}")
    
    def compare_step_detection_algorithms(
        self, 
        file_path: str,
        algorithms: Optional[List[StepDetectionAlgorithm]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple step detection algorithms.
        
        Args:
            file_path: Path to the data file
            algorithms: List of algorithms to compare
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Load data
            data = self._load_data_file(file_path)
            
            # Compare algorithms
            results = self.step_detector.compare_algorithms(data, algorithms=algorithms)
            
            # Format results
            comparison_results = {}
            for algorithm, result in results.items():
                comparison_results[algorithm.value] = {
                    "step_count": len(result.step_times),
                    "step_times": result.step_times.tolist(),
                    "processing_time": result.processing_time,
                    "parameters": result.parameters
                }
            
            return {
                "algorithms": comparison_results,
                "data_info": {
                    "sampling_frequency": self.step_detector.fs,
                    "duration": len(data) / self.step_detector.fs if hasattr(data, '__len__') else 0,
                    "data_points": len(data) if hasattr(data, '__len__') else 0
                }
            }
            
        except Exception as e:
            raise ValueError(f"Error comparing algorithms: {str(e)}")
    
    def update_trial_with_processing_results(
        self, 
        trial_id: int, 
        processing_results: Dict[str, Any]
    ) -> bool:
        """
        Update a trial with processing results.
        
        Args:
            trial_id: Trial ID to update
            processing_results: Results from data processing
            
        Returns:
            True if successful
        """
        try:
            trial = self.db.query(Trial).filter(Trial.id == trial_id).first()
            if not trial:
                return False
            
            # Extract step detection results
            step_detection = processing_results.get("step_detection", {})
            gait_metrics = processing_results.get("gait_metrics", {})
            
            # Update trial with processed features
            trial.processed_features = {
                "step_detection_algorithm": step_detection.get("algorithm"),
                "processing_time": step_detection.get("processing_time"),
                "data_info": processing_results.get("data_info", {})
            }
            
            # Update trial with steps
            trial.steps = step_detection.get("step_times")
            
            # Update trial with gait metrics
            if gait_metrics:
                trial.step_variance = gait_metrics.get("step_variance")
                trial.y_change = gait_metrics.get("y_change")
                trial.y_total = gait_metrics.get("y_total")
                trial.rms_load_cell_force = gait_metrics.get("rms_load_cell_force")
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            raise ValueError(f"Error updating trial: {str(e)}")
    
    def calculate_combined_loss(
        self, 
        trial_id: int, 
        objective: str,
        survey_responses: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate combined loss for a trial.
        
        Args:
            trial_id: Trial ID
            objective: Optimization objective
            survey_responses: Survey response data
            
        Returns:
            Combined loss value
        """
        from ..config import experiment_config
        
        trial = self.db.query(Trial).filter(Trial.id == trial_id).first()
        if not trial:
            raise ValueError("Trial not found")
        
        total_loss = 0.0
        
        # Add quantitative metrics
        quantitative_metrics = experiment_config.OBJECTIVE_TO_QUANTITATIVE_METRICS.get(objective, [])
        for metric in quantitative_metrics:
            if hasattr(trial, metric):
                value = getattr(trial, metric)
                if value is not None:
                    weight = experiment_config.METRIC_WEIGHTS.get(metric, 1.0)
                    total_loss += value * weight
        
        # Add survey metrics
        survey_metrics = experiment_config.OBJECTIVE_TO_SURVEY_METRICS.get(objective, [])
        survey_data = survey_responses or trial.survey_responses or {}
        
        for metric in survey_metrics:
            if metric in survey_data:
                value = survey_data[metric]
                if value is not None:
                    weight = experiment_config.METRIC_WEIGHTS.get(metric, 1.0)
                    total_loss += value * weight
        
        # Update trial with combined loss
        trial.total_combined_loss = total_loss
        self.db.commit()
        
        return total_loss
    
    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data file based on file type.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded data as DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load accordingly
        if file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.parquet':
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Validate required columns
        required_columns = ['acc_x_data', 'acc_z_data']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add time column if not present
        if 'time' not in data.columns:
            if 'acc_x_time' in data.columns:
                data['time'] = (data['acc_x_time'] - data['acc_x_time'].iloc[0]) / 1000
            else:
                # Assume 100 Hz sampling rate
                data['time'] = np.arange(len(data)) / 100.0
        
        return data
    
    def get_processing_status(self, file_id: int) -> Dict[str, Any]:
        """
        Get processing status for a data file.
        
        Args:
            file_id: Data file ID
            
        Returns:
            Processing status information
        """
        data_file = self.db.query(DataFile).filter(DataFile.id == file_id).first()
        if not data_file:
            raise ValueError("Data file not found")
        
        return {
            "id": data_file.id,
            "filename": data_file.filename,
            "processing_status": data_file.processing_status,
            "uploaded_at": data_file.uploaded_at.isoformat(),
            "processed_at": data_file.processed_at.isoformat() if data_file.processed_at else None,
            "file_size": data_file.file_size,
            "file_type": data_file.file_type
        }
