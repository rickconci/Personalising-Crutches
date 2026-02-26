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


def clean_for_json(obj):
    """Clean data structure to be JSON-safe by replacing NaN/inf with None."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_for_json(item) for item in obj]
    return obj


class DataProcessingService:
    """Service for processing accelerometer data and detecting steps."""
    
    def __init__(self, db: Session):
        """Initialize the data processing service."""
        self.db = db
        self.step_detector = StepDetector()
        self.gait_analyzer = GaitAnalyzer()
    
    def _encode_geometry_token(self, alpha: Optional[float], beta: Optional[float], gamma: Optional[float]) -> str:
        """
        Build geometry token like A85_B95_Gm9 (m for negative gamma, p for positive).
        
        Args:
            alpha: Alpha angle in degrees
            beta: Beta angle in degrees
            gamma: Gamma value in degrees
        
        Returns:
            Encoded geometry string token.
        """
        # Use integers for compactness
        a = "A" + (str(int(round(alpha))) if alpha is not None else "na")
        b = "B" + (str(int(round(beta))) if beta is not None else "na")
        if gamma is None:
            g = "Gna"
        else:
            sign = "m" if gamma < 0 else ("p" if gamma > 0 else "0")
            gval = str(int(round(abs(gamma)))) if sign in ("m", "p") else "0"
            g = f"G{sign}{gval}" if sign in ("m", "p") else "G0"
        return f"{a}_{b}_{g}"
    
    def _parse_existing_trial_numbers(self, files: List[Path]) -> List[Tuple[int, str, Optional[int]]]:
        """
        Parse filenames to extract trial number, geometry token and optional repeat index.
        Expected patterns:
            trial{n}_{geom}_{HHMMSS}.ext
            trial{n}_{geom}_{rep}_{HHMMSS}.ext  (when repeated)
        Returns list of tuples: (n, geom, rep)
        """
        results: List[Tuple[int, str, Optional[int]]] = []
        for f in files:
            name = f.stem  # without extension
            parts = name.split("_")
            if not parts:
                continue
            if not parts[0].startswith("trial"):
                continue
            try:
                n = int(parts[0].removeprefix("trial"))
            except ValueError:
                continue
            # geometry token could be three segments combined (Axx_Bxx_G?
            # We joined using underscores, but our token itself uses underscores.
            # Structure: trialN + [A..] + [B..] + [G..] + (optional rep) + time
            # Enforce new scheme strictly: trial + A + B + G + rep + time
            if len(parts) != 6:
                continue
            geom = "_".join(parts[1:4])
            try:
                rep = int(parts[4])
            except ValueError:
                continue
            results.append((n, geom, rep))
        return results
    
    def _compute_next_trial_filename(
        self,
        directory: Path,
        alpha: Optional[float],
        beta: Optional[float],
        gamma: Optional[float],
        time_str: str,
        file_extension: str
    ) -> str:
        """
        Compute next filename using the scheme:
          trial{n}_{Axx}_{Byy}_{G[m|p]v}[_{rep}]_{HHMMSS}{ext}
        - n increments with each new recording within directory, unless the previous
          recording has the same geometry token, in which case reuse n and increment rep.
        - rep starts at 1 when repeating same geometry consecutively.
        """
        geometry_token = self._encode_geometry_token(alpha, beta, gamma)
        existing_files = sorted(directory.glob("*.csv")) + sorted(directory.glob("*.parquet"))
        parsed = self._parse_existing_trial_numbers(existing_files)
        if not parsed:
            # first recording for this participant/date: start with rep 0
            return f"trial1_{geometry_token}_0_{time_str}{file_extension}"
        # Determine most recent by scanning filenames; since we don't have timestamps in a reliable sortable prefix,
        # we fallback to filesystem modification time to get the last one.
        last_file: Optional[Path] = None
        last_mtime = -1.0
        for f in existing_files:
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            if mtime > last_mtime:
                last_mtime = mtime
                last_file = f
        if last_file is None:
            return f"trial1_{geometry_token}_0_{time_str}{file_extension}"
        # Parse last file for n, geom, rep
        last_info = self._parse_existing_trial_numbers([last_file])
        if not last_info:
            # If last file doesn't follow pattern, continue incrementing max n
            max_n = max((n for n, _, _ in parsed), default=0)
            return f"trial{max_n + 1}_{geometry_token}_{time_str}{file_extension}"
        last_n, last_geom, last_rep = last_info[0]
        if last_geom == geometry_token:
            # same geometry: keep n, bump rep (first occurrence is 0 by definition)
            next_rep = 0 if last_rep is None else last_rep + 1
            return f"trial{last_n}_{geometry_token}_{next_rep}_{time_str}{file_extension}"
        # different geometry: increment n, no rep
        max_n = max((n for n, _, _ in parsed), default=last_n)
        # different geometry: increment n and set rep to 0
        return f"trial{max_n + 1}_{geometry_token}_0_{time_str}{file_extension}"
    
    def get_participant_data_directory(
        self, 
        participant_name: str, 
        data_type: str = "trials",
        date: Optional[str] = None
    ) -> Path:
        """
        Get or create an organized directory for participant data.
        
        Directory structure: data/raw/{participant_name}/{date}/{data_type}/
        
        Args:
            participant_name: Name of the participant (e.g., "MH1")
            data_type: Type of data ("trials", "metabolic", "surveys", "processed")
            date: Date string (YYYY-MM-DD). If None, uses today's date.
            
        Returns:
            Path object for the directory
        """
        from datetime import datetime
        
        # Use today's date if not specified
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        base_dir = Path(settings.raw_data_directory)
        participant_dir = base_dir / participant_name / date / data_type
        participant_dir.mkdir(parents=True, exist_ok=True)
        return participant_dir
    
    async def upload_data_file(
        self, 
        file: UploadFile, 
        trial_id: Optional[int] = None,
        participant_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Upload and store a data file with organized directory structure.
        
        Directory structure:
        data/raw/{participant_name}/trials/{trial_id}_{geometry_name}_{timestamp}.csv
        
        Args:
            file: Uploaded file
            trial_id: Associated trial ID
            participant_id: Associated participant ID
            
        Returns:
            Dictionary with file information
        """
        from database.models import Participant
        from datetime import datetime
        
        # Get participant and trial info for organized file structure
        participant_name = "unknown"
        geometry_name = "unknown"
        
        if trial_id:
            trial = self.db.query(Trial).filter(Trial.id == trial_id).first()
            if trial:
                if trial.participant:
                    participant_name = trial.participant.name
                if trial.geometry:
                    geometry_name = trial.geometry.name
        elif participant_id:
            participant = self.db.query(Participant).filter(Participant.id == participant_id).first()
            if participant:
                participant_name = participant.name
        
        # Get current date and time for organization
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        
        # Create organized directory structure: participant/date/trials/
        participant_dir = self.get_participant_data_directory(
            participant_name=participant_name,
            data_type="trials",
            date=date_str
        )
        
        # Generate descriptive filename with new scheme
        file_extension = Path(file.filename).suffix
        # Determine angles: prefer trial's denormalized fields, fallback to geometry
        trial_alpha: Optional[float] = None
        trial_beta: Optional[float] = None
        trial_gamma: Optional[float] = None
        if trial_id:
            trial = self.db.query(Trial).filter(Trial.id == trial_id).first()
            if trial:
                trial_alpha = trial.alpha if trial.alpha is not None else (trial.geometry.alpha if trial.geometry else None)
                trial_beta = trial.beta if trial.beta is not None else (trial.geometry.beta if trial.geometry else None)
                trial_gamma = trial.gamma if trial.gamma is not None else (trial.geometry.gamma if trial.geometry else None)
        # Compute filename within participant/date/trials directory
        unique_filename = self._compute_next_trial_filename(
            participant_dir,
            trial_alpha,
            trial_beta,
            trial_gamma,
            time_str,
            file_extension
        )
        
        file_path = participant_dir / unique_filename
        file_path_absolute = file_path.resolve()

        # Save file
        with open(file_path_absolute, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create database record (store absolute path so process finds file regardless of cwd)
        data_file = DataFile(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=str(file_path_absolute),
            file_size=len(content),
            file_type=file_extension[1:].lower(),
            trial_id=trial_id
        )
        
        self.db.add(data_file)
        
        # Update trial with raw_data_path if trial_id is provided
        if trial_id:
            trial = self.db.query(Trial).filter(Trial.id == trial_id).first()
            if trial:
                trial.raw_data_path = str(file_path_absolute)
        
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
            
            # Prepare plot data
            plot_data = {}
            if isinstance(data, pd.DataFrame):
                # Prepare time series for plotting
                time_data = data['time'].tolist() if 'time' in data.columns else list(range(len(data)))
                
                # Calculate force derivative if force data is available
                force_derivative = []
                if 'force' in data.columns:
                    force = data['force'].values
                    time = data['time'].values if 'time' in data.columns else np.arange(len(force)) / self.step_detector.fs
                    force_derivative = np.gradient(force, time).tolist()
                
                plot_data = {
                    "time": time_data,
                    "acc_x": data['acc_x_data'].tolist() if 'acc_x_data' in data.columns else [],
                    "acc_z": data['acc_z_data'].tolist() if 'acc_z_data' in data.columns else [],
                    "force": data['force'].tolist() if 'force' in data.columns else [],
                    "force_derivative": force_derivative,
                    "step_times": step_result.step_times.tolist(),
                    "step_indices": step_result.step_indices.tolist()
                }
            
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
                },
                "plots": plot_data
            }
            
            # Clean all data to ensure JSON compatibility
            return clean_for_json(results)
            
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
            file_path: Path to the data file (absolute or relative to raw_data_directory)
            
        Returns:
            Loaded data as DataFrame
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = (Path(settings.raw_data_directory).resolve() / file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load accordingly
        if file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.parquet':
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Map common column names to expected format
        column_mapping = {
            'accX': 'acc_x_data',
            'accY': 'acc_z_data',  # Map accY to acc_z_data for vertical acceleration
            'acc_x': 'acc_x_data',
            'acc_z': 'acc_z_data',
            'acc_x_data': 'acc_x_data',
            'acc_z_data': 'acc_z_data'
        }
        
        # Apply column mapping
        for old_name, new_name in column_mapping.items():
            if old_name in data.columns and new_name not in data.columns:
                data[new_name] = data[old_name]
        
        # Validate required columns
        required_columns = ['acc_x_data', 'acc_z_data']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            available_columns = list(data.columns)
            raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {available_columns}")
        
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
            "file_path": data_file.file_path,  # Added this
            "processing_status": data_file.processing_status,
            "uploaded_at": data_file.uploaded_at.isoformat(),
            "processed_at": data_file.processed_at.isoformat() if data_file.processed_at else None,
            "file_size": data_file.file_size,
            "file_type": data_file.file_type
        }
