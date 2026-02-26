"""
Data processing API endpoints.

This module provides endpoints for data upload, processing, and analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from pathlib import Path

from ..core.models import (
    DataUpload,
    DataUploadResponse,
    DataProcessingRequest,
    DataProcessingResponse,
    AlgorithmComparisonRequest,
    AlgorithmComparisonResponse
)
from ..core.services import get_data_processing_service
from ..core.algorithms.step_detection import StepDetectionAlgorithm
from database.connection import get_db

router = APIRouter()

# Dependency to get data processing service
def get_data_processing_service_instance(db: Session = Depends(get_db)):
    DataProcessingService = get_data_processing_service()
    return DataProcessingService(db)

@router.post("/upload", response_model=DataUploadResponse)
async def upload_data_file(
    file: UploadFile = File(...),
    trial_id: Optional[int] = Form(None),
    participant_id: Optional[int] = Form(None),
    service = Depends(get_data_processing_service_instance)
):
    """Upload a data file for processing."""
    try:
        result = await service.upload_data_file(
            file=file,
            trial_id=trial_id,
            participant_id=participant_id
        )
        return DataUploadResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _parse_form_bool(value: str) -> bool:
    """Parse form string to bool so frontend can send 'true'/'false' without 422."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


@router.post("/process/{file_id}")
async def process_data_file(
    file_id: int,
    algorithm: str = Form("algo6_javascript"),
    use_force_gradient: str = Form("false"),
    service = Depends(get_data_processing_service_instance)
):
    """Process a data file with step detection."""
    try:
        use_force_gradient_bool = _parse_form_bool(use_force_gradient)
        # Get file info
        file_info = service.get_processing_status(file_id)
        
        # Process the file
        result = service.process_accelerometer_data(
            file_path=file_info["file_path"],
            algorithm=StepDetectionAlgorithm(algorithm),
            use_force_gradient=use_force_gradient_bool
        )
        
        print(f"Processing completed successfully")
        
        return {
            "file_id": file_id,
            "processing_results": result,
            "status": "completed"
        }
    except ValueError as e:
        print(f"ValueError in process_data_file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Exception in process_data_file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/process-trial/{trial_id}")
async def process_trial_data(
    trial_id: int,
    algorithm: str = Form("algo6_javascript"),
    use_force_gradient: bool = Form(False),
    service = Depends(get_data_processing_service_instance)
):
    """Process data for a specific trial."""
    try:
        # This would need to be implemented to get the file path from trial
        # For now, we'll return a placeholder
        raise HTTPException(status_code=501, detail="Trial processing not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compare-algorithms/{file_id}")
async def compare_algorithms(
    file_id: int,
    algorithms: Optional[str] = None,
    service = Depends(get_data_processing_service_instance)
):
    """Compare multiple step detection algorithms on a data file."""
    try:
        # Get file info
        file_info = service.get_processing_status(file_id)
        
        # Parse algorithms if provided
        algorithm_list = None
        if algorithms:
            algorithm_list = [StepDetectionAlgorithm(algo.strip()) for algo in algorithms.split(",")]
        
        # Compare algorithms
        result = service.compare_step_detection_algorithms(
            file_path=file_info["file_path"],
            algorithms=algorithm_list
        )
        
        return {
            "file_id": file_id,
            "comparison_results": result,
            "status": "completed"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/status/{file_id}")
async def get_processing_status(
    file_id: int,
    service = Depends(get_data_processing_service_instance)
):
    """Get processing status for a data file."""
    try:
        status = service.get_processing_status(file_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/calculate-loss/{trial_id}")
async def calculate_combined_loss(
    trial_id: int,
    objective: str = Form(...),
    survey_responses: Optional[str] = Form(None),
    service = Depends(get_data_processing_service_instance)
):
    """Calculate combined loss for a trial."""
    try:
        # Parse survey responses if provided
        survey_data = None
        if survey_responses:
            import json
            survey_data = json.loads(survey_responses)
        
        # Calculate loss
        loss = service.calculate_combined_loss(
            trial_id=trial_id,
            objective=objective,
            survey_responses=survey_data
        )
        
        return {
            "trial_id": trial_id,
            "objective": objective,
            "total_combined_loss": loss,
            "status": "completed"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/algorithms")
async def get_available_algorithms():
    """Get list of available step detection algorithms."""
    algorithms = [
        {
            "name": algo.value,
            "description": _get_algorithm_description(algo)
        }
        for algo in StepDetectionAlgorithm
    ]
    return {"algorithms": algorithms}

@router.get("/test-data")
async def get_test_data():
    """Get test data CSV file for test mode."""
    # Get the path to the test data file
    base_path = Path(__file__).parent.parent.parent / "data" / "test_data" / "live_recorded_data.csv"
    
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Test data file not found")
    
    return FileResponse(
        path=str(base_path),
        media_type="text/csv",
        filename="live_recorded_data.csv"
    )

def _get_algorithm_description(algorithm: StepDetectionAlgorithm) -> str:
    """Get description for an algorithm."""
    descriptions = {
        StepDetectionAlgorithm.PEAKS: "Simple peak detection using height and distance thresholds",
        StepDetectionAlgorithm.DERIVATIVE: "Derivative-based detection with adaptive threshold",
        StepDetectionAlgorithm.ADAPTIVE: "Adaptive threshold detection with local statistics",
        StepDetectionAlgorithm.TKEO: "Teager-Kaiser Energy Operator (TKEO) based detection",
        StepDetectionAlgorithm.MATCHED: "Matched filter detection with template correlation",
        StepDetectionAlgorithm.JAVASCRIPT: "JavaScript port from original implementation",
        StepDetectionAlgorithm.FORCE_DERIVATIVE: "Force derivative based detection",
        StepDetectionAlgorithm.MIN_THRESHOLD: "Minimum threshold detection",
        StepDetectionAlgorithm.DEEP_LEARNING: "Deep learning based detection (TimesFM)"
    }
    return descriptions.get(algorithm, "Unknown algorithm")
