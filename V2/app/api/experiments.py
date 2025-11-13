"""
Experiment management API endpoints.

This module provides endpoints for managing experiments, participants, and trials.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from sqlalchemy.orm import Session

from ..core.models import (
    ParticipantBase, ParticipantCreate, ParticipantUpdate, Participant, ParticipantResponse,
    TrialBase, TrialCreate, TrialUpdate, Trial, TrialResponse, TrialCreateManual,
    CrutchGeometryBase, GeometryCreate, GeometryUpdate, CrutchGeometry, GeometryResponse
)
from ..core.services.experiment_service import ExperimentService
from database.connection import get_db

router = APIRouter()

# Dependency to get experiment service
def get_experiment_service(db: Session = Depends(get_db)) -> ExperimentService:
    return ExperimentService(db)

@router.get("/participants", response_model=List[Participant])
async def get_participants(
    skip: int = 0,
    limit: int = 100,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Get all participants."""
    participants = service.get_participants(skip=skip, limit=limit)
    return participants

@router.post("/participants", response_model=Participant)
async def create_participant(
    participant: ParticipantCreate,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Create a new participant."""
    try:
        return service.create_participant(participant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create participant: {str(e)}")

@router.get("/participants/{participant_id}", response_model=Participant)
async def get_participant(
    participant_id: int,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Get a specific participant."""
    participant = service.get_participant(participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    return participant

@router.put("/participants/{participant_id}", response_model=Participant)
async def update_participant(
    participant_id: int,
    participant_update: ParticipantUpdate,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Update a participant."""
    participant = service.update_participant(participant_id, participant_update)
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    return participant

@router.delete("/participants/{participant_id}")
async def delete_participant(
    participant_id: int,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Delete a participant."""
    success = service.delete_participant(participant_id)
    if not success:
        raise HTTPException(status_code=404, detail="Participant not found")
    return {"message": "Participant deleted successfully"}


@router.post("/participants/{participant_id}/trials", response_model=Trial)
async def create_trial_for_participant(
    participant_id: int,
    trial_data: TrialCreateManual,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Create a new trial for a specific participant with manual geometry."""
    try:
        return service.create_trial_with_geometry(participant_id, trial_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/geometries", response_model=List[CrutchGeometry])
async def get_geometries(
    skip: int = 0,
    limit: int = 100,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Get all crutch geometries."""
    geometries = service.get_geometries(skip=skip, limit=limit)
    return geometries

@router.post("/geometries", response_model=CrutchGeometry)
async def create_geometry(
    geometry: GeometryCreate,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Create a new crutch geometry."""
    return service.create_geometry(geometry)

@router.get("/geometries/{geometry_id}", response_model=CrutchGeometry)
async def get_geometry(
    geometry_id: int,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Get a specific crutch geometry."""
    geometry = service.get_geometry(geometry_id)
    if not geometry:
        raise HTTPException(status_code=404, detail="Geometry not found")
    return geometry

@router.put("/geometries/{geometry_id}", response_model=CrutchGeometry)
async def update_geometry(
    geometry_id: int,
    geometry_update: GeometryUpdate,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Update a crutch geometry."""
    geometry = service.update_geometry(geometry_id, geometry_update)
    if not geometry:
        raise HTTPException(status_code=404, detail="Geometry not found")
    return geometry

@router.get("/trials", response_model=List[Trial])
async def get_trials(
    participant_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Get trials, optionally filtered by participant."""
    trials = service.get_trials(participant_id=participant_id, skip=skip, limit=limit)
    return trials

@router.post("/trials", response_model=Trial)
async def create_trial(
    trial: TrialCreate,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Create a new trial."""
    return service.create_trial(trial)

@router.get("/trials/{trial_id}", response_model=Trial)
async def get_trial(
    trial_id: int,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Get a specific trial."""
    trial = service.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")
    return trial

@router.put("/trials/{trial_id}", response_model=Trial)
async def update_trial(
    trial_id: int,
    trial_update: TrialUpdate,
    service: ExperimentService = Depends(get_experiment_service)
):
    """Update a trial."""
    trial = service.update_trial(trial_id, trial_update)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")
    return trial

@router.delete("/trials/{trial_id}")
async def delete_trial(
    trial_id: int,
    soft_delete: bool = False,  # Default to hard delete for research use
    service: ExperimentService = Depends(get_experiment_service)
):
    """Delete a trial (hard delete by default, optional soft delete)."""
    success = service.delete_trial(trial_id, soft_delete=soft_delete)
    if not success:
        raise HTTPException(status_code=404, detail="Trial not found")
    return {"message": "Trial deleted successfully"}
