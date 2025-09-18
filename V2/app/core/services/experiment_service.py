"""
Experiment service for managing participants, geometries, and trials.

This service encapsulates all business logic related to experiment management.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime

from database.models import Participant as SQLParticipant, CrutchGeometry as SQLCrutchGeometry, Trial as SQLTrial, ExperimentSession
from ..models import (
    ParticipantCreate, ParticipantUpdate, Participant,
    CrutchGeometry as GeometryModel, GeometryCreate, GeometryUpdate, CrutchGeometry,
    TrialCreate, TrialUpdate, Trial, TrialSource
)


class ExperimentService:
    """Service for managing experiments, participants, and trials."""
    
    def __init__(self, db: Session):
        """Initialize the experiment service."""
        self.db = db
    
    # Participant management
    def get_participants(self, skip: int = 0, limit: int = 100) -> List[Participant]:
        """Get all participants with pagination."""
        sql_participants = (
            self.db.query(SQLParticipant)
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [self._sql_participant_to_pydantic(p) for p in sql_participants]
    
    def get_participant(self, participant_id: int) -> Optional[Participant]:
        """Get a specific participant by ID."""
        sql_participant = self.db.query(SQLParticipant).filter(SQLParticipant.id == participant_id).first()
        if not sql_participant:
            return None
        return self._sql_participant_to_pydantic(sql_participant)
    
    def create_participant(self, participant_data: ParticipantCreate) -> Participant:
        """Create a new participant."""
        sql_participant = SQLParticipant(
            name=participant_data.name,
            characteristics=participant_data.characteristics
        )
        self.db.add(sql_participant)
        self.db.commit()
        self.db.refresh(sql_participant)
        return self._sql_participant_to_pydantic(sql_participant)
    
    def update_participant(self, participant_id: int, participant_data: ParticipantUpdate) -> Optional[Participant]:
        """Update a participant."""
        sql_participant = self.db.query(SQLParticipant).filter(SQLParticipant.id == participant_id).first()
        if not sql_participant:
            return None
        
        update_data = participant_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(sql_participant, field, value)
        
        sql_participant.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(sql_participant)
        return self._sql_participant_to_pydantic(sql_participant)
    
    def delete_participant(self, participant_id: int) -> bool:
        """Delete a participant and all associated trials."""
        sql_participant = self.db.query(SQLParticipant).filter(SQLParticipant.id == participant_id).first()
        if not sql_participant:
            return False
        
        # Delete all associated trials
        self.db.query(SQLTrial).filter(SQLTrial.participant_id == participant_id).delete()
        
        # Delete the participant
        self.db.delete(sql_participant)
        self.db.commit()
        return True
    
    # Geometry management
    def get_geometries(self, skip: int = 0, limit: int = 100) -> List[CrutchGeometry]:
        """Get all crutch geometries with pagination."""
        sql_geometries = (
            self.db.query(SQLCrutchGeometry)
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [self._sql_geometry_to_pydantic(g) for g in sql_geometries]
    
    def get_geometry(self, geometry_id: int) -> Optional[CrutchGeometry]:
        """Get a specific geometry by ID."""
        sql_geometry = self.db.query(SQLCrutchGeometry).filter(SQLCrutchGeometry.id == geometry_id).first()
        if not sql_geometry:
            return None
        return self._sql_geometry_to_pydantic(sql_geometry)
    
    def create_geometry(self, geometry_data: GeometryCreate) -> CrutchGeometry:
        """Create a new crutch geometry."""
        sql_geometry = SQLCrutchGeometry(
            name=geometry_data.name,
            alpha=geometry_data.alpha,
            beta=geometry_data.beta,
            gamma=geometry_data.gamma,
            delta=geometry_data.delta
        )
        self.db.add(sql_geometry)
        self.db.commit()
        self.db.refresh(sql_geometry)
        return self._sql_geometry_to_pydantic(sql_geometry)
    
    def update_geometry(self, geometry_id: int, geometry_data: GeometryUpdate) -> Optional[CrutchGeometry]:
        """Update a crutch geometry."""
        sql_geometry = self.db.query(SQLCrutchGeometry).filter(SQLCrutchGeometry.id == geometry_id).first()
        if not sql_geometry:
            return None
        
        update_data = geometry_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(sql_geometry, field, value)
        
        sql_geometry.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(sql_geometry)
        return self._sql_geometry_to_pydantic(sql_geometry)
    
    def delete_geometry(self, geometry_id: int) -> bool:
        """Delete a crutch geometry."""
        sql_geometry = self.db.query(SQLCrutchGeometry).filter(SQLCrutchGeometry.id == geometry_id).first()
        if not sql_geometry:
            return False
        
        self.db.delete(sql_geometry)
        self.db.commit()
        return True
    
    # Trial management
    def get_trials(
        self, 
        participant_id: Optional[int] = None,
        skip: int = 0, 
        limit: int = 100,
        include_deleted: bool = False
    ) -> List[Trial]:
        """Get trials with optional filtering."""
        query = self.db.query(SQLTrial)
        
        if participant_id:
            query = query.filter(SQLTrial.participant_id == participant_id)
        
        if not include_deleted:
            query = query.filter(SQLTrial.deleted_at.is_(None))
        
        sql_trials = query.offset(skip).limit(limit).all()
        return [self._sql_trial_to_pydantic(t) for t in sql_trials]
    
    def get_trial(self, trial_id: int) -> Optional[Trial]:
        """Get a specific trial by ID."""
        sql_trial = self.db.query(SQLTrial).filter(SQLTrial.id == trial_id).first()
        if not sql_trial:
            return None
        return self._sql_trial_to_pydantic(sql_trial)
    
    def create_trial(self, trial_data: TrialCreate) -> Trial:
        """Create a new trial."""
        sql_trial = SQLTrial(
            participant_id=trial_data.participant_id,
            geometry_id=trial_data.geometry_id,
            alpha=trial_data.alpha,
            beta=trial_data.beta,
            gamma=trial_data.gamma,
            delta=trial_data.delta,
            source=trial_data.source,
            raw_data_path=trial_data.raw_data_path,
            processed_features=trial_data.processed_features,
            survey_responses=trial_data.survey_responses,
            steps=trial_data.steps,
            metabolic_cost=trial_data.metabolic_cost,
            total_combined_loss=trial_data.total_combined_loss
        )
        self.db.add(sql_trial)
        self.db.commit()
        self.db.refresh(sql_trial)
        return self._sql_trial_to_pydantic(sql_trial)
    
    def update_trial(self, trial_id: int, trial_data: TrialUpdate) -> Optional[Trial]:
        """Update a trial."""
        sql_trial = self.db.query(SQLTrial).filter(SQLTrial.id == trial_id).first()
        if not sql_trial:
            return None
        
        update_data = trial_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(sql_trial, field, value)
        
        sql_trial.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(sql_trial)
        return self._sql_trial_to_pydantic(sql_trial)
    
    def delete_trial(self, trial_id: int, soft_delete: bool = True) -> bool:
        """Delete a trial (soft or hard delete)."""
        sql_trial = self.db.query(SQLTrial).filter(SQLTrial.id == trial_id).first()
        if not sql_trial:
            return False
        
        if soft_delete:
            sql_trial.soft_delete()
        else:
            self.db.delete(sql_trial)
        
        self.db.commit()
        return True
    
    def get_participant_trials(self, participant_id: int, include_deleted: bool = False) -> List[Trial]:
        """Get all trials for a specific participant."""
        return self.get_trials(participant_id=participant_id, include_deleted=include_deleted)
    
    def get_trials_by_geometry(self, geometry_id: int) -> List[Trial]:
        """Get all trials using a specific geometry."""
        sql_trials = (
            self.db.query(SQLTrial)
            .filter(SQLTrial.geometry_id == geometry_id)
            .filter(SQLTrial.deleted_at.is_(None))
            .all()
        )
        return [self._sql_trial_to_pydantic(t) for t in sql_trials]
    
    # Experiment session management
    def create_experiment_session(
        self, 
        participant_id: int, 
        objective: str, 
        configuration: Optional[Dict[str, Any]] = None
    ) -> int:
        """Create a new experiment session."""
        session = ExperimentSession(
            participant_id=participant_id,
            objective=objective,
            configuration=configuration
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session.id
    
    def get_experiment_session(self, session_id: int) -> Optional[ExperimentSession]:
        """Get an experiment session by ID."""
        return self.db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
    
    def complete_experiment_session(self, session_id: int) -> bool:
        """Mark an experiment session as completed."""
        session = self.db.query(ExperimentSession).filter(ExperimentSession.id == session_id).first()
        if not session:
            return False
        
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        self.db.commit()
        return True
    
    # Helper methods
    
    def _sql_participant_to_pydantic(self, sql_participant: SQLParticipant) -> Participant:
        """Convert SQLAlchemy Participant to Pydantic Participant."""
        return Participant(
            id=sql_participant.id,
            name=sql_participant.name,
            characteristics=sql_participant.characteristics,
            created_at=sql_participant.created_at,
            updated_at=sql_participant.updated_at
        )
    
    def _sql_geometry_to_pydantic(self, sql_geometry: SQLCrutchGeometry) -> CrutchGeometry:
        """Convert SQLAlchemy CrutchGeometry to Pydantic CrutchGeometry."""
        return CrutchGeometry(
            id=sql_geometry.id,
            name=sql_geometry.name,
            alpha=sql_geometry.alpha,
            beta=sql_geometry.beta,
            gamma=sql_geometry.gamma,
            delta=sql_geometry.delta,
            created_at=sql_geometry.created_at,
            updated_at=sql_geometry.updated_at
        )
    
    def _sql_trial_to_pydantic(self, sql_trial: SQLTrial) -> Trial:
        """Convert SQLAlchemy Trial to Pydantic Trial."""
        # Convert string source to TrialSource enum
        try:
            source_enum = TrialSource(sql_trial.source)
        except ValueError:
            # If the source doesn't match any enum value, default to GRID_SEARCH
            source_enum = TrialSource.GRID_SEARCH
        
        return Trial(
            id=sql_trial.id,
            participant_id=sql_trial.participant_id,
            geometry_id=sql_trial.geometry_id,
            alpha=sql_trial.alpha,
            beta=sql_trial.beta,
            gamma=sql_trial.gamma,
            delta=sql_trial.delta,
            source=source_enum,
            raw_data_path=sql_trial.raw_data_path,
            processed_features=sql_trial.processed_features,
            survey_responses=sql_trial.survey_responses,
            steps=sql_trial.steps,
            metabolic_cost=sql_trial.metabolic_cost,
            total_combined_loss=sql_trial.total_combined_loss,
            timestamp=sql_trial.timestamp,
            deleted=sql_trial.deleted_at
        )
    
