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
        # If geometry parameters are provided but no geometry_id, find or create the geometry
        if trial_data.geometry_id is None and all([
            trial_data.alpha is not None,
            trial_data.beta is not None,
            trial_data.gamma is not None
        ]):
            # Find or create geometry
            geometry = self.db.query(SQLCrutchGeometry).filter(
                SQLCrutchGeometry.alpha == trial_data.alpha,
                SQLCrutchGeometry.beta == trial_data.beta,
                SQLCrutchGeometry.gamma == trial_data.gamma,
                SQLCrutchGeometry.delta == (trial_data.delta or 0)
            ).first()
            
            if not geometry:
                geometry_name = f"α{trial_data.alpha}_β{trial_data.beta}_γ{trial_data.gamma}"
                geometry = SQLCrutchGeometry(
                    name=geometry_name,
                    alpha=trial_data.alpha,
                    beta=trial_data.beta,
                    gamma=trial_data.gamma,
                    delta=trial_data.delta or 0
                )
                self.db.add(geometry)
                self.db.flush()  # Get the ID
            
            geometry_id = geometry.id
        else:
            geometry_id = trial_data.geometry_id
        
        # Convert Pydantic models to dict for JSON fields
        survey_responses_dict = None
        survey_fields = {}
        if trial_data.survey_responses:
            if hasattr(trial_data.survey_responses, 'dict'):
                survey_responses_dict = trial_data.survey_responses.dict()
            else:
                survey_responses_dict = trial_data.survey_responses
            
            # Extract individual survey fields for columns
            if survey_responses_dict:
                survey_fields = {
                    'sus_q1': survey_responses_dict.get('sus_q1'),
                    'sus_q2': survey_responses_dict.get('sus_q2'),
                    'sus_q3': survey_responses_dict.get('sus_q3'),
                    'sus_q4': survey_responses_dict.get('sus_q4'),
                    'sus_q5': survey_responses_dict.get('sus_q5'),
                    'sus_q6': survey_responses_dict.get('sus_q6'),
                    'sus_score': survey_responses_dict.get('sus_score'),
                    'nrs_score': survey_responses_dict.get('nrs_score'),
                    'tlx_mental_demand': survey_responses_dict.get('tlx_mental_demand'),
                    'tlx_physical_demand': survey_responses_dict.get('tlx_physical_demand'),
                    'tlx_performance': survey_responses_dict.get('tlx_performance'),
                    'tlx_effort': survey_responses_dict.get('tlx_effort'),
                    'tlx_frustration': survey_responses_dict.get('tlx_frustration'),
                    'tlx_score': survey_responses_dict.get('tlx_score')
                }
        
        # Extract processed features into individual columns
        processed_features_fields = {}
        if trial_data.processed_features:
            processed_features_fields = {
                'step_count': trial_data.processed_features.get('step_count'),
                'step_variance': trial_data.processed_features.get('step_variance'),
                'instability_loss': trial_data.processed_features.get('instability_loss'),
                'y_change': trial_data.processed_features.get('y_change'),
                'y_total': trial_data.processed_features.get('y_total'),
                'rms_load_cell_force': trial_data.processed_features.get('rms_load_cell_force'),
            }
        
        # Get geometry to copy alpha, beta, gamma, delta values
        geometry = self.db.query(SQLCrutchGeometry).filter(SQLCrutchGeometry.id == geometry_id).first()
        
        sql_trial = SQLTrial(
            participant_id=trial_data.participant_id,
            geometry_id=geometry_id,
            alpha=geometry.alpha if geometry else trial_data.alpha,
            beta=geometry.beta if geometry else trial_data.beta,
            gamma=geometry.gamma if geometry else trial_data.gamma,
            delta=geometry.delta if geometry else trial_data.delta,
            source=trial_data.source,
            raw_data_path=trial_data.raw_data_path,
            processed_features=trial_data.processed_features,
            survey_responses=survey_responses_dict,
            steps=trial_data.steps,
            metabolic_cost=trial_data.metabolic_cost,
            total_combined_loss=trial_data.total_combined_loss,
            **survey_fields,  # Unpack individual survey fields
            **processed_features_fields  # Unpack processed features into columns
        )
        self.db.add(sql_trial)
        self.db.commit()
        self.db.refresh(sql_trial)
        return self._sql_trial_to_pydantic(sql_trial)

    def create_trial_with_geometry(self, participant_id: int, trial_data: "TrialCreateManual") -> Trial:
        """
        Create a trial from manual geometry input.
        Finds an existing geometry or creates a new one.
        """
        # Check if participant exists
        participant = self.db.query(SQLParticipant).filter(SQLParticipant.id == participant_id).first()
        if not participant:
            raise ValueError("Participant not found")

        # Find or create the geometry
        geometry = self.db.query(SQLCrutchGeometry).filter(
            SQLCrutchGeometry.alpha == trial_data.alpha,
            SQLCrutchGeometry.beta == trial_data.beta,
            SQLCrutchGeometry.gamma == trial_data.gamma,
            SQLCrutchGeometry.delta == trial_data.delta
        ).first()

        if not geometry:
            geometry_name = f"α{trial_data.alpha}_β{trial_data.beta}_γ{trial_data.gamma}"
            geometry = SQLCrutchGeometry(
                name=geometry_name,
                alpha=trial_data.alpha,
                beta=trial_data.beta,
                gamma=trial_data.gamma,
                delta=trial_data.delta
            )
            self.db.add(geometry)
            self.db.flush()  # Use flush to get the ID before commit

        # Create the trial
        sql_trial = SQLTrial(
            participant_id=participant_id,
            geometry_id=geometry.id,
            alpha=geometry.alpha,
            beta=geometry.beta,
            gamma=geometry.gamma,
            delta=geometry.delta,
            source=trial_data.source
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
        
        # Extract survey responses if provided
        if 'survey_responses' in update_data and update_data['survey_responses']:
            survey_dict = update_data['survey_responses']
            if hasattr(survey_dict, 'dict'):
                survey_dict = survey_dict.dict()
            
            # Set individual survey columns
            if survey_dict:
                sql_trial.sus_q1 = survey_dict.get('sus_q1')
                sql_trial.sus_q2 = survey_dict.get('sus_q2')
                sql_trial.sus_q3 = survey_dict.get('sus_q3')
                sql_trial.sus_q4 = survey_dict.get('sus_q4')
                sql_trial.sus_q5 = survey_dict.get('sus_q5')
                sql_trial.sus_q6 = survey_dict.get('sus_q6')
                sql_trial.sus_score = survey_dict.get('sus_score')
                sql_trial.nrs_score = survey_dict.get('nrs_score')
                sql_trial.tlx_mental_demand = survey_dict.get('tlx_mental_demand')
                sql_trial.tlx_physical_demand = survey_dict.get('tlx_physical_demand')
                sql_trial.tlx_performance = survey_dict.get('tlx_performance')
                sql_trial.tlx_effort = survey_dict.get('tlx_effort')
                sql_trial.tlx_frustration = survey_dict.get('tlx_frustration')
                sql_trial.tlx_score = survey_dict.get('tlx_score')
        
        # Extract processed features into individual columns if provided
        if 'processed_features' in update_data and update_data['processed_features']:
            features_dict = update_data['processed_features']
            if features_dict:
                sql_trial.step_count = features_dict.get('step_count')
                sql_trial.step_variance = features_dict.get('step_variance')
                sql_trial.instability_loss = features_dict.get('instability_loss')
                sql_trial.y_change = features_dict.get('y_change')
                sql_trial.y_total = features_dict.get('y_total')
                sql_trial.rms_load_cell_force = features_dict.get('rms_load_cell_force')
                # Handle laps_completed from both direct field and processed_features
                if 'laps_completed' in features_dict:
                    sql_trial.laps = features_dict.get('laps_completed')
        
        # Update other fields
        for field, value in update_data.items():
            if field not in ['survey_responses', 'processed_features']:  # Already handled above
                # Map laps_completed to laps in the database
                if field == 'laps_completed':
                    sql_trial.laps = value
                else:
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
        
        # Get geometry parameters from the related geometry object
        alpha = sql_trial.alpha  # Use trial's alpha first
        beta = sql_trial.beta
        gamma = sql_trial.gamma
        delta = sql_trial.delta
        geometry_name = None
        
        if sql_trial.geometry:
            # Use geometry values if trial doesn't have them
            if alpha is None:
                alpha = sql_trial.geometry.alpha
            if beta is None:
                beta = sql_trial.geometry.beta
            if gamma is None:
                gamma = sql_trial.geometry.gamma
            if delta is None:
                delta = sql_trial.geometry.delta
            geometry_name = sql_trial.geometry.name
        
        # Construct survey_responses from individual columns
        survey_responses = None
        if any([sql_trial.sus_score, sql_trial.nrs_score, sql_trial.tlx_score]):
            survey_responses = {
                'sus_q1': sql_trial.sus_q1,
                'sus_q2': sql_trial.sus_q2,
                'sus_q3': sql_trial.sus_q3,
                'sus_q4': sql_trial.sus_q4,
                'sus_q5': sql_trial.sus_q5,
                'sus_q6': sql_trial.sus_q6,
                'sus_score': sql_trial.sus_score,
                'nrs_score': sql_trial.nrs_score,
                'tlx_mental_demand': sql_trial.tlx_mental_demand,
                'tlx_physical_demand': sql_trial.tlx_physical_demand,
                'tlx_performance': sql_trial.tlx_performance,
                'tlx_effort': sql_trial.tlx_effort,
                'tlx_frustration': sql_trial.tlx_frustration,
                'tlx_score': sql_trial.tlx_score
            }
        
        # Construct processed_features from individual columns
        processed_features = sql_trial.processed_features or {}
        if sql_trial.step_count is not None:
            processed_features['step_count'] = sql_trial.step_count
        if sql_trial.step_variance is not None:
            processed_features['step_variance'] = sql_trial.step_variance
        if sql_trial.instability_loss is not None:
            processed_features['instability_loss'] = sql_trial.instability_loss
        
        return Trial(
            id=sql_trial.id,
            participant_id=sql_trial.participant_id,
            geometry_id=sql_trial.geometry_id,
            geometry_name=geometry_name,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            source=source_enum,
            raw_data_path=sql_trial.raw_data_path,
            processed_features=processed_features if processed_features else None,
            survey_responses=survey_responses,
            steps=sql_trial.steps,
            metabolic_cost=sql_trial.metabolic_cost,
            total_combined_loss=sql_trial.total_combined_loss,
            instability_loss=sql_trial.instability_loss,
            laps_completed=sql_trial.laps,
            timestamp=sql_trial.timestamp,
            deleted=sql_trial.deleted_at
        )
    
