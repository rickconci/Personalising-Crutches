"""
SQLAlchemy database models for the Personalising Crutches application.

This module defines the database schema using SQLAlchemy ORM models.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any, List

from .connection import Base


class Participant(Base):
    """Represents a participant in the study."""
    
    __tablename__ = "participants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)  # Used as Participant ID (e.g., P001, Subject_123)
    characteristics = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)
    
    # Relationships
    trials = relationship("Trial", back_populates="participant", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Participant {self.name}>"
    
    @property
    def full_name(self) -> str:
        """Get the full name of the participant."""
        return self.name




class CrutchGeometry(Base):
    """Represents a crutch geometry configuration."""
    
    __tablename__ = "crutch_geometries"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    alpha = Column(Float, nullable=False)
    beta = Column(Float, nullable=False)
    gamma = Column(Float, nullable=False)
    delta = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)
    
    # Relationships
    trials = relationship("Trial", back_populates="geometry")
    
    def __repr__(self) -> str:
        return f"<CrutchGeometry α={self.alpha}° β={self.beta}° γ={self.gamma}°>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert geometry to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Trial(Base):
    """Represents a single experimental trial."""
    
    __tablename__ = "trials"
    
    id = Column(Integer, primary_key=True, index=True)
    participant_id = Column(Integer, ForeignKey("participants.id"), nullable=False, index=True)
    geometry_id = Column(Integer, ForeignKey("crutch_geometries.id"), nullable=False, index=True)
    
    # Geometry parameters (denormalized for easier querying and analysis)
    alpha = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    delta = Column(Float, nullable=True)
    
    # Trial metadata
    experiment_session_id = Column(Integer, ForeignKey("experiment_sessions.id"), nullable=True, index=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    source = Column(String(20), nullable=False, default="grid_search", index=True)
    raw_data_path = Column(String(500), nullable=True)
    
    # Processed data
    processed_features = Column(JSON, nullable=True)
    survey_responses = Column(JSON, nullable=True)  # Legacy JSON field for backward compatibility
    steps = Column(JSON, nullable=True)  # List of step timestamps
    
    # Detailed survey response columns
    # SUS (System Usability Scale) - 6 questions
    sus_q1 = Column(Integer, nullable=True)
    sus_q2 = Column(Integer, nullable=True)
    sus_q3 = Column(Integer, nullable=True)
    sus_q4 = Column(Integer, nullable=True)
    sus_q5 = Column(Integer, nullable=True)
    sus_q6 = Column(Integer, nullable=True)
    sus_score = Column(Float, nullable=True)
    
    # NRS (Numeric Rating Scale) - Pain assessment
    nrs_score = Column(Integer, nullable=True)
    
    # NASA TLX (Task Load Index) - 5 questions
    tlx_mental_demand = Column(Integer, nullable=True)
    tlx_physical_demand = Column(Integer, nullable=True)
    tlx_performance = Column(Integer, nullable=True)
    tlx_effort = Column(Integer, nullable=True)
    tlx_frustration = Column(Integer, nullable=True)
    tlx_score = Column(Integer, nullable=True)
    
    # Metrics
    metabolic_cost = Column(Float, nullable=True)
    y_change = Column(Float, nullable=True)
    y_total = Column(Float, nullable=True)
    step_count = Column(Integer, nullable=True)  # Number of steps detected
    step_variance = Column(Float, nullable=True)
    instability_loss = Column(Float, nullable=True)  # Instability metric
    rms_load_cell_force = Column(Float, nullable=True)
    total_combined_loss = Column(Float, nullable=True)
    laps = Column(Integer, nullable=True)  # Number of laps completed
    
    # Soft delete
    deleted_at = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)
    
    # Relationships
    participant = relationship("Participant", back_populates="trials")
    geometry = relationship("CrutchGeometry", back_populates="trials")
    experiment_session = relationship("ExperimentSession", back_populates="trials")
    
    def __repr__(self) -> str:
        return f"<Trial {self.id} for Participant {self.participant_id}>"
    
    @property
    def is_deleted(self) -> bool:
        """Check if trial is soft deleted."""
        return self.deleted_at is not None
    
    def soft_delete(self) -> None:
        """Soft delete the trial."""
        self.deleted_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trial to dictionary."""
        return {
            "id": self.id,
            "participant_id": self.participant_id,
            "participant_name": self.participant.name if self.participant else None,
            "geometry_id": self.geometry_id,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "geometry_name": self.geometry.name if self.geometry else None,
            "alpha": self.alpha if self.alpha is not None else (self.geometry.alpha if self.geometry else None),
            "beta": self.beta if self.beta is not None else (self.geometry.beta if self.geometry else None),
            "gamma": self.gamma if self.gamma is not None else (self.geometry.gamma if self.geometry else None),
            "delta": self.delta if self.delta is not None else (self.geometry.delta if self.geometry else None),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
            "raw_data_path": self.raw_data_path,
            "processed_features": self.processed_features,
            "survey_responses": self.survey_responses,  # Legacy JSON field
            "steps": self.steps,
            "metabolic_cost": self.metabolic_cost,
            "y_change": self.y_change,
            "y_total": self.y_total,
            "step_count": self.step_count,
            "step_variance": self.step_variance,
            "instability_loss": self.instability_loss,
            "rms_load_cell_force": self.rms_load_cell_force,
            "total_combined_loss": self.total_combined_loss,
            "laps": self.laps,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            # Detailed survey response fields
            "sus_q1": self.sus_q1,
            "sus_q2": self.sus_q2,
            "sus_q3": self.sus_q3,
            "sus_q4": self.sus_q4,
            "sus_q5": self.sus_q5,
            "sus_q6": self.sus_q6,
            "sus_score": self.sus_score,
            "nrs_score": self.nrs_score,
            "tlx_mental_demand": self.tlx_mental_demand,
            "tlx_physical_demand": self.tlx_physical_demand,
            "tlx_performance": self.tlx_performance,
            "tlx_effort": self.tlx_effort,
            "tlx_frustration": self.tlx_frustration,
            "tlx_score": self.tlx_score,
        }


class ExperimentSession(Base):
    """Represents an experiment session for a participant."""
    
    __tablename__ = "experiment_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    participant_id = Column(Integer, ForeignKey("participants.id"), nullable=False, index=True)
    objective = Column(String(20), nullable=False, index=True)  # 'effort', 'stability', 'pain'
    status = Column(String(20), nullable=False, default="active", index=True)  # 'active', 'completed', 'paused'
    
    # Session metadata
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Configuration
    configuration = Column(JSON, nullable=True)  # Store experiment configuration
    
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)
    
    # Relationships
    participant = relationship("Participant")
    trials = relationship("Trial", back_populates="experiment_session")
    
    def __repr__(self) -> str:
        return f"<ExperimentSession {self.id} for Participant {self.participant_id}>"


class DataFile(Base):
    """Represents uploaded data files."""
    
    __tablename__ = "data_files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)  # 'csv', 'parquet', etc.
    
    # File metadata
    uploaded_at = Column(DateTime, server_default=func.now(), nullable=False)
    processed_at = Column(DateTime, nullable=True)
    processing_status = Column(String(20), nullable=False, default="pending", index=True)  # 'pending', 'processing', 'completed', 'failed'
    
    # Associated trial
    trial_id = Column(Integer, ForeignKey("trials.id"), nullable=True, index=True)
    
    # File metadata
    file_metadata = Column(JSON, nullable=True)  # Store file-specific metadata
    
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)
    
    # Relationships
    trial = relationship("Trial", backref="data_files")
    
    def __repr__(self) -> str:
        return f"<DataFile {self.filename}>"


class OptimizationRun(Base):
    """Represents a Bayesian optimization run."""
    
    __tablename__ = "optimization_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    participant_id = Column(Integer, ForeignKey("participants.id"), nullable=False, index=True)
    experiment_session_id = Column(Integer, ForeignKey("experiment_sessions.id"), nullable=True, index=True)
    objective = Column(String(20), nullable=False, index=True)
    
    # Optimization parameters
    acquisition_function = Column(String(20), nullable=False, default="EI")
    kernel_type = Column(String(20), nullable=False, default="Matern52")
    max_iterations = Column(Integer, nullable=False, default=10)
    
    # Results
    suggested_geometry = Column(JSON, nullable=True)  # Suggested crutch geometry
    confidence_score = Column(Float, nullable=True)
    acquisition_value = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), nullable=False, default="pending", index=True)  # 'pending', 'running', 'completed', 'failed'
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Results and metadata
    optimization_history = Column(JSON, nullable=True)  # Store optimization history
    run_metadata = Column(JSON, nullable=True)  # Additional metadata
    
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)
    
    # Relationships
    participant = relationship("Participant")
    experiment_session = relationship("ExperimentSession")
    
    def __repr__(self) -> str:
        return f"<OptimizationRun {self.id} for Participant {self.participant_id}>"
