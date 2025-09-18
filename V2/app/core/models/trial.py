"""
Trial data models.

Defines Pydantic models for experimental trial validation and serialization.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class TrialSource(str, Enum):
    """Enumeration of trial sources."""
    GRID_SEARCH = "grid_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    PAIN_OPTIMIZATION = "pain_optimization"
    MANUAL = "manual"


class TrialBase(BaseModel):
    """Base trial model with common fields."""
    participant_id: int = Field(..., description="Participant identifier")
    geometry_id: Optional[int] = Field(None, description="Predefined geometry identifier")
    
    # BO trial parameters (for trials without predefined geometries)
    alpha: Optional[float] = Field(None, ge=70, le=125)
    beta: Optional[float] = Field(None, ge=90, le=145)
    gamma: Optional[float] = Field(None, ge=-12, le=12)
    delta: Optional[float] = Field(None, ge=-10, le=10)
    
    source: TrialSource = Field(default=TrialSource.GRID_SEARCH, description="Trial source")
    raw_data_path: Optional[str] = Field(None, description="Path to raw data file")
    
    # Processed data
    processed_features: Optional[Dict[str, Any]] = Field(None, description="Processed accelerometer features")
    survey_responses: Optional[Dict[str, Any]] = Field(None, description="Subjective survey responses")
    steps: Optional[List[float]] = Field(None, description="Detected step timestamps")
    
    # Metrics
    metabolic_cost: Optional[float] = Field(None, description="Metabolic cost metric")
    total_combined_loss: Optional[float] = Field(None, description="Total combined loss value")


class TrialCreate(TrialBase):
    """Model for creating a new trial."""
    pass


class TrialUpdate(BaseModel):
    """Model for updating trial data."""
    processed_features: Optional[Dict[str, Any]] = None
    survey_responses: Optional[Dict[str, Any]] = None
    steps: Optional[List[float]] = None
    metabolic_cost: Optional[float] = None
    total_combined_loss: Optional[float] = None


class Trial(TrialBase):
    """Complete trial model with database fields."""
    id: int = Field(..., description="Unique trial identifier")
    timestamp: datetime = Field(..., description="Trial timestamp")
    deleted: Optional[datetime] = Field(None, description="Soft delete timestamp")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrialResponse(BaseModel):
    """Response model for trial data."""
    id: int
    participant_id: int
    participant_name: str
    geometry_id: Optional[int]
    geometry_name: Optional[str]
    alpha: Optional[float]
    beta: Optional[float]
    gamma: Optional[float]
    delta: Optional[float]
    source: str
    timestamp: str
    processed_features: Optional[Dict[str, Any]] = None
    survey_responses: Optional[Dict[str, Any]] = None
    steps: Optional[List[float]] = None
    metabolic_cost: Optional[float] = None
    total_combined_loss: Optional[float] = None


class TrialAnalysis(BaseModel):
    """Model for trial analysis results."""
    trial_id: int
    step_count: int
    step_variance: Optional[float] = None
    y_change: Optional[float] = None
    y_total: Optional[float] = None
    rms_load_cell_force: Optional[float] = None
    processing_time: float = Field(..., description="Analysis processing time in seconds")


class SurveyResponse(BaseModel):
    """Model for survey response data."""
    effort_survey_answer: Optional[int] = Field(None, ge=0, le=6, description="Effort rating (0-6)")
    pain_survey_answer: Optional[int] = Field(None, ge=0, le=6, description="Pain rating (0-6)")
    stability_survey_answer: Optional[int] = Field(None, ge=0, le=6, description="Stability rating (0-6)")
    
    @validator('effort_survey_answer', 'pain_survey_answer', 'stability_survey_answer')
    def validate_survey_answers(cls, v):
        if v is not None and (v < 0 or v > 6):
            raise ValueError('Survey answers must be between 0 and 6')
        return v
