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


class SurveyResponse(BaseModel):
    """Model for detailed survey response data."""
    
    # Adapted SUS (System Usability Scale) - 6 questions
    sus_q1: Optional[int] = Field(None, ge=1, le=5, description="SUS Q1: I would like to use crutches frequently")
    sus_q2: Optional[int] = Field(None, ge=1, le=5, description="SUS Q2: Crutches were easy to use")
    sus_q3: Optional[int] = Field(None, ge=1, le=5, description="SUS Q3: Need technical support to use crutches")
    sus_q4: Optional[int] = Field(None, ge=1, le=5, description="SUS Q4: Most people would learn quickly")
    sus_q5: Optional[int] = Field(None, ge=1, le=5, description="SUS Q5: Felt confident using crutches")
    sus_q6: Optional[int] = Field(None, ge=1, le=5, description="SUS Q6: Needed to learn a lot before getting going")
    sus_score: Optional[float] = Field(None, ge=0, le=100, description="Calculated SUS score (0-100)")
    
    # NRS (Numeric Rating Scale) - Pain assessment
    nrs_score: Optional[int] = Field(None, ge=0, le=10, description="NRS pain score (0-10)")
    
    # NASA TLX (Task Load Index) - 5 questions
    tlx_mental_demand: Optional[int] = Field(None, ge=0, le=20, description="TLX: Mental demand")
    tlx_physical_demand: Optional[int] = Field(None, ge=0, le=20, description="TLX: Physical demand")
    tlx_performance: Optional[int] = Field(None, ge=0, le=20, description="TLX: Performance success")
    tlx_effort: Optional[int] = Field(None, ge=0, le=20, description="TLX: Effort required")
    tlx_frustration: Optional[int] = Field(None, ge=0, le=20, description="TLX: Frustration level")
    tlx_score: Optional[int] = Field(None, ge=0, le=100, description="Calculated TLX score (0-100)")
    
    # Legacy fields for backward compatibility
    effort_survey_answer: Optional[int] = Field(None, ge=0, le=6, description="Legacy effort rating (0-6)")
    pain_survey_answer: Optional[int] = Field(None, ge=0, le=6, description="Legacy pain rating (0-6)")
    stability_survey_answer: Optional[int] = Field(None, ge=0, le=6, description="Legacy stability rating (0-6)")
    
    @validator('sus_q1', 'sus_q2', 'sus_q3', 'sus_q4', 'sus_q5', 'sus_q6')
    def validate_sus_questions(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError('SUS questions must be between 1 and 5')
        return v
    
    @validator('nrs_score')
    def validate_nrs_score(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('NRS score must be between 0 and 10')
        return v
    
    @validator('tlx_mental_demand', 'tlx_physical_demand', 'tlx_performance', 'tlx_effort', 'tlx_frustration')
    def validate_tlx_questions(cls, v):
        if v is not None and (v < 0 or v > 20):
            raise ValueError('TLX questions must be between 0 and 20')
        return v
    
    @validator('effort_survey_answer', 'pain_survey_answer', 'stability_survey_answer')
    def validate_legacy_survey_answers(cls, v):
        if v is not None and (v < 0 or v > 6):
            raise ValueError('Legacy survey answers must be between 0 and 6')
        return v
    
    def calculate_sus_score(self) -> Optional[float]:
        """Calculate SUS score from individual question responses."""
        if not all([self.sus_q1, self.sus_q2, self.sus_q3, self.sus_q4, self.sus_q5, self.sus_q6]):
            return None
        
        # Positive questions: 1, 2, 4, 5 (subtract 1)
        # Negative questions: 3, 6 (subtract from 5)
        sus_score = (
            (self.sus_q1 - 1) +  # Q1 positive
            (self.sus_q2 - 1) +  # Q2 positive
            (5 - self.sus_q3) +  # Q3 negative
            (self.sus_q4 - 1) +  # Q4 positive
            (self.sus_q5 - 1) +  # Q5 positive
            (5 - self.sus_q6)    # Q6 negative
        )
        return (sus_score / 24) * 100
    
    def calculate_tlx_score(self) -> Optional[int]:
        """Calculate TLX score from individual question responses."""
        if not all([self.tlx_mental_demand, self.tlx_physical_demand, self.tlx_performance, 
                   self.tlx_effort, self.tlx_frustration]):
            return None
        
        return (self.tlx_mental_demand + self.tlx_physical_demand + self.tlx_performance + 
                self.tlx_effort + self.tlx_frustration)


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
    survey_responses: Optional[SurveyResponse] = Field(None, description="Detailed survey responses")
    steps: Optional[List[float]] = Field(None, description="Detected step timestamps")
    opencap_events: Optional[List[Dict[str, Any]]] = Field(None, description="OpenCap toggle events: [{timestamp, state, relativeTime}]")
    
    # Metrics
    metabolic_cost: Optional[float] = Field(None, description="Metabolic cost metric")
    total_combined_loss: Optional[float] = Field(None, description="Total combined loss value")
    instability_loss: Optional[float] = Field(None, description="Instability loss metric")
    laps_completed: Optional[float] = Field(None, description="Number of laps completed during trial (can be fractional)", ge=0)


class TrialCreate(TrialBase):
    """Model for creating a new trial."""
    pass


class TrialCreateManual(BaseModel):
    """Model for creating a trial from manual geometry input."""
    alpha: float = Field(..., ge=70, le=125)
    beta: float = Field(..., ge=90, le=145)
    gamma: float = Field(..., ge=-12, le=12)
    delta: float = Field(0.0, ge=0, le=120)
    source: TrialSource = Field(default=TrialSource.MANUAL, description="Trial source")


class TrialUpdate(BaseModel):
    """Model for updating trial data."""
    processed_features: Optional[Dict[str, Any]] = None
    survey_responses: Optional[SurveyResponse] = None
    steps: Optional[List[float]] = None
    opencap_events: Optional[List[Dict[str, Any]]] = None
    metabolic_cost: Optional[float] = None
    total_combined_loss: Optional[float] = None
    laps_completed: Optional[float] = None


class Trial(TrialBase):
    """Complete trial model with database fields."""
    id: int = Field(..., description="Unique trial identifier")
    geometry_name: Optional[str] = Field(None, description="Geometry name")
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
    survey_responses: Optional[SurveyResponse] = None
    steps: Optional[List[float]] = None
    opencap_events: Optional[List[Dict[str, Any]]] = None
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
