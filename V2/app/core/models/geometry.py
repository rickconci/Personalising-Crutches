"""
Crutch geometry data models.

Defines Pydantic models for crutch geometry validation and serialization.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class CrutchGeometryBase(BaseModel):
    """Base crutch geometry model with common fields."""
    name: str = Field(..., min_length=1, max_length=80, description="Geometry name")
    alpha: float = Field(..., ge=70, le=125, description="Handle angle from vertical (degrees)")
    beta: float = Field(..., ge=90, le=145, description="Angle between forearm and hand grip (degrees)")
    gamma: float = Field(..., ge=-12, le=12, description="Distance between forearm and vertical strut (degrees)")
    delta: float = Field(default=0, ge=-10, le=10, description="Additional parameter (degrees)")


class GeometryCreate(CrutchGeometryBase):
    """Model for creating a new crutch geometry."""
    pass


class GeometryUpdate(BaseModel):
    """Model for updating crutch geometry."""
    name: Optional[str] = Field(None, min_length=1, max_length=80)
    alpha: Optional[float] = Field(None, ge=70, le=125)
    beta: Optional[float] = Field(None, ge=90, le=145)
    gamma: Optional[float] = Field(None, ge=-12, le=12)
    delta: Optional[float] = Field(None, ge=-10, le=10)


class CrutchGeometry(CrutchGeometryBase):
    """Complete crutch geometry model with database fields."""
    id: int = Field(..., description="Unique geometry identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GeometryResponse(BaseModel):
    """Response model for crutch geometry data."""
    id: int
    name: str
    alpha: float
    beta: float
    gamma: float
    delta: float
    created_at: str
    updated_at: Optional[str] = None


class GeometrySuggestion(BaseModel):
    """Model for Bayesian optimization geometry suggestions."""
    alpha: float = Field(..., description="Suggested alpha angle")
    beta: float = Field(..., description="Suggested beta angle")
    gamma: float = Field(..., description="Suggested gamma angle")
    confidence: Optional[float] = Field(None, description="Optimization confidence score")
    acquisition_value: Optional[float] = Field(None, description="Acquisition function value")
