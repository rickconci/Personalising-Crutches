"""
Participant data models.

Defines Pydantic models for participant data validation and serialization.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ParticipantBase(BaseModel):
    """Base participant model with common fields."""
    name: str = Field(..., min_length=1, max_length=100, description="Participant ID (e.g., P001, Subject_123)")
    characteristics: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Participant characteristics (height, weight, etc.)"
    )


class ParticipantCreate(ParticipantBase):
    """Model for creating a new participant."""
    pass


class ParticipantUpdate(BaseModel):
    """Model for updating participant information."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    characteristics: Optional[Dict[str, Any]] = None


class Participant(ParticipantBase):
    """Complete participant model with database fields."""
    id: int = Field(..., description="Unique participant identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ParticipantResponse(BaseModel):
    """Response model for participant data."""
    id: int
    name: str
    characteristics: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: Optional[str] = None
