"""
Data upload models.

Defines Pydantic models for data upload and processing.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DataUpload(BaseModel):
    """Model for data upload request."""
    trial_id: Optional[int] = Field(None, description="Associated trial ID")
    participant_id: Optional[int] = Field(None, description="Associated participant ID")
    file_type: Optional[str] = Field(None, description="Expected file type")
    description: Optional[str] = Field(None, description="File description")


class DataUploadResponse(BaseModel):
    """Response model for data upload."""
    id: int = Field(..., description="File ID")
    filename: str = Field(..., description="Stored filename")
    file_path: str = Field(..., description="File path")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File type")
    uploaded_at: str = Field(..., description="Upload timestamp")


class DataProcessingRequest(BaseModel):
    """Model for data processing request."""
    file_id: int = Field(..., description="File ID to process")
    algorithm: str = Field(default="algo6_javascript", description="Step detection algorithm")
    use_force_gradient: bool = Field(default=False, description="Use force gradient signal")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Algorithm-specific parameters")


class DataProcessingResponse(BaseModel):
    """Response model for data processing."""
    file_id: int = Field(..., description="File ID")
    processing_results: Dict[str, Any] = Field(..., description="Processing results")
    status: str = Field(..., description="Processing status")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class AlgorithmComparisonRequest(BaseModel):
    """Model for algorithm comparison request."""
    file_id: int = Field(..., description="File ID to process")
    algorithms: Optional[list[str]] = Field(None, description="Algorithms to compare")
    use_force_gradient: bool = Field(default=False, description="Use force gradient signal")


class AlgorithmComparisonResponse(BaseModel):
    """Response model for algorithm comparison."""
    file_id: int = Field(..., description="File ID")
    comparison_results: Dict[str, Any] = Field(..., description="Comparison results")
    status: str = Field(..., description="Comparison status")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
