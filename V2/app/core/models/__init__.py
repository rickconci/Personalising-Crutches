"""
Data models for the Personalising Crutches application.

This module contains Pydantic models for data validation and serialization.
"""

from .participant import ParticipantBase, ParticipantCreate, ParticipantUpdate, Participant, ParticipantResponse
from .geometry import CrutchGeometryBase, GeometryCreate, GeometryUpdate, CrutchGeometry, GeometryResponse, GeometrySuggestion
from .trial import TrialBase, TrialCreate, TrialUpdate, Trial, TrialResponse, TrialAnalysis, SurveyResponse, TrialSource
from .data_upload import (
    DataUpload,
    DataUploadResponse,
    DataProcessingRequest,
    DataProcessingResponse,
    AlgorithmComparisonRequest,
    AlgorithmComparisonResponse
)

__all__ = [
    "ParticipantBase",
    "ParticipantCreate", 
    "ParticipantUpdate",
    "Participant",
    "ParticipantResponse",
    "CrutchGeometryBase",
    "GeometryCreate",
    "GeometryUpdate",
    "CrutchGeometry",
    "GeometryResponse",
    "GeometrySuggestion",
    "TrialBase",
    "TrialCreate",
    "TrialUpdate",
    "Trial",
    "TrialResponse",
    "TrialAnalysis",
    "SurveyResponse",
    "TrialSource",
    "DataUpload",
    "DataUploadResponse",
    "DataProcessingRequest",
    "DataProcessingResponse",
    "AlgorithmComparisonRequest",
    "AlgorithmComparisonResponse",
]
