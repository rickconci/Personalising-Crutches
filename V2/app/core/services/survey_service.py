"""
Survey service for handling detailed survey response data.

This service provides methods for processing and validating survey responses
from SUS, NRS, and NASA TLX surveys.
"""

from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from ..models.trial import SurveyResponse
from database.models import Trial


class SurveyService:
    """Service for handling survey response operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def process_survey_responses(self, survey_data: Dict[str, Any]) -> SurveyResponse:
        """
        Process raw survey data into a structured SurveyResponse object.
        
        Args:
            survey_data: Raw survey data from frontend
            
        Returns:
            Processed SurveyResponse object
        """
        # Extract SUS data
        sus_data = {}
        for i in range(1, 7):
            key = f"sus_q{i}"
            if key in survey_data:
                sus_data[key] = survey_data[key]
        if 'sus_score' in survey_data:
            sus_data['sus_score'] = survey_data['sus_score']
        
        # Extract NRS data
        nrs_data = {}
        if 'nrs_score' in survey_data:
            nrs_data['nrs_score'] = survey_data['nrs_score']
        
        # Extract TLX data
        tlx_data = {}
        tlx_field_names = ['tlx_mental_demand', 'tlx_physical_demand', 'tlx_performance', 'tlx_effort', 'tlx_frustration']
        for field_name in tlx_field_names:
            if field_name in survey_data:
                tlx_data[field_name] = survey_data[field_name]
        
        # Don't trust the tlx_score from survey_data - we'll calculate it properly
        # if 'tlx_score' in survey_data:
        #     tlx_data['tlx_score'] = survey_data['tlx_score']
        
        # Create SurveyResponse object
        survey_response = SurveyResponse(
            **sus_data,
            **nrs_data,
            **tlx_data,
            # Legacy fields for backward compatibility
            effort_survey_answer=survey_data.get('effort_survey_answer'),
            pain_survey_answer=survey_data.get('pain_survey_answer'),
            stability_survey_answer=survey_data.get('stability_survey_answer'),
        )
        
        # Calculate scores if individual questions are provided but scores are missing
        if not survey_response.sus_score and all([
            survey_response.sus_q1, survey_response.sus_q2, survey_response.sus_q3,
            survey_response.sus_q4, survey_response.sus_q5, survey_response.sus_q6
        ]):
            survey_response.sus_score = survey_response.calculate_sus_score()
        
        # Always calculate TLX score from individual dimensions to ensure it's correct
        if all([
            survey_response.tlx_mental_demand, survey_response.tlx_physical_demand,
            survey_response.tlx_performance, survey_response.tlx_effort, survey_response.tlx_frustration
        ]):
            survey_response.tlx_score = survey_response.calculate_tlx_score()
        
        return survey_response
    
    def update_trial_survey_data(self, trial_id: int, survey_data: Dict[str, Any]) -> Optional[Trial]:
        """
        Update trial with detailed survey response data.
        
        Args:
            trial_id: ID of the trial to update
            survey_data: Raw survey data from frontend
            
        Returns:
            Updated Trial object or None if not found
        """
        trial = self.db.query(Trial).filter(Trial.id == trial_id).first()
        if not trial:
            return None
        
        # Process survey responses
        survey_response = self.process_survey_responses(survey_data)
        
        # Update trial with detailed survey data
        trial.sus_q1 = survey_response.sus_q1
        trial.sus_q2 = survey_response.sus_q2
        trial.sus_q3 = survey_response.sus_q3
        trial.sus_q4 = survey_response.sus_q4
        trial.sus_q5 = survey_response.sus_q5
        trial.sus_q6 = survey_response.sus_q6
        trial.sus_score = survey_response.sus_score
        
        trial.nrs_score = survey_response.nrs_score
        
        trial.tlx_mental_demand = survey_response.tlx_mental_demand
        trial.tlx_physical_demand = survey_response.tlx_physical_demand
        trial.tlx_performance = survey_response.tlx_performance
        trial.tlx_effort = survey_response.tlx_effort
        trial.tlx_frustration = survey_response.tlx_frustration
        trial.tlx_score = survey_response.tlx_score
        
        # Also update the legacy JSON field for backward compatibility
        trial.survey_responses = survey_data
        
        self.db.commit()
        self.db.refresh(trial)
        
        return trial
    
    def get_survey_summary(self, trial_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a summary of survey responses for a trial.
        
        Args:
            trial_id: ID of the trial
            
        Returns:
            Dictionary with survey summary or None if not found
        """
        trial = self.db.query(Trial).filter(Trial.id == trial_id).first()
        if not trial:
            return None
        
        return {
            "trial_id": trial_id,
            "sus": {
                "questions": {
                    "q1": trial.sus_q1,
                    "q2": trial.sus_q2,
                    "q3": trial.sus_q3,
                    "q4": trial.sus_q4,
                    "q5": trial.sus_q5,
                    "q6": trial.sus_q6,
                },
                "score": trial.sus_score,
                "interpretation": self._interpret_sus_score(trial.sus_score)
            },
            "nrs": {
                "score": trial.nrs_score,
                "interpretation": self._interpret_nrs_score(trial.nrs_score)
            },
            "tlx": {
                "questions": {
                    "mental_demand": trial.tlx_mental_demand,
                    "physical_demand": trial.tlx_physical_demand,
                    "performance": trial.tlx_performance,
                    "effort": trial.tlx_effort,
                    "frustration": trial.tlx_frustration,
                },
                "score": trial.tlx_score,
                "interpretation": self._interpret_tlx_score(trial.tlx_score)
            }
        }
    
    def _interpret_sus_score(self, score: Optional[float]) -> str:
        """Interpret SUS score."""
        if score is None:
            return "No data"
        elif score >= 80:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "OK"
        else:
            return "Poor"
    
    def _interpret_nrs_score(self, score: Optional[int]) -> str:
        """Interpret NRS pain score."""
        if score is None:
            return "No data"
        elif score == 0:
            return "No pain"
        elif score <= 3:
            return "Mild pain"
        elif score <= 6:
            return "Moderate pain"
        else:
            return "Severe pain"
    
    def _interpret_tlx_score(self, score: Optional[int]) -> str:
        """Interpret NASA TLX score."""
        if score is None:
            return "No data"
        elif score <= 20:
            return "Very low workload"
        elif score <= 40:
            return "Low workload"
        elif score <= 60:
            return "Moderate workload"
        elif score <= 80:
            return "High workload"
        else:
            return "Very high workload"
