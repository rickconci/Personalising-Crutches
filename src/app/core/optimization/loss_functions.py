"""
Loss functions for crutch optimization.

This module provides various loss functions for different optimization objectives.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from ..config import experiment_config


class LossFunction(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def calculate(self, trial_data: Dict[str, Any], objective: str) -> float:
        """
        Calculate loss for a trial.
        
        Args:
            trial_data: Trial data dictionary
            objective: Optimization objective
            
        Returns:
            Loss value
        """
        pass


class CombinedLossFunction(LossFunction):
    """
    Combined loss function that incorporates both quantitative and survey metrics.
    
    This is the main loss function used in the original system.
    """
    
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the combined loss function.
        
        Args:
            custom_weights: Custom metric weights (optional)
        """
        self.weights = custom_weights or experiment_config.METRIC_WEIGHTS
    
    def calculate(self, trial_data: Dict[str, Any], objective: str) -> float:
        """
        Calculate combined loss for a trial.
        
        Args:
            trial_data: Trial data dictionary
            objective: Optimization objective
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Get relevant metrics for the objective
        quantitative_metrics = experiment_config.OBJECTIVE_TO_QUANTITATIVE_METRICS.get(objective, [])
        survey_metrics = experiment_config.OBJECTIVE_TO_SURVEY_METRICS.get(objective, [])
        
        # Add quantitative metrics
        for metric in quantitative_metrics:
            if metric in trial_data and trial_data[metric] is not None:
                weight = self.weights.get(metric, 1.0)
                total_loss += trial_data[metric] * weight
        
        # Add survey metrics
        survey_responses = trial_data.get('survey_responses', {})
        for metric in survey_metrics:
            if metric in survey_responses and survey_responses[metric] is not None:
                weight = self.weights.get(metric, 1.0)
                total_loss += survey_responses[metric] * weight
        
        return total_loss


class EffortLossFunction(LossFunction):
    """Loss function specifically for effort optimization."""
    
    def calculate(self, trial_data: Dict[str, Any], objective: str) -> float:
        """Calculate effort-based loss."""
        if objective != 'effort':
            raise ValueError("EffortLossFunction can only be used for 'effort' objective")
        
        loss = 0.0
        
        # Metabolic cost (primary metric)
        if 'metabolic_cost' in trial_data and trial_data['metabolic_cost'] is not None:
            loss += trial_data['metabolic_cost'] * self.weights.get('metabolic_cost', 20.0)
        
        # Survey response
        survey_responses = trial_data.get('survey_responses', {})
        if 'effort_survey_answer' in survey_responses:
            loss += survey_responses['effort_survey_answer'] * self.weights.get('effort_survey_answer', 1.0)
        
        return loss
    
    def __init__(self):
        self.weights = experiment_config.METRIC_WEIGHTS


class StabilityLossFunction(LossFunction):
    """Loss function specifically for stability optimization."""
    
    def calculate(self, trial_data: Dict[str, Any], objective: str) -> float:
        """Calculate stability-based loss."""
        if objective != 'stability':
            raise ValueError("StabilityLossFunction can only be used for 'stability' objective")
        
        loss = 0.0
        
        # Step variance (primary stability metric)
        if 'step_variance' in trial_data and trial_data['step_variance'] is not None:
            loss += trial_data['step_variance'] * self.weights.get('step_variance', 100.0)
        
        # Y-axis metrics
        if 'y_change' in trial_data and trial_data['y_change'] is not None:
            loss += trial_data['y_change'] * self.weights.get('y_change', 2.0)
        
        if 'y_total' in trial_data and trial_data['y_total'] is not None:
            loss += trial_data['y_total'] * self.weights.get('y_total', 100.0)
        
        # Survey response
        survey_responses = trial_data.get('survey_responses', {})
        if 'stability_survey_answer' in survey_responses:
            loss += survey_responses['stability_survey_answer'] * self.weights.get('stability_survey_answer', 1.0)
        
        return loss
    
    def __init__(self):
        self.weights = experiment_config.METRIC_WEIGHTS


class PainLossFunction(LossFunction):
    """Loss function specifically for pain optimization."""
    
    def calculate(self, trial_data: Dict[str, Any], objective: str) -> float:
        """Calculate pain-based loss."""
        if objective != 'pain':
            raise ValueError("PainLossFunction can only be used for 'pain' objective")
        
        loss = 0.0
        
        # Only survey response for pain (no quantitative metrics)
        survey_responses = trial_data.get('survey_responses', {})
        if 'pain_survey_answer' in survey_responses:
            loss += survey_responses['pain_survey_answer'] * self.weights.get('pain_survey_answer', 1.0)
        
        return loss
    
    def __init__(self):
        self.weights = experiment_config.METRIC_WEIGHTS


class MultiObjectiveLossFunction(LossFunction):
    """
    Multi-objective loss function for optimizing multiple objectives simultaneously.
    
    This function combines multiple objectives using weighted sum approach.
    """
    
    def __init__(
        self, 
        objectives: List[str], 
        objective_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize multi-objective loss function.
        
        Args:
            objectives: List of objectives to optimize
            objective_weights: Weights for each objective
        """
        self.objectives = objectives
        self.objective_weights = objective_weights or {obj: 1.0 for obj in objectives}
        
        # Create individual loss functions
        self.loss_functions = {
            'effort': EffortLossFunction(),
            'stability': StabilityLossFunction(),
            'pain': PainLossFunction()
        }
    
    def calculate(self, trial_data: Dict[str, Any], objective: str) -> float:
        """
        Calculate multi-objective loss.
        
        Args:
            trial_data: Trial data dictionary
            objective: Not used in multi-objective (should be 'multi')
            
        Returns:
            Combined multi-objective loss value
        """
        total_loss = 0.0
        
        for obj in self.objectives:
            if obj in self.loss_functions:
                obj_loss = self.loss_functions[obj].calculate(trial_data, obj)
                weight = self.objective_weights.get(obj, 1.0)
                total_loss += obj_loss * weight
        
        return total_loss


def get_loss_function(objective: str, custom_weights: Optional[Dict[str, float]] = None) -> LossFunction:
    """
    Get appropriate loss function for an objective.
    
    Args:
        objective: Optimization objective
        custom_weights: Custom metric weights
        
    Returns:
        Loss function instance
    """
    if objective == 'effort':
        return EffortLossFunction()
    elif objective == 'stability':
        return StabilityLossFunction()
    elif objective == 'pain':
        return PainLossFunction()
    elif objective == 'multi':
        # Default multi-objective with all objectives
        return MultiObjectiveLossFunction(['effort', 'stability', 'pain'])
    else:
        return CombinedLossFunction(custom_weights)
