"""
Bayesian optimization for crutch geometry optimization.

This module provides a clean interface for Bayesian optimization using GPy and GPyOpt,
consolidating the optimization logic from the original codebase.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field
try:
    import GPy
    from GPyOpt.methods import BayesianOptimization
    GPY_AVAILABLE = True
except ImportError:
    GPY_AVAILABLE = False
    print("Warning: GPy and GPyOpt not available. Install with: pip install gpy gpyopt")

from ..config import experiment_config
from .loss_functions import LossFunction, CombinedLossFunction


class AcquisitionFunction(str, Enum):
    """Available acquisition functions for Bayesian optimization."""
    EXPECTED_IMPROVEMENT = "EI"
    PROBABILITY_IMPROVEMENT = "PI"
    UPPER_CONFIDENCE_BOUND = "UCB"



class OptimizationResult(BaseModel):
    """Result of Bayesian optimization."""
    suggested_geometry: Dict[str, float] = Field(..., description="Suggested crutch geometry")
    confidence: Optional[float] = Field(None, description="Confidence score")
    acquisition_value: Optional[float] = Field(None, description="Acquisition function value")
    optimization_history: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optimization history as list of records"
    )
    processing_time: float = Field(0.0, description="Processing time")


class BayesianOptimizer:
    """
    Bayesian optimization for crutch geometry optimization.
    
    This class provides a clean interface for optimizing crutch parameters
    using Gaussian Processes and various acquisition functions.
    """
    
    def __init__(
        self,
        acquisition_type: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT,
        exact_feval: bool = True,
        kernel_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            acquisition_type: Acquisition function to use
            exact_feval: Whether to use exact function evaluations
            kernel_params: Custom kernel parameters
        """
        if not GPY_AVAILABLE:
            raise ImportError("GPy and GPyOpt are required for Bayesian optimization")
        
        self.acquisition_type = acquisition_type.value
        self.exact_feval = exact_feval
        self.kernel_params = kernel_params or experiment_config.KERNEL_PARAMS
        self.tested_geometries = set()
        
    def optimize(
        self,
        experiment_data: pd.DataFrame,
        objective: str,
        user_characteristics: Optional[Dict[str, float]] = None,
        max_iterations: int = 10,
        **kwargs
    ) -> OptimizationResult:
        """
        Run Bayesian optimization to suggest next crutch geometry.
        
        Args:
            experiment_data: Historical trial data
            objective: Optimization objective ('effort', 'stability', 'pain')
            user_characteristics: User characteristics for personalization
            max_iterations: Maximum optimization iterations
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult with suggested geometry
        """
        import time
        start_time = time.time()
        
        # Validate objective
        if objective not in experiment_config.OBJECTIVES:
            raise ValueError(f"Invalid objective: {objective}. Must be one of {experiment_config.OBJECTIVES}")
        
        # Prepare data for optimization
        X, Y = self._prepare_optimization_data(experiment_data, objective, user_characteristics)
        
        if len(X) < 2:
            # Not enough data for optimization, return default geometry
            suggested_geometry = experiment_config.INITIAL_CRUTCH_GEOMETRY.copy()
            return OptimizationResult(
                suggested_geometry=suggested_geometry,
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Define search space
        search_space = self._create_search_space()
        
        # Create dummy objective function (we provide data via X, Y)
        def dummy_objective(x):
            return np.array([[0]])
        
        # Create Bayesian optimization object
        bo = BayesianOptimization(
            f=dummy_objective,
            domain=search_space,
            model_type='GP',
            kernel=self._create_kernel(),
            acquisition_type=self.acquisition_type,
            exact_feval=self.exact_feval,
            X=X,
            Y=Y
        )
        
        # Suggest next location
        next_params = bo.suggest_next_locations()
        
        # Round to nearest allowed values
        suggested_geometry = self._round_to_discrete_values(next_params[0])
        
        # Calculate confidence (acquisition function value)
        acquisition_value = self._calculate_acquisition_value(bo, next_params[0])
        
        processing_time = time.time() - start_time
        
        return OptimizationResult(
            suggested_geometry=suggested_geometry,
            confidence=acquisition_value,
            acquisition_value=acquisition_value,
            optimization_history=experiment_data.to_dict(orient="records") if experiment_data is not None else None,
            processing_time=processing_time
        )
    
    def _prepare_optimization_data(
        self,
        experiment_data: pd.DataFrame,
        objective: str,
        user_characteristics: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Bayesian optimization.
        
        Args:
            experiment_data: Historical trial data
            objective: Optimization objective
            user_characteristics: User characteristics
            
        Returns:
            Tuple of (X, Y) arrays for optimization
        """
        # Extract crutch parameters
        crutch_params = ['alpha', 'beta', 'gamma']
        X_crutch = experiment_data[crutch_params].values
        
        # Add user characteristics if provided
        if user_characteristics:
            user_chars = [user_characteristics.get(char, 0) for char in experiment_config.USER_CHARACTERISTICS]
            X_user = np.tile(user_chars, (len(X_crutch), 1))
            X = np.hstack([X_user, X_crutch])
        else:
            X = X_crutch
        
        # Calculate loss values
        Y = self._calculate_loss_values(experiment_data, objective)
        
        return X, Y
    
    def _calculate_loss_values(self, experiment_data: pd.DataFrame, objective: str) -> np.ndarray:
        """
        Calculate loss values for optimization.
        
        Args:
            experiment_data: Historical trial data
            objective: Optimization objective
            
        Returns:
            Array of loss values
        """
        # Get relevant metrics for the objective
        quantitative_metrics = experiment_config.OBJECTIVE_TO_QUANTITATIVE_METRICS.get(objective, [])
        survey_metrics = experiment_config.OBJECTIVE_TO_SURVEY_METRICS.get(objective, [])
        
        loss_values = []
        
        for _, trial in experiment_data.iterrows():
            total_loss = 0.0
            
            # Add quantitative metrics
            for metric in quantitative_metrics:
                if metric in trial and pd.notna(trial[metric]):
                    weight = experiment_config.METRIC_WEIGHTS.get(metric, 1.0)
                    total_loss += trial[metric] * weight
            
            # Add survey metrics
            for metric in survey_metrics:
                if metric in trial and pd.notna(trial[metric]):
                    weight = experiment_config.METRIC_WEIGHTS.get(metric, 1.0)
                    total_loss += trial[metric] * weight
            
            loss_values.append(total_loss)
        
        return np.array(loss_values).reshape(-1, 1)
    
    def _create_search_space(self) -> List[Dict[str, Any]]:
        """Create search space for Bayesian optimization."""
        return experiment_config.CRUTCH_PARAM_BOUNDARIES.copy()
    
    def _create_kernel(self) -> GPy.kern.Kern:
        """Create Gaussian Process kernel."""
        # Calculate input dimension
        num_user_chars = len(experiment_config.USER_CHARACTERISTICS)
        num_crutch_params = len(experiment_config.CRUTCH_PARAM_BOUNDARIES)
        total_dim = num_user_chars + num_crutch_params
        
        return GPy.kern.Matern52(
            input_dim=total_dim,
            variance=self.kernel_params['variance'],
            lengthscale=self.kernel_params['lengthscale']
        )
    
    def _round_to_discrete_values(self, suggested_params: np.ndarray) -> Dict[str, float]:
        """
        Round suggested parameters to nearest discrete values.
        
        Args:
            suggested_params: Suggested parameter values
            
        Returns:
            Dictionary of rounded parameters
        """
        # Extract crutch parameters (assuming user characteristics come first)
        num_user_chars = len(experiment_config.USER_CHARACTERISTICS)
        crutch_params = suggested_params[num_user_chars:]
        
        rounded_params = {}
        for i, param_config in enumerate(experiment_config.CRUTCH_PARAM_BOUNDARIES):
            param_name = param_config['name']
            domain = param_config['domain']
            suggested_value = crutch_params[i]
            
            # Round to nearest allowed value
            rounded_value = min(domain, key=lambda x: abs(x - suggested_value))
            rounded_params[param_name] = float(rounded_value)
        
        return rounded_params
    
    def _calculate_acquisition_value(self, bo: BayesianOptimization, suggested_params: np.ndarray) -> float:
        """
        Calculate acquisition function value for confidence estimation.
        
        Args:
            bo: Bayesian optimization object
            suggested_params: Suggested parameter values
            
        Returns:
            Acquisition function value
        """
        try:
            # Get acquisition function value
            acquisition_value = bo.acquisition.acquisition_function(suggested_params.reshape(1, -1))
            return float(acquisition_value[0, 0])
        except Exception:
            return 0.0
    
    def suggest_alternative_geometry(
        self,
        base_geometry: Dict[str, float],
        experiment_data: pd.DataFrame,
        max_attempts: int = 10
    ) -> Dict[str, float]:
        """
        Suggest an alternative geometry that hasn't been tested.
        
        Args:
            base_geometry: Base geometry to modify
            experiment_data: Historical trial data
            max_attempts: Maximum attempts to find alternative
            
        Returns:
            Alternative geometry
        """
        # Get tested geometries
        tested_geometries = set()
        for _, trial in experiment_data.iterrows():
            if all(pd.notna(trial[param]) for param in ['alpha', 'beta', 'gamma']):
                geometry_tuple = (trial['alpha'], trial['beta'], trial['gamma'])
                tested_geometries.add(geometry_tuple)
        
        # Try to find alternative
        for _ in range(max_attempts):
            # Add small random variations
            alternative = base_geometry.copy()
            for param in ['alpha', 'beta', 'gamma']:
                if param in alternative:
                    # Add random variation within step size
                    step_size = experiment_config.CRUTCH_PARAM_STEPS.get(param, 1)
                    variation = np.random.randint(-step_size, step_size + 1)
                    alternative[param] = max(0, alternative[param] + variation)
            
            # Check if this geometry has been tested
            geometry_tuple = (alternative['alpha'], alternative['beta'], alternative['gamma'])
            if geometry_tuple not in tested_geometries:
                return alternative
        
        # If no alternative found, return base geometry
        return base_geometry
