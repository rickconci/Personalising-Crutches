"""
Optimization API endpoints.

This module provides endpoints for Bayesian optimization of crutch geometry.
"""

from fastapi import APIRouter, HTTPException, Depends, Form
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from ..core.services import get_optimization_service
from ..core.optimization.bayesian_optimizer import AcquisitionFunction
from ..core.models.geometry import CrutchGeometryBase
from database.connection import get_db

router = APIRouter()

# Dependency to get optimization service
def get_optimization_service_instance(db: Session = Depends(get_db)):
    OptimizationService = get_optimization_service()
    return OptimizationService(db)

@router.post("/suggest-geometry/{participant_id}")
async def suggest_next_geometry(
    participant_id: int,
    objective: str = Form(...),
    acquisition_function: str = Form("EI"),
    max_iterations: int = Form(10),
    service = Depends(get_optimization_service_instance)
):
    """Suggest next crutch geometry for a participant using Bayesian optimization."""
    try:
        # Validate acquisition function
        try:
            acq_func = AcquisitionFunction(acquisition_function)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid acquisition function: {acquisition_function}. Must be one of {[af.value for af in AcquisitionFunction]}"
            )
        
        # Run optimization
        result = service.suggest_next_geometry(
            participant_id=participant_id,
            objective=objective,
            acquisition_function=acq_func,
            max_iterations=max_iterations
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get("/history/{participant_id}")
async def get_optimization_history(
    participant_id: int,
    objective: Optional[str] = None,
    service = Depends(get_optimization_service_instance)
):
    """Get optimization history for a participant."""
    try:
        history = service.get_optimization_history(
            participant_id=participant_id,
            objective=objective
        )
        return {
            "participant_id": participant_id,
            "objective": objective,
            "optimization_runs": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-geometries/{participant_id}")
async def compare_geometries(
    participant_id: int,
    geometries: str,  # JSON string of geometries
    service = Depends(get_optimization_service_instance)
):
    """Compare multiple geometries for a participant."""
    try:
        import json
        geometry_list = json.loads(geometries)
        
        if not isinstance(geometry_list, list):
            raise ValueError("Geometries must be a list")
        
        # Validate geometry format
        for geom in geometry_list:
            if not all(key in geom for key in ['alpha', 'beta', 'gamma']):
                raise ValueError("Each geometry must have alpha, beta, and gamma parameters")
        
        result = service.compare_geometries(
            participant_id=participant_id,
            geometries=geometry_list
        )
        
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for geometries")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{participant_id}")
async def get_optimization_recommendations(
    participant_id: int,
    objective: str,
    service = Depends(get_optimization_service_instance)
):
    """Get optimization recommendations for a participant."""
    try:
        recommendations = service.get_optimization_recommendations(
            participant_id=participant_id,
            objective=objective
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/acquisition-functions")
async def get_acquisition_functions():
    """Get list of available acquisition functions."""
    functions = [
        {
            "name": af.value,
            "description": _get_acquisition_function_description(af)
        }
        for af in AcquisitionFunction
    ]
    return {"acquisition_functions": functions}

@router.get("/objectives")
async def get_optimization_objectives():
    """Get list of available optimization objectives."""
    from ..core.config import experiment_config
    
    objectives = [
        {
            "name": obj,
            "description": _get_objective_description(obj),
            "quantitative_metrics": experiment_config.OBJECTIVE_TO_QUANTITATIVE_METRICS.get(obj, []),
            "survey_metrics": experiment_config.OBJECTIVE_TO_SURVEY_METRICS.get(obj, [])
        }
        for obj in experiment_config.OBJECTIVES
    ]
    return {"objectives": objectives}

def _get_acquisition_function_description(acquisition_function: AcquisitionFunction) -> str:
    """Get description for an acquisition function."""
    descriptions = {
        AcquisitionFunction.EXPECTED_IMPROVEMENT: "Expected Improvement - balances exploration and exploitation",
        AcquisitionFunction.PROBABILITY_IMPROVEMENT: "Probability of Improvement - focuses on exploitation",
        AcquisitionFunction.UPPER_CONFIDENCE_BOUND: "Upper Confidence Bound - emphasizes exploration"
    }
    return descriptions.get(acquisition_function, "Unknown acquisition function")

def _get_objective_description(objective: str) -> str:
    """Get description for an optimization objective."""
    descriptions = {
        "effort": "Minimize effort required for crutch use (metabolic cost)",
        "stability": "Maximize stability during crutch use (step variance, Y-axis metrics)",
        "pain": "Minimize pain and discomfort during crutch use (survey responses)"
    }
    return descriptions.get(objective, "Unknown objective")
