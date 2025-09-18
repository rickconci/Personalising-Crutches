"""
Optimization service for Bayesian optimization of crutch geometry.

This service provides high-level optimization capabilities using the
Bayesian optimization framework.
"""

from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from database.models import Trial, OptimizationRun, Participant
from ..optimization.bayesian_optimizer import BayesianOptimizer, OptimizationResult, AcquisitionFunction
from ..config import experiment_config


class OptimizationService:
    """Service for Bayesian optimization of crutch geometry."""
    
    def __init__(self, db: Session):
        """Initialize the optimization service."""
        self.db = db
    
    def suggest_next_geometry(
        self,
        participant_id: int,
        objective: str,
        acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT,
        max_iterations: int = 10,
        user_characteristics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Suggest next crutch geometry for a participant.
        
        Args:
            participant_id: Participant ID
            objective: Optimization objective ('effort', 'stability', 'pain')
            acquisition_function: Acquisition function to use
            max_iterations: Maximum optimization iterations
            user_characteristics: User characteristics for personalization
            
        Returns:
            Dictionary with suggested geometry and metadata
        """
        # Validate objective
        if objective not in experiment_config.OBJECTIVES:
            raise ValueError(f"Invalid objective: {objective}. Must be one of {experiment_config.OBJECTIVES}")
        
        # Get participant
        participant = self.db.query(Participant).filter(Participant.id == participant_id).first()
        if not participant:
            raise ValueError("Participant not found")
        
        # Get user characteristics
        if user_characteristics is None:
            user_characteristics = participant.characteristics or {}
        
        # Get historical trial data
        trials = (
            self.db.query(Trial)
            .filter(Trial.participant_id == participant_id)
            .filter(Trial.deleted_at.is_(None))
            .filter(Trial.total_combined_loss.isnot(None))
            .all()
        )
        
        if len(trials) < 2:
            # Not enough data for optimization, return default geometry
            return {
                "suggested_geometry": experiment_config.INITIAL_CRUTCH_GEOMETRY.copy(),
                "confidence": 0.0,
                "acquisition_value": 0.0,
                "optimization_status": "insufficient_data",
                "trial_count": len(trials),
                "message": "Not enough trial data for optimization. Using default geometry."
            }
        
        # Convert trials to DataFrame
        trial_data = self._trials_to_dataframe(trials)
        
        # Create optimization run record
        optimization_run = OptimizationRun(
            participant_id=participant_id,
            objective=objective,
            acquisition_function=acquisition_function.value,
            max_iterations=max_iterations,
            status="running"
        )
        self.db.add(optimization_run)
        self.db.commit()
        
        try:
            # Run Bayesian optimization
            optimizer = BayesianOptimizer(acquisition_type=acquisition_function)
            result = optimizer.optimize(
                experiment_data=trial_data,
                objective=objective,
                user_characteristics=user_characteristics,
                max_iterations=max_iterations
            )
            
            # Update optimization run
            optimization_run.status = "completed"
            optimization_run.completed_at = pd.Timestamp.now()
            optimization_run.suggested_geometry = result.suggested_geometry
            optimization_run.confidence_score = result.confidence
            optimization_run.acquisition_value = result.acquisition_value
            optimization_run.optimization_history = trial_data.to_dict('records')
            
            self.db.commit()
            
            return {
                "suggested_geometry": result.suggested_geometry,
                "confidence": result.confidence,
                "acquisition_value": result.acquisition_value,
                "optimization_status": "completed",
                "trial_count": len(trials),
                "optimization_run_id": optimization_run.id,
                "processing_time": result.processing_time
            }
            
        except Exception as e:
            # Update optimization run with error
            optimization_run.status = "failed"
            optimization_run.completed_at = pd.Timestamp.now()
            optimization_run.metadata = {"error": str(e)}
            self.db.commit()
            
            raise ValueError(f"Optimization failed: {str(e)}")
    
    def get_optimization_history(
        self, 
        participant_id: int, 
        objective: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get optimization history for a participant.
        
        Args:
            participant_id: Participant ID
            objective: Optional objective filter
            
        Returns:
            List of optimization runs
        """
        query = self.db.query(OptimizationRun).filter(OptimizationRun.participant_id == participant_id)
        
        if objective:
            query = query.filter(OptimizationRun.objective == objective)
        
        runs = query.order_by(OptimizationRun.started_at.desc()).all()
        
        return [
            {
                "id": run.id,
                "objective": run.objective,
                "acquisition_function": run.acquisition_function,
                "status": run.status,
                "suggested_geometry": run.suggested_geometry,
                "confidence_score": run.confidence_score,
                "acquisition_value": run.acquisition_value,
                "started_at": run.started_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "trial_count": len(run.optimization_history) if run.optimization_history else 0
            }
            for run in runs
        ]
    
    def compare_geometries(
        self, 
        participant_id: int, 
        geometries: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compare multiple geometries for a participant.
        
        Args:
            participant_id: Participant ID
            geometries: List of geometry dictionaries
            
        Returns:
            Comparison results
        """
        # Get participant's trial data
        trials = (
            self.db.query(Trial)
            .filter(Trial.participant_id == participant_id)
            .filter(Trial.deleted_at.is_(None))
            .all()
        )
        
        if not trials:
            return {"error": "No trial data found for participant"}
        
        # Group trials by geometry
        geometry_performance = {}
        
        for trial in trials:
            # Create geometry key
            if trial.geometry_id:
                geometry_key = f"geometry_{trial.geometry_id}"
            else:
                geometry_key = f"custom_{trial.alpha}_{trial.beta}_{trial.gamma}"
            
            if geometry_key not in geometry_performance:
                geometry_performance[geometry_key] = {
                    "geometry": {
                        "alpha": trial.alpha,
                        "beta": trial.beta,
                        "gamma": trial.gamma,
                        "delta": trial.delta
                    },
                    "trials": [],
                    "metrics": {}
                }
            
            geometry_performance[geometry_key]["trials"].append({
                "trial_id": trial.id,
                "total_combined_loss": trial.total_combined_loss,
                "metabolic_cost": trial.metabolic_cost,
                "step_variance": trial.step_variance,
                "y_change": trial.y_change,
                "y_total": trial.y_total,
                "timestamp": trial.timestamp.isoformat()
            })
        
        # Calculate aggregate metrics for each geometry
        for geometry_key, data in geometry_performance.items():
            trials_data = data["trials"]
            if trials_data:
                losses = [t["total_combined_loss"] for t in trials_data if t["total_combined_loss"] is not None]
                if losses:
                    data["metrics"] = {
                        "mean_loss": np.mean(losses),
                        "std_loss": np.std(losses),
                        "min_loss": np.min(losses),
                        "max_loss": np.max(losses),
                        "trial_count": len(trials_data)
                    }
        
        return {
            "participant_id": participant_id,
            "geometry_performance": geometry_performance,
            "total_trials": len(trials)
        }
    
    def get_optimization_recommendations(
        self, 
        participant_id: int,
        objective: str
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations for a participant.
        
        Args:
            participant_id: Participant ID
            objective: Optimization objective
            
        Returns:
            Optimization recommendations
        """
        # Get recent trials
        recent_trials = (
            self.db.query(Trial)
            .filter(Trial.participant_id == participant_id)
            .filter(Trial.deleted_at.is_(None))
            .filter(Trial.total_combined_loss.isnot(None))
            .order_by(Trial.timestamp.desc())
            .limit(10)
            .all()
        )
        
        if not recent_trials:
            return {
                "recommendations": [],
                "status": "no_data",
                "message": "No trial data available for recommendations"
            }
        
        # Calculate improvement trends
        losses = [t.total_combined_loss for t in recent_trials if t.total_combined_loss is not None]
        
        if len(losses) < 2:
            return {
                "recommendations": [],
                "status": "insufficient_data",
                "message": "Insufficient data for trend analysis"
            }
        
        # Calculate trend
        recent_losses = losses[:5] if len(losses) >= 5 else losses
        older_losses = losses[5:10] if len(losses) >= 10 else losses[5:]
        
        if older_losses:
            recent_avg = np.mean(recent_losses)
            older_avg = np.mean(older_losses)
            improvement_rate = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0
        else:
            improvement_rate = 0
        
        # Generate recommendations
        recommendations = []
        
        if improvement_rate > 0.1:
            recommendations.append({
                "type": "positive_trend",
                "message": f"Good improvement trend: {improvement_rate:.1%} reduction in loss",
                "priority": "low"
            })
        elif improvement_rate < -0.1:
            recommendations.append({
                "type": "negative_trend",
                "message": f"Concerning trend: {abs(improvement_rate):.1%} increase in loss",
                "priority": "high"
            })
        
        # Check for convergence
        if len(losses) >= 5:
            recent_std = np.std(recent_losses)
            if recent_std < 0.1 * np.mean(recent_losses):
                recommendations.append({
                    "type": "convergence",
                    "message": "Optimization appears to be converging",
                    "priority": "medium"
                })
        
        # Check for high variance
        if len(losses) >= 3:
            cv = np.std(losses) / np.mean(losses) if np.mean(losses) > 0 else 0
            if cv > 0.5:
                recommendations.append({
                    "type": "high_variance",
                    "message": "High variance in results - consider more trials",
                    "priority": "medium"
                })
        
        return {
            "recommendations": recommendations,
            "status": "success",
            "trial_count": len(recent_trials),
            "improvement_rate": improvement_rate,
            "recent_loss_avg": np.mean(recent_losses),
            "recent_loss_std": np.std(recent_losses)
        }
    
    def _trials_to_dataframe(self, trials: List[Trial]) -> pd.DataFrame:
        """Convert trials to DataFrame for optimization."""
        data = []
        
        for trial in trials:
            row = {
                'alpha': trial.alpha or (trial.geometry.alpha if trial.geometry else None),
                'beta': trial.beta or (trial.geometry.beta if trial.geometry else None),
                'gamma': trial.gamma or (trial.geometry.gamma if trial.geometry else None),
                'delta': trial.delta or (trial.geometry.delta if trial.geometry else 0.0),
                'total_combined_loss': trial.total_combined_loss,
                'metabolic_cost': trial.metabolic_cost,
                'step_variance': trial.step_variance,
                'y_change': trial.y_change,
                'y_total': trial.y_total,
                'rms_load_cell_force': trial.rms_load_cell_force,
                'timestamp': trial.timestamp
            }
            data.append(row)
        
        return pd.DataFrame(data)
