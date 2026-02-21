#!/usr/bin/env python3
"""
Database utilities for accessing participant data.

This module provides convenient functions to query participant data from the database,
useful for data analysis in Jupyter notebooks and Python scripts.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy.orm import Session
import json
# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure we're in the correct directory for database access
os.chdir(project_root)

from database.connection import SessionLocal, engine
from database.models import Participant, Trial, CrutchGeometry
from app.core.config import settings


class DatabaseAccessor:
    """Utility class for accessing database data."""
    
    def __init__(self):
        """Initialize database accessor."""
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def get_all_participants(self) -> List[Dict[str, Any]]:
        """
        Get all participants with their characteristics.
        
        Returns:
            List of participant dictionaries with characteristics
        """
        with self.get_session() as session:
            participants = session.query(Participant).all()
            
            result = []
            for participant in participants:
                participant_dict = {
                    'id': participant.id,
                    'name': participant.name,
                    'created_at': participant.created_at,
                    'updated_at': participant.updated_at,
                    'characteristics': participant.characteristics if participant.characteristics else {}
                }
                result.append(participant_dict)
            
            return result
    
    def get_participant_by_id(self, participant_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific participant by ID.
        
        Args:
            participant_id: The participant ID
            
        Returns:
            Participant dictionary with characteristics or None
        """
        with self.get_session() as session:
            participant = session.query(Participant).filter(Participant.id == participant_id).first()
            
            if not participant:
                return None
            
            participant_dict = {
                'id': participant.id,
                'name': participant.name,
                'created_at': participant.created_at,
                'updated_at': participant.updated_at
            }
            
            # Add characteristics if they exist
            if participant.characteristics:
                participant_dict.update(participant.characteristics)
            
            return participant_dict
    
    def get_participants_dataframe(self) -> pd.DataFrame:
        """
        Get all participants as a pandas DataFrame.
        
        Returns:
            DataFrame with participant data and characteristics
        """
        participants = self.get_all_participants()
        
        # Convert to DataFrame and unpack characteristics
        df = pd.DataFrame(participants)
        
        # Unpack characteristics into separate columns
        if not df.empty and 'characteristics' in df.columns:
            characteristics_df = pd.json_normalize(df['characteristics'])
            df = pd.concat([df.drop('characteristics', axis=1), characteristics_df], axis=1)
        
        return df
    
    def get_participant_trials(self, participant_id: int) -> List[Dict[str, Any]]:
        """
        Get all trials for a specific participant.
        
        Args:
            participant_id: The participant ID
            
        Returns:
            List of trial dictionaries
        """
        with self.get_session() as session:
            trials = session.query(Trial).filter(Trial.participant_id == participant_id).all()
            
            result = []
            for trial in trials:
                trial_dict = {
                    'id': trial.id,
                    'participant_id': trial.participant_id,
                    'participant_name': trial.participant.name if trial.participant else None,
                    'geometry_id': trial.geometry_id,
                    'alpha': trial.alpha,
                    'beta': trial.beta,
                    'gamma': trial.gamma,
                    'delta': trial.delta,
                    'timestamp': trial.timestamp,
                    'source': trial.source,
                    'raw_data_path': trial.raw_data_path,
                    'metabolic_cost': trial.metabolic_cost,
                    'y_change': trial.y_change,
                    'y_total': trial.y_total,
                    'step_count': trial.step_count,
                    'step_variance': trial.step_variance,
                    'instability_loss': trial.instability_loss,
                    'rms_load_cell_force': trial.rms_load_cell_force,
                    'total_combined_loss': trial.total_combined_loss,
                    'laps': trial.laps,
                    'created_at': trial.created_at,
                    'updated_at': trial.updated_at
                }
                result.append(trial_dict)
            
            return result
    
    def get_participant_trials_dataframe(self, participant_id: int) -> pd.DataFrame:
        """
        Get all trials for a specific participant as a pandas DataFrame.
        
        Args:
            participant_id: The participant ID
            
        Returns:
            DataFrame with trial data
        """
        trials = self.get_participant_trials(participant_id)
        return pd.DataFrame(trials)
    
    def get_all_trials_with_participants(self) -> pd.DataFrame:
        """
        Get all trials with participant characteristics joined.
        
        Returns:
            DataFrame with trial data and participant characteristics
        """
        with self.get_session() as session:
            # Query trials with participant data
            query = session.query(
                Trial.id,
                Trial.participant_id,
                Participant.name.label('participant_name'),
                Trial.geometry_id,
                Trial.alpha,
                Trial.beta,
                Trial.gamma,
                Trial.delta,
                Trial.timestamp,
                Trial.source,
                Trial.metabolic_cost,
                Trial.y_change,
                Trial.y_total,
                Trial.step_count,
                Trial.step_variance,
                Trial.instability_loss,
                Trial.rms_load_cell_force,
                Trial.total_combined_loss,
                Trial.laps,
                Trial.created_at,
                Trial.updated_at,
                # Participant characteristics
                Participant.characteristics
            ).join(Participant, Trial.participant_id == Participant.id)
            
            results = query.all()
            
            # Convert to list of dictionaries
            data = []
            for row in results:
                row_dict = {
                    'trial_id': row.id,
                    'participant_id': row.participant_id,
                    'participant_name': row.participant_name,
                    'geometry_id': row.geometry_id,
                    'alpha': row.alpha,
                    'beta': row.beta,
                    'gamma': row.gamma,
                    'delta': row.delta,
                    'timestamp': row.timestamp,
                    'source': row.source,
                    'metabolic_cost': row.metabolic_cost,
                    'y_change': row.y_change,
                    'y_total': row.y_total,
                    'step_count': row.step_count,
                    'step_variance': row.step_variance,
                    'instability_loss': row.instability_loss,
                    'rms_load_cell_force': row.rms_load_cell_force,
                    'total_combined_loss': row.total_combined_loss,
                    'laps': row.laps,
                    'trial_created_at': row.created_at,
                    'trial_updated_at': row.updated_at
                }
                
                # Add participant characteristics
                if row.characteristics:
                    for key, value in row.characteristics.items():
                        row_dict[f'participant_{key}'] = value
                
                data.append(row_dict)
            
            return pd.DataFrame(data)


# Convenience functions for easy access
def get_participants_df() -> pd.DataFrame:
    """Get all participants as a DataFrame."""
    db = DatabaseAccessor()
    return db.get_participants_dataframe()


def get_participant_by_id(participant_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific participant by ID."""
    db = DatabaseAccessor()
    return db.get_participant_by_id(participant_id)


def get_all_trials_df() -> pd.DataFrame:
    """Get all trials with participant data as a DataFrame."""
    db = DatabaseAccessor()
    return db.get_all_trials_with_participants()


def get_participant_trials_df(participant_id: int) -> pd.DataFrame:
    """Get trials for a specific participant as a DataFrame."""
    db = DatabaseAccessor()
    return db.get_participant_trials_dataframe(participant_id)


if __name__ == "__main__":
    # Example usage
    print("Database Accessor Example")
    print("=" * 50)
    
    # Initialize database accessor
    db = DatabaseAccessor()
    
    # Get all participants
    print("\nAll Participants:")
    participants_df = db.get_participants_dataframe()
    print(participants_df.head())
    
    # Get all trials with participant data
    print("\nAll Trials with Participant Data:")
    trials_df = db.get_all_trials_with_participants()
    print(trials_df.head())
    
    # Get specific participant
    if not participants_df.empty:
        first_participant_id = participants_df.iloc[0]['id']
        print(f"\nFirst Participant Details (ID: {first_participant_id}):")
        participant = db.get_participant_by_id(first_participant_id)
        print(participant)
