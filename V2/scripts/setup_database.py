#!/usr/bin/env python3
"""
Database setup script for Personalising Crutches.

This script initializes the database with tables and initial data.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after adding to path
from database.connection import create_tables, engine
from database.models import Participant
from sqlalchemy.orm import sessionmaker
import numpy as np


# Note: Geometry creation removed - now using dynamic geometry system


def create_sample_participant(session):
    """Create a sample participant for testing."""
    print("Creating sample participant...")
    
    # Check if participants already exist
    if session.query(Participant).first() is not None:
        print("Participants already exist, skipping creation.")
        return
    
    sample_participant = Participant(
        name="Test User",
        characteristics={
            "height": 175.0,
            "weight": 70.0,
            "forearm_length": 25.0,
            "fitness_level": 3
        }
    )
    session.add(sample_participant)
    session.commit()
    print("Created sample participant: Test User")


def main():
    """Main setup function."""
    print("Setting up Personalising Crutches database...")
    
    try:
        # Create all tables
        print("Creating database tables...")
        create_tables()
        print("Database tables created successfully.")
        
        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        try:
    # Create initial data
    create_sample_participant(session)
            
            print("Database setup completed successfully!")
            print("\nNext steps:")
            print("1. Run the FastAPI server: uvicorn app.main:app --reload")
            print("2. Access the API documentation at: http://localhost:8000/api/docs")
            print("3. Access the web interface at: http://localhost:8000")
            
        finally:
            session.close()
            
    except Exception as e:
        print(f"Error setting up database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
