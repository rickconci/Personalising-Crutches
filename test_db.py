#!/usr/bin/env python3
"""Test script to verify database creation."""

import os
import sys
sys.path.append('server')

from database import db, create_and_populate_database
from flask import Flask

# Create a minimal Flask app for testing
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///server/experiments.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

print("Creating database...")
create_and_populate_database(app)

print("Database created successfully!")
print("Checking if tables exist...")

with app.app_context():
    # Test if we can query the tables
    from database import Participant, Geometry, Trial
    
    try:
        participants = Participant.query.all()
        geometries = Geometry.query.all()
        trials = Trial.query.all()
        
        print(f"✓ Participants table: {len(participants)} records")
        print(f"✓ Geometries table: {len(geometries)} records")
        print(f"✓ Trials table: {len(trials)} records")
        
        # Check if Trial table has the new columns
        if hasattr(Trial, 'alpha') and hasattr(Trial, 'beta') and hasattr(Trial, 'gamma'):
            print("✓ Trial table has alpha, beta, gamma columns")
        else:
            print("✗ Trial table missing alpha, beta, gamma columns")
            
    except Exception as e:
        print(f"✗ Error querying database: {e}") 