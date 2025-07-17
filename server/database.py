import os
import json
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np

db = SQLAlchemy()

class Participant(db.Model):
    """Represents a participant in the study."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    characteristics = db.Column(db.JSON, nullable=True)
    
    trials = db.relationship('Trial', backref='participant', lazy=True)

    def __repr__(self):
        return f'<Participant {self.name}>'
    
    @property
    def full_name(self):
        return self.name

class Geometry(db.Model):
    """Represents a predefined crutch geometry."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    alpha = db.Column(db.Float, nullable=False)
    beta = db.Column(db.Float, nullable=False)
    gamma = db.Column(db.Float, nullable=False)

    trials = db.relationship('Trial', backref='geometry', lazy=True)

    def __repr__(self):
        return f'<Geometry {self.name}>'

class Trial(db.Model):
    """Represents a single experimental trial."""
    id = db.Column(db.Integer, primary_key=True)
    participant_id = db.Column(db.Integer, db.ForeignKey('participant.id'), nullable=False)
    geometry_id = db.Column(db.Integer, db.ForeignKey('geometry.id'), nullable=False)
    
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    raw_data_path = db.Column(db.String(255), nullable=True)
    source = db.Column(db.String(20), nullable=False, server_default='systematic')
    
    # Store featurized and survey data as JSON
    processed_features = db.Column(db.JSON, nullable=True)
    survey_responses = db.Column(db.JSON, nullable=True)
    
    def __repr__(self):
        return f'<Trial {self.id} for Participant {self.participant_id}>'

def create_and_populate_database(app):
    """Initializes the database and populates it with initial data."""
    with app.app_context():
        db.create_all()

        # Check if geometries are already populated
        if Geometry.query.first() is None:
            print("Populating geometries...")
            
            # Grid search parameters (specific combinations for grid search)
            alphas = [75, 95, 115]
            betas = [95, 115, 135]
            gammas = [-9, 0, 9]
            
            # Create grid search geometries with exact mapping
            # G1-G9 for gamma -9, G10-G18 for gamma 0, G19-G27 for gamma 9
            # Pattern: iterate through alpha, then beta, then gamma
            count = 1
            for alpha in alphas:
                for beta in betas:
                    for gamma in gammas:
                        geom = Geometry(name=f"G{count}", alpha=alpha, beta=beta, gamma=gamma)
                        db.session.add(geom)
                        count += 1
            
            # Add control trial (always alpha=95, beta=115, gamma=0)
            # This is the same geometry run before all other trials
            control_geom = Geometry(name="Control", alpha=95, beta=115, gamma=0)
            db.session.add(control_geom)
            
            db.session.commit()
            print("Geometries populated.")
        else:
            print("Geometries already exist.") 