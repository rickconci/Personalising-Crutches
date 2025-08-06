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
    geometry_id = db.Column(db.Integer, db.ForeignKey('geometry.id'), nullable=True)  # Nullable for BO trials
    
    # BO trial parameters (for trials without predefined geometries)
    alpha = db.Column(db.Float, nullable=True)
    beta = db.Column(db.Float, nullable=True)
    gamma = db.Column(db.Float, nullable=True)
    
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    raw_data_path = db.Column(db.String(255), nullable=True)
    source = db.Column(db.String(20), nullable=False, server_default='grid_search')
    
    # Store featurized and survey data as JSON
    processed_features = db.Column(db.JSON, nullable=True)
    survey_responses = db.Column(db.JSON, nullable=True)
    steps = db.Column(db.JSON, nullable=True) # To store the final list of step timestamps
    
    # Metabolic cost field for effort optimization
    metabolic_cost = db.Column(db.Float, nullable=True)
    
    # Soft delete flag
    deleted = db.Column(db.DateTime, nullable=True)  # Timestamp when deleted, NULL if not deleted
    
    def __repr__(self):
        return f'<Trial {self.id} for Participant {self.participant_id}>'

def create_and_populate_database(app):
    """Initializes the database and populates it with initial data."""
    with app.app_context():
        db.create_all()

        # Check if geometries are already populated
        if Geometry.query.first() is None:
            print("Populating geometries...")
            
            # --- Define Geometries ---
            # Gamma (γ) values: -9, 0, 9
            # Beta (β) values: 110, 125, 140
            # Alpha (α) values: 80, 95, 110
            
            # Add a "Control" geometry first, using the new median values
            control_geometry = Geometry(name='Control', alpha=95, beta=125, gamma=0)
            db.session.add(control_geometry)
            
            geometries = []
            gammas = [-9, 0, 9]
            betas = [110, 125, 140]
            alphas = [80, 95, 110]
            
            g_number = 1
            for gamma in gammas:
                for beta in betas:
                    for alpha in alphas:
                        # Skip the one that matches the control
                        if alpha == 95 and beta == 125 and gamma == 0:
                            continue
                        
                        geom_name = f'G{g_number}'
                        geom = Geometry(name=geom_name, alpha=alpha, beta=beta, gamma=gamma)
                        db.session.add(geom)
                        geometries.append(geom)
                        g_number += 1
            
            db.session.commit()
            print("Geometries populated.")
        else:
            print("Geometries already exist.") 