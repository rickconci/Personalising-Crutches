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
    user_id = db.Column(db.String(80), unique=True, nullable=False)
    characteristics = db.Column(db.JSON, nullable=True)
    
    trials = db.relationship('Trial', backref='participant', lazy=True)

    def __repr__(self):
        return f'<Participant {self.user_id}>'

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
            
            # Add a control geometry
            control_geometry = Geometry(name="Control", alpha=15.0, beta=12.0, gamma=10.0)
            db.session.add(control_geometry)

            # Add 26 other placeholder geometries
            # This creates a simple grid for demonstration
            alphas = np.linspace(10, 20, 3)
            betas = np.linspace(8, 16, 3)
            gammas = np.linspace(5, 15, 3)
            
            count = 1
            for a in alphas:
                for b in betas:
                    for g in gammas:
                        # Simple check to avoid duplicating control, assuming it's one of the generated values
                        if count > 26: break
                        if abs(a - 15.0) < 0.1 and abs(b - 12.0) < 0.1 and abs(g - 10.0) < 0.1:
                            continue

                        geom = Geometry(name=f"G{count}", alpha=round(a, 1), beta=round(b, 1), gamma=round(g, 1))
                        db.session.add(geom)
                        count += 1
            
            db.session.commit()
            print("Geometries populated.")
        else:
            print("Geometries already exist.") 