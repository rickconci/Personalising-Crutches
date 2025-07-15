import os
import json
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- Database & Core Logic ---
from database import db, Participant, Geometry, Trial, create_and_populate_database
from core.experiment import Experiment 
from core import config as core_config

# --- App Initialization ---
app = Flask(__name__)
# Be more explicit with CORS for development to avoid preflight issues
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Configuration ---
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Initialize Database & App ---
db.init_app(app)
create_and_populate_database(app)


# --- Helper Functions ---
def participant_to_dict(p):
    return {'id': p.id, 'user_id': p.user_id, 'characteristics': p.characteristics}

def geometry_to_dict(g):
    return {'id': g.id, 'name': g.name, 'alpha': g.alpha, 'beta': g.beta, 'gamma': g.gamma}

def trial_to_dict(t):
    # This helper now needs to handle trials that might not have a predefined geometry (from BO)
    geom_name = t.geometry.name if t.geometry else "BO"
    return {
        'id': t.id,
        'participant_id': t.participant_id,
        'participant_user_id': t.participant.user_id,
        'geometry_id': t.geometry_id,
        'geometry_name': geom_name,
        'alpha': t.alpha,
        'beta': t.beta,
        'gamma': t.gamma,
        'timestamp': t.timestamp.isoformat(),
        'survey_responses': t.survey_responses,
        'processed_features': t.processed_features,
        'source': t.source
    }


# =================================================================================
# === MODE 1: SYSTEMATIC EXPERIMENT API ===========================================
# =================================================================================

@app.route('/api/participants', methods=['GET'])
def get_participants():
    participants = Participant.query.all()
    return jsonify([participant_to_dict(p) for p in participants])

@app.route('/api/participants', methods=['POST'])
def create_participant():
    data = request.json
    user_id = data.get('userId')
    if not user_id: return jsonify({'error': 'User ID is required'}), 400
    if Participant.query.filter_by(user_id=user_id).first():
        return jsonify({'error': f'Participant with ID "{user_id}" already exists'}), 409
    
    new_participant = Participant(user_id=user_id, characteristics=data.get('userCharacteristics'))
    db.session.add(new_participant)
    db.session.commit()
    return jsonify(participant_to_dict(new_participant)), 201

@app.route('/api/participants/<int:participant_id>', methods=['GET'])
def get_participant_details(participant_id):
    participant = Participant.query.get_or_404(participant_id)
    all_geometries = Geometry.query.all()
    # We only care about systematic trials for the checklist
    completed_trials = Trial.query.filter_by(participant_id=participant.id, source='systematic').all()
    completed_geometry_ids = {t.geometry_id for t in completed_trials}

    return jsonify({
        'participant': participant_to_dict(participant),
        'completed_trials_count': len(completed_geometry_ids),
        'remaining_geometries': [g for g in all_geometries if g.id not in completed_geometry_ids],
    })

@app.route('/api/geometries', methods=['GET'])
def get_geometries():
    return jsonify([geometry_to_dict(g) for g in Geometry.query.all()])

@app.route('/api/trials', methods=['GET'])
def get_all_trials():
    trials = Trial.query.order_by(Trial.timestamp.desc()).all()
    return jsonify([trial_to_dict(t) for t in trials])

@app.route('/api/trials', methods=['POST'])
def create_systematic_trial():
    # This endpoint is now specifically for systematic trials
    form_data = json.loads(request.form.get('data'))
    participant_id = form_data.get('participantId')
    geometry_id = form_data.get('geometryId')
    
    # --- Data Processing Placeholder ---
    processed_features = {'feature1': 0.5, 'feature2': 0.8}

    new_trial = Trial(
        participant_id=participant_id,
        geometry_id=geometry_id,
        survey_responses=form_data.get('surveyResponses'),
        processed_features=processed_features,
        source='systematic' # Explicitly set source
    )
    db.session.add(new_trial)
    db.session.commit()
    return jsonify(trial_to_dict(new_trial)), 201


# =================================================================================
# === MODE 2: PERSONALIZED BAYESIAN OPTIMIZATION API ==============================
# =================================================================================

class BayesianOptimizationManager:
    """Manages active BO experiment sessions, now integrated with the database."""
    def __init__(self):
        self.sessions = {}

    def start_session(self, user_id):
        """Starts a new BO session or loads an existing one for a user."""
        participant = Participant.query.filter_by(user_id=user_id).first_or_404()
        
        if user_id in self.sessions:
            return self.sessions[user_id]

        # Load all previous data for this participant from the DB
        previous_trials_query = Trial.query.filter_by(participant_id=participant.id).all()
        
        previous_data = None
        if previous_trials_query:
            records = [trial_to_dict(t) for t in previous_trials_query]
            previous_data = pd.DataFrame(records)
            # Ensure required columns for BO are present
            # This part needs to align with what the Experiment class expects
            if 'Total_Combined_Loss' not in previous_data.columns:
                 # Create a dummy loss if not present, BO class needs a target
                previous_data['Total_Combined_Loss'] = 0.5 

        print(f"Starting BO session for '{user_id}' with {len(previous_trials_query)} prior data points.")

        experiment = Experiment(
            user_id=user_id,
            user_characteristics=participant.characteristics,
            # These can be default or configured in the BO start screen
            objective=core_config.objective_weights, 
            initial_crutch_geometry=core_config.initial_crutch_geometry,
            data_manager=None, # We handle data via DB now
            manual_correction=False,
            visualize_steps=True,
            previous_data=previous_data
        )
        self.sessions[user_id] = experiment
        return experiment

    def get_session(self, user_id):
        return self.sessions.get(user_id)

bo_manager = BayesianOptimizationManager()

@app.route('/api/bo/start', methods=['POST'])
def start_bo_session():
    user_id = request.json.get('userId')
    if not user_id: return jsonify({'error': 'userId is required'}), 400
    
    try:
        session = bo_manager.start_session(user_id)
        history = session.experiment_data.to_dict('records') if session.experiment_data is not None else []
        return jsonify({
            'message': f"BO session started for {user_id}",
            'userId': user_id,
            'history': history
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to start BO session: {str(e)}'}), 500

@app.route('/api/bo/next-geometry', methods=['GET'])
def get_bo_next_geometry():
    user_id = request.args.get('userId')
    session = bo_manager.get_session(user_id)
    if not session: return jsonify({'error': 'BO session not found'}), 404
    
    next_geometry = session.get_next_suggestion()
    return jsonify(next_geometry)

@app.route('/api/bo/trial', methods=['POST'])
def process_bo_trial():
    form_data = json.loads(request.form.get('data'))
    user_id = form_data.get('userId')
    session = bo_manager.get_session(user_id)
    if not session: return jsonify({'error': 'BO session not found'}), 404

    # --- Run BO analysis (this is from the original Experiment class) ---
    # This assumes `run_and_process_trial` returns success and updates the internal model
    trial_num = len(session.experiment_data) + 1 if session.experiment_data is not None else 1
    
    # This part needs to be adapted from your core.experiment logic
    # It should take the file, process it, calculate loss, and update the BO model.
    # For now, let's simulate this.
    
    # 1. Save file (placeholder)
    # file = request.files['file']
    # raw_data_path = ...
    
    # 2. Process data and get loss (placeholder)
    # success = session.run_and_process_trial(...)
    
    # 3. Get the latest trial data from the BO model's dataframe
    # latest_trial_data = session.experiment_data.iloc[-1]
    
    # --- Save the BO trial result to our central database ---
    # Placeholder data for now
    crutch_geometry = form_data.get('crutchGeometry')
    processed_features = {'bo_feature': 1.0}
    
    new_trial = Trial(
        participant_id=Participant.query.filter_by(user_id=user_id).first().id,
        geometry_id=None, # BO trials are not from the predefined grid
        alpha=crutch_geometry['alpha'],
        beta=crutch_geometry['beta'],
        gamma=crutch_geometry['gamma'],
        survey_responses=form_data.get('subjectiveMetrics'),
        processed_features=processed_features,
        source='bo' # Set source to 'bo'
    )
    db.session.add(new_trial)
    db.session.commit()
    
    history = session.experiment_data.to_dict('records') if session.experiment_data is not None else []
    return jsonify({'message': 'BO trial processed', 'history': history}), 200


# =================================================================================
# === COMMON AND STATIC ROUTES ====================================================
# =================================================================================

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serves files from the main data directory."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    return send_from_directory(data_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 