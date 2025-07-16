import os
import json
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from plotly.io import to_html

# --- Database & Core Logic ---
from database import db, Participant, Geometry, Trial, create_and_populate_database
from core.experiment import Experiment
from core import config as core_config
from core.gait_analysis import detect_steps_unsupervised, _postprocess_steps, compute_cycle_variance

# --- App Initialization ---
# Configure Flask to serve the frontend static files
app = Flask(__name__, static_folder='../site', static_url_path='/')

# A more robust CORS configuration to handle preflight (OPTIONS) requests
CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type"], "methods": ["GET", "POST", "OPTIONS"]}})

# --- Configuration ---
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# The upload folder path needs to be absolute or relative to the instance folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')
app.config['PLOTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'plots')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)

# --- Initialize Database & App ---
db.init_app(app)
create_and_populate_database(app)

# --- Root endpoint to serve the frontend ---
@app.route('/')
def serve_index():
    # The static_folder is '../site', so we serve index.html from there.
    return send_from_directory(app.static_folder, 'index.html')


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
        'remaining_geometries': [geometry_to_dict(g) for g in all_geometries if g.id not in completed_geometry_ids],
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

@app.route('/api/trials/analyze', methods=['POST'])
def analyze_trial_data():
    """
    Receives raw collected data (either as JSON or a file upload),
    saves it, analyzes it, and returns plot paths.
    """
    raw_data_path = ''
    
    # Check if the request is JSON or a file upload
    if request.is_json:
        # --- Handle Live Data from Bluetooth ---
        data = request.json
        participant_id = data.get('participantId')
        geometry_id = data.get('geometryId')
        trial_data = data.get('trialData')

        if not all([participant_id, geometry_id, trial_data]):
            return jsonify({'error': 'Missing required JSON data for analysis'}), 400
        
        trial_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(participant_id), str(geometry_id))
        os.makedirs(trial_dir, exist_ok=True)
        raw_data_path = os.path.join(trial_dir, 'live_recorded_data.csv')
        
        df = pd.DataFrame(trial_data)
        df['relative_time_ms'] = [i * 5 for i in range(len(df))] 
        df.to_csv(raw_data_path, index=False)

    elif 'file' in request.files:
        # --- Handle File Upload ---
        file = request.files['file']
        participant_id = request.form.get('participantId')
        geometry_id = request.form.get('geometryId')

        if not all([file, participant_id, geometry_id]):
            return jsonify({'error': 'Missing file or IDs for analysis'}), 400
            
        trial_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(participant_id), str(geometry_id))
        os.makedirs(trial_dir, exist_ok=True)
        filename = secure_filename(file.filename)
        raw_data_path = os.path.join(trial_dir, filename)
        file.save(raw_data_path)
    
    else:
        return jsonify({'error': 'Unsupported request format. Must be JSON or file upload.'}), 415


    # --- Run Analysis & Plotting ---
    try:
        df = pd.read_csv(raw_data_path)
        # Ensure standard column names
        df.rename(columns={
            "relative_time_ms": "timestamp",
            "accX": "acc_x_data",
            "accY": "acc_y_data"
        }, inplace=True, errors='ignore')

        required_cols = ['timestamp', 'force', 'acc_x_data', 'acc_y_data']
        if not all(col in df.columns for col in required_cols):
            return jsonify({'error': f'CSV must contain all required columns: {required_cols}'}), 400

        # Ensure timestamp is in seconds
        if df['timestamp'].max() > 1000: # Heuristic for milliseconds
             df['timestamp'] = (pd.to_numeric(df['timestamp'], errors='coerce') - df['timestamp'].iloc[0]) / 1000.0
        df.dropna(subset=required_cols, inplace=True)
        
        fs = 1.0 / np.median(np.diff(df['timestamp'].values))
        
        # Detect steps from the FORCE signal
        raw_steps_time = detect_steps_unsupervised(df['force'].values, df['timestamp'].values, fs)
        final_steps_time = _postprocess_steps(raw_steps_time)
        
        # Get the force values at the detected step times for plotting
        # Create a temporary series for quick lookup
        force_at_step_time = df.set_index('timestamp')['force'].reindex(final_steps_time, method='nearest')

        # --- Calculate Metrics ---
        instability_loss = compute_cycle_variance(df, final_steps_time)
        # Placeholder for metabolic cost until live data is available
        effort_loss = instability_loss * 1500 + np.random.rand() * 5 # Simulate a plausible relationship

        # --- Create Plots ---
        # Figure 1: Time Series Data
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df['timestamp'], y=df['force'], mode='lines', name='Force'))
        fig_ts.add_trace(go.Scatter(x=df['timestamp'], y=df['acc_x_data'], mode='lines', name='Accel X', visible='legendonly'))
        fig_ts.add_trace(go.Scatter(x=df['timestamp'], y=df['acc_y_data'], mode='lines', name='Accel Y', visible='legendonly'))
        fig_ts.add_trace(go.Scatter(
            x=final_steps_time, y=force_at_step_time, mode='markers', 
            name='Detected Steps', marker=dict(symbol='x', color='red', size=10)
        ))
        fig_ts.update_layout(title="Time Series Data", xaxis_title="Time (s)", yaxis_title="Signal Value")

        # Figure 2: Step Duration Histogram
        step_durations = np.diff(final_steps_time)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=step_durations, nbinsx=20, name='Step Durations'))
        fig_hist.update_layout(title="Step Duration Distribution", xaxis_title="Duration (s)", yaxis_title="Count")

        # --- Save Plots ---
        def save_plot(fig, filename_suffix):
            plot_filename = f'plot_{participant_id}_{geometry_id}_{filename_suffix}.html'
            plot_save_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
            raw_plot_json = fig.to_json()
            html_output = to_html(fig, full_html=False, include_plotlyjs=False)
            html_output_with_data = html_output.replace('<div ', f'<div data-raw=\'{raw_plot_json}\' ', 1)
            with open(plot_save_path, 'w') as f:
                f.write(html_output_with_data)
            return f'/data/plots/{plot_filename}'

        ts_plot_path = save_plot(fig_ts, 'timeseries')
        hist_plot_path = save_plot(fig_hist, 'histogram')

        return jsonify({
            'message': f'Analysis complete. Detected {len(final_steps_time)} steps.',
            'plots': {
                'timeseries': ts_plot_path,
                'histogram': hist_plot_path
            },
            'metrics': {
                'instability_loss': instability_loss,
                'effort_loss': effort_loss
            },
            'steps': final_steps_time.tolist()
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500

@app.route('/api/trials/save', methods=['POST'])
def save_trial_results():
    """Saves the final trial data including metrics and survey responses."""
    data = request.json
    try:
        new_trial = Trial(
            participant_id=data['participantId'],
            geometry_id=data['geometryId'],
            survey_responses=data['surveyResponses'],
            instability_loss=data['metrics']['instability_loss'],
            effort_loss=data['metrics']['effort_loss'],
            source='systematic'
        )
        db.session.add(new_trial)
        db.session.commit()
        return jsonify(trial_to_dict(new_trial)), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to save trial: {str(e)}'}), 500


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

@app.route('/data/<path:path>')
def serve_data(path):
    """Serves files from the main data directory, including plots."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    return send_from_directory(data_dir, path)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 