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

# --- App Initialization ---
DEBUG = True
def debug_print(message):
    """Prints a debug message only if the DEBUG flag is True."""
    if DEBUG:
        # Using a distinct prefix to easily spot these messages in the log
        print(f"[DATA_FLOW] {message}")

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
    return {'id': p.id, 'name': p.name, 'full_name': p.full_name, 'characteristics': p.characteristics}

def geometry_to_dict(g):
    return {'id': g.id, 'name': g.name, 'alpha': g.alpha, 'beta': g.beta, 'gamma': g.gamma}

def trial_to_dict(t):
    # This helper now needs to handle trials that might not have a predefined geometry (from BO)
    geom_name = t.geometry.name if t.geometry else "BO"
    return {
        'id': t.id,
        'participant_id': t.participant_id,
        'participant_full_name': t.participant.full_name,
        'geometry_id': t.geometry_id,
        'geometry_name': geom_name,
        'alpha': t.geometry.alpha if t.geometry else None,
        'beta': t.geometry.beta if t.geometry else None,
        'gamma': t.geometry.gamma if t.geometry else None,
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
    name = data.get('name')
    if not name: 
        return jsonify({'error': 'Name is required'}), 400
    
    # Check if participant already exists
    existing = Participant.query.filter_by(name=name).first()
    if existing:
        return jsonify({'error': f'Participant "{name}" already exists'}), 409
    
    new_participant = Participant(name=name, characteristics=data.get('userCharacteristics'))
    db.session.add(new_participant)
    db.session.commit()
    return jsonify(participant_to_dict(new_participant)), 201

@app.route('/api/participants/<int:participant_id>', methods=['GET', 'DELETE'])
def get_participant_details(participant_id):
    participant = Participant.query.get_or_404(participant_id)
    
    if request.method == 'DELETE':
        # Delete all trials for this participant first
        Trial.query.filter_by(participant_id=participant_id).delete()
        # Delete the participant
        db.session.delete(participant)
        db.session.commit()
        return jsonify({'message': 'Participant deleted successfully'}), 200
    
    all_geometries = Geometry.query.all()
    # We only care about systematic trials for the checklist
    completed_trials = Trial.query.filter_by(participant_id=participant.id, source='systematic').all()
    completed_geometry_ids = {t.geometry_id for t in completed_trials}

    return jsonify({
        'participant': participant_to_dict(participant),
        'completed_trials_count': len(completed_geometry_ids),
        'all_geometries': [geometry_to_dict(g) for g in all_geometries],
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
    Receives raw collected data and analyzes it, returning plot paths.
    Raw data is saved immediately when trial stops.
    """
    debug_print("--- /api/trials/analyze endpoint hit ---")
    # --- Handle Live Data from Bluetooth ---
    data = request.json
    if not data:
        debug_print("Request failed: No JSON data received.")
        return jsonify({'error': 'No JSON data received'}), 400
        
    participant_id = data.get('participantId')
    geometry_id = data.get('geometryId')
    trial_data = data.get('trialData')
    debug_print(f"Received data for Participant ID: {participant_id}, Geometry ID: {geometry_id}")

    if not all([participant_id, geometry_id, trial_data]):
        debug_print("Request failed: Missing required fields in JSON payload.")
        return jsonify({'error': 'Missing required JSON data for analysis'}), 400
    
    # --- DATA SHAPE CHECK 1: The data arrives from the frontend ---
    if isinstance(trial_data, list) and len(trial_data) > 0:
        debug_print(f"Received 'trialData' is a LIST with {len(trial_data)} items.")
        debug_print(f"Shape of first item: {list(trial_data[0].keys())}")
    elif isinstance(trial_data, dict):
        debug_print(f"Received 'trialData' is a DICTIONARY. Keys: {list(trial_data.keys())}")
        # This is where the bug was! This branch would have been taken on the second analysis call.
    else:
        debug_print(f"Received 'trialData' is of an unexpected type: {type(trial_data)}")


    trial_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(participant_id), str(geometry_id))
    os.makedirs(trial_dir, exist_ok=True)
    raw_data_path = os.path.join(trial_dir, 'live_recorded_data.csv')
    
    # --- DATA TRANSFORMATION 1: From JSON list/dict to Pandas DataFrame ---
    df = pd.DataFrame(trial_data)
    debug_print("Converted JSON to DataFrame.")
    debug_print(f"DataFrame shape: {df.shape}")
    debug_print(f"DataFrame columns: {list(df.columns)}")
    df['relative_time_ms'] = [i * 5 for i in range(len(df))] 
    df.to_csv(raw_data_path, index=False)
    
    debug_print(f"Saved raw data to: {raw_data_path}")


    # --- Run Analysis & Plotting ---
    try:
        debug_print(f"Attempting to read back the CSV for analysis from {raw_data_path}")
        if not os.path.exists(raw_data_path):
            return jsonify({'error': f'CSV file not found at {raw_data_path}'}), 400
            
        df = pd.read_csv(raw_data_path)
        debug_print(f"Successfully read CSV. Shape: {df.shape}")
        
        # --- DATA TRANSFORMATION 2: Standardizing column names ---
        df.rename(columns={
            "relative_time_ms": "timestamp",
            "accX": "acc_x_data",
            "accY": "acc_y_data"
        }, inplace=True, errors='ignore')
        debug_print(f"Renamed columns. Current columns: {list(df.columns)}")


        required_cols = ['timestamp', 'force', 'acc_x_data', 'acc_y_data']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            debug_print(f"Analysis failed: Missing required columns: {missing_cols}")
            return jsonify({'error': f'CSV missing required columns: {missing_cols}. Available: {list(df.columns)}'}), 400

        # Ensure timestamp is in seconds
        if df['timestamp'].max() > 1000: # Heuristic for milliseconds
             debug_print("Timestamp seems to be in ms, converting to seconds.")
             df['timestamp'] = (pd.to_numeric(df['timestamp'], errors='coerce') - df['timestamp'].iloc[0]) / 1000.0
        df.dropna(subset=required_cols, inplace=True)
        
        fs = 1.0 / np.median(np.diff(df['timestamp'].values))
        debug_print(f"Calculated sampling frequency (fs): {fs:.2f} Hz")
        
        # Detect steps from the FORCE signal
        raw_steps_time = detect_steps_unsupervised(df['force'].values, df['timestamp'].values, fs)
        final_steps_time = _postprocess_steps(raw_steps_time)
        debug_print(f"Detected {len(final_steps_time)} steps after post-processing.")
        
        # Get the force values at the detected step times for plotting
        # Create a temporary series for quick lookup
        force_at_step_time = df.set_index('timestamp')['force'].reindex(final_steps_time, method='nearest')

        # --- Calculate Metrics ---
        instability_loss = compute_cycle_variance(df, final_steps_time)
        debug_print(f"Calculated instability loss (cycle variance): {instability_loss:.4f}")
        # Dummy metabolic cost for now
        metabolic_cost = 3.2

        # --- Create Plots ---
        debug_print("Creating Time Series and Histogram plots.")
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

        # Figure 2: Step Duration Histogram with Mean and Std Dev
        step_durations = np.diff(final_steps_time)
        mean_duration = np.mean(step_durations)
        std_duration = np.std(step_durations)
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=step_durations, nbinsx=20, name='Step Durations'))
        
        # Add mean line
        fig_hist.add_vline(x=mean_duration, line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: {mean_duration:.3f}s")
        
        # Add standard deviation range
        fig_hist.add_vrect(x0=mean_duration-std_duration, x1=mean_duration+std_duration,
                          fillcolor="red", opacity=0.2, layer="below", line_width=0,
                          annotation_text=f"±1σ: {std_duration:.3f}s")
        
        fig_hist.update_layout(
            title=f"Step Duration Distribution (Mean: {mean_duration:.3f}s, Std: {std_duration:.3f}s)", 
            xaxis_title="Duration (s)", 
            yaxis_title="Count"
        )

        # --- Save Plots ---
        def save_plot(fig, filename_suffix):
            plot_filename = f'plot_{participant_id}_{geometry_id}_{filename_suffix}.html'
            plot_save_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
            raw_plot_json = fig.to_json()
            html_output = to_html(fig, full_html=False, include_plotlyjs=False)
            html_output_with_data = html_output.replace('<div ', f'<div data-raw=\'{raw_plot_json}\' ', 1)
            with open(plot_save_path, 'w') as f:
                f.write(html_output_with_data)
            debug_print(f"Saved plot to {plot_save_path}")
            return f'/data/plots/{plot_filename}'

        ts_plot_path = save_plot(fig_ts, 'timeseries')
        hist_plot_path = save_plot(fig_hist, 'histogram')

        final_payload = {
            'message': f'Analysis complete. Detected {len(final_steps_time)} steps.',
            'plots': {
                'timeseries': ts_plot_path,
                'histogram': hist_plot_path
            },
            'metrics': {
                'instability_loss': instability_loss,
                'metabolic_cost': metabolic_cost,
                'step_count': len(final_steps_time)
            },
            'steps': final_steps_time.tolist()
        }
        debug_print("--- Analysis successful. Sending response to frontend. ---")
        return jsonify(final_payload)

    except Exception as e:
        debug_print(f"--- Analysis CRASHED ---")
        debug_print(f"Error: {str(e)}")
        import traceback
        debug_print(f"Full traceback: {traceback.format_exc()}")
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
            processed_features=data['metrics'],
            source='systematic'
        )
        db.session.add(new_trial)
        db.session.commit()
        return jsonify(trial_to_dict(new_trial)), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to save trial: {str(e)}'}), 500

@app.route('/api/trials/<int:trial_id>', methods=['DELETE'])
def delete_trial(trial_id):
    """Deletes a specific trial."""
    trial = Trial.query.get_or_404(trial_id)
    try:
        db.session.delete(trial)
        db.session.commit()
        return jsonify({'message': 'Trial deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to delete trial: {str(e)}'}), 500


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