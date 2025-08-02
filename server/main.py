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
import re

# Import GPy and GPyOpt for Bayesian Optimization
try:
    import GPy
    from GPyOpt.methods import BayesianOptimization
except ImportError:
    print("GPy and GPyOpt are required for Bayesian Optimization. Install with:")
    print("    pip install GPy GPyOpt")
    GPy = None
    BayesianOptimization = None

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
    return {'id': p.id, 'name': p.name, 'characteristics': p.characteristics}

def geometry_to_dict(g):
    return {'id': g.id, 'name': g.name, 'alpha': g.alpha, 'beta': g.beta, 'gamma': g.gamma}

def trial_to_dict(t):
    # This helper now needs to handle trials that might not have a predefined geometry (from BO)
    geom_name = t.geometry.name if t.geometry else ("Pain BO" if t.source == 'pain_bo' else "BO")
    
    # Ensure geometry_id is properly handled to avoid NaN in JSON
    geometry_id = t.geometry_id
    if geometry_id is None or (hasattr(geometry_id, 'isna') and geometry_id.isna()):
        geometry_id = None
    
    # Debug logging to track NaN values
    if geometry_id is not None and (hasattr(geometry_id, 'isna') and geometry_id.isna()):
        debug_print(f"WARNING: Found NaN geometry_id for trial {t.id}, source: {t.source}")
    
    result = {
        'id': t.id,
        'participant_id': t.participant_id,
        'participant_name': t.participant.name,
        'geometry_id': geometry_id,
        'geometry_name': geom_name,
        'alpha': t.geometry.alpha if t.geometry else t.alpha,
        'beta': t.geometry.beta if t.geometry else t.beta,
        'gamma': t.geometry.gamma if t.geometry else t.gamma,
        'timestamp': t.timestamp.isoformat(),
        'survey_responses': t.survey_responses,
        'processed_features': t.processed_features,
        'steps': t.steps,
        'source': t.source
    }
    
    # Check for any NaN values in the result
    for key, value in result.items():
        if hasattr(value, 'isna') and value.isna():
            debug_print(f"WARNING: Found NaN in {key} for trial {t.id}")
    
    return result


def _perform_analysis_and_plotting(df, final_steps_time, participant_id, geometry_id, plot_folder):
    """
    Helper function to perform analysis and generate plots based on a dataframe and step times.
    This is extracted to be reusable for recalculation.
    """
    debug_print(f"Performing analysis for P:{participant_id}, G:{geometry_id} with {len(final_steps_time)} steps.")
    
    # Ensure steps are sorted, as adding steps on the frontend might not preserve order
    final_steps_time = sorted(final_steps_time)
    
    # --- Calculate Metrics ---
    instability_loss = compute_cycle_variance(df, final_steps_time)
    debug_print(f"Calculated instability loss (cycle variance): {instability_loss:.4f}")

    # Get the force values at the detected step times for plotting
    # Create a temporary series for quick lookup
    force_at_step_time = df.set_index('timestamp')['force'].reindex(final_steps_time, method='nearest')

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
    mean_duration = np.mean(step_durations) if len(step_durations) > 0 else 0
    std_duration = np.std(step_durations) if len(step_durations) > 0 else 0
    
    fig_hist = go.Figure()
    if len(step_durations) > 0:
        fig_hist.add_trace(go.Histogram(x=step_durations, nbinsx=20, name='Step Durations'))
        fig_hist.add_vline(x=mean_duration, line_dash="dash", line_color="red", 
                        annotation_text=f"Mean: {mean_duration:.3f}s")
        fig_hist.add_vrect(x0=mean_duration-std_duration, x1=mean_duration+std_duration,
                        fillcolor="red", opacity=0.2, layer="below", line_width=0,
                        annotation_text=f"±1σ: {std_duration:.3f}s")
    
    fig_hist.update_layout(
        title=f"Step Duration Distribution (Mean: {mean_duration:.3f}s, Std: {std_duration:.3f}s)", 
        xaxis_title="Duration (s)", 
        yaxis_title="Count"
    )

    # --- Save Plots ---
    # To avoid cache issues on the frontend, let's add a timestamp to the plot filenames
    import time
    timestamp_ms = int(time.time() * 1000)

    def save_plot(fig, filename_suffix):
        plot_filename = f'plot_{participant_id}_{geometry_id}_{filename_suffix}_{timestamp_ms}.html'
        plot_save_path = os.path.join(plot_folder, plot_filename)
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
        'message': f'Analysis complete. Recalculated with {len(final_steps_time)} steps.',
        'plots': {
            'timeseries': ts_plot_path,
            'histogram': hist_plot_path
        },
        'metrics': {
            'instability_loss': instability_loss,
            'step_count': len(final_steps_time)
        },
        'steps': final_steps_time, # Already a list from JSON or tolist()
        'processed_data': df.to_dict('records')
    }
    return final_payload


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
    completed_trials = Trial.query.filter_by(participant_id=participant.id, source='grid_search').all()
    completed_geometry_ids = {t.geometry_id for t in completed_trials}

    # --- Prepare data for the 3D instability plot ---
    instability_plot_data = []
    for t in completed_trials:
        if t.geometry and t.processed_features and 'instability_loss' in t.processed_features:
            instability_plot_data.append({
                'alpha': t.geometry.alpha,
                'beta': t.geometry.beta,
                'gamma': t.geometry.gamma,
                'instability_loss': t.processed_features['instability_loss'],
                'geometry_name': t.geometry.name,
                'trial_id': t.id
            })


    return jsonify({
        'participant': participant_to_dict(participant),
        'completed_trials_count': len(completed_geometry_ids),
        'all_geometries': [geometry_to_dict(g) for g in all_geometries],
        'instability_plot_data': instability_plot_data
    })

@app.route('/api/geometries', methods=['GET'])
def get_geometries():
    return jsonify([geometry_to_dict(g) for g in Geometry.query.all()])

@app.route('/api/trials', methods=['GET'])
def get_all_trials():
    trials = Trial.query.order_by(Trial.timestamp.desc()).all()
    trial_dicts = [trial_to_dict(t) for t in trials]
    # Clean any NaN values before returning
    trial_dicts = clean_nan_values(trial_dicts)
    return jsonify(trial_dicts)

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
        source='grid_search' # Explicitly set source
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
    
    try:
        trial_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(participant_id), str(geometry_id))
        os.makedirs(trial_dir, exist_ok=True)
        raw_data_path = os.path.join(trial_dir, 'live_recorded_data.csv')
        
        # --- DATA TRANSFORMATION 1: From JSON list/dict to Pandas DataFrame ---
        df = pd.DataFrame(trial_data)
        debug_print("Converted JSON to DataFrame.")
        debug_print(f"DataFrame shape: {df.shape}")
        df['relative_time_ms'] = [i * 5 for i in range(len(df))] 
        df.to_csv(raw_data_path, index=False)
        debug_print(f"Saved raw data to: {raw_data_path}")

        # --- Run Analysis & Plotting ---
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
        
        final_payload = _perform_analysis_and_plotting(df, final_steps_time, participant_id, geometry_id, app.config['PLOTS_FOLDER'])
        
        debug_print("--- Analysis successful. Sending response to frontend. ---")
        return jsonify(final_payload)

    except Exception as e:
        debug_print(f"--- Analysis CRASHED ---")
        debug_print(f"Error: {str(e)}")
        import traceback
        debug_print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500


@app.route('/api/trials/recalculate', methods=['POST'])
def recalculate_trial_metrics():
    """
    Receives an updated list of step times and recalculates metrics and plots.
    """
    debug_print("--- /api/trials/recalculate endpoint hit ---")
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400
        
    participant_id = data.get('participantId')
    geometry_id = data.get('geometryId')
    final_steps_time = data.get('steps') # The frontend-modified list of step times
    
    debug_print(f"Recalculating for P:{participant_id}, G:{geometry_id} with {len(final_steps_time)} steps.")

    if not all([participant_id is not None, geometry_id is not None, isinstance(final_steps_time, list)]):
        return jsonify({'error': 'Missing required JSON data: participantId, geometryId, steps (as a list)'}), 400
    
    try:
        # --- Load the original raw data ---
        trial_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(participant_id), str(geometry_id))
        raw_data_path = os.path.join(trial_dir, 'live_recorded_data.csv')
        
        if not os.path.exists(raw_data_path):
            return jsonify({'error': f'Could not find raw data file for this trial at {raw_data_path}'}), 404

        df = pd.read_csv(raw_data_path)
        
        # --- Perform the same data prep as in the initial analysis ---
        df.rename(columns={"accX": "acc_x_data", "accY": "acc_y_data"}, inplace=True, errors='ignore')
        
        # This column is added during the initial processing, so we need to recreate it if it doesn't exist.
        if 'relative_time_ms' not in df.columns:
            df['relative_time_ms'] = [i * 5 for i in range(len(df))] 

        df.rename(columns={"relative_time_ms": "timestamp"}, inplace=True, errors='ignore')
        
        required_cols = ['timestamp', 'force', 'acc_x_data', 'acc_y_data']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             return jsonify({'error': f'Loaded CSV missing required columns: {missing_cols}. Available: {list(df.columns)}'}), 400
        
        if df['timestamp'].max() > 1000: # Heuristic for milliseconds
             df['timestamp'] = (pd.to_numeric(df['timestamp'], errors='coerce') - df['timestamp'].iloc[0]) / 1000.0
        df.dropna(subset=required_cols, inplace=True)

        # --- CALL SHARED ANALYSIS/PLOTTING FUNCTION ---
        # The `steps` are provided by the client, so we don't detect them again.
        final_payload = _perform_analysis_and_plotting(df, final_steps_time, participant_id, geometry_id, app.config['PLOTS_FOLDER'])
        
        debug_print("--- Recalculation successful. Sending response to frontend. ---")
        return jsonify(final_payload)
        
    except Exception as e:
        debug_print(f"--- Recalculation CRASHED ---")
        import traceback
        debug_print(f"Error: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({'error': f'An error occurred during recalculation: {str(e)}'}), 500


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
            steps=data['steps'], # Save the final step timestamps
            source='grid_search'
        )
        db.session.add(new_trial)
        db.session.commit()
        db.session.refresh(new_trial) # Refresh the object to get the saved data
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


@app.route('/api/trials/<int:trial_id>/details', methods=['GET'])
def get_trial_details(trial_id):
    """
    Fetches the detailed data for a single trial, including the raw data file,
    to allow for re-analysis or editing.
    """
    debug_print(f"--- /api/trials/{trial_id}/details endpoint hit ---")
    trial = Trial.query.get_or_404(trial_id)
    
    # Only allow editing of Grid Search and Instability BO trials (they have raw data)
    if trial.source not in ['grid_search', 'instability_bo']:
        return jsonify({'error': f'Only Grid Search and Instability BO trials can be edited. This is a {trial.source} trial with no raw data.'}), 400
    
    # Ensure trial has a geometry_id (Grid Search trials should always have this)
    if trial.geometry_id is None:
        return jsonify({'error': 'This trial has no geometry_id and cannot be edited.'}), 400
    
    try:
        # --- Load the original raw data ---
        # The path to the raw data is not currently saved, so we must construct it.
        # This assumes a consistent directory structure.
        trial_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(trial.participant_id), str(trial.geometry_id))
        raw_data_path = os.path.join(trial_dir, 'live_recorded_data.csv')
        
        if not os.path.exists(raw_data_path):
            debug_print(f"Data file not found at {raw_data_path}")
            return jsonify({'error': f'Could not find raw data file for this trial at {raw_data_path}'}), 404

        df = pd.read_csv(raw_data_path)
        
        # --- Perform the same data prep as in the initial analysis ---
        # This logic is duplicated from analyze/recalculate, could be refactored.
        df.rename(columns={"accX": "acc_x_data", "accY": "acc_y_data"}, inplace=True, errors='ignore')
        if 'relative_time_ms' not in df.columns:
            df['relative_time_ms'] = [i * 5 for i in range(len(df))] 
        df.rename(columns={"relative_time_ms": "timestamp"}, inplace=True, errors='ignore')
        
        required_cols = ['timestamp', 'force', 'acc_x_data', 'acc_y_data']
        if df['timestamp'].max() > 1000:
             df['timestamp'] = (pd.to_numeric(df['timestamp'], errors='coerce') - df['timestamp'].iloc[0]) / 1000.0
        df.dropna(subset=required_cols, inplace=True)

        # --- Use the saved steps to generate plots ---
        final_steps_time = trial.steps
        if not final_steps_time:
            # Fallback if steps aren't saved for some reason (e.g., older trials)
            debug_print("No steps saved with trial, re-detecting from raw data...")
            fs = 1.0 / np.median(np.diff(df['timestamp'].values))
            raw_steps_time = detect_steps_unsupervised(df['force'].values, df['timestamp'].values, fs)
            final_steps_time = _postprocess_steps(raw_steps_time)

        # Generate the plots and metrics using the shared function
        final_payload = _perform_analysis_and_plotting(df, final_steps_time, trial.participant_id, trial.geometry_id, app.config['PLOTS_FOLDER'])
        
        # Also include the original trial info
        final_payload['trial_info'] = trial_to_dict(trial)

        debug_print(f"Successfully prepared details for trial {trial_id}.")
        return jsonify(final_payload)

    except Exception as e:
        debug_print(f"--- get_trial_details CRASHED for trial {trial_id} ---")
        import traceback
        debug_print(f"Error: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({'error': f'An error occurred while fetching trial details: {str(e)}'}), 500


@app.route('/api/trials/<int:trial_id>', methods=['PUT'])
def update_trial(trial_id):
    """
    Updates an existing trial with new metrics, steps, and survey responses.
    """
    debug_print(f"--- PUT /api/trials/{trial_id} endpoint hit ---")
    trial = Trial.query.get_or_404(trial_id)
    data = request.json
    
    try:
        # Update fields from the request payload
        trial.processed_features = data.get('metrics', trial.processed_features)
        trial.steps = data.get('steps', trial.steps)
        trial.survey_responses = data.get('surveyResponses', trial.survey_responses)
        
        db.session.commit()
        db.session.refresh(trial)
        
        debug_print(f"Successfully updated trial {trial_id}.")
        return jsonify(trial_to_dict(trial))
        
    except Exception as e:
        db.session.rollback()
        debug_print(f"--- update_trial CRASHED for trial {trial_id} ---")
        debug_print(f"Error: {str(e)}")
        return jsonify({'error': f'Failed to update trial: {str(e)}'}), 500

import zipfile
import shutil

def get_g_number(name):
    """Helper to extract the number from a geometry name like 'G10' for sorting."""
    if name and name.startswith('G'):
        match = re.match(r'G(\d+)', name)
        if match:
            return int(match.group(1))
    return float('inf') # Sort non-G names after G-names

@app.route('/api/participants/<int:participant_id>/download', methods=['GET'])
def download_participant_data(participant_id):
    """
    Zips up all data for a participant, including a summary CSV,
    and provides it for download.
    """
    participant = Participant.query.get_or_404(participant_id)
    debug_print(f"--- Download request for participant {participant.id} ({participant.name}) ---")
    
    try:
        # --- 1. Fetch all trials and sort them to match the UI ---
        trials = Trial.query.filter_by(participant_id=participant.id).all()
        if not trials:
            return jsonify({'error': 'No trials found for this participant.'}), 404
            
        trial_records = [trial_to_dict(t) for t in trials]
        
        # Sort the records exactly like the frontend
        trial_records.sort(key=lambda t: (
            0 if t['geometry_name'] == 'Control' else 1,
            get_g_number(t['geometry_name']),
            t.get('geometry_name', '')
        ))

        # Add the UI-consistent trial number to each record
        non_control_count = 0
        for record in trial_records:
            if record['geometry_name'] == 'Control':
                record['ui_trial_number'] = 'Control'
            else:
                non_control_count += 1
                record['ui_trial_number'] = non_control_count
        
        # --- 2. Create a summary DataFrame for the CSV ---
        summary_df = pd.DataFrame(trial_records)
        
        if 'processed_features' in summary_df.columns:
            features_df = summary_df['processed_features'].apply(pd.Series)
            summary_df = pd.concat([summary_df.drop('processed_features', axis=1), features_df], axis=1)
        if 'survey_responses' in summary_df.columns:
            surveys_df = summary_df['survey_responses'].apply(pd.Series)
            summary_df = pd.concat([summary_df.drop('survey_responses', axis=1), surveys_df], axis=1)

        # Reorder columns, putting the new UI trial number first
        final_columns = [
            'ui_trial_number', 'geometry_name', 'alpha', 'beta', 'gamma', 'timestamp',
            'instability_loss', 'step_count', 'sus_score', 'nrs_score', 'tlx_score', 'source', 'id'
        ]
        final_columns = [col for col in final_columns if col in summary_df.columns]
        summary_df = summary_df[final_columns]
        summary_df.rename(columns={'id': 'trial_db_id'}, inplace=True)
        
        # --- 3. Create the zip file ---
        zip_filename = f"{participant.name}_data_export.zip"
        zip_path = os.path.join(app.config['PLOTS_FOLDER'], zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the summary CSV to the zip
            summary_csv_path = os.path.join(app.config['PLOTS_FOLDER'], 'temp_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            zipf.write(summary_csv_path, 'trials_summary.csv')
            os.remove(summary_csv_path)
            debug_print("Added trials_summary.csv to zip.")

            # --- 4. Add raw data files with UI-consistent folder names ---
            participant_data_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(participant_id))
            if os.path.isdir(participant_data_folder):
                for trial in trial_records:
                    geom_id = trial.get('geometry_id')
                    geom_name = trial.get('geometry_name', f"geom_{geom_id}")
                    # Use the new UI-consistent number for the folder name
                    archive_folder_name = f"Trial {trial['ui_trial_number']} - {geom_name}"
                    
                    raw_data_folder = os.path.join(participant_data_folder, str(geom_id))
                    if os.path.isdir(raw_data_folder):
                        for file in os.listdir(raw_data_folder):
                            file_path = os.path.join(raw_data_folder, file)
                            archive_path = os.path.join(archive_folder_name, file)
                            zipf.write(file_path, archive_path)
                            debug_print(f"Adding {file_path} to zip as {archive_path}.")

        download_url = f"/data/plots/{zip_filename}"
        
        debug_print(f"Data for participant {participant.name} zipped successfully.")
        return jsonify({'download_url': download_url})

    except Exception as e:
        db.session.rollback() # Rollback in case of DB error during trial fetch
        debug_print(f"--- download_participant_data CRASHED ---")
        import traceback
        debug_print(f"Error: {str(e)}\nFull traceback: {traceback.format_exc()}")
        return jsonify({'error': f'An error occurred while preparing the download: {str(e)}'}), 500


# =================================================================================
# === MODE 2: PERSONALIZED BAYESIAN OPTIMIZATION API ==============================
# =================================================================================

class BayesianOptimizationManager:
    """Manages active BO experiment sessions, now integrated with the database."""
    def __init__(self):
        self.sessions = {}

    def start_session(self, user_id):
        """Starts a new BO session or loads an existing one for a user."""
        participant = Participant.query.get_or_404(int(user_id))
        
        if user_id in self.sessions:
            return self.sessions[user_id]

        # Load all previous data for this participant from the DB
        previous_trials_query = Trial.query.filter_by(participant_id=participant.id).all()
        
        previous_data = None
        if previous_trials_query:
            records = [trial_to_dict(t) for t in previous_trials_query]
            # Clean any NaN values from the records
            records = clean_nan_values(records)
            previous_data = pd.DataFrame(records)
            # Replace any NaN values with None to avoid JSON serialization issues
            previous_data = previous_data.where(pd.notnull(previous_data), None)
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
            objective=getattr(core_config, 'objective_weights', getattr(core_config, 'objective_preference', None)),
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
        history = []
        if session.experiment_data is not None:
            # Convert to dict and ensure no NaN values
            history_df = session.experiment_data.where(pd.notnull(session.experiment_data), None)
            history = clean_nan_values(history_df.to_dict('records'))
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
        participant_id=int(user_id),
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
# === PAIN OPTIMIZATION BAYESIAN OPTIMIZATION API ===============================
# =================================================================================

class PainOptimizationManager:
    """Manages pain optimization BO sessions with discrete geometry suggestions."""
    def __init__(self):
        self.sessions = {}

    def mark_previous_trials_as_deleted(self, user_id):
        """Marks all previous pain BO trials for a user as deleted (soft delete)."""
        participant = Participant.query.get_or_404(int(user_id))
        
        # Mark all existing pain BO trials as deleted
        from datetime import datetime
        deleted_count = Trial.query.filter_by(
            participant_id=participant.id,
            source='pain_bo'
        ).filter(Trial.deleted.is_(None)).update({
            'deleted': datetime.utcnow()
        })
        
        db.session.commit()
        debug_print(f"Marked {deleted_count} previous pain BO trials as deleted for {user_id}")
        return deleted_count

    def start_session(self, user_id, restart_mode=False):
        """Starts a new pain optimization session for a user."""
        participant = Participant.query.get_or_404(int(user_id))
        
        # Use a different session key to avoid conflicts with original BO
        session_key = f"pain_bo_{user_id}"
        
        if session_key in self.sessions and not restart_mode:
            return self.sessions[session_key]

        if restart_mode:
            # In restart mode, ignore all previous data
            previous_trials = []
            # Mark existing pain BO trials as deleted
            self.mark_previous_trials_as_deleted(user_id)
            debug_print(f"Starting pain optimization in RESTART mode for {user_id} - ignoring all previous data")
        else:
            # Load previous pain optimization trials for this participant (only non-deleted ones)
            previous_trials = Trial.query.filter_by(
                participant_id=participant.id,
                source='pain_bo'
            ).filter(Trial.deleted.is_(None)).all()  # Only non-deleted trials
            debug_print(f"Starting pain optimization in CONTINUE mode for {user_id} - loaded {len(previous_trials)} previous pain BO trials")
        
        session_data = {
            'participant_id': participant.id,
            'user_id': user_id,
            'trials': [trial_to_dict(t) for t in previous_trials],
            'tested_geometries': set(),
            'trial_count': len(previous_trials),
            'restart_mode': restart_mode
        }
        
        # Track previously tested geometries (only if not in restart mode)
        if not restart_mode:
            for trial in previous_trials:
                if trial.geometry:
                    session_data['tested_geometries'].add((trial.geometry.alpha, trial.geometry.beta, trial.geometry.gamma))
                elif trial.alpha is not None and trial.beta is not None and trial.gamma is not None:
                    session_data['tested_geometries'].add((trial.alpha, trial.beta, trial.gamma))
        
        self.sessions[session_key] = session_data
        return session_data

    def get_first_geometry(self, user_id):
        """Gets the first geometry for a new pain optimization session."""
        session_key = f"pain_bo_{user_id}"
        session = self.sessions.get(session_key)
        if not session:
            raise ValueError("No active session found")
        
        # For first trial, suggest a reasonable starting point
        # Use the initial geometry from config or a reasonable default
        from core import config as core_config
        initial_geometry = core_config.initial_crutch_geometry
        return {
            'alpha': initial_geometry['alpha'],
            'beta': initial_geometry['beta'], 
            'gamma': initial_geometry['gamma'],
            'trial_number': session['trial_count'] + 1,
            'is_first_trial': True
        }

    def get_next_geometry(self, user_id):
        """Uses BO to suggest the next geometry based on previous pain scores."""
        session_key = f"pain_bo_{user_id}"
        session = self.sessions.get(session_key)
        if not session:
            raise ValueError("No active session found")
        
        if session['trial_count'] == 0:
            return self.get_first_geometry(user_id)
        
        # Use BO logic similar to InstabilityBO.py
        import pandas as pd
        import numpy as np
        
        # Convert trials to DataFrame for BO
        trials_data = []
        
        # Add pain BO trials
        for trial_dict in session['trials']:
            if trial_dict.get('survey_responses') and 'nrs_score' in trial_dict['survey_responses']:
                row = {
                    'alpha': trial_dict.get('alpha'),
                    'beta': trial_dict.get('beta'),
                    'gamma': trial_dict.get('gamma'),
                    'pain_score': trial_dict['survey_responses']['nrs_score'],
                    # Use pain score as loss (higher pain = higher loss)
                    'Total_Combined_Loss': trial_dict['survey_responses']['nrs_score'],
                    'source': 'pain_bo'
                }
                trials_data.append(row)
        
        # In continue mode, also include NRS data from Grid Search trials
        if not session.get('restart_mode', False):
            participant = Participant.query.get(session['participant_id'])
            grid_search_trials = Trial.query.filter_by(
                participant_id=participant.id,
                source='grid_search'
            ).all()
            
            grid_search_nrs_count = 0
            for trial in grid_search_trials:
                if trial.survey_responses and 'nrs_score' in trial.survey_responses:
                    trial_dict = trial_to_dict(trial)
                    row = {
                        'alpha': trial_dict['alpha'],
                        'beta': trial_dict['beta'],
                        'gamma': trial_dict['gamma'],
                        'pain_score': trial.survey_responses['nrs_score'],
                        'Total_Combined_Loss': trial.survey_responses['nrs_score'],
                        'source': 'grid_search'
                    }
                    trials_data.append(row)
                    grid_search_nrs_count += 1
                    # Also track these geometries as tested
                    session['tested_geometries'].add((trial_dict['alpha'], trial_dict['beta'], trial_dict['gamma']))
            
            if grid_search_nrs_count > 0:
                debug_print(f"Including {grid_search_nrs_count} NRS scores from Grid Search trials in BO optimization")
        
        if not trials_data:
            return self.get_first_geometry(user_id)
        
        df = pd.DataFrame(trials_data)
        
        # Use the same approach as InstabilityBO.py
        try:
            # Create BO optimizer with discrete domains
            alpha_range = list(range(70, 125, 5))   # α: handle angle from vertical (70-120°)
            beta_range  = list(range(90, 145, 5))   # β: angle between forearm and hand grip (90-140°)
            gamma_range = list(range(-12, 13, 3))   # γ: distance between forearm and vertical strut (-12 to +12°)
            
            # Define search space for GPyOpt (3D optimization)
            SEARCH_SPACE = [
                {'name': 'alpha', 'type': 'discrete', 'domain': alpha_range},
                {'name': 'beta',  'type': 'discrete', 'domain': beta_range},
                {'name': 'gamma', 'type': 'discrete', 'domain': gamma_range}
            ]
            
            # Prepare data for BO
            X = df[['alpha', 'beta', 'gamma']].values
            Y = df[['Total_Combined_Loss']].values
            
            # Dummy objective always returns 0 (real data provided via X, Y)
            def objective(x):
                return np.array([[0]])
            
            # Create BO optimizer (same approach as InstabilityBO)
            bo = BayesianOptimization(
                f=objective,
                domain=SEARCH_SPACE,
                model_type='GP',
                kernel=GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=3.0),
                acquisition_type='EI',
                exact_feval=True,
                X=X,
                Y=Y
            )
            
            # Use suggest_next_locations() method (same as InstabilityBO)
            next_params_array = bo.suggest_next_locations()
            a, b, g = next_params_array[0]
            
            # Round to nearest allowed values
            def round_float(value, bounds):
                return min(bounds, key=lambda x: abs(x - value))
            
            a = round_float(a, alpha_range)
            b = round_float(b, beta_range)
            g = round_float(g, gamma_range)
            
            # Check if this geometry has already been tested
            if (a, b, g) in session['tested_geometries']:
                debug_print(f"Geometry α={a}°, β={b}°, γ={g}° already tested. Finding alternative...")
                return self._suggest_alternative_geometry(session['tested_geometries'])
            
            return {
                'alpha': a,
                'beta': b,
                'gamma': g,
                'trial_number': session['trial_count'] + 1,
                'is_first_trial': False
            }
            
        except Exception as e:
            debug_print(f"BO suggestion failed: {e}. Using random geometry.")
            return self._suggest_random_geometry(session['tested_geometries'])

    def _suggest_alternative_geometry(self, tested_geometries):
        """Suggests an alternative geometry when BO suggests a duplicate."""
        # Use the same ranges as InstabilityBO.py
        alpha_range = list(range(70, 125, 5))   # α: handle angle from vertical (70-120°)
        beta_range  = list(range(90, 145, 5))   # β: angle between forearm and hand grip (90-140°)
        gamma_range = list(range(-12, 13, 3))   # γ: distance between forearm and vertical strut (-12 to +12°)
        
        import itertools
        all_geometries = set(itertools.product(alpha_range, beta_range, gamma_range))
        available = list(all_geometries - tested_geometries)
        
        if available:
            import random
            choice = random.choice(available)
            debug_print(f"→ Alternative: α={choice[0]}°, β={choice[1]}°, γ={choice[2]}°")
            return {'alpha': choice[0], 'beta': choice[1], 'gamma': choice[2]}
        else:
            # Fallback if all tested
            debug_print("⚠️ All geometries tested! Returning default.")
            return {'alpha': 95, 'beta': 125, 'gamma': 0}

    def _suggest_random_geometry(self, tested_geometries):
        """Suggests a random untested geometry."""
        result = self._suggest_alternative_geometry(tested_geometries)
        result['trial_number'] = len(tested_geometries) + 1
        result['is_first_trial'] = False
        return result

    def record_trial(self, user_id, geometry, pain_score, is_high_loss=False):
        """Records a pain optimization trial."""
        session_key = f"pain_bo_{user_id}"
        session = self.sessions.get(session_key)
        if not session:
            raise ValueError("No active session found")
        
        participant = Participant.query.get(session['participant_id'])
        
        # Create a new trial record
        survey_responses = {
            'nrs_score': pain_score,
            'is_high_loss_penalty': is_high_loss
        }
        
        # If high loss penalty, use maximum pain score
        final_pain_score = 10 if is_high_loss else pain_score
        
        processed_features = {
            'pain_optimization_loss': final_pain_score,
            'trial_number': session['trial_count'] + 1
        }
        
        new_trial = Trial(
            participant_id=session['participant_id'],
            geometry_id=None,  # BO trials don't use predefined geometries
            alpha=geometry['alpha'],
            beta=geometry['beta'],
            gamma=geometry['gamma'],
            survey_responses=survey_responses,
            processed_features=processed_features,
            source='pain_bo'
        )
        
        db.session.add(new_trial)
        db.session.commit()
        db.session.refresh(new_trial)
        
        # Update session data
        session['trials'].append(trial_to_dict(new_trial))
        session['tested_geometries'].add((geometry['alpha'], geometry['beta'], geometry['gamma']))
        session['trial_count'] += 1
        
        return trial_to_dict(new_trial)

pain_bo_manager = PainOptimizationManager()

@app.route('/api/pain-bo/check-existing-data', methods=['GET'])
def check_existing_pain_data():
    """Check if a participant has existing NRS score data from any source."""
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    try:
        participant = Participant.query.get_or_404(int(user_id))
        
        # Check for NRS scores in Grid Search trials
        grid_search_trials = Trial.query.filter_by(
            participant_id=participant.id, 
            source='grid_search'
        ).all()
        
        grid_search_nrs_data = []
        for trial in grid_search_trials:
            if trial.survey_responses and 'nrs_score' in trial.survey_responses:
                trial_dict = trial_to_dict(trial)
                grid_search_nrs_data.append({
                    'trial_id': trial.id,
                    'alpha': trial_dict['alpha'],
                    'beta': trial_dict['beta'], 
                    'gamma': trial_dict['gamma'],
                    'nrs_score': trial.survey_responses['nrs_score'],
                    'source': 'grid_search',
                    'geometry_name': trial_dict['geometry_name']
                })
        
        # Check for NRS scores in Pain BO trials
        pain_bo_trials = Trial.query.filter_by(
            participant_id=participant.id,
            source='pain_bo'
        ).filter(Trial.deleted.is_(None)).all()  # Only non-deleted trials
        
        pain_bo_nrs_data = []
        for trial in pain_bo_trials:
            if trial.survey_responses and 'nrs_score' in trial.survey_responses:
                pain_bo_nrs_data.append({
                    'trial_id': trial.id,
                    'alpha': trial.alpha,
                    'beta': trial.beta,
                    'gamma': trial.gamma,
                    'nrs_score': trial.survey_responses['nrs_score'],
                    'source': 'pain_bo',
                    'is_high_loss': trial.survey_responses.get('is_high_loss_penalty', False)
                })
        
        total_nrs_trials = len(grid_search_nrs_data) + len(pain_bo_nrs_data)
        has_existing_data = total_nrs_trials > 0
        
        return jsonify({
            'has_existing_data': has_existing_data,
            'total_nrs_trials': total_nrs_trials,
            'grid_search_trials': len(grid_search_nrs_data),
            'pain_bo_trials': len(pain_bo_nrs_data),
            'grid_search_data': grid_search_nrs_data,
            'pain_bo_data': pain_bo_nrs_data
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to check existing data: {str(e)}'}), 500

@app.route('/api/pain-bo/start', methods=['POST'])
def start_pain_bo_session():
    """Start a new pain optimization session."""
    user_id = request.json.get('userId')
    restart_mode = request.json.get('restartMode', False)  # New parameter
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    try:
        session = pain_bo_manager.start_session(user_id, restart_mode)
        
        # Include Grid Search trials in the history for display
        all_trials = session['trials'].copy()  # Start with Pain BO trials
        
        # Add Grid Search trials with NRS scores to the history
        if not restart_mode:
            participant = Participant.query.get(session['participant_id'])
            grid_search_trials = Trial.query.filter_by(
                participant_id=participant.id,
                source='grid_search'
            ).all()
            
            debug_print(f"Found {len(grid_search_trials)} Grid Search trials")
            for trial in grid_search_trials:
                if trial.survey_responses and 'nrs_score' in trial.survey_responses:
                    trial_dict = trial_to_dict(trial)
                    # Mark as Grid Search trial for frontend display
                    trial_dict['source'] = 'grid_search'
                    all_trials.append(trial_dict)
                    debug_print(f"Added Grid Search trial {trial.id} with NRS score {trial.survey_responses['nrs_score']}")
                else:
                    debug_print(f"Grid Search trial {trial.id} has no NRS score")
        
        debug_print(f"Total trials in history: {len(all_trials)} (Pain BO: {len(session['trials'])}, Grid Search: {len(all_trials) - len(session['trials'])})")
        mode_text = "RESTART" if restart_mode else "CONTINUE"
        return jsonify({
            'message': f"Pain optimization session started for {user_id} in {mode_text} mode",
            'userId': user_id,
            'trial_count': session['trial_count'],
            'history': all_trials,  # Include both Pain BO and Grid Search trials
            'restart_mode': restart_mode
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to start pain optimization session: {str(e)}'}), 500

@app.route('/api/pain-bo/first-geometry', methods=['GET'])
def get_first_pain_geometry():
    """Get the first geometry for pain optimization."""
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    try:
        geometry = pain_bo_manager.get_first_geometry(user_id)
        return jsonify(geometry), 200
    except Exception as e:
        return jsonify({'error': f'Failed to get first geometry: {str(e)}'}), 500

@app.route('/api/pain-bo/next-geometry', methods=['GET'])
def get_next_pain_geometry():
    """Get the next BO-suggested geometry for pain optimization."""
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    try:
        geometry = pain_bo_manager.get_next_geometry(user_id)
        return jsonify(geometry), 200
    except Exception as e:
        return jsonify({'error': f'Failed to get next geometry: {str(e)}'}), 500

@app.route('/api/pain-bo/record-trial', methods=['POST'])
def record_pain_trial():
    """Record a pain optimization trial with NRS score."""
    data = request.json
    user_id = data.get('userId')
    geometry = data.get('geometry')
    pain_score = data.get('painScore')
    is_high_loss = data.get('isHighLoss', False)
    
    if not all([user_id, geometry, pain_score is not None]):
        return jsonify({'error': 'userId, geometry, and painScore are required'}), 400
    
    try:
        trial = pain_bo_manager.record_trial(user_id, geometry, pain_score, is_high_loss)
        return jsonify({
            'message': 'Pain trial recorded successfully',
            'trial': trial
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to record trial: {str(e)}'}), 500

@app.route('/api/pain-bo/suggest-alternative', methods=['POST'])
def suggest_alternative_pain_geometry():
    """Suggest an alternative geometry when user rejects BO suggestion."""
    user_id = request.json.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    session_key = f"pain_bo_{user_id}"
    session = pain_bo_manager.sessions.get(session_key)
    if not session:
        return jsonify({'error': 'No active session found'}), 404
    
    try:
        alternative = pain_bo_manager._suggest_alternative_geometry(session['tested_geometries'])
        alternative['trial_number'] = session['trial_count'] + 1
        alternative['is_alternative'] = True
        return jsonify(alternative), 200
    except Exception as e:
        return jsonify({'error': f'Failed to suggest alternative: {str(e)}'}), 500


# =================================================================================
# === INSTABILITY OPTIMIZATION BAYESIAN OPTIMIZATION API ==========================
# =================================================================================

class InstabilityOptimizationManager:
    """Manages instability optimization BO sessions with discrete geometry suggestions."""
    def __init__(self):
        self.sessions = {}

    def mark_previous_trials_as_deleted(self, user_id):
        """Marks all previous instability BO trials for a user as deleted (soft delete)."""
        participant = Participant.query.get_or_404(int(user_id))
        
        # Mark all existing instability BO trials as deleted
        from datetime import datetime
        deleted_count = Trial.query.filter_by(
            participant_id=participant.id,
            source='instability_bo'
        ).filter(Trial.deleted.is_(None)).update({
            'deleted': datetime.utcnow()
        })
        
        db.session.commit()
        debug_print(f"Marked {deleted_count} previous instability BO trials as deleted for {user_id}")
        return deleted_count

    def start_session(self, user_id, restart_mode=False):
        """Starts a new instability optimization session for a user."""
        participant = Participant.query.get_or_404(int(user_id))
        
        # Use a different session key to avoid conflicts
        session_key = f"instability_bo_{user_id}"
        
        if session_key in self.sessions and not restart_mode:
            return self.sessions[session_key]

        if restart_mode:
            # In restart mode, ignore all previous data
            previous_trials = []
            # Mark existing instability BO trials as deleted
            self.mark_previous_trials_as_deleted(user_id)
            debug_print(f"Starting instability optimization in RESTART mode for {user_id} - ignoring all previous data")
        else:
            # Load previous instability optimization trials for this participant (only non-deleted ones)
            previous_trials = Trial.query.filter_by(
                participant_id=participant.id,
                source='instability_bo'
            ).filter(Trial.deleted.is_(None)).all()  # Only non-deleted trials
            debug_print(f"Starting instability optimization in CONTINUE mode for {user_id} - loaded {len(previous_trials)} previous instability BO trials")
        
        session_data = {
            'participant_id': participant.id,
            'user_id': user_id,
            'trials': [trial_to_dict(t) for t in previous_trials],
            'tested_geometries': set(),
            'trial_count': len(previous_trials),
            'restart_mode': restart_mode
        }
        
        # Track previously tested geometries (only if not in restart mode)
        if not restart_mode:
            for trial in previous_trials:
                if trial.geometry:
                    session_data['tested_geometries'].add((trial.geometry.alpha, trial.geometry.beta, trial.geometry.gamma))
                elif trial.alpha is not None and trial.beta is not None and trial.gamma is not None:
                    session_data['tested_geometries'].add((trial.alpha, trial.beta, trial.gamma))
        
        self.sessions[session_key] = session_data
        return session_data

    def get_first_geometry(self, user_id):
        """Gets the first geometry for a new instability optimization session."""
        session_key = f"instability_bo_{user_id}"
        session = self.sessions.get(session_key)
        if not session:
            raise ValueError("No active session found")
        
        # For first trial, suggest a reasonable starting point
        from core import config as core_config
        initial_geometry = core_config.initial_crutch_geometry
        return {
            'alpha': initial_geometry['alpha'],
            'beta': initial_geometry['beta'], 
            'gamma': initial_geometry['gamma'],
            'trial_number': session['trial_count'] + 1,
            'is_first_trial': True
        }

    def get_next_geometry(self, user_id):
        """Uses BO to suggest the next geometry based on previous instability losses."""
        session_key = f"instability_bo_{user_id}"
        session = self.sessions.get(session_key)
        if not session:
            raise ValueError("No active session found")
        
        # Use BO logic similar to PainOptimizationManager but with instability loss
        import pandas as pd
        import numpy as np
        
        # Convert trials to DataFrame for BO
        trials_data = []
        
        # Add instability BO trials (only non-deleted ones if in continue mode)
        for trial_dict in session['trials']:
            if trial_dict.get('processed_features') and 'instability_loss' in trial_dict['processed_features']:
                row = {
                    'alpha': trial_dict.get('alpha'),
                    'beta': trial_dict.get('beta'),
                    'gamma': trial_dict.get('gamma'),
                    'instability_loss': trial_dict['processed_features']['instability_loss'],
                    # Use instability loss as objective (lower is better)
                    'Total_Combined_Loss': trial_dict['processed_features']['instability_loss'],
                    'source': 'instability_bo'
                }
                trials_data.append(row)
        
        # In continue mode, also include instability data from Grid Search trials
        if not session.get('restart_mode', False):
            participant = Participant.query.get(session['participant_id'])
            grid_search_trials = Trial.query.filter_by(
                participant_id=participant.id,
                source='grid_search'
            ).all()
            
            grid_search_instability_count = 0
            for trial in grid_search_trials:
                if trial.processed_features and 'instability_loss' in trial.processed_features:
                    trial_dict = trial_to_dict(trial)
                    row = {
                        'alpha': trial_dict['alpha'],
                        'beta': trial_dict['beta'],
                        'gamma': trial_dict['gamma'],
                        'instability_loss': trial.processed_features['instability_loss'],
                        'Total_Combined_Loss': trial.processed_features['instability_loss'],
                        'source': 'grid_search'
                    }
                    trials_data.append(row)
                    grid_search_instability_count += 1
                    # Also track these geometries as tested
                    session['tested_geometries'].add((trial_dict['alpha'], trial_dict['beta'], trial_dict['gamma']))
            
            if grid_search_instability_count > 0:
                debug_print(f"Including {grid_search_instability_count} instability losses from Grid Search trials in BO optimization")
        
        # If we have ANY historical data, run BO optimization
        # Only use first geometry if truly no data exists (restart mode with no new trials)
        if not trials_data:
            debug_print("No historical instability data available, getting first geometry")
            return self.get_first_geometry(user_id)
        
        debug_print(f"Running BO optimization with {len(trials_data)} historical data points")
        df = pd.DataFrame(trials_data)
        
        # Use the same approach as PainOptimizationManager but for instability
        try:
            # Create BO optimizer with discrete domains
            alpha_range = list(range(70, 125, 5))   # α: handle angle from vertical (70-120°)
            beta_range  = list(range(90, 145, 5))   # β: angle between forearm and hand grip (90-140°)
            gamma_range = list(range(-12, 13, 3))   # γ: distance between forearm and vertical strut (-12 to +12°)
            
            # Define search space for GPyOpt (3D optimization)
            SEARCH_SPACE = [
                {'name': 'alpha', 'type': 'discrete', 'domain': alpha_range},
                {'name': 'beta',  'type': 'discrete', 'domain': beta_range},
                {'name': 'gamma', 'type': 'discrete', 'domain': gamma_range}
            ]
            
            # Prepare data for BO
            X = df[['alpha', 'beta', 'gamma']].values
            Y = df[['Total_Combined_Loss']].values
            
            # Dummy objective always returns 0 (real data provided via X, Y)
            def objective(x):
                return np.array([[0]])
            
            # Create BO optimizer (same approach as PainOptimizationManager)
            bo = BayesianOptimization(
                f=objective,
                domain=SEARCH_SPACE,
                model_type='GP',
                kernel=GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=3.0),
                acquisition_type='EI',
                exact_feval=True,
                X=X,
                Y=Y
            )
            
            # Use suggest_next_locations() method
            next_params_array = bo.suggest_next_locations()
            a, b, g = next_params_array[0]
            
            # Round to nearest allowed values
            def round_float(value, bounds):
                return min(bounds, key=lambda x: abs(x - value))
            
            a = round_float(a, alpha_range)
            b = round_float(b, beta_range)
            g = round_float(g, gamma_range)
            
            # Check if this geometry has already been tested
            if (a, b, g) in session['tested_geometries']:
                debug_print(f"Geometry α={a}°, β={b}°, γ={g}° already tested. Finding alternative...")
                return self._suggest_alternative_geometry(session['tested_geometries'])
            
            return {
                'alpha': a,
                'beta': b,
                'gamma': g,
                'trial_number': session['trial_count'] + 1,
                'is_first_trial': False
            }
            
        except Exception as e:
            debug_print(f"BO suggestion failed: {e}. Using random geometry.")
            return self._suggest_random_geometry(session['tested_geometries'])

    def _suggest_alternative_geometry(self, tested_geometries):
        """Suggests an alternative geometry when BO suggests a duplicate."""
        # Use the same ranges as PainOptimizationManager
        alpha_range = list(range(70, 125, 5))   # α: handle angle from vertical (70-120°)
        beta_range  = list(range(90, 145, 5))   # β: angle between forearm and hand grip (90-140°)
        gamma_range = list(range(-12, 13, 3))   # γ: distance between forearm and vertical strut (-12 to +12°)
        
        import itertools
        all_geometries = set(itertools.product(alpha_range, beta_range, gamma_range))
        available = list(all_geometries - tested_geometries)
        
        if available:
            import random
            choice = random.choice(available)
            debug_print(f"→ Alternative: α={choice[0]}°, β={choice[1]}°, γ={choice[2]}°")
            return {'alpha': choice[0], 'beta': choice[1], 'gamma': choice[2]}
        else:
            # Fallback if all tested
            debug_print("⚠️ All geometries tested! Returning default.")
            return {'alpha': 95, 'beta': 125, 'gamma': 0}

    def _suggest_random_geometry(self, tested_geometries):
        """Suggests a random untested geometry."""
        result = self._suggest_alternative_geometry(tested_geometries)
        result['trial_number'] = len(tested_geometries) + 1
        result['is_first_trial'] = False
        return result

    def record_trial(self, user_id, geometry, instability_loss, sus_score):
        """Records an instability optimization trial."""
        session_key = f"instability_bo_{user_id}"
        session = self.sessions.get(session_key)
        if not session:
            raise ValueError("No active session found")
        
        participant = Participant.query.get(session['participant_id'])
        
        # Create a new trial record
        survey_responses = {
            'sus_score': sus_score
        }
        
        processed_features = {
            'instability_loss': instability_loss,
            'trial_number': session['trial_count'] + 1
        }
        
        new_trial = Trial(
            participant_id=session['participant_id'],
            geometry_id=None,  # BO trials don't use predefined geometries
            alpha=geometry['alpha'],
            beta=geometry['beta'],
            gamma=geometry['gamma'],
            survey_responses=survey_responses,
            processed_features=processed_features,
            source='instability_bo'
        )
        
        db.session.add(new_trial)
        db.session.commit()
        db.session.refresh(new_trial)
        
        # Update session data
        session['trials'].append(trial_to_dict(new_trial))
        session['tested_geometries'].add((geometry['alpha'], geometry['beta'], geometry['gamma']))
        session['trial_count'] += 1
        
        return trial_to_dict(new_trial)

instability_bo_manager = InstabilityOptimizationManager()

@app.route('/api/instability-bo/check-existing-data', methods=['GET'])
def check_existing_instability_data():
    """Check if a participant has existing instability loss data from any source."""
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    try:
        participant = Participant.query.get_or_404(int(user_id))
        
        # Check for instability losses in Grid Search trials
        grid_search_trials = Trial.query.filter_by(
            participant_id=participant.id, 
            source='grid_search'
        ).all()
        
        grid_search_instability_data = []
        for trial in grid_search_trials:
            if trial.processed_features and 'instability_loss' in trial.processed_features:
                trial_dict = trial_to_dict(trial)
                grid_search_instability_data.append({
                    'trial_id': trial.id,
                    'alpha': trial_dict['alpha'],
                    'beta': trial_dict['beta'], 
                    'gamma': trial_dict['gamma'],
                    'instability_loss': trial.processed_features['instability_loss'],
                    'source': 'grid_search',
                    'geometry_name': trial_dict['geometry_name']
                })
        
        # Check for instability losses in Instability BO trials
        instability_bo_trials = Trial.query.filter_by(
            participant_id=participant.id,
            source='instability_bo'
        ).filter(Trial.deleted.is_(None)).all()  # Only non-deleted trials
        
        instability_bo_data = []
        for trial in instability_bo_trials:
            if trial.processed_features and 'instability_loss' in trial.processed_features:
                instability_bo_data.append({
                    'trial_id': trial.id,
                    'alpha': trial.alpha,
                    'beta': trial.beta,
                    'gamma': trial.gamma,
                    'instability_loss': trial.processed_features['instability_loss'],
                    'source': 'instability_bo',
                    'sus_score': trial.survey_responses.get('sus_score', 0)
                })
        
        total_instability_trials = len(grid_search_instability_data) + len(instability_bo_data)
        has_existing_data = total_instability_trials > 0
        
        return jsonify({
            'has_existing_data': has_existing_data,
            'total_instability_trials': total_instability_trials,
            'grid_search_trials': len(grid_search_instability_data),
            'instability_bo_trials': len(instability_bo_data),
            'grid_search_data': grid_search_instability_data,
            'instability_bo_data': instability_bo_data
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to check existing data: {str(e)}'}), 500

@app.route('/api/instability-bo/start', methods=['POST'])
def start_instability_bo_session():
    """Start a new instability optimization session."""
    user_id = request.json.get('userId')
    restart_mode = request.json.get('restartMode', False)
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    try:
        session = instability_bo_manager.start_session(user_id, restart_mode)
        
        # Include Grid Search trials in the history for display
        all_trials = session['trials'].copy()  # Start with Instability BO trials
        
        # Add Grid Search trials with instability losses to the history
        if not restart_mode:
            participant = Participant.query.get(session['participant_id'])
            grid_search_trials = Trial.query.filter_by(
                participant_id=participant.id,
                source='grid_search'
            ).all()
            
            debug_print(f"Found {len(grid_search_trials)} Grid Search trials")
            for trial in grid_search_trials:
                if trial.processed_features and 'instability_loss' in trial.processed_features:
                    trial_dict = trial_to_dict(trial)
                    # Mark as Grid Search trial for frontend display
                    trial_dict['source'] = 'grid_search'
                    all_trials.append(trial_dict)
                    debug_print(f"Added Grid Search trial {trial.id} with instability loss {trial.processed_features['instability_loss']}")
                else:
                    debug_print(f"Grid Search trial {trial.id} has no instability loss")
        
        debug_print(f"Total trials in history: {len(all_trials)} (Instability BO: {len(session['trials'])}, Grid Search: {len(all_trials) - len(session['trials'])})")
        
        mode_text = "RESTART" if restart_mode else "CONTINUE"
        return jsonify({
            'message': f"Instability optimization session started for {user_id} in {mode_text} mode",
            'userId': user_id,
            'trial_count': session['trial_count'],
            'history': all_trials,  # Include both Instability BO and Grid Search trials
            'restart_mode': restart_mode
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to start instability optimization session: {str(e)}'}), 500

@app.route('/api/instability-bo/first-geometry', methods=['GET'])
def get_first_instability_geometry():
    """Get the first geometry for instability optimization."""
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    try:
        geometry = instability_bo_manager.get_first_geometry(user_id)
        return jsonify(geometry), 200
    except Exception as e:
        return jsonify({'error': f'Failed to get first geometry: {str(e)}'}), 500

@app.route('/api/instability-bo/next-geometry', methods=['GET'])
def get_next_instability_geometry():
    """Get the next BO-suggested geometry for instability optimization."""
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    try:
        geometry = instability_bo_manager.get_next_geometry(user_id)
        return jsonify(geometry), 200
    except Exception as e:
        return jsonify({'error': f'Failed to get next geometry: {str(e)}'}), 500

@app.route('/api/instability-bo/record-trial', methods=['POST'])
def record_instability_trial():
    """Record an instability optimization trial with instability loss and SUS score."""
    data = request.json
    user_id = data.get('userId')
    geometry = data.get('geometry')
    instability_loss = data.get('instabilityLoss')
    sus_score = data.get('susScore')
    
    if not all([user_id, geometry, instability_loss is not None, sus_score is not None]):
        return jsonify({'error': 'userId, geometry, instabilityLoss, and susScore are required'}), 400
    
    try:
        trial = instability_bo_manager.record_trial(user_id, geometry, instability_loss, sus_score)
        return jsonify({
            'message': 'Instability trial recorded successfully',
            'trial': trial
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to record trial: {str(e)}'}), 500

@app.route('/api/instability-bo/suggest-alternative', methods=['POST'])
def suggest_alternative_instability_geometry():
    """Suggest an alternative geometry when user rejects BO suggestion."""
    user_id = request.json.get('userId')
    if not user_id:
        return jsonify({'error': 'userId is required'}), 400
    
    session_key = f"instability_bo_{user_id}"
    session = instability_bo_manager.sessions.get(session_key)
    if not session:
        return jsonify({'error': 'No active session found'}), 404
    
    try:
        alternative = instability_bo_manager._suggest_alternative_geometry(session['tested_geometries'])
        alternative['trial_number'] = session['trial_count'] + 1
        alternative['is_alternative'] = True
        return jsonify(alternative), 200
    except Exception as e:
        return jsonify({'error': f'Failed to suggest alternative: {str(e)}'}), 500

# --- Effort Optimization Manager (copied from Pain) ---
class EffortOptimizationManager:
    def __init__(self):
        self.sessions = {}
    
    def mark_previous_trials_as_deleted(self, participant_id):
        """Soft delete previous effort_bo trials for this participant"""
        try:
            from datetime import datetime
            now = datetime.utcnow()
            
            previous_trials = Trial.query.filter(
                Trial.participant_id == participant_id,
                Trial.source == 'effort_bo',
                Trial.deleted.is_(None)
            ).all()
            
            for trial in previous_trials:
                trial.deleted = now
            
            db.session.commit()
            debug_print(f"Marked {len(previous_trials)} previous effort BO trials as deleted for participant {participant_id}")
            
        except Exception as e:
            debug_print(f"Error marking previous effort BO trials as deleted: {str(e)}")
            db.session.rollback()
    
    def start_session(self, participant_id, restart_mode=False):
        """Initialize or restart a effort optimization session"""
        try:
            # Get participant
            participant = Participant.query.get(participant_id)
            if not participant:
                raise ValueError(f"Participant {participant_id} not found")
            
            # If restart mode, mark previous effort_bo trials as deleted
            if restart_mode:
                self.mark_previous_trials_as_deleted(participant_id)
            
            # Get all trials for this participant (excluding deleted ones)
            all_trials = Trial.query.filter(
                Trial.participant_id == participant_id,
                Trial.deleted.is_(None)
            ).all()
            
            # For continue mode, include grid_search and effort_bo trials with NASA TLX data
            # For restart mode, only include grid_search trials (if any)
            if restart_mode:
                # Only include grid_search trials for baseline data
                trials_data = []
                for trial in all_trials:
                    if (trial.source == 'grid_search' and 
                        trial.survey_responses and 
                        'nasa_tlx' in trial.survey_responses):
                        
                        scores = trial.survey_responses['nasa_tlx']
                        effort_score = (scores.get('mental_demand', 0) + scores.get('physical_demand', 0) + 
                                      scores.get('temporal_demand', 0) + scores.get('performance', 0) + 
                                      scores.get('effort', 0) + scores.get('frustration', 0)) / 6
                        
                        trials_data.append({
                            'geometry': [trial.alpha, trial.beta, trial.gamma],
                            'effort_score': effort_score
                        })
            else:
                # Include both grid_search and effort_bo trials with effort data
                trials_data = []
                for trial in all_trials:
                    effort_score = None
                    
                    # Check for NASA TLX data (from Grid Search or Effort BO)
                    if (trial.survey_responses and 'nasa_tlx' in trial.survey_responses):
                        scores = trial.survey_responses['nasa_tlx']
                        effort_score = (scores.get('mental_demand', 0) + scores.get('physical_demand', 0) + 
                                      scores.get('temporal_demand', 0) + scores.get('performance', 0) + 
                                      scores.get('effort', 0) + scores.get('frustration', 0)) / 6
                    
                    # Check for processed features (direct effort score from Effort BO)
                    elif (trial.processed_features and 'effort_score' in trial.processed_features):
                        effort_score = trial.processed_features['effort_score']
                    
                    if effort_score is not None:
                        trials_data.append({
                            'geometry': [trial.alpha, trial.beta, trial.gamma],
                            'effort_score': effort_score
                        })
            
            # Store session data
            session_data = {
                'participant_id': participant_id,
                'trials_data': trials_data,
                'tested_geometries': set((t['geometry'][0], t['geometry'][1], t['geometry'][2]) 
                                       for t in trials_data),
                'restart_mode': restart_mode
            }
            
            session_key = f"effort_bo_{participant_id}"
            self.sessions[session_key] = session_data
            
            debug_print(f"Started effort optimization session for participant {participant_id}")
            debug_print(f"Loaded {len(trials_data)} trials with effort data")
            debug_print(f"Restart mode: {restart_mode}")
            
            return session_data
            
        except Exception as e:
            debug_print(f"Error starting effort optimization session: {str(e)}")
            raise
    
    def get_next_geometry(self, participant_id):
        """Get the next geometry suggestion using Bayesian Optimization"""
        session_key = f"effort_bo_{participant_id}"
        
        if session_key not in self.sessions:
            raise ValueError("No active session found")
        
        session = self.sessions[session_key]
        trials_data = session['trials_data']
        
        # If no trials yet, suggest a random starting geometry
        if len(trials_data) == 0:
            return self._suggest_random_geometry()
        
        # Run Bayesian Optimization
        if GPy is None or BayesianOptimization is None:
            debug_print("GPy/GPyOpt not available, using random suggestion")
            return self._suggest_random_geometry()
        
        try:
            # Extract X (geometries) and Y (effort scores) from trials
            X = np.array([trial['geometry'] for trial in trials_data])
            Y = np.array([[trial['effort_score']] for trial in trials_data])
            
            debug_print(f"BO input - X shape: {X.shape}, Y shape: {Y.shape}")
            debug_print(f"X: {X}")
            debug_print(f"Y: {Y}")
            
            # Define discrete parameter space (same as Pain BO)
            alpha_range = list(range(70, 125, 5))  # 70, 75, 80, ..., 120
            beta_range = list(range(90, 145, 5))   # 90, 95, 100, ..., 140  
            gamma_range = list(range(-12, 15, 3))  # -12, -9, -6, ..., 12
            
            domain = [
                {'name': 'alpha', 'type': 'discrete', 'domain': alpha_range},
                {'name': 'beta', 'type': 'discrete', 'domain': beta_range}, 
                {'name': 'gamma', 'type': 'discrete', 'domain': gamma_range}
            ]
            
            # Create BO object for MINIMIZATION (lower effort score = better)
            bo = BayesianOptimization(
                f=None,  # We're not optimizing a function directly
                domain=domain,
                X=X,
                Y=Y,
                model_type='GP',
                acquisition_type='EI',  # Expected Improvement
                normalize_Y=True,
                exact_feval=True
            )
            
            # Use suggest_next_locations() method
            next_params_array = bo.suggest_next_locations()
            a, b, g = next_params_array[0]
            
            # Round to nearest allowed values
            def round_float(value, bounds):
                return min(bounds, key=lambda x: abs(x - value))
            
            a = round_float(a, alpha_range)
            b = round_float(b, beta_range)
            g = round_float(g, gamma_range)
            
            # Check if this geometry has already been tested
            if (a, b, g) in session['tested_geometries']:
                debug_print(f"Geometry α={a}°, β={b}°, γ={g}° already tested. Finding alternative...")
                return self._suggest_alternative_geometry(session['tested_geometries'])
            
            return {
                'alpha': a,
                'beta': b,
                'gamma': g,
                'trial_number': len(trials_data) + 1,
                'source': 'BO_suggestion'
            }
            
        except Exception as e:
            debug_print(f"Error in BO optimization: {str(e)}")
            return self._suggest_random_geometry()
    
    def _suggest_alternative_geometry(self, tested_geometries):
        """Suggest an alternative geometry not in tested set"""
        return self._suggest_random_geometry(exclude=tested_geometries)
    
    def _suggest_random_geometry(self, exclude=None):
        """Suggest a random geometry from the discrete parameter space"""
        import random
        
        alpha_range = list(range(70, 125, 5))
        beta_range = list(range(90, 145, 5))  
        gamma_range = list(range(-12, 15, 3))
        
        if exclude is None:
            exclude = set()
        
        # Generate all possible combinations
        all_geometries = [(a, b, g) for a in alpha_range for b in beta_range for g in gamma_range]
        available_geometries = [g for g in all_geometries if g not in exclude]
        
        if not available_geometries:
            # If all geometries tested, suggest from tested ones
            available_geometries = all_geometries
        
        a, b, g = random.choice(available_geometries)
        
        return {
            'alpha': a,
            'beta': b, 
            'gamma': g,
            'trial_number': 1,
            'source': 'random_suggestion'
        }
    
    def record_trial(self, participant_id, geometry, effort_score, survey_data=None, is_high_loss=False):
        """Record a trial result"""
        try:
            # Create trial record
            trial = Trial(
                participant_id=participant_id,
                geometry_id=None,  # BO trials don't use predefined geometries
                alpha=geometry['alpha'],
                beta=geometry['beta'], 
                gamma=geometry['gamma'],
                source='effort_bo',
                survey_responses={'nasa_tlx': survey_data} if survey_data else None,
                processed_features={'effort_score': effort_score} if not survey_data else None
            )
            
            db.session.add(trial)
            db.session.commit()
            
            # Update session with this trial
            session_key = f"effort_bo_{participant_id}"
            if session_key in self.sessions:
                session = self.sessions[session_key]
                session['trials_data'].append({
                    'geometry': [geometry['alpha'], geometry['beta'], geometry['gamma']],
                    'effort_score': effort_score
                })
                session['tested_geometries'].add((geometry['alpha'], geometry['beta'], geometry['gamma']))
            
            debug_print(f"Recorded effort BO trial: effort_score={effort_score}")
            return trial
            
        except Exception as e:
            debug_print(f"Error recording effort trial: {str(e)}")
            db.session.rollback()
            raise

# Global effort optimization manager
effort_bo_sessions = EffortOptimizationManager()

# --- Effort Optimization API Endpoints ---

@app.route('/api/effort-bo/check-existing-data', methods=['GET'])
def check_existing_effort_data():
    """Check if participant has existing effort data"""
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        participant_id = int(user_id)
        
        # Get all trials for this participant (excluding deleted ones)
        all_trials = Trial.query.filter(
            Trial.participant_id == participant_id,
            Trial.deleted.is_(None)
        ).all()
        
        # Count trials with effort data
        grid_search_count = 0
        effort_bo_count = 0
        
        for trial in all_trials:
            has_effort_data = False
            
            # Debug: Print trial info
            debug_print(f"Checking trial {trial.id}: source={trial.source}, survey_responses={trial.survey_responses}, processed_features={trial.processed_features}")
            
            # Check for NASA TLX data (look for TLX fields or TLX score)
            if trial.survey_responses:
                survey_data = trial.survey_responses
                # Check for TLX score or individual TLX questions
                if ('tlx_score' in survey_data or 
                    'tlx_q1' in survey_data or 
                    'tlx_q2' in survey_data or 
                    'tlx_q3' in survey_data or 
                    'tlx_q4' in survey_data or 
                    'tlx_q5' in survey_data):
                    has_effort_data = True
                    debug_print(f"Found NASA TLX data in trial {trial.id}")
            
            # Check for processed effort score
            if (trial.processed_features and 'effort_score' in trial.processed_features):
                has_effort_data = True
                debug_print(f"Found effort score in trial {trial.id}")
            
            if has_effort_data:
                if trial.source == 'grid_search':
                    grid_search_count += 1
                elif trial.source == 'effort_bo':
                    effort_bo_count += 1
        
        total_count = grid_search_count + effort_bo_count
        has_existing_data = total_count > 0
        
        debug_print(f"Effort data check result: has_existing_data={has_existing_data}, total_count={total_count}, grid_search_count={grid_search_count}, effort_bo_count={effort_bo_count}")
        
        return jsonify({
            'has_existing_data': has_existing_data,
            'trialCount': total_count,
            'gridSearchCount': grid_search_count,
            'effortBoCount': effort_bo_count
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to check existing data: {str(e)}'}), 500

@app.route('/api/effort-bo/start', methods=['POST'])
def start_effort_bo_session():
    """Start or restart effort optimization session"""
    try:
        data = request.get_json()
        user_id = data.get('userId')
        restart_mode = data.get('restartMode', False)
        
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        participant_id = int(user_id)
        
        # Start session
        session_data = effort_bo_sessions.start_session(participant_id, restart_mode)
        
        # Get history for frontend (including Grid Search trials with NASA TLX)
        all_trials = Trial.query.filter(
            Trial.participant_id == participant_id,
            Trial.deleted.is_(None)
        ).all()
        
        history = []
        for trial in all_trials:
            # Include trials with effort data
            has_effort_data = False
            if trial.survey_responses:
                survey_data = trial.survey_responses
                # Check for TLX score or individual TLX questions
                if ('tlx_score' in survey_data or 
                    'tlx_q1' in survey_data or 
                    'tlx_q2' in survey_data or 
                    'tlx_q3' in survey_data or 
                    'tlx_q4' in survey_data or 
                    'tlx_q5' in survey_data):
                    has_effort_data = True
            
            if ((trial.source == 'grid_search' and has_effort_data) or
                (trial.source == 'effort_bo')):
                
                trial_dict = trial_to_dict(trial)
                trial_dict = clean_nan_values(trial_dict)
                history.append(trial_dict)
        
        response_data = {
            'success': True,
            'message': 'Effort optimization session started',
            'history': history,
            'sessionData': {
                'participantId': participant_id,
                'trialsCount': len(session_data['trials_data']),
                'restartMode': restart_mode
            }
        }
        
        return jsonify(clean_nan_values(response_data))
        
    except Exception as e:
        return jsonify({'error': f'Failed to start BO session: {str(e)}'}), 500

@app.route('/api/effort-bo/first-geometry', methods=['GET'])
def get_first_effort_geometry():
    """Get first geometry for effort optimization"""
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        participant_id = int(user_id)
        geometry = effort_bo_sessions.get_next_geometry(participant_id)
        
        return jsonify({'geometry': geometry})
        
    except Exception as e:
        return jsonify({'error': f'Failed to get first geometry: {str(e)}'}), 500

@app.route('/api/effort-bo/next-geometry', methods=['GET'])
def get_next_effort_geometry():
    """Get next geometry suggestion from BO"""
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        participant_id = int(user_id)
        geometry = effort_bo_sessions.get_next_geometry(participant_id)
        
        return jsonify({'geometry': geometry})
        
    except Exception as e:
        return jsonify({'error': f'Failed to get next geometry: {str(e)}'}), 500

@app.route('/api/effort-bo/record-trial', methods=['POST'])
def record_effort_trial():
    """Record a effort optimization trial"""
    try:
        data = request.get_json()
        user_id = data.get('userId')
        geometry = data.get('geometry')
        effort_score = data.get('effortScore')
        survey_data = data.get('surveyData')
        is_high_loss = data.get('isHighLoss', False)
        
        if not all([user_id, geometry, effort_score is not None]):
            return jsonify({'error': 'userId, geometry, and effortScore are required'}), 400
        
        participant_id = int(user_id)
        
        # Record the trial
        trial = effort_bo_sessions.record_trial(
            participant_id, geometry, effort_score, survey_data, is_high_loss
        )
        
        trial_dict = trial_to_dict(trial)
        trial_dict = clean_nan_values(trial_dict)
        
        return jsonify({
            'success': True,
            'message': 'Trial recorded successfully',
            'trial': trial_dict
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to record trial: {str(e)}'}), 500

@app.route('/api/effort-bo/suggest-alternative', methods=['POST'])
def suggest_alternative_effort_geometry():
    """Suggest alternative geometry"""
    try:
        data = request.get_json()
        user_id = data.get('userId')
        
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        participant_id = int(user_id)
        
        # Get session and suggest alternative
        session_key = f"effort_bo_{participant_id}"
        if session_key not in effort_bo_sessions.sessions:
            return jsonify({'error': 'No active session found'}), 400
        
        session = effort_bo_sessions.sessions[session_key]
        geometry = effort_bo_sessions._suggest_alternative_geometry(session['tested_geometries'])
        
        return jsonify({'geometry': geometry})
        
    except Exception as e:
        return jsonify({'error': f'Failed to suggest alternative: {str(e)}'}), 500

# =================================================================================
# === COMMON AND STATIC ROUTES ====================================================
# =================================================================================

@app.route('/data/<path:path>')
def serve_data(path):
    """Serves files from the main data directory, including plots."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    return send_from_directory(data_dir, path)

def clean_nan_values(obj):
    """Recursively clean NaN values from dictionaries and lists, replacing them with None."""
    import math
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or np.isnan(obj)):
        return None
    elif hasattr(obj, 'isna') and obj.isna():
        return None
    elif hasattr(obj, 'isna') and hasattr(obj, 'iloc') and obj.isna().any():
        # Handle pandas Series with NaN values
        return None
    elif isinstance(obj, (np.integer, np.floating)) and (np.isnan(obj) if hasattr(obj, 'isna') else False):
        return None
    else:
        return obj

if __name__ == '__main__':
    app.run(debug=True, port=5000) 