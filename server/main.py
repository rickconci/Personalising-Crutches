import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json

from core.experiment import Experiment
from core.data_manager import DataManager
from core import config as core_config

app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Experiment Management ---
class ExperimentManager:
    """Manages active experiment sessions."""
    def __init__(self):
        self.experiments = {}
        self.data_manager = DataManager()

    def start_or_load_experiment(self, user_id, user_characteristics, objective, manual_correction, visualize_steps):
        """Starts a new experiment or loads an existing one."""
        if user_id in self.experiments:
            return self.experiments[user_id]

        previous_data = self.data_manager.load_user_data(user_id)
        initial_crutch_geometry = core_config.initial_crutch_geometry.copy()

        if previous_data is not None and not previous_data.empty:
            print(f"Continuing experiment for user '{user_id}'")
            last_trial = previous_data.iloc[-1]
            # Use the best geometry from the past as the new starting point
            best_past_trial = previous_data.loc[previous_data['Total_Combined_Loss'].idxmin()]
            initial_crutch_geometry = {
                'alpha': best_past_trial['alpha'],
                'beta': best_past_trial['beta'],
                'gamma': best_past_trial['gamma']
            }
        
        experiment = Experiment(
            user_id=user_id,
            user_characteristics=user_characteristics,
            objective=objective,
            initial_crutch_geometry=initial_crutch_geometry,
            data_manager=self.data_manager,
            manual_correction=manual_correction,
            visualize_steps=visualize_steps,
            previous_data=previous_data
        )
        self.experiments[user_id] = experiment
        return experiment

    def get_experiment(self, user_id):
        """Retrieves an experiment by user ID."""
        return self.experiments.get(user_id)

experiment_manager = ExperimentManager()

# --- API Endpoints ---
@app.route('/api/experiment/start', methods=['POST'])
def start_experiment():
    """
    Starts a new experiment session for a user.
    Expects a JSON body with user_id, user_characteristics, and objective.
    """
    data = request.json
    user_id = data.get('userId')
    user_characteristics = data.get('userCharacteristics')
    objective = data.get('objective')
    
    if not all([user_id, user_characteristics, objective]):
        return jsonify({'error': 'Missing required fields: userId, userCharacteristics, or objective'}), 400

    # These can be made configurable from the UI later if needed
    manual_correction = False
    visualize_steps = True

    try:
        experiment = experiment_manager.start_or_load_experiment(
            user_id, user_characteristics, objective, manual_correction, visualize_steps
        )
        history = []
        if experiment.experiment_data is not None and not experiment.experiment_data.empty:
            history = experiment.experiment_data.to_dict('records')

        return jsonify({
            'message': f"Experiment started for user {user_id}",
            'userId': user_id,
            'initialGeometry': experiment.current_crutch_geometry,
            'history': history
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to start experiment: {str(e)}'}), 500

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serves files from the data directory (e.g., plots)."""
    data_dir = os.path.abspath(core_config.DATA_DIRECTORY)
    return send_from_directory(data_dir, filename)

@app.route('/api/experiment/next-geometry', methods=['GET'])
def get_next_geometry():
    """Gets the next suggested geometry from the optimizer."""
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'Missing userId parameter'}), 400

    experiment = experiment_manager.get_experiment(user_id)
    if not experiment:
        return jsonify({'error': 'Experiment not found for this user'}), 404
        
    try:
        next_geometry = experiment.get_next_suggestion()
        return jsonify(next_geometry), 200
    except Exception as e:
        return jsonify({'error': f'Failed to get next suggestion: {str(e)}'}), 500

@app.route('/api/experiment/trial', methods=['POST'])
def process_trial():
    """
    Processes a single trial: saves file, runs analysis, returns results.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    form_data = request.form.get('data')
    if not form_data:
        return jsonify({'error': 'No metadata provided for the trial'}), 400
    
    try:
        trial_data = json.loads(form_data)
        user_id = trial_data.get('userId')
        crutch_geometry = trial_data.get('crutchGeometry')
        subjective_metrics = trial_data.get('subjectiveMetrics')

        experiment = experiment_manager.get_experiment(user_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found for this user'}), 404
            
        filename = secure_filename(file.filename)
        raw_data_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(raw_data_path)
        
        trial_num = len(experiment.experiment_data) + 1

        success = experiment.run_and_process_trial(
            trial_num, raw_data_path, subjective_metrics, crutch_geometry
        )
        
        if success:
            history = experiment.experiment_data.to_dict('records')
            return jsonify({
                'message': 'Trial processed successfully',
                'history': history,
                'plot_path': f'/data/{user_id}/plots/optimization_progress_trial_{trial_num}.png'
            }), 200
        else:
            return jsonify({'error': 'Failed to process trial'}), 500

    except (json.JSONDecodeError, KeyError) as e:
        return jsonify({'error': f'Invalid form data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/experiment/penalty', methods=['POST'])
def record_penalty():
    """Records a penalty for a rejected geometry."""
    data = request.json
    user_id = data.get('userId')
    penalty_loss = data.get('penaltyLoss')
    crutch_geometry = data.get('crutchGeometry')

    if not all([user_id, penalty_loss, crutch_geometry]):
        return jsonify({'error': 'Missing required fields'}), 400

    experiment = experiment_manager.get_experiment(user_id)
    if not experiment:
        return jsonify({'error': 'Experiment not found for this user'}), 404

    try:
        experiment.record_penalty_trial(float(penalty_loss), crutch_geometry)
        history = experiment.experiment_data.to_dict('records')
        return jsonify({
            'message': 'Penalty recorded',
            'history': history
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to record penalty: {str(e)}'}), 500

@app.route('/api/experiment/best', methods=['GET'])
def get_best_trial():
    """Gets the best performing trial for a user."""
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({'error': 'Missing userId parameter'}), 400

    experiment = experiment_manager.get_experiment(user_id)
    if not experiment:
        return jsonify({'error': 'Experiment not found for this user'}), 404

    best_trial = experiment.get_best_trial()
    if best_trial:
        return jsonify(best_trial), 200
    else:
        return jsonify({'message': 'No valid trials completed yet'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000) 