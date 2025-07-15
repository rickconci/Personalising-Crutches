from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
import os
import tempfile
from werkzeug.utils import secure_filename
from analysis import analyze_accelerometer_data
from optimization import bayesian_optimization

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store data in memory (for demo purposes)
# In a production app, you'd use a database
uploaded_files = {}
analysis_results = {}
experiment_history = []

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for accelerometer data"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate a file ID (for simplicity, use the filename)
        file_id = filename
        uploaded_files[file_id] = filepath
        
        return jsonify({
            'fileId': file_id,
            'message': 'File uploaded successfully'
        })

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Analyze accelerometer data with subjective metrics"""
    data = request.json
    file_id = data.get('accelerometerData')
    num_bouts = data.get('numBouts', 5)
    weightings = data.get('weightings', {
        'pain': 0.33,
        'effort': 0.33,
        'instability': 0.34
    })
    subjective_metrics = data.get('subjectiveMetrics', {
        'pain': 0,
        'effort': 0,
        'instability': 0
    })
    
    # Get file path from file ID
    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404
    
    filepath = uploaded_files[file_id]
    
    # Run analysis on the accelerometer data
    # In a real app, this would use actual accelerometer data processing
    # For demo, we'll use our simulated function
    result = analyze_accelerometer_data(
        filepath, 
        num_bouts,
        subjective_metrics,
        weightings
    )
    
    # Store result
    analysis_id = f"analysis_{len(analysis_results) + 1}"
    analysis_results[analysis_id] = result
    
    return jsonify(result)

@app.route('/api/optimize', methods=['POST'])
def optimize_geometry():
    """Run Bayesian optimization on experiment history"""
    data = request.json
    history = data.get('history', [])
    kernel_type = data.get('kernelType', 'rbf')
    iterations = data.get('iterations', 5)
    
    # Update global experiment history
    global experiment_history
    experiment_history = history
    
    # Convert history to numpy arrays for BO
    if not history:
        return jsonify({'error': 'No experiment history available'}), 400
    
    X = np.array([[entry['geometry']['alpha'], 
                   entry['geometry']['beta'],
                   entry['geometry']['gamma'],
                   entry['geometry']['delta']] for entry in history])
    
    y = np.array([float(entry['weightedLoss']) for entry in history])
    
    # Run Bayesian optimization
    result = bayesian_optimization(X, y, kernel_type, iterations)
    
    return jsonify(result)

@app.route('/api/history', methods=['GET'])
def get_history():
    """Return the experiment history"""
    return jsonify(experiment_history)

if __name__ == '__main__':
    app.run(debug=True) 