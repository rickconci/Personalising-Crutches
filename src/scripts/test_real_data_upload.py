#!/usr/bin/env python3
"""
Test script to upload real Katie Ho data and verify the new directory structure.
"""

import os
import requests
import json
from datetime import datetime

def test_real_data_upload():
    """Test uploading real Katie Ho data and verify directory structure."""
    
    base_url = "http://127.0.0.1:8000"
    
    print("=== Testing Real Data Upload with Katie Ho Data ===")
    
    # Use real Katie Ho data
    real_data_file = "/Users/riccardoconci/Local_documents/Personalising-Crutches/data/2025.08.05/Katie Ho_data_export (2)/Trial Control - Control/live_recorded_data.csv"
    
    if not os.path.exists(real_data_file):
        print(f"Real data file not found: {real_data_file}")
        return
    
    print(f"Using real data from: {real_data_file}")
    
    # Check file size and structure
    file_size = os.path.getsize(real_data_file)
    print(f"File size: {file_size / (1024*1024):.2f} MB")
    
    # Show first few lines
    with open(real_data_file, 'r') as f:
        lines = f.readlines()[:5]
        print("First 5 lines of real data:")
        for i, line in enumerate(lines):
            print(f"  {i+1}: {line.strip()}")
    
    # Step 1: Create trial first
    print("Step 1: Creating trial...")
    trial_data = {
        "participant_id": 1,  # MH1
        "geometry_id": 1,     # Control geometry
        "alpha": 95,
        "beta": 125,
        "gamma": 0,
        "delta": 0,
        "source": "grid_search",
        "survey_responses": {
            "effort_survey_answer": 3,
            "pain_survey_answer": 3,
            "stability_survey_answer": 3
        }
    }
    
    trial_response = requests.post(f"{base_url}/api/experiments/trials", json=trial_data)
    print(f"Trial creation status: {trial_response.status_code}")
    
    if trial_response.status_code != 200:
        print(f"Trial creation failed: {trial_response.text}")
        return
    
    trial = trial_response.json()
    trial_id = trial['id']
    print(f"✅ Trial created with ID: {trial_id}")
    
    # Step 2: Upload the real data file
    print("Step 2: Uploading real data file...")
    with open(real_data_file, "rb") as f:
        filename = os.path.basename(real_data_file)
        files = {"file": (filename, f, "text/csv")}
        data = {"trial_id": trial_id, "participant_id": 1}
        response = requests.post(f"{base_url}/api/data/upload", files=files, data=data)
    
    print(f"Upload response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return
    
    upload_result = response.json()
    file_id = upload_result['id']
    print(f"✅ File uploaded with ID: {file_id}")
    print(f"File path: {upload_result['file_path']}")
    
    # Step 3: Process the uploaded file
    print("Step 3: Processing uploaded file...")
    process_response = requests.post(f"{base_url}/api/data/process/{file_id}")
    print(f"Process response status: {process_response.status_code}")
    
    if process_response.status_code != 200:
        print(f"Processing failed: {process_response.text}")
        return
    
    process_result = process_response.json()
    print(f"✅ File processed successfully")
    print(f"Processing results: {json.dumps(process_result, indent=2)}")
    
    # Step 4: Update trial with results
    print("Step 4: Updating trial with results...")
    if 'processing_results' in process_result and 'data_info' in process_result['processing_results']:
        update_data = {
            "processed_features": process_result['processing_results']['data_info'],
            "steps": process_result['processing_results'].get('step_detection', {}).get('step_times', []),
            "step_variance": process_result['processing_results'].get('gait_metrics', {}).get('step_variance', 0),
            "y_change": process_result['processing_results'].get('gait_metrics', {}).get('y_change', 0),
            "y_total": process_result['processing_results'].get('gait_metrics', {}).get('y_total', 0),
            "total_combined_loss": process_result['processing_results'].get('gait_metrics', {}).get('step_variance', 0)
        }
        
        update_response = requests.put(f"{base_url}/api/experiments/trials/{trial_id}", json=update_data)
        print(f"Update response status: {update_response.status_code}")
        
        if update_response.status_code == 200:
            updated_trial = update_response.json()
            print(f"✅ Trial updated successfully")
            print(f"Updated trial: {json.dumps(updated_trial, indent=2)}")
        else:
            print(f"Trial update failed: {update_response.text}")
    
    # Verify the file was saved in the expected location
    file_path = upload_result['file_path']
    if file_path:
        print(f"Verifying file exists at: {file_path}")
        if os.path.exists(file_path):
            print("✅ File successfully saved to expected location!")
            # Show the directory structure
            print("Directory structure:")
            import subprocess
            result = subprocess.run(['ls', '-la', os.path.dirname(file_path)], capture_output=True, text=True)
            print(result.stdout)
        else:
            print("❌ File not found at expected location")
    
    # Test the directory structure
    print("\n=== Testing Directory Structure ===")
    base_dir = "/Users/riccardoconci/Local_documents/Personalising-Crutches/V2/data/raw"
    
    if os.path.exists(base_dir):
        print("Raw data directory structure:")
        import subprocess
        result = subprocess.run(['find', base_dir, '-type', 'd'], capture_output=True, text=True)
        print(result.stdout)
    else:
        print("Raw data directory not found")

if __name__ == "__main__":
    test_real_data_upload()
