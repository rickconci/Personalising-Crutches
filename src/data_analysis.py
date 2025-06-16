import os
import pandas as pd
import numpy as np
from gait_analysis import detect_initial_peaks, calculate_gait_metrics_from_steps
import config
from ble_MCU_2 import record_trial_data
import asyncio

class DataProcessor:
    """
    Orchestrates the data handling pipeline for each trial, including
    recording, initial analysis, and final processing.
    """
    def __init__(self, user_id, trial_num):
        self.user_id = user_id
        self.trial_num = trial_num
        
        # Create a specific directory for the current user
        self.user_data_path = os.path.join(config.DATA_DIRECTORY, self.user_id)
        os.makedirs(self.user_data_path, exist_ok=True)
        
        # Define file paths for the current trial
        base_filename = f"{self.user_id}_trial_{self.trial_num}"
        self.raw_data_path = os.path.join(self.user_data_path, base_filename + config.RAW_DATA_SUFFIX)
        self.step_file_path = os.path.join(self.user_data_path, base_filename + config.STEP_FILE_SUFFIX)

    def record_and_detect_peaks(self):
        """
        Manages data recording by calling the BLE controller and then performs
        an initial, automated peak detection. Returns True on success, False on failure.
        """
        # 1. Record raw data by calling the integrated controller
        print("\n--- Preparing for Data Recording ---")
        try:
            # Run the asynchronous recording function
            asyncio.run(record_trial_data(self.raw_data_path))
        except Exception as e:
            print(f"\nAn error occurred during data recording: {e}")
            return False

        # 2. Perform initial peak detection on the new file
        try:
            # Check if the file was created and is not empty
            if not os.path.exists(self.raw_data_path) or os.path.getsize(self.raw_data_path) == 0:
                print("\nError: Raw data file was not created or is empty.")
                print("The trial cannot proceed without data.")
                return False

            df = pd.read_csv(self.raw_data_path)
            
            # Rename columns from HTML tool format to Python format
            df.rename(columns={
                "acc_x_time": "timestamp",
                "acc_x_data": "accX",
                "acc_y_data": "roll",
                "acc_z_data": "force"
            }, inplace=True)

            if df.empty:
                print("Warning: Raw data file is empty. Cannot perform peak detection.")
                return True
                
            # Convert timestamp to seconds for analysis
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
            
            initial_peaks = detect_initial_peaks(df)
            
            # Save the initial peaks to the step file
            pd.DataFrame(initial_peaks, columns=['step_index']).to_csv(self.step_file_path, index=False)
            print(f"Initial peaks saved to {self.step_file_path}")
            
        except FileNotFoundError:
            print(f"Error: Raw data file not found at {self.raw_data_path}. Cannot perform peak detection.")
            return False
        except Exception as e:
            print(f"An error occurred during peak detection: {e}")
            return False
            
        return True # Signal that this stage was successful

    def featurize_trial_data(self):
        """
        Performs the final feature extraction using the raw data and the
        (potentially manually corrected) step file.
        """
        try:
            raw_df = pd.read_csv(self.raw_data_path)
            # The raw data file has headers: 'acc_x_time', 'force', 'roll', 'acc_x_data'
            # We need to convert time to seconds and rename columns for the analysis functions.
            raw_df['timestamp'] = (raw_df['acc_x_time'] - raw_df['acc_x_time'].iloc[0]) / 1000.0
            raw_df.rename(columns={"acc_x_data": "accX"}, inplace=True)
            # 'force' and 'roll' are kept as-is, assuming downstream functions use these names.

            # The HTML tool saves steps with a header. We skip it and name the columns.
            steps_df = pd.read_csv(self.step_file_path, skiprows=1, header=None, names=['step_time', 'value'])
            
            # Ensure the step_time column is numeric, handling potential read errors
            steps_df['step_time'] = pd.to_numeric(steps_df['step_time'], errors='coerce')
            steps_df.dropna(subset=['step_time'], inplace=True) # Remove any rows that failed conversion

            # To robustly find the step indices, we find the closest timestamp
            # in the raw data for each step time you identified.
            raw_df_sorted = raw_df.sort_values('timestamp').reset_index()
            steps_df_sorted = steps_df.sort_values('step_time')

            # Convert both timestamp columns to float64 to ensure compatibility
            raw_df_sorted['timestamp'] = raw_df_sorted['timestamp'].astype('float64')
            steps_df_sorted['step_time'] = steps_df_sorted['step_time'].astype('float64')

            # Find the row in raw_df that corresponds to each step via the timestamp
            merged = pd.merge_asof(
                left=steps_df_sorted,
                right=raw_df_sorted,
                left_on='step_time',
                right_on='timestamp',
                direction='nearest'
            )
            
            # Get the original row indices to pass to the gait calculation function
            step_indices = merged['index'].to_numpy()
            gait_metrics = calculate_gait_metrics_from_steps(raw_df, step_indices)

        except FileNotFoundError:
            print(f"Error: Could not find data files. Ensure both '{self.raw_data_path}' and '{self.step_file_path}' exist.")
            gait_metrics = {'step_variance': 0, 'Y_change': 0, 'Y_total': 0}
        except Exception as e:
            print(f"An error occurred during data featurization: {e}")
            print("Raw data shape:", raw_df.shape if 'raw_df' in locals() else "Not loaded")
            print("Steps data shape:", steps_df.shape if 'steps_df' in locals() else "Not loaded")
            gait_metrics = {'step_variance': 0, 'Y_change': 0, 'Y_total': 0}

        # Placeholder for other data sources (surveys, metabolic, etc.)
        processed_data = {
            'metabolic_cost': 3.5,
            'effort_survey_answer': 3,
            'stability_survey_answer': 4,
            'pain_survey_answer': 1,
            'RMS_load_cell_force': 120,
            **gait_metrics
        }
        return processed_data

if __name__ == "__main__":
    processor = DataProcessor("test_user_01", 1)
    
    # Manually get the data path
    trial_data_file = processor.get_trial_data_path()
    
    # Process the data
    metrics = processor.process_trial_data(trial_data_file)
    print(f"Processed metrics: {metrics}")