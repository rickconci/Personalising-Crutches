import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, correlate
from gait_analysis import calculate_gait_metrics_from_steps
import config
from ble_MCU_2 import record_trial_data
import asyncio
import plotly.graph_objs as go
import shutil

def _postprocess_steps(preds: np.ndarray, tolerance_ratio: float = 0.2, isolation_threshold: float = 5.0) -> np.ndarray:
    """
    Applies post-processing filters to a series of detected step times.
    1. Regularizes step intervals based on a data-driven expected interval.
    2. Removes isolated steps that are too far from any other step.
    """
    if len(preds) < 2:
        return preds

    # 1. Determine the data-driven expected interval from the median of diffs
    intervals = np.diff(preds)
    expected_interval = np.median(intervals) if len(intervals) >= 3 else 1.1

    # --- Interval-based regularization ---
    min_conflict_interval = expected_interval * tolerance_ratio
    
    regularized_preds = [preds[0]]
    for i in range(1, len(preds)):
        current_pred = preds[i]
        last_accepted_pred = regularized_preds[-1]
        
        if current_pred - last_accepted_pred < min_conflict_interval:
            prev_accepted_pred = regularized_preds[-2] if len(regularized_preds) > 1 else 0.0
            error_if_keep_last = abs((last_accepted_pred - prev_accepted_pred) - expected_interval)
            error_if_use_current = abs((current_pred - prev_accepted_pred) - expected_interval)
            if error_if_use_current < error_if_keep_last:
                regularized_preds[-1] = current_pred  # Swap
        else:
            regularized_preds.append(current_pred)
    
    regularized_preds = np.array(regularized_preds)

    # --- Isolation filter ---
    if len(regularized_preds) < 2:
        return np.array([])
        
    final_preds = []
    padded_preds = np.concatenate(([np.NINF], regularized_preds, [np.PINF]))
    
    for i in range(1, len(padded_preds) - 1):
        dist_to_prev = padded_preds[i] - padded_preds[i-1]
        dist_to_next = padded_preds[i+1] - padded_preds[i]
        if not ((dist_to_prev > isolation_threshold) and (dist_to_next > isolation_threshold)):
            final_preds.append(padded_preds[i])
            
    return np.array(final_preds)


def detect_steps_unsupervised(force_signal: np.ndarray, time_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Detects steps using an unsupervised matched-filter approach on the force gradient.
    """
    if len(force_signal) < int(fs): # Need at least 1s of data
        return np.array([])

    # 1. Calculate force gradient
    force_grad = np.gradient(force_signal, time_signal)

    # 2. Auto-generate a template from the most prominent peaks in the signal
    template_len_s = 0.6
    L = int(template_len_s * fs)
    
    # Find a few strong, initial peaks to serve as "pseudo ground truth"
    initial_peaks, _ = find_peaks(force_grad, prominence=np.std(force_grad) * 1.5, distance=fs*0.5)
    
    segments = []
    for i0 in initial_peaks[:5]: # Use up to the first 5 found peaks
        if i0 - L//2 >= 0 and i0 + L//2 < len(force_grad):
            seg = force_grad[i0-L//2:i0+L//2]
            segments.append(seg)
            
    if not segments:
        print("Warning: Could not auto-generate a step template. No initial prominent peaks found.")
        return np.array([])
        
    template = np.mean(segments, axis=0)
    template = (template - template.mean()) / (template.std() + 1e-6)

    # 3. Correlate the signal with the auto-generated template
    sig_n = (force_grad - force_grad.mean()) / (force_grad.std() + 1e-6)
    corr = correlate(sig_n, template, mode='same')
    
    # 4. Find peaks on the correlation trace
    prom = np.median(np.abs(corr)) * 5
    peaks, _ = find_peaks(corr, prominence=prom, distance=0.3 * fs)
    
    return time_signal[peaks]

class DataProcessor:
    """
    Orchestrates the data handling pipeline for each trial, including
    recording, initial analysis, and final processing.
    """
    def __init__(self, user_id, trial_num, visualize_steps=False):
        self.user_id = user_id
        self.trial_num = trial_num
        self.visualize_steps = visualize_steps
        
        # Create a specific directory for the current user
        self.user_data_path = os.path.join(config.DATA_DIRECTORY, self.user_id)
        os.makedirs(self.user_data_path, exist_ok=True)
        
        # Define file paths for the current trial
        base_filename = f"{self.user_id}_trial_{self.trial_num}"
        self.raw_data_path = os.path.join(self.user_data_path, base_filename + config.RAW_DATA_SUFFIX)
        self.step_file_path = os.path.join(self.user_data_path, base_filename + config.STEP_FILE_SUFFIX)
        self.visualization_path = os.path.join(self.user_data_path, base_filename + '_steps_visualization.html')

    def _create_and_save_plot(self, df, steps):
        """
        Generates and saves an interactive Plotly plot of the step detection results.
        """
        fig = go.Figure()

        # 1. Accelerometer Signal
        if 'accX' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['accX'],
                                     mode='lines', name='Accelerometer X', yaxis='y1'))

        # 2. Force Signal on secondary axis
        if 'force' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['force'],
                                     mode='lines', name='Force',
                                     line=dict(dash='dot'),
                                     yaxis='y2'))

        # 3. Detected Steps
        if 'accX' in df.columns and len(steps) > 0:
            step_y_values = np.interp(steps, df['timestamp'], df['accX'])
            fig.add_trace(go.Scatter(x=steps, y=step_y_values,
                                     mode='markers', marker=dict(color='red', symbol='cross', size=8),
                                     name='Detected Steps'))

        # Configure layout
        layout_update = {
            'title_text': f'Step Detection Visualization for {self.user_id} Trial {self.trial_num}',
            'xaxis_title': 'Time (s)',
            'yaxis_title': 'Accelerometer X',
            'legend_title_text': 'Signals'
        }
        if 'force' in df.columns:
            layout_update['yaxis2'] = {
                'title': 'Force',
                'overlaying': 'y',
                'side': 'right'
            }
        
        fig.update_layout(**layout_update)
        fig.write_html(self.visualization_path)
        print(f"Saved step detection visualization to {self.visualization_path}")

    def record_and_detect_peaks(self):
        """
        Manages data recording. If recording fails, offers a fallback to use existing data.
        Then performs automated peak detection.
        """
        # 1. Attempt to record raw data by calling the integrated controller
        print("\n--- Preparing for Data Recording ---")
        try:
            asyncio.run(record_trial_data(self.raw_data_path))
        except Exception as e:
            print(f"\nAn error occurred during data recording: {e}")
            # The file existence check below will handle the failure.

        # 2. Verify data was recorded. If not, prompt user for existing file.
        if not os.path.exists(self.raw_data_path) or os.path.getsize(self.raw_data_path) == 0:
            print("\nWarning: New data was not recorded (e.g., Bluetooth connection failed).")
            use_existing = input("Would you like to use an existing raw data file for this trial? (yes/no): ").lower()
            
            if use_existing == 'yes':
                existing_path = input("Please enter the full path to the existing raw data file: ")
                if os.path.exists(existing_path):
                    try:
                        shutil.copy(existing_path, self.raw_data_path)
                        print(f"Successfully copied existing data to '{self.raw_data_path}'.")
                    except Exception as e:
                        print(f"Error: Failed to copy the file. {e}")
                        return False
                else:
                    print("Error: The file path you entered does not exist.")
                    return False
            else:
                print("The trial cannot proceed without data.")
                return False
        
        # 3. Perform peak detection on the new or copied file
        try:
            df = pd.read_csv(self.raw_data_path)
            
            # More robustly handle column renaming for compatibility.
            # The goal is to have 'timestamp', 'accX', 'roll', and 'force'.
            
            # Standardize timestamp: prefer 'timestamp', fall back to 'acc_x_time'.
            if 'acc_x_time' in df.columns and 'timestamp' not in df.columns:
                df.rename(columns={"acc_x_time": "timestamp"}, inplace=True)

            # Standardize force: prefer 'force', fall back to 'acc_z_data'.
            if 'force' not in df.columns and 'acc_z_data' in df.columns:
                df.rename(columns={"acc_z_data": "force"}, inplace=True)
            
            # Standardize other columns, ignoring errors if they don't exist.
            df.rename(columns={"acc_x_data": "accX", "acc_y_data": "roll"}, inplace=True, errors='ignore')

            if df.empty or 'force' not in df.columns or 'timestamp' not in df.columns:
                print("Warning: Raw data file is empty or missing required columns ('force', 'timestamp'). Aborting.")
                return False
                
            # Convert timestamp to seconds for analysis
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
            
            # Calculate sampling frequency
            fs = 1.0 / np.median(np.diff(df['timestamp'].values))

            # Detect steps using the best algorithm
            raw_steps = detect_steps_unsupervised(df['force'].values, df['timestamp'].values, fs)
            
            # Apply post-processing filters
            final_steps = _postprocess_steps(raw_steps)
            
            # Conditionally visualize the results before saving
            if self.visualize_steps:
                self._create_and_save_plot(df, final_steps)

            # Save the final step times to the step file
            pd.DataFrame({'step_time': final_steps}).to_csv(self.step_file_path, index=False)
            print(f"Detected and saved {len(final_steps)} steps to {self.step_file_path}")
            
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

            # The step file now contains a single column 'step_time' with a header.
            steps_df = pd.read_csv(self.step_file_path)
            
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