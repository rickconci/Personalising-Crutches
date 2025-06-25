import os
import pickle
import itertools
import webbrowser

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from data_analysis import DataProcessor
from GP_BO import BayesOpt
import config
from data_manager import DataManager
import ast

class Experiment:
    """
    Manages a single user experiment, guiding through the process
    of data collection, analysis, and optimization.
    """
    def __init__(self, user_id, user_characteristics, objective, initial_crutch_geometry, data_manager, manual_correction=False):
        self.user_id = user_id
        self.user_characteristics = user_characteristics
        self.objective = objective
        self.current_crutch_geometry = initial_crutch_geometry.copy()
        self.data_manager = data_manager
        self.bayesian_optimizer = BayesOpt()
        self.experiment_data = pd.DataFrame()
        self.manual_correction = manual_correction
        
        # Create plots directory for this user
        self.plots_dir = os.path.join(config.DATA_DIRECTORY, self.user_id, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def _calculate_loss(self, processed_data):
        """
        Calculates the total loss based on the selected objective and its associated metrics.
        """
        loss = 0
        objective_weights = {self.objective: 1.0}

        for metric in config.objective_to_quantitative_measurements_mapping.get(self.objective, []):
            if metric in processed_data:
                loss += objective_weights[self.objective] * \
                        processed_data[metric] * \
                        config.Metric_weighting_values_dict.get(metric, 1)
        
        for metric in config.objective_to_survey_measureemnts_mapping.get(self.objective, []):
            if metric in processed_data:
                loss += objective_weights[self.objective] * \
                        processed_data[metric] * \
                        config.Metric_weighting_values_dict.get(metric, 1)
        return loss

    def _plot_optimization_progress(self, trial_num):
        """
        Creates and saves a 3D plot of the optimization progress.
        """
        # Create custom colormap
        colors = ["#6AA84F", "#FFD966", "#CC0000"]
        cmap_name = 'green_red'
        n_bins = 100
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # Extract data
        alpha = self.experiment_data['alpha']
        beta = self.experiment_data['beta']
        gamma = self.experiment_data['gamma']
        values = self.experiment_data['Total_Combined_Loss']

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        img = ax.scatter(alpha, beta, gamma, c=values, cmap=custom_cmap, alpha=1.0, s=40)

        # Connect points with line
        ax.plot(alpha, beta, gamma, color='gray', marker=None, linestyle="-", alpha=.6)

        # Add start and end labels
        ax.text(alpha.iloc[0], beta.iloc[0], gamma.iloc[0], "Start", color='black')
        ax.text(alpha.iloc[-1], beta.iloc[-1], gamma.iloc[-1], "End", color='black')

        # Customize plot
        ax.set_xlabel('Alpha Axis')
        ax.set_ylabel('Beta Axis')
        ax.set_zlabel('Gamma Axis')
        ax.set_title(f'Optimization Progress - Trial {trial_num}\nObjective: {self.objective}')
        
        # Add colorbar
        fig.colorbar(img, ax=ax, label='Loss Value')

        # Save plot
        plot_path = os.path.join(self.plots_dir, f'optimization_progress_trial_{trial_num}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nOptimization progress plot saved to: {plot_path}")

    def run_trial(self, trial_num):
        """
        Runs a single, complete trial.
        """
        print(f"\n--- Running Trial {trial_num} for User '{self.user_id}' ---")
        print(f"Current Geometry: {self.current_crutch_geometry}")
        
        # Instantiate the processor and enable visualization by default.
        data_processor = DataProcessor(self.user_id, trial_num, visualize_steps=True)

        # 1. Record data and perform initial peak detection
        # If this step fails, abort the trial.
        success = data_processor.record_and_detect_peaks()
        if not success:
            print(f"\n--- Trial {trial_num} Aborted ---")
            print("Please resolve the issue with the recording script before trying again.")
            return False # Signal failure to the main run loop

        # 2. (Optional) Pause for manual step file correction using the HTML tool
        if self.manual_correction:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            html_file_path = os.path.join(script_dir, 'Accelerometer_Processing_Program.html')
            
            if os.path.exists(html_file_path):
                webbrowser.open(f'file://{os.path.abspath(html_file_path)}')
                print("\n--- Manual Step Correction (HTML Tool) ---")
                print("The HTML analysis tool has been opened in your browser.")
                print("\nInstructions:")
                print(f"1. In the browser, click the FIRST 'Choose File' button (for Polar Sensor data).")
                print(f"2. Navigate to and select the raw data file for this trial:")
                print(f"   {data_processor.raw_data_path}")
                print(f"3. (Optional) Load existing steps by clicking the SECOND 'Choose File' (next to Save) and selecting:")
                print(f"   {data_processor.step_file_path}")
                print(f"4. Manually correct the steps using the visual interface.")
                print(f"5. When finished, click the 'Save' button. The corrected step file will likely go to your Downloads folder.")
                print(f"6. You MUST manually move this downloaded '.step' file from Downloads into the following directory, overwriting the original file:")
                print(f"   {data_processor.user_data_path}")
                input("\nOnce you have MOVED THE FILE and are ready, press Enter here to continue...")
            else:
                # Fallback to the original method if the HTML file isn't found
                input(f"\nPlease open and manually correct the step file: {data_processor.step_file_path}\n"
                      "Once you have saved your changes, press Enter to continue...")

        # 3. Featurize data using the corrected steps
        processed_data = data_processor.featurize_trial_data()

        # 4. Collect survey responses
        print("\n--- Survey Responses ---")
        print("Please rate your experience on a scale of 0-6 (0 being best, 6 being worst)")
        
        while True:
            try:
                effort = float(input("How much effort did you feel? (0-6): "))
                if 0 <= effort <= 6:
                    break
                print("Please enter a value between 0 and 6")
            except ValueError:
                print("Please enter a numeric value")
        
        while True:
            try:
                pain = float(input("How much pain did you experience? (0-6): "))
                if 0 <= pain <= 6:
                    break
                print("Please enter a value between 0 and 6")
            except ValueError:
                print("Please enter a numeric value")
        
        while True:
            try:
                stability = float(input("How unstable did you feel? (0-6): "))
                if 0 <= stability <= 6:
                    break
                print("Please enter a value between 0 and 6")
            except ValueError:
                print("Please enter a numeric value")

        # Add survey responses to processed data
        processed_data.update({
            'effort_survey_answer': effort,
            'pain_survey_answer': pain,
            'stability_survey_answer': stability
        })

        # 5. Calculate loss
        loss = self._calculate_loss(processed_data)
        print(f"Loss for trial {trial_num}: {loss:.4f}")

        # 6. Save all data to master log
        trial_results = {
            'objective': self.objective,
            'user_id': self.user_id,
            **self.user_characteristics,
            **self.current_crutch_geometry,
            **processed_data,
            'Total_Combined_Loss': loss
        }
        self.data_manager.save_trial_data(trial_results)
        
        # 7. Append data for the BO
        # Include user characteristics in the experiment data
        trial_data_for_bo = {
            **self.user_characteristics,
            **self.current_crutch_geometry,
            'Total_Combined_Loss': loss
        }
        self.experiment_data = pd.concat([
            self.experiment_data,
            pd.DataFrame([trial_data_for_bo])
        ], ignore_index=True)
        
        # 8. Plot and save optimization progress
        self._plot_optimization_progress(trial_num)
        
        return True # Signal success

    def run(self, num_iterations=10):
        """
        Runs the full experimental loop.
        """
        for i in range(num_iterations):
            success = self.run_trial(trial_num=i + 1)
            
            # If a trial was aborted, don't proceed to the next one automatically.
            if not success:
                print("\nExperiment paused due to an error in the last trial.")
                # We could add a prompt here to ask the user if they want to retry or exit.
                # For now, we will exit the experiment loop.
                break

            if i < num_iterations - 1:
                # Get next geometry suggestion
                next_params = self.bayesian_optimizer.get_next_parameters(self.experiment_data)
                
                print("\n--- Crutch Adjustment ---")
                print("Bayesian Optimizer suggests the following parameters:")
                for param, value in next_params.items():
                    print(f"  - {param}: {value:.2f}")

                # Allow user to accept or edit
                user_input = input("Press Enter to accept, or type a dictionary to edit (e.g., {'alpha': 95.0, 'gamma': 5.0}): ")
                if user_input:
                    try:
                        edited_params = ast.literal_eval(user_input)
                        next_params.update(edited_params)
                        print("Using edited parameters.")
                    except (ValueError, SyntaxError):
                        print("Invalid input. Using the original suggested parameters.")

                self.current_crutch_geometry.update(next_params)
        
        print("\n----- Experiment Finished -----")
        if not self.experiment_data.empty:
            best_trial = self.experiment_data.loc[self.experiment_data['Total_Combined_Loss'].idxmin()]
            print(f"Best trial found in this experiment:\n{best_trial}")
        else:
            print("No trials were successfully completed. Cannot determine a best trial.")

def run_application():
    """ Main application loop to manage experiments. """
    data_manager = DataManager()
    
    while True:
        print("\n--- Welcome to the Crutch Personalization Framework ---")
        
        print("Available objectives:", config.objective_preference)
        objective = ""
        while objective not in config.objective_preference:
            objective = input("1. Select the primary objective for this experiment: ")

        user_id = input("2. Enter the user ID: ")
        
        user_characteristics = {}
        for char in config.USER_CHARACTERISTICS:
            while True:
                try:
                    value = float(input(f"   - Enter user's {char}: "))
                    user_characteristics[char] = value
                    break
                except ValueError:
                    print(f"Please enter a numeric value for {char}")
        
        enable_manual_check = input("3. Enable manual step correction for this experiment? (yes/no): ").lower() == 'yes'

        experiment = Experiment(
            user_id=user_id,
            user_characteristics=user_characteristics,
            objective=objective,
            initial_crutch_geometry=config.initial_crutch_geometry,
            data_manager=data_manager,
            manual_correction=enable_manual_check
        )
        
        num_iterations = int(input("4. Enter the number of trials for this experiment: "))
        experiment.run(num_iterations=num_iterations)

        if input("\nRun another experiment? (yes/no): ").lower() != 'yes':
            break
            
    print("Exiting application.")

if __name__ == "__main__":
    run_application()