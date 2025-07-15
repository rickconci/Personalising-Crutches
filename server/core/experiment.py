import os
import pickle
import itertools
import webbrowser
import ast

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config
from .data_manager import DataManager
from .bo import BayesOpt
from .data_analysis import TrialRecorder, TrialAnalyzer


class Experiment:
    """
    Manages a single user experiment, guiding through the process
    of data collection, analysis, and optimization.
    """
    def __init__(self, user_id, user_characteristics, objective, initial_crutch_geometry, data_manager, manual_correction=False, visualize_steps=True, previous_data=None):
        self.user_id = user_id
        self.user_characteristics = user_characteristics
        self.objective = objective
        self.current_crutch_geometry = initial_crutch_geometry.copy()
        self.data_manager = data_manager
        self.bayesian_optimizer = BayesOpt()
        self.experiment_data = previous_data if previous_data is not None else pd.DataFrame()
        self.manual_correction = manual_correction
        self.visualize_steps = visualize_steps
        
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

    def run_trial(self, trial_num: int, raw_data_path: str, subjective_metrics: dict) -> dict | None:
        """
        Runs a single trial, including analysis and featurization, using pre-recorded data.

        Args:
            trial_num: The current trial number.
            raw_data_path: The path to the raw data file for the trial.
            subjective_metrics: A dictionary of subjective responses from the user.

        Returns:
            A dictionary of processed data if successful, otherwise None.
        """
        print(f"--- Running Analysis for Trial {trial_num} ---")
        
        if not raw_data_path or not os.path.exists(raw_data_path):
            print(f"Trial {trial_num} aborted. Invalid raw_data_path: {raw_data_path}")
            return None

        # 1. Initial Analysis (Step Detection)
        # The data is already recorded and path is provided.
        analyzer = TrialAnalyzer(raw_data_path)
        step_file_path = analyzer.run_initial_analysis(visualize=self.visualize_steps)

        # Manual correction is handled by the frontend, so we skip it here.
        # The frontend will be responsible for providing a corrected step file if needed.
        
        # 2. Final Featurization
        processed_data = analyzer.featurize_trial(step_file_path)
        if not processed_data:
            print(f"Trial {trial_num} aborted due to featurization failure.")
            return None

        # 3. Add survey responses to processed data
        processed_data.update(subjective_metrics)
        
        return processed_data

    def run_and_process_trial(self, trial_num: int, raw_data_path: str, subjective_metrics: dict, crutch_geometry: dict) -> bool:
        """
        Runs a full trial processing loop: analysis, loss calculation, and data saving.

        Args:
            trial_num: The current trial number.
            raw_data_path: Path to the raw data file.
            subjective_metrics: Dictionary of user's subjective feedback.
            crutch_geometry: The crutch geometry used for this trial.

        Returns:
            True if the trial was processed successfully, False otherwise.
        """
        self.current_crutch_geometry.update(crutch_geometry)

        # 1. Run the analysis on the trial data
        trial_results_dict = self.run_trial(trial_num, raw_data_path, subjective_metrics)
        
        if not trial_results_dict:
            print("\nExperiment paused due to an error in the last trial.")
            return False

        # 2. Calculate loss
        loss = self._calculate_loss(trial_results_dict)
        print(f"Loss for trial {trial_num}: {loss:.4f}")

        # 3. Save all data to master log
        full_trial_data = {
            'objective': self.objective,
            'user_id': self.user_id,
            'is_penalty': False,
            **self.user_characteristics,
            **self.current_crutch_geometry,
            **trial_results_dict,
            'Total_Combined_Loss': loss
        }
        self.data_manager.save_trial_data(full_trial_data)
        
        # 4. Append data for the BO
        trial_data_for_bo = {
            **self.user_characteristics,
            **self.current_crutch_geometry,
            'Total_Combined_Loss': loss
        }
        self.experiment_data = pd.concat([
            self.experiment_data,
            pd.DataFrame([trial_data_for_bo])
        ], ignore_index=True)
    
        # 5. Plot and save optimization progress
        self._plot_optimization_progress(trial_num)

        return True

    def record_penalty_trial(self, penalty_loss: float, crutch_geometry: dict):
        """
        Records a trial with a penalty loss for a rejected geometry.
        """
        self.current_crutch_geometry.update(crutch_geometry)
        trial_results = {
            'objective': self.objective,
            'user_id': self.user_id,
            'is_penalty': True,
            **self.user_characteristics,
            **self.current_crutch_geometry,
            'Total_Combined_Loss': penalty_loss
        }
        self.data_manager.save_trial_data(trial_results)
        
        self.experiment_data = pd.concat([
            self.experiment_data,
            pd.DataFrame([trial_results])
        ], ignore_index=True)
        
        print(f"Penalty trial recorded with loss {penalty_loss}.")

    def get_next_suggestion(self):
        """
        Gets the next geometry suggestion from the Bayesian Optimizer.
        """
        if self.experiment_data.empty:
            print("This is the first trial. Using the initial default geometry.")
            return self.current_crutch_geometry.copy()
        else:
            print("Asking Bayesian Optimizer for the next suggestion...")
            return self.bayesian_optimizer.get_next_parameters(self.experiment_data)

    def get_best_trial(self):
        """
        Finds and returns the best performing trial from the experiment data.
        """
        if self.experiment_data.empty:
            return None
        
        valid_trials = self.experiment_data[self.experiment_data.get('is_penalty', False) == False]
        if not valid_trials.empty:
            best_trial = valid_trials.loc[valid_trials['Total_Combined_Loss'].idxmin()]
            return best_trial.to_dict()
        
        return None