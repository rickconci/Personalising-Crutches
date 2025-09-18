import os
import pandas as pd
from . import config

class DataManager:
    """
    Manages the master log file for all experiments.
    """
    def __init__(self):
        # Create data directory if it doesn't exist
        os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
        self.master_log_path = os.path.join(config.DATA_DIRECTORY, 'master_experiment_log.csv')
        
        # Initialize or load the master log
        if os.path.exists(self.master_log_path):
            self.master_log = pd.read_csv(self.master_log_path)
        else:
            self.master_log = pd.DataFrame()

    def save_trial_data(self, trial_data):
        """
        Saves a single trial's data to the master log.
        """
        # Convert trial_data to DataFrame
        trial_df = pd.DataFrame([trial_data])
        
        # Append to master log
        self.master_log = pd.concat([self.master_log, trial_df], ignore_index=True)
        
        # Save to CSV
        self.master_log.to_csv(self.master_log_path, index=False)
        print(f"Successfully saved trial data to {self.master_log_path}")

    def load_user_data(self, user_id: str) -> pd.DataFrame | None:
        """
        Loads all previous trial data for a specific user.

        Args:
            user_id: The ID of the user to load data for.

        Returns:
            A DataFrame containing the user's data, or None if no data exists.
        """
        if self.master_log.empty or 'user_id' not in self.master_log.columns:
            return None
        
        user_data = self.master_log[self.master_log['user_id'] == user_id].copy()
        
        if user_data.empty:
            return None
            
        return user_data 