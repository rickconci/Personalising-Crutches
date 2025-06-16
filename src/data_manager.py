import os
import pandas as pd
import config

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

if __name__ == '__main__':
    # Example usage:
    dm = DataManager()
    
    # Example data for one trial
    example_data = {
        'objective': 'stability',
        'user_id': 'test_user_02',
        'height': 180,
        'weight': 75,
        'forearm_length': 45,
        'fitness_level': 4,
        'alpha': 95,
        'beta': 115,
        'gamma': 5,
        'delta': 2,
        'Y_change': 1.5,
        'step_variance': 0.02,
        'Total_Combined_Loss': 15.7
    }
    
    dm.save_trial_data(example_data)
    print("Example trial data saved.") 