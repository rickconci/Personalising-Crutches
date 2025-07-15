import numpy as np

objective_preference = ['effort', 'stability', 'pain'] #pick one of the three for single objective optimization, or all three for multi-objective optimization

# --- File Paths and Naming ---
DATA_DIRECTORY = "data"
MASTER_LOG_FILE = 'master_experiment_log.csv'
RAW_DATA_SUFFIX = "_raw.csv"
STEP_FILE_SUFFIX = "_steps.csv"

# Define the columns for the master log file to ensure consistency.
# This includes user info, crutch params, and all possible loss metrics.
MASTER_LOG_COLUMNS = [
    'objective', 'user_id', 'height', 'weight', 'forearm_length', 'fitness_level',
    'alpha', 'beta', 'gamma', 'delta',
    'effort_survey_answer', 'pain_survey_answer', 'stability_survey_answer',
    'metabolic_cost', 'Y_change', 'Y_total', 'step_variance', 'RMS_load_cell_force',
    'Total_Combined_Loss'
]

# Characteristics to prompt for from the user at the start of an experiment
USER_CHARACTERISTICS = ['height', 'weight', 'forearm_length', 'fitness_level']

initial_crutch_geometry = {'alpha': 90, 'beta': 110, 'gamma': 0, 'delta': 0}

crutch_params_boundaries = [
    {'name': 'alpha', 'type': 'discrete', 'domain': np.arange(70, 125, 5).tolist()}, # alpha
    {'name': 'beta',  'type': 'discrete', 'domain': np.arange(90, 145, 5).tolist()}, # beta
    {'name': 'gamma', 'type': 'discrete', 'domain': np.arange(-12, 13, 3).tolist()},    # gamma
    {'name': 'delta', 'type': 'discrete', 'domain': np.arange(0, 21, 2).tolist()}     # delta
]


kernel_params = {
    'lengthscale': 3,
    'variance': 1,
    'noise': 1.05
}


Luke_height = 119

objective_to_quantitative_measurements_mapping = {
    'effort': ['metabolic_cost'],
    'stability': ['Y_change','Y_total', 'step_variance'],
    'pain': []
}



objective_to_survey_measureemnts_mapping = {
    'effort': ['effort_survey_answer'],
    'stability': ['stability_survey_answer'],
    'pain': ['pain_survey_answer']
}


Metric_weighting_values_dict = {
    'metabolic_cost': 20,
    'Y_change':2,
    'Y_total': 100,
    'step_variance': 100,
    'RMS_load_cell_force': 3000,
    'effort_survey_answer':1,
    'stability_survey_answer': 1,
    'pain_survey_answer':1, 
    }


