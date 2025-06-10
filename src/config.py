import numpy as np

objective_preference = ['effort', 'stability', 'pain'] #pick one of the three for single objective optimization, or all three for multi-objective optimization



crutch_params_boundaries = [
    {'name': 'alpha', 'type': 'discrete', 'domain': np.arange(70, 125, 5).tolist()}, # alpha
    {'name': 'beta',  'type': 'discrete', 'domain': np.arange(90, 145, 5).tolist()}, # beta
    {'name': 'gamma', 'type': 'discrete', 'domain': np.arange(0, 33, 3).tolist()}    # gamma
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


