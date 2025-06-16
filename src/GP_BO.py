import GPy
import numpy as np
import pandas as pd
from GPyOpt.methods import BayesianOptimization

from config import crutch_params_boundaries, kernel_params
import config

class BayesOpt():
    def __init__(self, acquisition_type='EI', exact_feval=True):
        self.acquisition_type = acquisition_type
        self.exact_feval = exact_feval
        self.user_characteristics = None

    def _get_kernel(self):
        """Creates and returns a GPy kernel."""
        # Get the number of user characteristics from config
        num_user_chars = len(config.USER_CHARACTERISTICS)
        # Total input dimension is user characteristics + crutch parameters
        total_dim = num_user_chars + 3  # 3 crutch parameters (alpha, beta, gamma)
        
        return GPy.kern.Matern52(
            input_dim=total_dim,
            variance=kernel_params['variance'],
            lengthscale=kernel_params['lengthscale']
        )

    def get_next_parameters(self, experiment_data: pd.DataFrame):
        """
        Takes the existing experimental data and suggests the next parameters to test.
        Only optimizes over crutch parameters while considering user characteristics.
        """
        # Store user characteristics from the first row (they should be constant)
        if self.user_characteristics is None:
            self.user_characteristics = experiment_data[config.USER_CHARACTERISTICS].iloc[0].astype(float)
        
        # Prepare X (inputs) including both user characteristics and crutch parameters
        X = experiment_data[config.USER_CHARACTERISTICS + ['alpha', 'beta', 'gamma']].astype(float).to_numpy()
        Y = experiment_data[['Total_Combined_Loss']].astype(float).to_numpy()

        # Define a dummy objective function for GPyOpt
        def dummy_objective(x):
            return 0

        # Create domain that includes both user characteristics and crutch parameters
        # but only allow optimization over crutch parameters
        domain = []
        
        # Add user characteristics as fixed parameters
        for char in config.USER_CHARACTERISTICS:
            domain.append({
                'name': char,
                'type': 'continuous',
                'domain': (float(self.user_characteristics[char]), float(self.user_characteristics[char]))  # Fixed value
            })
        
        # Add crutch parameters as optimizable parameters
        for param in crutch_params_boundaries:
            if param['name'] in ['alpha', 'beta', 'gamma']:
                # Ensure the domain values are floats
                param_copy = param.copy()
                param_copy['domain'] = (float(param['domain'][0]), float(param['domain'][1]))
                domain.append(param_copy)

        optimizer = BayesianOptimization(
            f=dummy_objective,
            domain=domain,
            model_type='GP',
            kernel=self._get_kernel(),
            acquisition_type=self.acquisition_type,
            exact_feval=self.exact_feval,
            X=X,
            Y=Y
        )

        next_params_array = optimizer.suggest_next_locations()
        
        # Extract only the crutch parameters from the result
        # The first len(config.USER_CHARACTERISTICS) values are user characteristics
        crutch_param_indices = [i for i, param in enumerate(domain) if param['name'] in ['alpha', 'beta', 'gamma']]
        next_params = {
            'alpha': float(next_params_array[0, crutch_param_indices[0]]),
            'beta': float(next_params_array[0, crutch_param_indices[1]]),
            'gamma': float(next_params_array[0, crutch_param_indices[2]])
        }
        
        print(f"Next suggested parameters: {next_params}")
        return next_params


   


