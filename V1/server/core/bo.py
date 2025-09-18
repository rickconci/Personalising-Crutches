import GPy
import numpy as np
import pandas as pd
import itertools
from GPyOpt.methods import BayesianOptimization

from .config import crutch_params_boundaries, kernel_params, CRUTCH_PARAM_STEPS
from . import config

def _get_discrete_domain(param_name: str) -> list[int]:
    """
    Generates a discrete domain (list of allowed values) for a parameter.
    
    Reads the parameter's boundaries and step from the config file.
    """
    bounds = next((p['domain'] for p in crutch_params_boundaries if p['name'] == param_name), None)
    step = CRUTCH_PARAM_STEPS.get(param_name)
    if bounds and step:
        # The 'stop' argument in range() is exclusive, so we add the step 
        # to the upper bound to ensure it's included in the domain.
        return list(range(int(bounds[0]), int(bounds[1]) + int(step), int(step)))
    return []

def _round_to_nearest(value: float, domain: list[int]) -> int:
    """Rounds a value to the nearest allowed value in a discrete domain."""
    if not domain:
        return int(value)
    return min(domain, key=lambda x: abs(x - value))

class BayesOpt():
    def __init__(self, acquisition_type='EI', exact_feval=True):
        self.acquisition_type = acquisition_type
        self.exact_feval = exact_feval
        self.user_characteristics = None
        self._tested_geometries = set()

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
    
    def _suggest_alternative(self, alpha: int, beta: int, gamma: int) -> tuple[int, int, int]:
        """
        Suggests an alternative geometry if the original suggestion was a duplicate.

        Tries small variations first, then falls back to a random untested geometry.
        """
        alpha_range = _get_discrete_domain('alpha')
        beta_range = _get_discrete_domain('beta')
        gamma_range = _get_discrete_domain('gamma')
        
        # Try small variations around the suggested point
        variations = [
            (alpha + CRUTCH_PARAM_STEPS['alpha'], beta, gamma),
            (alpha - CRUTCH_PARAM_STEPS['alpha'], beta, gamma),
            (alpha, beta + CRUTCH_PARAM_STEPS['beta'], gamma),
            (alpha, beta - CRUTCH_PARAM_STEPS['beta'], gamma),
            (alpha, beta, gamma + CRUTCH_PARAM_STEPS['gamma']),
            (alpha, beta, gamma - CRUTCH_PARAM_STEPS['gamma']),
        ]
        
        for a, b, g in variations:
            if (a in alpha_range and b in beta_range and g in gamma_range and 
                (a, b, g) not in self._tested_geometries):
                print(f"→ Proposing a close alternative: α={a}, β={b}, γ={g}")
                return a, b, g
        
        # If no close alternative is found, suggest a random untested geometry
        all_geometries = set(itertools.product(alpha_range, beta_range, gamma_range))
        available_geometries = list(all_geometries - self._tested_geometries)
        
        if available_geometries:
            choice_idx = np.random.choice(len(available_geometries))
            a, b, g = available_geometries[choice_idx]
            print(f"→ Proposing a random alternative: α={a}, β={b}, γ={g}")
            return a, b, g
            
        # This should rarely happen, only if all possible geometries have been tested.
        print("⚠️ All possible geometries have been tested! Returning original suggestion.")
        return alpha, beta, gamma

    def get_next_parameters(self, experiment_data: pd.DataFrame):
        """
        Takes the existing experimental data and suggests the next parameters to test.
        Only optimizes over crutch parameters while considering user characteristics.
        """
        # Store user characteristics from the first row (they should be constant)
        if self.user_characteristics is None:
            self.user_characteristics = experiment_data[config.USER_CHARACTERISTICS].iloc[0].astype(float)
        
        # Populate the set of tested geometries from the historical data
        self._tested_geometries = set()
        if not experiment_data.empty:
            for _, row in experiment_data[['alpha', 'beta', 'gamma']].iterrows():
                self._tested_geometries.add(tuple(row.astype(int)))
                
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
                param_copy = param.copy()
                param_copy['type'] = 'discrete'
                param_copy['domain'] = _get_discrete_domain(param['name'])
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
        
        # Extract the suggested crutch parameters from the full array
        crutch_param_names = ['alpha', 'beta', 'gamma']
        crutch_param_indices = [i for i, p in enumerate(domain) if p['name'] in crutch_param_names]
        
        a, b, g = next_params_array[0, crutch_param_indices]

        # The optimizer should return discrete values, but we ensure they are rounded
        # to handle any floating point inaccuracies.
        a = _round_to_nearest(a, _get_discrete_domain('alpha'))
        b = _round_to_nearest(b, _get_discrete_domain('beta'))
        g = _round_to_nearest(g, _get_discrete_domain('gamma'))
        
        # Check if this geometry has already been tested and find an alternative if so.
        if (a, b, g) in self._tested_geometries:
            print(f"⚠️  Geometry α={a}, β={b}, γ={g} has already been tested. Finding alternative...")
            a, b, g = self._suggest_alternative(a, b, g)

        next_params = {'alpha': a, 'beta': b, 'gamma': g}
        
        print(f"Next suggested parameters: {next_params}")
        return next_params


   


