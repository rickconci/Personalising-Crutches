
import GPy
import numpy as np
from GPyOpt import BayesianOptimization

from config import crutch_params_boundaries, Metric_weighting_values_dict, kernel_params


class Luke:
    def __init__(self, acquisition_type = 'EI', exact_feval = True, max_iter = 1):
        self.acquisition_type = acquisition_type
        self.exact_feval = exact_feval
        self.max_iter = max_iter
        self.height = 119

    def objective_function(self, X):
        return np.array([[0]])

    def get_kernel(self):
        ker = GPy.kern.src.stationary.Matern52(len(crutch_params_boundaries), variance=kernel_params['variance'], lengthscale=kernel_params['lengthscale'])
        return ker

    def run_BO(self, Experiment_metrics_df):
        Y = np.array(list(Experiment_metrics_df['Total_Combined_Loss'])).reshape(-1, 1)
        X = Experiment_metrics_df[['alpha', 'beta', 'gamma']].to_numpy()

        myBopt = BayesianOptimization(f=self.objective_function,               # Objective function
                              domain = crutch_params_boundaries,                    # Domain
                              model_type='GP',                    # Model type, using Gaussian Process
                              acquisition_type='EI',              # Acquisition type, using Expected Improvement
                              exact_feval=True,                   # Exact evaluations of the objective
                              initial_design_numdata=X.shape[0],  # Number of initial points
                              kernel = self.get_kernel())

        myBopt.X = X
        myBopt.Y = Y
        myBopt.run_optimization(self.max_iter)


    def run_evrythign(self):
        self.run_BO()




Luke = Luke(acquisition_type = 'EI', exact_feval = True, max_iter = 1)


Luke.run_evrythign()

Luke.height

