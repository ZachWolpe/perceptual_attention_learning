"""
---------------------------------------------------------------------------------
cognitive_science_learning_model_base.py

Cog Sci Learning model base class.

Source:
-------
    : https://ccn.studentorg.berkeley.edu/pdfs/papers/WilsonCollins_modelFitting.pdf

Modifaction Logs:
: 03 July 24     : zachcolinwolpe@gmail.com      : init
---------------------------------------------------------------------------------
"""

import plotly.graph_objects as go
from scipy.optimize import minimize  # finding optimal params in models
import numpy as np                   # matrix/array functions
from abc import ABC, abstractmethod

# deprecated packages
# from src.rescorla_wagner_model_diagnostics import (RoscorlaWagerModelDiagnostics)
# from src.rescorla_wagner_model_simulation import (RescorlaWagnerSimulate)
# from src.rescorla_wagner_model_plots import (RescorlaWagnerPlots)
# from src.rescorla_wagner_model import (RoscorlaWagner)
# from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt
# import ipywidgets as widgets
# from scipy import stats
# from tqdm import tqdm
# import pandas as pd


# " +-- Helpers ----------------------------------------------------------------->> "
def add_diag_line(fig, xmin=0, xmax=1, marker_color='black', row=1, col=1):
    _x = np.linspace(xmin, xmax, 100)
    _y = _x
    fig.add_trace(go.Scatter(x=_x, y=_y, mode='lines', marker_color=marker_color, name='y=x', showlegend=False),row=row, col=col)


# " +-- Base Class -------------------------------------------------------------->> "
class MultiArmedBanditModels(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def simulate(self, **kwargs):
        self.simulated_params = kwargs

    @abstractmethod
    def neg_log_likelihood(self):
        pass
    
    @abstractmethod
    def perform_sensitivity_analysis(self):
        pass

    @abstractmethod
    def optimize_brute_force(self, loss_function, loss_kwargs, parameter_ranges):

        neg_log_likelihoods = [loss_function(**params) for params in loss_kwargs]
        min_loss_func_idx = np.argmin(neg_log_likelihoods)
        min_loss_value = min(neg_log_likelihoods)
        return min_loss_func_idx, loss_kwargs[min_loss_func_idx], min_loss_value
        # best_epsilon = epsilon_values[np.argmin(neg_log_likelihoods)]
        # return best_epsilon

    def optimize_scikit(self, loss_function, init_guess, args, bounds):
        result = minimize(
                        loss_function,
                        init_guess,
                        args=args,
                        bounds=bounds
            )
        res_nll = result.fun
        param_fits = result.x
        return result, res_nll, param_fits

    @abstractmethod
    def plot_neg_log_likelihood(self):
        pass

    def compare_fitting_procedures(self):
        pass

    def compute_BIC(self, negLL, T, k_params=2):
        return 2 * negLL + k_params * np.log(T)
        # return -2 * LL + k_params * np.log(T)
        # bic = k * np.log(N) + 2 * neg_log_likelihood
