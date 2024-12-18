"""
-------------------------------------------------------------------------------
rescorla_wagner_model_diagostics.py

Rescorla Wagner Model model diagnostics.

Modifaction Logs:
: 20 June 24     : zachcolinwolpe@gmail.com      : init
-------------------------------------------------------------------------------
"""

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy


class RoscorlaWagerModelDiagnostics:

    @staticmethod
    def generate_Q_values(
            params,
            action_vector,
            reward_vector,
            Q_init=[0.5, 0.5],
            noisy_choice=False):
        """
        Generate Q-values for the Rescorla-Wagner model.

        Arguments:
        ----------
            : params (list: [alpha, theta]):
                alpha: learning rate
                theta: inverse temperature
            : action_vector (list, shape (T,)): array of known actions.
            : reward_vector (list, shape (T,)): array of known rewards.
            : Q_init (list, shape (2,)): initial Q-values. Default is [0.5, 0.5] (no bias).
            : noisy_choice (bool): whether to add noise to the predicted choice. Default is False.
                True is not implemented. To implement a sampling probability should be specificed.

        """
        alpha, theta = params

        assert len(reward_vector) == len(action_vector), 'Reward and action vectors must be the same length.'
        T = len(reward_vector)
        
        predicted_action_vector = np.zeros((T), dtype=int)

        Q_stored = np.zeros((2, T), dtype=float)
        Q = Q_init  # Q: current Q value
        action_probabilities = []

        # data input checks
        assert np.unique(action_vector).size == 2, 'Action vector must have two unique values.'
        assert list(np.unique(action_vector)) == [0,1], 'Action vector must have values of 0 and 1.'

        for _idx, (_reward, _action) in enumerate(zip(reward_vector, action_vector)):

            # store values for Q_{t+1}
            Q_stored[:, _idx] = Q
            
            # compute choice probabilities
            p0 = np.exp(theta*Q[0]) / (np.exp(theta*Q[0]) + np.exp(theta*Q[1]))
            p1 = 1 - p0
            action_probabilities.append([p0, p1])

            # predict choice: assumn
            if noisy_choice:
                raise NotImplementedError('Noisy prediction not implemented. See rescorla_wagner_model_simulation.py for an example implementation.')
        
            # else
            predicted_action_vector[_idx] = np.argmax([p0, p1])

            # update Q-values based on actual action taken
            # assuming _action in (0,1)
            Q[_action] = Q[_action] + alpha * (_reward - Q[_action])
            
        return Q_stored, predicted_action_vector, action_probabilities


    @staticmethod
    def calc_prob_of_success(rewards, bin_size=20):
        """
        Calculate the probability of taking the correct action over <bin_size> trials.

        Arguments:
        ----------
            : rewards (list, shape (T,)) in [0,1]: array of known rewards.
            : bin_size (int or None):
                int: number of trials to average over.
                None: use the entire set.

        """
        # data checks
        assert np.unique(rewards).size == 2, 'Reward vector must have two unique values.'
        assert list(np.unique(rewards)) == [0,1], 'Reward vector must have values of 0 and 1.'

        # compute rolling prob
        prob_of_success = []
        x_vector = []

        if bin_size is None:
            return np.arange(rewards), np.mean(rewards)

        for i in range(0, len(rewards), bin_size):
            x_vector.append(i)
            prob_of_success.append(np.mean(rewards[i: i+bin_size]))

        return x_vector, prob_of_success

    @staticmethod
    def fit_loess(x, y, frac=0.1):
        """
        Fit a loess curve to the data.
        """
        # fit loess
        loess = lowess(y, x, frac=frac)
        x_loess, y_loess = loess[:, 0], loess[:, 1]
        return x_loess, y_loess

    @staticmethod
    def compute_binary_metrics(TP, FP, TN, FN):

        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else np.nan
        specificity = TN / (TN + FP) if (TN + FP) != 0 else np.nan
        precision   = TP / (TP + FP) if (TP + FP) != 0 else np.nan
        recall      = TP / (TP + FN) if (TP + FN) != 0 else np.nan
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        F_score = 2 * (precision * recall) / (precision + recall)

        return {'senstivity': sensitivity, 'specificity': specificity, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'F_score': F_score}

    @staticmethod
    def compute_TP_FP_TN_FN(actions, rewards):
            # compute TP, FP, TN, FN
        TP, FP, TN, FN = 0, 0, 0, 0
        for _action, _reward in zip(actions, rewards):

            # TP = action = 1, reward = 1
            if _action == 1 and _reward == 1:
                TP += 1
        
            # FP = action = 1, reward = 0
            if _action == 1 and _reward == 0:
                FP += 1
            
            # TN = action = 0, reward = 0
            if _action == 0 and _reward == 0:
                TN += 1

            # FN = action = 0, reward = 1
            if _action == 0 and _reward == 1:
                FN += 1

        return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
 
    
    @staticmethod
    def calc_confusion_matrix(actions, predicted_action, bin_size=20):
        """
        Calculate the confusion matrix over <bin_size> trials.

        use <predicted_action> or <reward> as the model estimate.

        Arguments:
        ----------
            : actions (list, shape (T,), [0,1]): binary array of actions.
            : predicted_action (list, shape (T,), [0,1]): binary array of predicted action.
            : bin_size (int or None):
                int: number of trials to average over.
                None: use the entire set.

        """

        # data checks
        assert len(actions) == len(predicted_action), 'Actions and rewards must be the same length.'
        assert np.unique(actions).size == 2, 'Action vector must have two unique values.'
        # assert np.unique(predicted_action).size == 2, 'Reward vector must have two unique values.'
        assert list(np.unique(actions)) == [0,1], 'Action vector must have values of 0 and 1.'
        # assert list(np.unique(predicted_action)) == [0,1], 'Reward vector must have values of 0 and 1.'

        if bin_size is None:
            _results = RoscorlaWagerModelDiagnostics.compute_TP_FP_TN_FN(actions, predicted_action)
            _metrics = RoscorlaWagerModelDiagnostics.compute_binary_metrics(*_results.values())
            return {**_results, **_metrics}

        # compute rolling prob
        result_columns = ['start_index', 'end_index', 'TP', 'FP', 'TN', 'FN', 'senstivity', 'specificity', 'precision', 'recall', 'accuracy', 'F_score']
        
        # init results
        _result_dict = {k: [] for k in result_columns}
        
        for i in range(0, len(predicted_action), bin_size):
            i_end = i+bin_size
            _reward_slice = predicted_action[i: i_end]
            _action_slice = actions[i: i_end]
            _results = RoscorlaWagerModelDiagnostics.compute_TP_FP_TN_FN(_action_slice, _reward_slice)
            _metrics = RoscorlaWagerModelDiagnostics.compute_binary_metrics(*_results.values())
            _results = {**_results, **_metrics}
            _result_dict['start_index'].append(i)
            _result_dict['end_index'].append(i_end)
            for k, v in _results.items():
                _result_dict[k].append(v)

        return pd.DataFrame(_result_dict)
    