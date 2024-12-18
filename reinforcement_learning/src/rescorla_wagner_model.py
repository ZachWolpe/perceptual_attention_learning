"""
-------------------------------------------------------------------------------
rescorla_wagner_model.py

Rescorla Wagner Model model implementation & optimization (training).

#### Config

- *`T`* trails
- *`K`* number of states (multi-armed bandits)
- Goal is to maximise reward


During each trial $t$ each action $k$ pays out reward $r_t$ with probability $µ^k_t$ and reward $0$ with probability $1-µ^k_t$.

The goal is to maximise the reward over the $T$ trials.


Parameter Interpretation:
-------------------------
    : alpha: [0,1] : learning rate: how much the value of the action is updated after each trial.
        0 : no learning
        1 : deterministic updating (Q-value = reward)

    : beta: [0, inf] : inverse temperature: how much the agent exploits vs explores.
        0 : random choice
        inf : deterministic choice

Source:
-------
    : https://shawnrhoads.github.io/gu-psyc-347/module-03-01_Models-of-Learning.html


Modifaction Logs:
: 04 June 24     : zachcolinwolpe@gmail.com      : init
: 20 June 24     : zachcolinwolpe@gmail.com      : Add: (fit_rescorla_wagner_model_n_times)
-------------------------------------------------------------------------------
"""
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy


class RoscorlaWagner:

    @staticmethod
    def negll_RescorlaWagner(model_parameters, actions, rewards):
        """
        Compute the Negative Log-Likelihood of the Rescorla-Wagner model

        Parameters:
        -----------
            : model_parameters (list[float]) : list of model parameters (alpha, theta)
                : alpha (float) : learning rate
                : theta (float) : inverse temperature
            : actions (array) : choices
            : rewards (array) : rewards

        Returns:
        --------
            : negLL (float) : negative log-likelihood

        Source:
        -------
            : https://shawnrhoads.github.io/gu-psyc-347/module-03-01_Models-of-Learning.html
        """
        
        # extract params
        alpha, theta = model_parameters

        # assuming no starting bias
        Q = [0.5, 0.5]
        T = len(actions)
        choiceProb = np.zeros((T), dtype=float)

        # assuming action space K = 2
        assert len(np.unique(actions)) == 2, f'Action space should be 2. Larger action space not implemented. Got: {np.unique(actions)} unique actions.'
        assert list(np.unique(rewards)) == [0, 1], f'Rewards should be binary. Got: {np.unique(rewards)}.'
        assert len(actions) == len(rewards), f'Length of actions and rewards should be equal. Got: {len(actions)} and {len(rewards)}.'
        
        for t in range(T):
            
            # compute choice probabilities for k=2
            p0 = np.exp(theta*Q[0]) / (np.exp(theta*Q[0]) + np.exp(theta*Q[1]))
            p = [p0, 1-p0]

            # compute choice probability for actual choice
            choiceProb[t] = p[actions[t]]

            # update values
            delta = rewards[t] - Q[actions[t]]
            Q[actions[t]] = Q[actions[t]] + alpha * delta
        
        negLL = -np.sum(np.log(choiceProb))
        
        return negLL


    @staticmethod
    def optize_RL_negLL(
            actions,
            rewards,
            loss_function=None,
            alpha_range=np.linspace(0, 1, 10),
            theta_range=np.linspace(1, 25, 10)
            ):
        """
        Optimize the Rescorla-Wagner model using gradient descent.

        Parameters:
        -----------
            : actions (array) : choices
            : rewards (array) : rewards
            : alpha_range (array) : range of initial alpha values to search
            : theta_range (array) : range of initial theta values to search

        Returns:
        --------
            : res_nll (float) : negative log-likelihood
            : param_fits (list[float]) : list of optimal parameters (alpha, theta)

        Source:
        -------
            : https://shawnrhoads.github.io/gu-psyc-347/module-03-01_Models-of-Learning.html
            : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        """

        if loss_function is None:
            loss_function = RoscorlaWagner.negll_RescorlaWagner

        res_nll = np.inf  # set initial neg LL to be inf

        # guess several different starting points for alpha & theta
        for alpha_hat in alpha_range:
            for theta_hat in theta_range:
                
                # guesses for alpha, theta will change on each loop
                init_guess = (alpha_hat, theta_hat)
                
                # minimize neg LL
                result = scipy.optimize.minimize(
                    loss_function, #RoscorlaWagner.negll_RescorlaWagner 
                    init_guess, 
                    (actions, rewards), 
                    bounds=((0,1),(1,50)))
                
                # if current negLL is smaller than the last negLL,
                # then store current data
                if result.fun < res_nll:
                    res_nll = result.fun
                    param_fits = result.x

        # also, compute BIC
        # note: we don't need the -1 because 
        # we already have the negative log likelihood!
        BIC = len(init_guess) * np.log(len(actions)) + 2*res_nll

        return res_nll, param_fits, BIC


    @staticmethod
    def grid_search(actions, rewards, alpha_range=np.linspace(0,0.5,100), theta_range=np.linspace(0, 1, 100)):
        """
        Perform a grid search over the parameter space for the Rescorla-Wagner model. 

        Parameters:
        -----------
            actions : array[int] in [0,K]: action vector of the rat.
            rewards : array[int] in [0,1]: reward vector of the rat.

        Returns:
        --------
            nLL : list[float]: negative log-likelihood values for each parameter combination.
            plot_opt_grid : pd.DataFrame: sorted grid search results.
        """
        nLL = []
        _pd_rows = []
        for alpha_val in tqdm(alpha_range):
            for theta in theta_range:
                nLL.append(RoscorlaWagner.negll_RescorlaWagner([alpha_val, theta], actions, rewards))
                _pd_rows.append({'alpha': alpha_val, 'theta': theta, 'nLL': nLL[-1]})

        grid_df_to_plot = pd.DataFrame(_pd_rows).sort_values(by='nLL')

        return nLL, grid_df_to_plot

    @staticmethod
    def fit_rescorla_wagner_model_n_times(
            action_vector,
            reward_vector,
            alpha_range=np.linspace(0, 1, 10),
            theta_range=np.linspace(1, 25, 10),
            alpha_bound=(0, 1),
            theta_bound=(1, 50),
            log=True):
        """
        Fit N Rescorla-Wagner models to the data and return the best fit.

        N fits are required to account for local minima in the optimization. Each iteration takes different starting points for alpha and theta.

        The starting points are defined by the alpha_range and theta_range parameters.


        Parameters:
        -----------
            : action_vector (array) : choices
            : reward_vector (array) : rewards
            : alpha_range (array) : range of initial alpha values to search
            : theta_range (array) : range of initial theta values to search
            : alpha_bound (tuple) : bounds for alpha
            : theta_bound (tuple) : bounds for theta
        
        """

        # gradient descent to minimize neg LL
        res_nll = np.inf # set initial neg LL to be inf

        # guess several different starting points for alpha
        generator = tqdm if log else lambda x: x
        alpha_range = generator(alpha_range)

        for _alpha in alpha_range:
            for _theta in theta_range:
                
                # guesses for alpha, theta will change on each loop
                init_guess = (_alpha, _theta)
                
                # minimize neg LL
                result = minimize(
                    RoscorlaWagner.negll_RescorlaWagner,
                    init_guess,
                    (action_vector, reward_vector),
                    bounds=(alpha_bound, theta_bound)
                    )
                
                # if current negLL is smaller than the last negLL,
                # then store current data
                if result.fun < res_nll:
                    res_nll = result.fun
                    param_fits = result.x

        # also, compute BIC
        # note: we don't need the -1 because 
        # we already have the negative log likelihood!
        BIC = len(init_guess) * np.log(len(action_vector)) + 2*res_nll

        if log:
            print('Neg Log Likelihood fit complete.')
            print(fr'alpha_hat = {param_fits[0]:.2f}, theta_hat = {param_fits[1]:.2f}')
            print(fr'BIC = {BIC:.2f}')

        return param_fits, BIC