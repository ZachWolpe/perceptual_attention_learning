"""
-------------------------------------------------------------------------------
rescorla_wagner_model_simulation.py

Simulate a RW model. Used to verify the model fitting implementation.

Source: https://shawnrhoads.github.io/gu-psyc-347/module-03-01_Models-of-Learning.html

Modifaction Logs:
: 13.06.24          : zachcolinwolpe@gmail.com      : init
-------------------------------------------------------------------------------
"""

import numpy as np
from src.rescorla_wagner_model_plots import RescorlaWagnerPlots


class RescorlaWagnerSimulate(RescorlaWagnerPlots):

    def __init__(self):
        super().__init__()

    def simulate(self, params, T, mu, noisy_choice=True):
        """
        Simulate the Rescorla-Wagner model

        Parameters:
        -----------
            : params (list[float]) : list of model parameters (alpha, theta)
                : alpha (float) : learning rate
                : theta (float) : inverse temperature
            : T (int) : number of trials
            : mu (list[float]) : list of reward probabilities for each state
            : noisy_choice (bool) : add noise to choice
            : update_properties (bool): whether or not to update internal properties.
        """

        alpha, theta = params
        
        c = np.zeros((T), dtype=int)
        r = np.zeros((T), dtype=int)

        Q_stored = np.zeros((2, T), dtype=float)
        Q = [0.5, 0.5]

        for t in range(T):

            # store values for Q_{t+1}
            Q_stored[:, t] = Q
            
            # compute choice probabilities
            p0 = np.exp(theta*Q[0]) / (np.exp(theta*Q[0]) + np.exp(theta*Q[1]))
            p1 = 1 - p0

            # make choice according to choice probababilities
            # as weighted coin flip to make a choice
            # choose stim 0 if random number is in the [0 p0] interval
            # and 1 otherwise
            if noisy_choice:
                if np.random.random_sample(1) < p0:
                    c[t] = 0
                else:
                    c[t] = 1
            else:
                # make choice without noise
                c[t] = np.argmax([p0, p1])
            
            # generate reward based on reward probability
            r[t] = np.random.rand() < mu[c[t]]

            # update values
            delta = r[t] - Q[c[t]]
            Q[c[t]] = Q[c[t]] + alpha * delta

        self.params = params
        self.c = c
        self.r = r
        self.Q_stored = Q_stored
        return c, r, Q_stored
    
    def plot_reward(self, reward_vector=None, choice_vector=None, T=None):

        if reward_vector is None:
            reward_vector = self.r
        
        if choice_vector is None:
            choice_vector = self.c

        if T is None:
            T = len(reward_vector)
            
        # plot the simulation
        return RescorlaWagnerPlots.plot_reward(reward_vector=reward_vector, choice_vector=choice_vector, T=T)

    def plot_Q_estimates(self, Q_values=None, choice_vector=None, T=None):
        
        if Q_values is None:
            Q_values = self.Q_stored
        
        if choice_vector is None:
            choice_vector = self.c
        
        if T is None:
            T = Q_values.shape[1]

        labels = ['80% machine', '20% machine']


        # plot the simulation
        return RescorlaWagnerPlots.plot_Q_estimates(Q_values=Q_values, choice_vector=choice_vector, T=T, labels=labels)