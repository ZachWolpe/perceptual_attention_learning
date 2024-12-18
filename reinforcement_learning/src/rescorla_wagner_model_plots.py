"""
-------------------------------------------------------------------------------
rescorla_wagner_model_plots.py

Plots to support the RW model and analysis.

Modifaction Logs:
: 19 June 24     : zachcolinwolpe@gmail.com      : init
-------------------------------------------------------------------------------
"""
# fit loess
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy


class RescorlaWagnerPlots:

    @staticmethod
    def plot_reward(reward_vector, choice_vector, T=None):

        if T is None:
            T = len(reward_vector)
            
        # plot the simulation
        plt.plot(range(T), reward_vector, 'r--', alpha=.6)
        plt.plot(range(T), choice_vector, '+', label='choice')
        plt.xlabel('trials')
        plt.ylabel('outcome (1=reward, 0=no reward)')
        plt.title('Rescorla-Wagner Learning')
        plt.legend()
        return plt

    @staticmethod
    def plot_Q_estimates(Q_values=None, choice_vector=None, T=None, labels=[None, None]):
        
        if T is None:
            T = Q_values.shape[1]
    
        # plot the simulation
        plt.plot(range(T), Q_values[1, :], 'r--', alpha=.6, label=labels[0])
        plt.plot(range(T), Q_values[0, :], 'm-', alpha=.6, label=labels[1])
        plt.plot(range(T), choice_vector, 'b+', label='choice')
        plt.xlabel('trials')
        plt.ylabel('value')
        plt.title('Rescorla-Wagner Learning')
        plt.legend()
        return plt

    @staticmethod
    def plot_metric(x_vector, prob_of_success, name=None, title='Metric', fig=None, color='darkblue', dash='dash'):
        # plot go,nogo
        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vector, y=prob_of_success,
            line=dict(color=color, dash=dash),
            mode='lines', name=name))

        fig.update_layout(title=title, template='none')

        return fig

    @staticmethod
    def plot_grid_opt(grid_df_to_plot):
        """
        Plot the results of a grid search optimization.

        Parameters:
        -----------
            grid_df_to_plot : pd.DataFrame: sorted grid search results.
                This can be obtained by calling: RoscorlaWagner.grid_search()

        Returns:
        --------
            fig : plotly.graph_objects.Figure: plot of the grid search optimization.
        """
        # otimal results
        grid_df_to_plot = grid_df_to_plot.sort_values(by='nLL')
        alpha_hat, theta_hat = grid_df_to_plot.iloc[0]['alpha'], grid_df_to_plot.iloc[0]['theta']
        fig = px.scatter_3d(grid_df_to_plot.sample(10000), x='alpha', y='theta', z='nLL', color='nLL')
    
    # plot optimal point (argmin(neg logliklihood)
        fig.add_trace(
            go.Scatter3d(x=[alpha_hat], y=[theta_hat], z=[grid_df_to_plot['nLL'].min()],
            mode='markers',
            marker_symbol='x',
            marker_line_color="midnightblue", marker_color= "lightskyblue",
            marker_line_width=4, marker_size=10,
            name='Optimal Point',
            showlegend=False))

        # update layout
        fig.update_layout(
            title='Grid Search Optimization. argmax(nLL): alpha_hat = {:.2f}, theta_hat = {:.2f}'.format(alpha_hat, theta_hat),
            xaxis_title='alpha',
            yaxis_title='theta',
            template='none',
            )
        
        return fig


    @staticmethod
    def plot_Q_values(actions, rewards, Q_values):
        _, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(actions))

        Qs = Q_values
        ax.plot(x, Qs[:, 0] - 0.5 + 0, c="C0", lw=3, alpha=0.3)
        ax.plot(x, Qs[:, 1] - 0.5 + 1, c="C1", lw=3, alpha=0.3)

        s = 7
        lw = 2

        cond = (actions == 0) & (rewards == 0)
        ax.plot(x[cond], actions[cond], "o", ms=s, mfc="None", mec="C0", mew=lw)

        cond = (actions == 0) & (rewards == 1)
        ax.plot(x[cond], actions[cond], "o", ms=s, mfc="C0", mec="C0", mew=lw)

        cond = (actions == 1) & (rewards == 0)
        ax.plot(x[cond], actions[cond], "o", ms=s, mfc="None", mec="C1", mew=lw)

        cond = (actions == 1) & (rewards == 1)
        ax.plot(x[cond], actions[cond], "o", ms=s, mfc="C1", mec="C1", mew=lw)

        ax.set_yticks([0, 1], ["action=0", "action=1"])
        ax.set_ylim(-1, 2)
        ax.set_ylabel("action")
        ax.set_xlabel("trial")

        reward_artist = Line2D([], [], c="k", ls="none", marker="o", ms=s, mew=lw, label="Reward")
        no_reward_artist = Line2D(
            [], [], ls="none", marker="o", mfc="w", mec="k", ms=s, mew=lw, label="No reward"
        )
        Qvalue_artist = Line2D([], [], c="k", ls="-", lw=3, alpha=0.3, label="Qvalue (centered)")

        ax.legend(handles=[no_reward_artist, Qvalue_artist, reward_artist], fontsize=12, loc=(1.01, 0.27))

        return ax
