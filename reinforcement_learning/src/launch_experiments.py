"""
--------------------------------------------------------------------------------
run_experiments.py

Helper functions to run experiments for the reinforcement learning algorithms.



: 30 Sep 2024
: zach.wolpe@medibio.com.au
--------------------------------------------------------------------------------
"""

import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import pandas as pd

import numpy as np
import statistics
import scipy.io
import logging
import pprint
import os
from collections import Counter
import time
import sys

import mlflow
# internals
from src.helpers import log_rat_metadata, log_sequence_data, load_config
from src.query_dataset import (QuerySequenceData, load_data)

from src.rescorla_wagner_model import (RoscorlaWagner)
from src.rescorla_wagner_model_plots import (RescorlaWagnerPlots)
from src.rescorla_wagner_model_simulation import (RescorlaWagnerSimulate)
from src.rescorla_wagner_model_diagnostics import (RoscorlaWagerModelDiagnostics)

from src.cog_sci_random_response_model import (RandomResponseModel)
from src.cog_sci_win_stay_lose_shift_model import (WinStayLoseShiftModel)
from src.cog_sci_learning_model_base import (MultiArmedBanditModels)
from src.cog_sci_roscorla_wagner_model import RoscorlaWagnerModel


# config# load yaml config
_config = load_config('config.yaml')


# ----------------------------------------------------------------------------------------------------------->>
# Experiment Helper Functions ------------------------------------------------------------------------------->>

def print_unique_keys(StimCode):
    keys = np.unique([k[1] for k in StimCode.keys()])
    print(f'Unique keys: {keys}')


def extract_experiment(experiment, StimCode, RespCode):
    qsd = QuerySequenceData(StimCode, RespCode)

    for _key in experiment:
        # extract data
        qsd.filter_sequences(*_key, update_existing_stim_resp=True)

    qsd.extract_stim_resp_data()
    qsd.infer_action_reward_pairs()
    action_vector = qsd._action
    reward_vector = qsd._reward
    return qsd, action_vector, reward_vector


def run_experiment_model_1(action_vector, reward_vector, init_guess=[0.5]):
    # fit model 1. Random Response Model
    rrm = RandomResponseModel()
    results = rrm.optimize_scikit(
        loss_function=rrm.neg_log_likelihood,
        init_guess=init_guess,
        args=(action_vector, reward_vector), bounds=[(0,1)])
    return results


def run_experiment_model_2(action_vector, reward_vector, init_guess=[0.5]):
    wsls = WinStayLoseShiftModel()
    results = wsls.optimize_scikit(
        loss_function=wsls.neg_log_likelihood,
        init_guess=init_guess,
        args=(action_vector, reward_vector), bounds=[(0,1)])
    return results


def run_experiment_model_3(action_vector, reward_vector):
    rwm = RoscorlaWagnerModel()
    results = rwm.optimize_scikit_model_over_init_parameters(
        actions=action_vector,
        rewards=reward_vector,
        loss_function=None,
        alpha_init_range=np.linspace(0, 1, 5),
        theta_init_range=np.linspace(.1, 10, 7),
        bounds=((0,1), (0.1, 10)),
        log_progress=False
        )
    negLL, params_opt, BIC, optimal_init_params = results
    return negLL, params_opt, BIC, optimal_init_params


def init_results():
    return {
        # describe experiment
        'experiment_ID': [],
        'experiment': [],

        # describe input data
        'reward_rate': [],
        'action_rate': [],
        'corr_action_reward': [],
        'corr_stim_resp': [],

        'model_1_b_pred': [],
        'model_1_negLL': [],
        'model_1_BIC': [],

        'model_2_epsilon_pred': [],
        'model_2_negLL': [],
        'model_2_BIC': [],

        'model_3_alpha_pred': [],
        'model_3_theta_pred': [],
        'model_3_negLL': [],
        'model_3_BIC': [],
        'model_3_opt_init_params': []
    }


def update_results(_result, results):
    for key, val in _result.items():
        results[key].append(val)


def compute_experiment_features(qsd):
    _stimCount = Counter(qsd._stimCodeFlat)
    _respCount = Counter(qsd._respCodeFlat)
    action, reward = qsd._action, qsd._reward
    action_mean = np.mean(action)
    reward_mean = np.mean(reward)
    action_reward_corr = np.corrcoef(action, reward)[0, 1]
    stim_resp_corr = np.corrcoef(qsd._stimCodeFlat, qsd._respCodeFlat)[0, 1]

    return _stimCount, _respCount, action, reward, action_mean, reward_mean, action_reward_corr, stim_resp_corr


def print_progress_bar(iteration, total, length=50):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent}% Complete')
    sys.stdout.flush()

# # Example loop with manual progress bar
# total = 100
# for i in range(total):
#     time.sleep(0.1)  # Simulate work being done
#     print_progress_bar(i + 1, total)

# print()  # Move to the next line after the progress bar is complete


def generate_experiment(experiments):
    for idx, experiment in enumerate(experiments):
        yield idx, experiment


def run_experiment(idx, exp, StimCode, RespCode, drop_n_trails=0):
    # extract experiment data
    qsd, action_vector, reward_vector = extract_experiment(exp, StimCode, RespCode)

    # drop n trails
    if drop_n_trails > 0:
        action_vector = action_vector[drop_n_trails:]
        reward_vector = reward_vector[drop_n_trails:]

    # generate metadata
    _stimCount, _respCount, action, reward, action_mean, reward_mean, action_reward_corr, stim_resp_corr = \
        compute_experiment_features(qsd)

    # print(f'exp: {idx}, action rate: {np.mean(action_vector)}, reward rate: {np.mean(reward_vector)}')

    # fit experiment 1
    result_model_1 = run_experiment_model_1(action_vector, reward_vector)
    result_model_2 = run_experiment_model_2(action_vector, reward_vector)
    result_model_3 = run_experiment_model_3(action_vector, reward_vector)

    # extract model 3 data (not dict)
    negLL, params_opt, BIC, optimal_init_params = result_model_3

    del qsd

    # result object
    _result = {
        # describe experiment
        'experiment_ID': idx, 'experiment': exp,

        # describe input data
        'reward_rate': reward_mean, 'action_rate': action_mean,
        'corr_action_reward': action_reward_corr, 'corr_stim_resp': stim_resp_corr,

        'model_1_b_pred': result_model_1['param_opt'][0],
        'model_1_negLL': result_model_1['negLL'],
        'model_1_BIC': result_model_1['BIC'],
        'model_2_epsilon_pred': result_model_2['param_opt'][0],
        'model_2_negLL': result_model_2['negLL'],
        'model_2_BIC': result_model_2['BIC'],
        'model_3_alpha_pred': params_opt[0],
        'model_3_theta_pred': params_opt[1],
        'model_3_negLL': negLL, 'model_3_BIC': BIC,
        'model_3_opt_init_params': optimal_init_params
    }

    return _result


def run_experiment_suite(exp_gen, StimCode, RespCode, n_experiments=100, drop_n_trails=0, mlflow_tracking=True):
    results = init_results()

    for idx, exp in exp_gen:
        # print(f'Experiment: {idx}')
        print_progress_bar(idx + 1, n_experiments)

        _result = run_experiment(idx, exp, StimCode, RespCode, drop_n_trails=drop_n_trails)
        update_results(_result, results)  # external store

        if mlflow_tracking:
            with mlflow.start_run(run_name=f'Experiment {idx}', nested=True):
                # extract experiment data
                _metrics = {k: v for k, v in _result.items() if isinstance(v, (int, float))}
                _params = {k: str(v) for k, v in _result.items() if not isinstance(v, (int, float))}

                # Log parameters and metrics to MLFlow
                mlflow.log_params(_result)
                mlflow.log_metrics(_metrics)

                mlflow.log_param('experiment_ID', idx)
                mlflow.log_param('experiment', exp)
                # [mlflow.log_param(k,v) for k,v in _params.items()]

    print('Runtime complete :).')
    return pd.DataFrame(results)


# ----------------------------------------------------------------------------------------------------------->>
# Configure Experiments ------------------------------------------------------------------------------------->>

def build_experiment_set_v2(StimCode, interest_group='EDS'):

    # Subject ID
    subject_IDs = np.unique([k[0] for k in StimCode.keys()])

    # groupings: (subject_ID, <session_Groups>, <all>)
    experiments = []
    for _subject in subject_IDs:
        new_experiment = (_subject, interest_group, None)

        # double wrap to enforce backwards compatibility
        experiments.append([new_experiment])

    return experiments


# Set up a local directory for MLflow tracking
def configure_mlflow_tracking(
        mlflow_tracking_dir="./mlruns",
        experiment_name="RL Experiment - Rat Data - EDS baseline"):
    if not os.path.exists(mlflow_tracking_dir):
        os.makedirs(mlflow_tracking_dir)

    print('Warning: using file based method for mlflow tracking.')
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_tracking_dir)}")
    mlflow.set_experiment(experiment_name)


# ----------------------------------------------------------------------------------------------------------->>
# Filter, Select Models ------------------------------------------------------------------------------------->>

def filter_results(results, subset=['experiment_ID', 'experiment', 'reward_rate', 'action_rate', 'model_1_b_pred', 'model_2_epsilon_pred', 'model_3_alpha_pred', 'model_3_theta_pred', 'model_1_BIC', 'model_2_BIC', 'model_3_BIC','corr_action_reward'], add_var=None):
    if add_var:
        if isinstance(add_var, str):
            subset += [add_var]
        elif isinstance(add_var, list):
            subset += add_var
        else:
            raise NotImplementedError()

    _sub = results[subset]
    return _sub.round(3)


def select_model(_row):
    BICs = _row['model_1_BIC'], _row['model_2_BIC'], _row['model_3_BIC']
    best_model = np.argmin(BICs)
    opt_models = ['model_1', 'model_2', 'model_3']
    return opt_models[best_model]


def filter_results_select_model(results_data):
    results_data['opt_models'] = results_data.apply(axis=1, func=select_model)
    _sub = filter_results(results=results_data, add_var='opt_models')
    # drop duplicate columns
    _sub = _sub.loc[:, ~_sub.columns.duplicated()]
    return _sub


def save_results(_sub, experiment_name, loc='experiment_logs'):
    _sub.to_csv(f'{loc}/{experiment_name}.csv', index=False)

# ----------------------------------------------------------------------------------------------------------->>
# Check Data Quality and Optimisation Space ----------------------------------------------------------------->>
#   - 1. Assess the input data quality.
#   - 2. Test the Optimisation space.


def extract_experiment_for_data_quality(
    experiment=(422, 'EDS', None),
    experiment_class='rat_experiment'
):
    (data,
        sequence_data,
        meta_data,
        StimCode,
        RespCode) = load_data(config=_config, experiment_class=experiment_class)

    qsd = QuerySequenceData(StimCode, RespCode)
    qsd.filter_sequences(
        subjectID=experiment[0], # all subjects
        sessionType=experiment[1],
        sessionNum=None, # all sessions 
        update_existing_stim_resp=False)

    qsd.extract_stim_resp_data()
    qsd.infer_action_reward_pairs()

    return qsd


def compute_experiment_stats(experiment, qsd):
    n_samples = len(qsd._reward)
    n_sessions = len(qsd._stimCode.keys())
    samples_per_session = n_samples / n_sessions if n_sessions > 0 else 0
    avg_reward_rate = np.mean(qsd._reward)

    # quality checks
    assert len(qsd._respCode.keys()) == len(qsd._stimCode.keys())
    assert len(qsd._action) == len(qsd._reward)

    msg = """
    ----------------------------------------
    Experiment:
    -----------

        - subjectID:            {}
        - sessionType:          {}
        - sessionNum:           {}
        - n_samples:            {}
        - n_sessions:           {}
        - samples_per_session:  {}
        - avg_reward_rate:      {:.5f}
    ----------------------------------------
    """.format(*experiment, n_samples, n_sessions, samples_per_session, avg_reward_rate)

    print(msg)

    return n_samples, n_sessions, samples_per_session, avg_reward_rate


def compute_reward_per_session(qsd):
    reward_rates = []
    n = 0
    for key, val in qsd._respCode.items():
        # compute reward rate
        # print(f'Key: {key}, n_samples: {len(val)}')
        n += len(val)
        qsd.infer_action_reward_pairs(response_vector=val)
        _reward = qsd._reward
        _action = qsd._action
        reward_rate = np.mean(_reward)
        reward_rates.append(reward_rate)
    return reward_rates, n


def plot_reward_per_session(reward_rates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(reward_rates)), y=reward_rates, mode='lines+markers', marker=dict(color='lightblue', size=2)))
    fig.update_layout(title=f'Reward Rate per Session. Avg reward: {np.mean(reward_rates):.5f}',
                      xaxis_title='Session',
                      yaxis_title='Reward Rate',
                      template='plotly_dark')

    # set y-axis range
    fig.update_yaxes(range=[0, 1])

    # plot points
    fig.add_trace(go.Scatter(x=np.arange(len(reward_rates)), y=reward_rates, mode='markers', marker=dict(color='red', size=8)))

    # plot bar from x-axis to point
    for i, rate in enumerate(reward_rates):
        fig.add_shape(
            dict(
                type='line',
                x0=i,
                y0=0,
                x1=i,
                y1=rate,
                line=dict(color='red', width=1)
            )
        )

    # remove legend
    fig.update_layout(showlegend=False)
    return fig


def optimize_model_2(action_vector, reward_vector, init_guess=[0.5]):

    # fit experiment 1
    # result_model_1 = run_experiment_model_1(action_vector, reward_vector)
    res_sckit_run_exper = run_experiment_model_2(action_vector, reward_vector)

    wsls = WinStayLoseShiftModel()
    res_scikit = wsls.optimize_scikit(
            loss_function=wsls.neg_log_likelihood,
            init_guess=init_guess,
            args=(action_vector, reward_vector), bounds=[(0,1)])

    res_brute_force = wsls.optimize_brute_force(
        bounds=(0,1),
        actions=action_vector,
        rewards=reward_vector,
        loss_function=wsls.neg_log_likelihood,
    )
    return wsls, res_sckit_run_exper, res_scikit, res_brute_force


# plot data
def plot_prob_success(qsd, title='', loess_frac=0.5, bin_size=5):

    REWARD_VECTOR = qsd._reward
    x_vector, prob_of_success = RoscorlaWagerModelDiagnostics.calc_prob_of_success(REWARD_VECTOR, bin_size=bin_size)

    frac = loess_frac
    x_loess, y_loess = RoscorlaWagerModelDiagnostics.fit_loess(x_vector, prob_of_success, frac=frac)
    fig = RescorlaWagnerPlots.plot_metric(x_vector, prob_of_success, name='Prob of Success.', title=title)
    fig.add_trace(go.Scatter(x=x_loess, y=y_loess, mode='lines', line_color='red', name=f'loess frac:{frac}'))
    fig.show()


# ----------------------------------------------------------------------------------------------------------->>
# Instantiate: Run Data Quality Checks ---------------------------------------------------------------------->>


def run_data_quality_check(_experiment=(422, 'EDS', None), experiment_class='rat_experiment'):

    # Extract data ---------------------------------------------->>
    qsd = extract_experiment_for_data_quality(experiment=_experiment, experiment_class=experiment_class)

    n_samples, n_sessions, samples_per_session, avg_reward_rate \
        = compute_experiment_stats(_experiment, qsd)

    # Reward rate per session ----------------------------------->>
    reward_rates, n = compute_reward_per_session(qsd)
    print(f'N (total):{n} - Reward rates: {reward_rates}')
    fig = plot_reward_per_session(reward_rates)
    fig.show()

    # Optimisation Space
    (data,
        sequence_data,
        meta_data,
        StimCode,
        RespCode) = load_data(config=_config, experiment_class=experiment_class)
    qsd, action_vector, reward_vector = extract_experiment([_experiment], StimCode, RespCode)

    # generate metadata
    _stimCount, _respCount, action, reward, action_mean, reward_mean, action_reward_corr, stim_resp_corr = \
        compute_experiment_features(qsd)

    # compare
    wsls, res_sckit_run_exper, res_scikit, res_brute_force \
        = optimize_model_2(action_vector, reward_vector)

    b_a, b_b, b_c = res_sckit_run_exper['param_opt'], res_scikit['param_opt'], res_brute_force['epsilon_pred']
    print(f'Optimisation results: {b_a}, {b_b}, {b_c}')

    wsls.plot_neg_log_likelihood(epsilon_true=1, action=action_vector, reward=reward_vector)

    plot_prob_success(qsd, title='EDS Baseline - Prob of Success', loess_frac=0.5, bin_size=30)


# ----------------------------------------------------------------------------------------------------------->>
# Execute Experiment Suite ---------------------------------------------------------------------------------->>

def exe_experiment(
    DROP_N_TRAILS=200,
    EXPERIMENT_NAME='EDS_BL_Easy',
    EXPERIMENT_CLASS='rat_experiment',
    DATA_SET='Rat Data',
):

    EXPERIMENT_NAME_MLflow = f"RL Experiment - {DATA_SET} - {EXPERIMENT_NAME}"
    print('DROP_N_TRAILS={} introduced to account for rebasing.'.format(DROP_N_TRAILS))
    # load data
    (data,
        sequence_data,
        meta_data,
        StimCode,
        RespCode) = load_data(config=_config, experiment_class=EXPERIMENT_CLASS)
    qsd = QuerySequenceData(StimCode, RespCode)

    # configure mlflow
    configure_mlflow_tracking(
        mlflow_tracking_dir="./mlruns",
        experiment_name=EXPERIMENT_NAME_MLflow
    )

    experiments = build_experiment_set_v2(StimCode, interest_group=EXPERIMENT_NAME)

    # generate and run experiments
    exp_gen = generate_experiment(experiments)
    N = len(experiments)
    msg = f"""
        --------------------------------------------------------------------------------
        >> Experiment:
        --------------

        Experiment Name:            {EXPERIMENT_NAME}
        Experiment Class:           {EXPERIMENT_CLASS}
        Experiment MLFlow Name:     {EXPERIMENT_NAME_MLflow}

        Experiment Suite:
        -----------------
            N Experiments:          {N}
            DROP_N_TRAILS:          {DROP_N_TRAILS}
        --------------------------------------------------------------------------------
    """
    print(msg)
    _results = run_experiment_suite(exp_gen, StimCode, RespCode, N, drop_n_trails=DROP_N_TRAILS)

    _sub = filter_results_select_model(_results)
    save_results(_sub, EXPERIMENT_NAME_MLflow, loc='experiment_logs')
    print(f'Saved results to experiment_logs/{EXPERIMENT_NAME_MLflow}.')

    return _sub

# :: VERSION 2 >>>>>>>>>>>>>>>> ----------------------------------------------------------------------------------------------------------->>
# :: VERSION 2 >>>>>>>>>>>>>>>> ----------------------------------------------------------------------------------------------------------->>
# :: VERSION 2 >>>>>>>>>>>>>>>> ----------------------------------------------------------------------------------------------------------->>
# :: VERSION 2 >>>>>>>>>>>>>>>> ----------------------------------------------------------------------------------------------------------->>
# :: VERSION 2 >>>>>>>>>>>>>>>> ----------------------------------------------------------------------------------------------------------->>

# ----------------------------------------------------------------------------------------------------------->>
# Updated experiment Functions: V2 -------------------------------------------------------------------------->>
# Updated to handle the new experiment structure ------------------------------------------------------------>>
# ----------------------------------------------------------------------------------------------------------->>

# Build experiment structure


class build_experiment_set_v3_EDS_post_learning:
    """
    Build experiments that follow new group structure

    Groups
    ------
        > EDS_BL_EASY --> EDS
        > EDS_BL_HARD --> EDS

    """

    def __init__(self, StimCode, RespCode):

        # Generate experiments
        self.StimCode = StimCode
        self.RespCode = RespCode
        self.extract_experiment_type(StimCode)
        self.build_experiment_set_v3()

    def extract_experiment_type(self, StimCode, key='EDS'):
        all_experiments = [i for i in StimCode.keys() if (key in i[1])]
        subjects = np.unique([i[0] for i in all_experiments])
        self.all_experiments = all_experiments
        self.subjects = list(subjects)
        return self

    def extract_subject_experiments(self, all_experiments, subject=4151):
        experiments = [i for i in all_experiments if i[0] == subject]
        return experiments

    def split_experiments(self, subject_experiments):

        EXPERIMENT_CLASS = None
        _experiment_set = {}
        _experiment_set['EDS_BL_Easy'] = []
        _experiment_set['EDS_BL_Hard'] = []

        for _experiment in subject_experiments:
            if (_experiment[1] == 'EDS_BL_Easy') or (_experiment[1] == 'EDS_BL_Hard'):
                EXPERIMENT_CLASS = _experiment[1]
                continue
            elif (_experiment[1] != 'EDS'):
                raise ValueError('Unexpected Experiment Value. Should be one of: [EDS_BL_Hard, EDS_BL_Easy, EDS]')
            
            # At this stage we have an EDS experiment that --> we should now group the experiment type based the previous experiment type.
            assert _experiment[1] == 'EDS', 'Unexpected error type.'

            _experiment_set[EXPERIMENT_CLASS].append(_experiment)

        return _experiment_set
    
    def build_experiment_set_v3(self):
        """
        We want to build experiments with the following interest groups:

        Groups
        ------
            : EDS_BL_EASY --> EDS
            : EDS_BL_HARD --> EDS
        """

        EXPERIMENT_SET = {}
        EXPERIMENT_SET['EDS_BL_Easy'] = []
        EXPERIMENT_SET['EDS_BL_Hard'] = []
        # sort experiments
        for _subject in self.subjects:
            subject_experiments = self.extract_subject_experiments(self.all_experiments, _subject)
            experiment_set = self.split_experiments(subject_experiments)

            for _key in ['EDS_BL_Easy', 'EDS_BL_Hard']:
                EXPERIMENT_SET[_key].append(experiment_set[_key])

        self.EXPERIMENT_SET = EXPERIMENT_SET

        # compute number of experiments
        self.n_experiments = len(self.EXPERIMENT_SET['EDS_BL_Easy'])
        self.n_experiments =+ len(self.EXPERIMENT_SET['EDS_BL_Hard'])

    @staticmethod
    def extract_experiment_from_keys(StimCode, RespCode, experiment_set):
        """
        Extract experiment data from a list of experiments.

        Args
        ----
            : experiment_set : list[set()] : list[sets] containing all sub-experiments to form an experiment. In particular we want to stack contiguous trails (within a experiment grouping).
                
        Example Input
        -------------

            : experiment_set=[
                (4151, 'EDS', 0),
                (4151, 'EDS', 1),
                (4151, 'EDS', 2),
                (4151, 'EDS', 3),
                (4151, 'EDS', 4),
                (4151, 'EDS', 5),
                (4151, 'EDS', 6),
                (4151, 'EDS', 7),
            ]
                
        Return
        ------
            : qsd (QuerySequenceData) object.
            : action_vector (array): concatonated vector of action sequences.
            : reward_vector (array): concateonated vector  of reward sequences.
        
        """
        qsd = QuerySequenceData(StimCode, RespCode)

        for _key in experiment_set:
            # extract data
            qsd.filter_sequences(*_key, update_existing_stim_resp=True)
        qsd.extract_stim_resp_data()
        qsd.infer_action_reward_pairs()
        action_vector = qsd._action
        reward_vector = qsd._reward

        return qsd, action_vector, reward_vector

    def generate_experiments(self):
        def _retrieve_subject(_experiment):
            _subject = list(set(i[0] for i in _experiment))
            if len(_subject) != 1:
                raise ValueError(f'Unexpected number of subjects in group. Should be 1, got {len(_subject)}')
            return _subject[0]

        exp_idx = -1
        for experiment_class_key in self.EXPERIMENT_SET.keys():
            # print('Experiment class: ', experiment_class_key)
            _experiments = self.EXPERIMENT_SET[experiment_class_key]
            for _experiment in _experiments:

                # extract meta data used to describe experiment
                _subject_ID = _retrieve_subject(_experiment)
                _experiment_class = experiment_class_key + ' -> EDS'

                # print('subject: ', _subject_ID, ' - ', _experiment)
                qsd, action_vector, reward_vector = self.extract_experiment_from_keys(
                    self.StimCode,
                    self.RespCode,
                    _experiment)
                exp_idx += 1
                yield (
                    exp_idx,
                    _subject_ID,
                    _experiment_class,
                    qsd,
                    action_vector,
                    reward_vector
                )


# ----------------------------------------------------------------------------------------------------------->>
# 1. Build an Experiment Based on the new Experiment Structure ---------------------------------------------->>


# build_exp.EXPERIMENT_SET
def run_experiment_v2(subject_ID, experiment_class, qsd, action_vector, reward_vector, drop_n_trails=200, experiment_ID=None):
    """
    Updatd run_experiment to account for the new data structure.

    Args
    ----
        : subject (str)
        : experiment_class (str) 
        : qsd (QueryDataObject)
        : action_vector (array)
        : reward_vector (array)
        : drop_n_trails (int)

    """

    # drop n trails
    if drop_n_trails > 0:
        action_vector = action_vector[drop_n_trails:]
        reward_vector = reward_vector[drop_n_trails:]

    # generate metadata
    _stimCount, _respCount, action, reward, action_mean, reward_mean, action_reward_corr, stim_resp_corr = \
        compute_experiment_features(qsd)

    # print(f'exp: {idx}, action rate: {np.mean(action_vector)}, reward rate: {np.mean(reward_vector)}')

    # fit experiment 1
    result_model_1 = run_experiment_model_1(action_vector, reward_vector)
    result_model_2 = run_experiment_model_2(action_vector, reward_vector)
    result_model_3 = run_experiment_model_3(action_vector, reward_vector)

    # extract model 3 data (not dict)
    negLL, params_opt, BIC, optimal_init_params = result_model_3

    del qsd

    # result object
    _result = {
        # metadata /descriptives
        'subject_ID': str(subject_ID),
        'experiment_ID': experiment_ID,
        'experiment_class': experiment_class,

        # describe input data
        'reward_rate': reward_mean, 'action_rate': action_mean,
        'corr_action_reward': action_reward_corr, 'corr_stim_resp': stim_resp_corr,

        'model_1_b_pred': result_model_1['param_opt'][0],
        'model_1_negLL': result_model_1['negLL'],
        'model_1_BIC': result_model_1['BIC'],
        'model_2_epsilon_pred': result_model_2['param_opt'][0],
        'model_2_negLL': result_model_2['negLL'],
        'model_2_BIC': result_model_2['BIC'],
        'model_3_alpha_pred': params_opt[0],
        'model_3_theta_pred': params_opt[1],
        'model_3_negLL': negLL, 'model_3_BIC': BIC,
        'model_3_opt_init_params': optimal_init_params
    }

    return _result

# ----------------------------------------------------------------------------------------------------------->>
# 2. Abstraction 1. Run multiple Experiments ---------------------------------------------------------------->>


def init_results_v2():
    return {
        # describe experiment
        'subject_ID': [],
        'experiment_ID': [],
        'experiment_class': [],

        # describe input data
        'reward_rate': [],
        'action_rate': [],
        'corr_action_reward': [],
        'corr_stim_resp': [],

        # performance
        'model_1_b_pred': [],
        'model_1_negLL': [],
        'model_1_BIC': [],

        'model_2_epsilon_pred': [],
        'model_2_negLL': [],
        'model_2_BIC': [],

        'model_3_alpha_pred': [],
        'model_3_theta_pred': [],
        'model_3_negLL': [],
        'model_3_BIC': [],
        'model_3_opt_init_params': []
    }


class sort_mlflow_params:
    # describe input data
    params = ['subject_ID', 'experment_ID', 'experiment_class',
              'reward_rate', 'action_rate', 'corr_action_reward', 'corr_stim_resp',]
    # performance
    metrics = [
        'model_1_b_pred', 'model_1_negLL', 'model_1_BIC', 'model_2_epsilon_pred', 'model_2_negLL', 'model_2_BIC',
        'model_3_alpha_pred', 'model_3_theta_pred', 'model_3_negLL', 'model_3_BIC']
    metrics_tuple = ['model_3_opt_init_params']
    artifacts = []
    text = []


# ----------------------------------------------------------------------------------------------------------->>
# 3. Abstraction 2. Launch & Run all Experiments ------------------------------------------------------------>>


def log_tuple_metric(mlflow, metric_name, tuple_value):
    for i, value in enumerate(tuple_value):
        mlflow.log_metric(f"{metric_name}_{i}", np.float64(value))


def exe_experiment_suite_v2(exp_gen, DATA_SET, EXPERIMENT_NAME, drop_n_trails=200, mlflow_tracking=True, debug_mode=False):
    """
    Udated run experiment suite to account for the new data structure.
    """
    results = init_results_v2()

    EXPERIMENT_NAME_MLflow = f"RL Experiment - {DATA_SET} - {EXPERIMENT_NAME}"
    print('DROP_N_TRAILS={} introduced to account for rebasing.'.format(drop_n_trails))

    # configure mlflow
    configure_mlflow_tracking(
        mlflow_tracking_dir="./mlruns",
        experiment_name=EXPERIMENT_NAME_MLflow
    )

    for (exp_idx, _subject_ID, _experiment_class, qsd, action_vector, reward_vector) in exp_gen:
        print_progress_bar(exp_idx + 1, 100)

        # add experiment ID
        experiment_ID = f'[Experiment={exp_idx}]-[subject={_subject_ID}]-[exp_class={_experiment_class}]'

        # run experiment
        _result = run_experiment_v2(
            _subject_ID,
            _experiment_class,
            qsd,
            action_vector,
            reward_vector,
            experiment_ID=experiment_ID,
            drop_n_trails=drop_n_trails)

        if debug_mode:
            [print(f'{k:<25} - {str(v):<25} - {type(v)}') for k,v in _result.items()]

        update_results(_result, results)  # external store

        if mlflow_tracking:
            with mlflow.start_run(run_name=experiment_ID, nested=True):

                # extract experiment data
                _params = {k: str(v) for k, v in _result.items() if k in sort_mlflow_params.params}
                _metrics = {k: v for k, v in _result.items() if k in sort_mlflow_params.metrics}

                # extract tuple of parmaters 
                for _tuple_of_metrics in sort_mlflow_params.metrics_tuple:
                    log_tuple_metric(mlflow, _tuple_of_metrics, _result[_tuple_of_metrics])

                # Log parameters and metrics to MLFlow
                mlflow.log_param(key='experiment_ID', value=experiment_ID)
                mlflow.log_params(_params)
                mlflow.log_metrics(_metrics)

    print(results)
    print('Runtime complete :).')
    return pd.DataFrame(results)
