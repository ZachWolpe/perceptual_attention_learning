"""
--------------------------------------------------------------------------------
launch_experiments_filter_and_select_model.py

> Filter results.
> Select best model based on BIC.
> Save results.

PARTIAL DUPLICATE OF .SRC/launch_experiments.py -- Final Structure TBD.
PARTIAL DUPLICATE OF .SRC/launch_experiments.py -- Final Structure TBD.
PARTIAL DUPLICATE OF .SRC/launch_experiments.py -- Final Structure TBD.
PARTIAL DUPLICATE OF .SRC/launch_experiments.py -- Final Structure TBD.
PARTIAL DUPLICATE OF .SRC/launch_experiments.py -- Final Structure TBD.
PARTIAL DUPLICATE OF .SRC/launch_experiments.py -- Final Structure TBD.

: 30 Sep 2024
: zach.wolpe@medibio.com.au
--------------------------------------------------------------------------------
"""


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
