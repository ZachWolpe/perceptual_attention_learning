"""
--------------------------------------------------------------------------------
launch_experiments_config_runtime.py

> Configure Experiments.
> Set up a local directory for MLflow tracking.

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

import numpy as np
import mlflow
import os


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
