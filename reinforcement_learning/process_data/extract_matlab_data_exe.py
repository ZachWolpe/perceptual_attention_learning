"""
-------------------------------------------------------------------------------
extract_matlab_dataset_exe.py

Excute modules from extract_matlab_dataset.py

Modifaction Logs:
: 31 May 24     : zachcolinwolpe@gmail.com      : init
-------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import scipy.io
import logging
import os
import pickle
from extract_matlab_data import (
    convert_structured_array_to_df,
    convert_to_dict,
    fetch_matlab_data,
    extract_values_only,
    save_data,
    log_matlab_datastruct
)

# globals
PATH_TO_MATLAB_DATA = './data/raw_matlab_data/'
OUTPUT_PATH = './data/processed_data/'

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    logging.debug('...........................\n\n\n')

    # for debugging
    # log_matlab_datastruct(PATH_TO_MATLAB_DATA)

    # fetch matlab data
    datasets, dataset_names = fetch_matlab_data(PATH_TO_MATLAB_DATA)

    for dataset, name in zip(datasets, dataset_names):
 
        if 'human' in name:
            metadata, action_vector = extract_values_only(
                dataset,
                int_value_keys=['participantID', 'SessionType', 'SessionNum', 'CondNum', 'ReleMode'],
                single_value_keys=['SessionType'],
                array_keys=['StimCode', 'RespCode'],
                array_id_vars=['participantID', 'SessionType']
            )
            save_data(name, metadata, action_vector, output_path=OUTPUT_PATH)

        # rat data
        else:
            metadata, action_vector = extract_values_only(
                datasets[0],
                int_value_keys=['ratID', 'SessionType', 'SessionNum', 'CondNum', 'ReleMode'],
                single_value_keys=['SessionType'],
                array_keys=['StimCode', 'RespCode'],
                array_id_vars=['ratID', 'SessionType']
                )
            save_data(name, metadata, action_vector, output_path=OUTPUT_PATH)
        logging.info('Run complete.')

