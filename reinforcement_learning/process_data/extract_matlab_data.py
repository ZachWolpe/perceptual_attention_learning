"""
-------------------------------------------------------------------------------
extract_matlab_dataset.py

Extract data from a .mat file and save it as a
    - metadata (.csv)
    - action sequences (.pkl)

Data Schema

    % Make some docs
    info = {
    'SessionType',  {'EDS' 'EDS_BL_Easy' 'EDS_BL_Hard' 'IDS' 'IDS_BL_Easy' 'RE' 'Stair'}, {'Acronyms for each session type that occured'};
    'SessionNum',   [1 2 3 4 5 6 7 8 9 10 11 12 13], {'Consecutive number of each session within each session type'};
    'CondNum',      [1 2 3 4 5 6 7 8 9 10 11 12 13 14], {'Numeric ordering of occurence of conditions - same as numeric code for SessionType IFF these occured in same order for every rat.'};
    'ReleMode',     [0 1], {'Logical coding for relevant modality: 0=visual, 1=tone'};
    'StimCode',     [0 1], {'Logical coding for stimulus class: 0=nogo, 1=go'};
    'RespCode',     [1 2 3 4], {'Numeric coding for response type: 1=correct_rejection, 2=hit, 3=false_alarm, 4=omissions'}};

Modifaction Logs:
: 31 May 24     : zachcolinwolpe@gmail.com      : init
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import scipy.io
import logging
import os
import pickle
# logging.basicConfig(level=logging.INFO)


def convert_structured_array_to_df(data):
    """
    Convert structured numpy array to pandas DataFrame
    """
    cols = data.dtype.names
    df = pd.DataFrame({col: data[col].squeeze() for col in cols})
    return df


def convert_to_dict(data):
    """Convert structured numpy array to dictionary"""
    cols = data.dtype.names
    return {col: data[col].squeeze() for col in cols}


def fetch_matlab_data(path_to_data):
    """
    Fetch matlab data.

    Args:
        path_to_data (str): path to data

    Returns:
        datasets (list): list of datasets
        dataset_names (list): list of dataset names
    """
    datasets = []
    dataset_names = []
    logging.info(f'fetching data from: {path_to_data}')
    for file in os.listdir(path_to_data):
        if file.endswith('.mat'):
            dataset_names.append(file)
            mat = scipy.io.loadmat(path_to_data + file)
            for key in ['dat', 'human_data']:
                if key in mat.keys():
                    data = mat[key]
                    data_dict = convert_to_dict(data)
                    datasets.append(data_dict)
                    logging.info('  available keys: ' + str(data_dict.keys()))
    return datasets, dataset_names


def extract_values_only(dataset, int_value_keys, single_value_keys, array_keys, array_id_vars=['ratID', 'SessionType'], array_id_types=[int, str]):
    """Extract values from the .mat dataset"""

    def _log_data_for_debugging():
        pass

    def _extract_actions(data, action_var, array_id_vars=array_id_vars, array_id_types=array_id_types):

        assert array_id_types == [int, str], 'array_id_types does not match its expected value. Although implemented, this function should be examined before use.'

        def _extract_key_value(key, id_type):
            """Internal function: extract the correct key value."""
            if isinstance(id_type, int):
                return key[0][0]
            
            if isinstance(id_type, str):
                return key[0]
            raise NotImplementedError('Type not expected.')
        
        # extract keys
        # keys: (ID, SessionID, SessionNum)
        results = {}
        data_arrs = [data[a] for a in array_id_vars]

        # keep track of key indices ------------>
        _key_count = {}
        
        def _insert_key_count(key, _key_count):
            if key in list(_key_count.keys()):
                _key_count[key] += 1
            else:
                _key_count[key] = 0
            return _key_count
        # keep track of key indices ------------>

        for keys_vals in zip(*data_arrs, data[action_var]):
            keys = keys_vals[:-1]
            val = keys_vals[-1]
            
            # new method: account for multiple actions per key.
            _array = np.array([i[0] for i in val])
            
            # Keys: (ID, SessionID, SessionNum)
            key = [_extract_key_value(k, i) for k, i in zip(keys, array_id_types)]

            key = tuple(key)
            _key_count = _insert_key_count(key, _key_count)
            _key = (*key, _key_count[key])

    
            # Insert (debugging...)
            # if _key[0] == 4151:
            #     print('_key: ', _key)
            # # print('_array: ', _array)
            results[_key] = _array

        return results

    extraction_function = {
        'int_value': lambda dataset, key: [i[0][0] for i in dataset[key]],
        'single_value': lambda dataset, key: [i[0] for i in dataset[key]],
        'array': lambda dataset, key: _extract_actions(dataset, key)
    }

    results = {}
    action_vector = {}

    logging.debug('.........\n\n\n')

    # Check if the current logging level is set to DEBUG
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        _log_data_for_debugging()

        logging.debug('<int_value>')
        _all_keys = [key for subkey in [int_value_keys, single_value_keys, array_keys] for key in subkey]

        for key in _all_keys:
            logging.debug(f'\n\n {key}')
            _ds = dataset[key]
            np.shape(_ds)
            # logging.debug(_ds[0])
        logging.debug('End.')

    else:

        # single numeric (int) values
        for key in int_value_keys:
            results[key] = extraction_function['int_value'](dataset, key)
            logging.debug(f'results[{key}]      ', results[key][:3])

        # save single values to metadata
        for key in single_value_keys:
            results[key] = extraction_function['single_value'](dataset, key)
            logging.debug(f'results[{key}]      ', results[key][:3])

        # Create dataframe that constitute single values
        metadata = pd.DataFrame(results)
        # logging.debug(metadata.head(2))
        # logging.debug(metadata.info())

        # save action data (vectors)
        for key in array_keys:
            action_vector[key] = extraction_function['array'](dataset, key)

        # safety checks: BUG
        # for _vectors in action_vector.values():
        #     print(_vectors)
        #     for key, id in _vectors.keys():
        #         assert metadata[metadata[array_id_var]==key].loc[id,:].shape[0] > 0, f"ID mismatch: {key} != {metadata[metadata[array_id_var]==key].iloc[id,:].shape[0]}"

        # logs
        logmsg = f"""
            ------------------------------------------
            Extract Dataset:

            metadata ({metadata.shape})
                : metadata columns: {metadata.columns}
            
            action_vector:
                : action_vector['StimCode'] ({len(action_vector['StimCode'])})
                : action_vector['RespCode'] ({len(action_vector['RespCode'])})

            Structure:
                :metadata (pd.DataFrame) containing single parameters (metadata) describing a particular run

            ------------------------------------------
            """
        logging.info(logmsg)

        return metadata, action_vector


def log_matlab_datastruct(path_to_data='./data/'):
    for file in os.listdir(path_to_data):
        if file.endswith('.mat'):
            mat = scipy.io.loadmat(path_to_data + file)
            print(mat.keys())
            msg = f"""
                    headers:    {mat['__header__']}
                    version:    {mat['__version__']}
                    globals:    {mat['__globals__']}
                """
            logging.info(msg)
            for key in ['dat', 'human_data']:
                if key in mat.keys():
                    data = mat[key]
                    logging.info(f'data.shape: {data.shape}')
                    mat.keys()


def save_data(dataset_name, metadata, action_vector, output_path):
    """Save metadata and action vector to .csv and .pkl files"""
    output_filename = dataset_name.split('.')[0]
    output_filename = f'{output_path}{output_filename}'
    metadata.to_csv(f'{output_filename}_metadata.csv', index=False)
    with open(f'{output_filename}_actions.pkl', 'wb') as f:
        pickle.dump(action_vector, f)
    logmsg = f"""
        Saved data to
            : {output_filename}_metadata.csv
            : {output_filename}_actions.pkl
    """
    logging.info(logmsg)


def load_processed_data(path_to_processed_files):
    """: MIGRATED TO .src.query_dataset.py"""
    pass