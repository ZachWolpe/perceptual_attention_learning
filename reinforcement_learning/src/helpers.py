"""
-------------------------------------------------------------------------------
helpers.py

Helper modules.

Modifaction Logs:
: 18.06.24          : zachcolinwolpe@gmail.com      : init
-------------------------------------------------------------------------------
"""

import logging
import pprint
import yaml


def load_config(path_to_yaml_config):
    with open(path_to_yaml_config, 'r') as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.FullLoader)


def log_rat_metadata(rat_metadata):

    print('--------------------------------------------------------------- \nmeta_data:')
    pprint.pprint(rat_metadata.info())
    print('\nExample:')
    pprint.pprint(rat_metadata.head(2))
    print('---------------------------------------------------------------')

    experiment_stats = f"""
    ---------------------------------------------------------------
    No. Rats (IDs):     {len(rat_metadata.ratID.unique())}
    ---------------------------------------------------------------
    """
    print(experiment_stats)

    exeriments_per_session = rat_metadata.groupby(['SessionType']).count()[['ratID']]
    exeriments_per_session.columns = ['No. Experiments per SessionType']
    print(exeriments_per_session)


def log_sequence_data(sequence_data, log=True):
    # extract sequence data
    stim_code = sequence_data['StimCode']
    resp_code = sequence_data['RespCode']

    # log size
    msg = f"""
        StimCode (len):     {len(stim_code)}
        RespCode (len):     {len(resp_code)}
        StimCode (keys):    {stim_code.keys()}
        RespCode (keys):    {resp_code.keys()}

    """
    logging.info(msg)
    return stim_code, resp_code