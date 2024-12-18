"""
-------------------------------------------------------------------------------
query_dataset.py

Functions to help query the data.

Infer Action & Reward:
----------------------

    Action & reward is inferred from a `response` vector.
        - `Reward`: [0,1] (1=correct reject, 2=hit, 3=false alarm, 4=omission)
        - `Action`: [0,1] (1=nogo, 2=go, 3=go, 4=omissions)

Modifaction Logs:
: 31.05.24          : zachcolinwolpe@gmail.com      : init
-------------------------------------------------------------------------------
"""

from pprint import pprint
import pandas as pd
import numpy as np
import logging
import pickle
import os


# BUG::: REDEFINED. Finalize project format.
def load_processed_data(path_to_processed_files):
    """Load processed data from .pkl & .csv files"""
    results = {}
    for file in os.listdir(path_to_processed_files):
        if file.endswith('.pkl'):
            with open(path_to_processed_files + file, 'rb') as f:
                data = pickle.load(f)
                results[file] = data
        elif file.endswith('.csv'):
            data = pd.read_csv(path_to_processed_files + file)
            results[file] = data
        elif file.endswith('.gitkeep'):
            pass
        else:
            raise ValueError(f'File type not supported: {file}.')
    return results


def load_data(config, experiment_class='human_pilot_experiment', log=True):
    # fetch file locations
    _processed_data_loc = config['processed_data']['location']
    _data_file = config['processed_data'][experiment_class]['trials']
    _metadata_file = config['processed_data'][experiment_class]['metadata']

    # load data
    data = load_processed_data(_processed_data_loc)
    sequence_data = data[_data_file]
    meta_data = data[_metadata_file]

    # extract sequence data
    StimCode = sequence_data['StimCode']
    RespCode = sequence_data['RespCode']

    if log:
        logging.info(f'StimCode ({len(StimCode)}): {np.unique(list(StimCode.values())[0])}')
        logging.info(f'RespCode ({len(RespCode)}): {np.unique(list(RespCode.values())[0])}')

    return (
        data,
        sequence_data,
        meta_data,
        StimCode,
        RespCode
    )


class QuerySequenceData:
    def __init__(self, StimCode, RespCode):
        """
        Class to manage the sequence data.

        Args:
        -----
            StimCode (dict): Dictionary of stimulus data
            RespCode (dict): Dictionary of response data

        Properties:
        -----------
            _stimCode (dict): A subject of <self.StimCode>, after applying filters & transformations.
            _respCode (dict): A subject of <self.RespCode>, after applying filters & transformations.
        """
        self.StimCode = StimCode
        self.RespCode = RespCode
        self._stimCode = None
        self._respCode = None

    def reset_properties(self):
        self._stimCode = self.StimCode
        self._respCode = self.RespCode

    def filter_sequences(self, subjectID=None, sessionType=None, sessionNum=None,
                         update_existing_stim_resp=False):
        """
        Extract all experiments that match (subjectID, sessionType, sessionNumber)

        if None: return all experiments.

        Update <self._stimCode> and <self._respCode> with the filtered data.

        If update_existing_stim_resp: update <self._stimCode> and <self._respCode> with the filtered data.
            This allows you to concatenate multiple experiments into a single list of stim and resp data.

        Args:
        -----
            subjectID (int or None): Subject ID
            sessionType (str or None): Session type
            sessionNumber (int or None): Session number
            update_existing_stim_resp (bool): Update <self._stimCode> and <self._respCode> with the filtered data or initialize new sets.

        """
        # store session keys
        self.subjectID = subjectID
        self.sessionType = sessionType
        self.sessionNum = sessionNum

        # initialize values
        if self._stimCode is None or (not update_existing_stim_resp):
            _stimCode = {}
        if self._respCode is None or (not update_existing_stim_resp):
            _respCode = {}

        if update_existing_stim_resp and (self._stimCode is not None):
            _stimCode = self._stimCode
        if update_existing_stim_resp and (self._respCode is not None):
            _respCode = self._respCode

        for (_subjectID, _sessionID, _sessionNum), _data in self.StimCode.items():

            # extraction filters
            subjectID_filter = False
            sessionType_filter = False
            sessionNum_filter = False

            # apply filter
            if (subjectID is None) or (_subjectID == subjectID):
                subjectID_filter = True

            if (sessionType is None) or (_sessionID == sessionType):
                sessionType_filter = True

            if (sessionNum is None) or (_sessionNum == sessionNum):
                sessionNum_filter = True

            if subjectID_filter and sessionType_filter and sessionNum_filter:
                _stimCode[(_subjectID, _sessionID, _sessionNum)] = _data
                _respCode[(_subjectID, _sessionID, _sessionNum)] = self.RespCode[(_subjectID, _sessionID, _sessionNum)]

        self._stimCode = _stimCode
        self._respCode = _respCode
        return self

    def return_experiment_vectors(self, subjectID=None, sessionType=None, sessionNumber=None):
        # _stimCode, _respCode = self.query(ratID, sessionType, sessionNumber)
        key = (subjectID, sessionType, sessionNumber)
        return self.StimCode[key], self.RespCode[key]

    # infer action / reward pairs
    def infer_action_reward_pairs(self, response_vector=None):
        """
        Given a signal response vector, infer the action and reward pairs.

        Args:
        -----
            response_vector (list): List of response signals. [1, 2, 3, 4]
                if response_vector is None: --> use self._respCodeFlat
        """
        _reward = []
        _action = []

        if response_vector is None:
            response_vector = self._respCodeFlat

        for resp in response_vector:
            if resp == 1:
                # _action.append(1)
                _action.append(0)
                _reward.append(1)
            elif resp == 2:
                # _action.append(0)
                _action.append(1)

                _reward.append(1)
            elif resp == 3:
                _action.append(1)
                _reward.append(0)
            elif resp == 4:
                _action.append(0)
                _reward.append(0)

        self._reward = _reward
        self._action = _action

        return self

    def extract_stim_resp_data(self, _stimCode=None, _respCode=None):
        """
        Concatenate multiple experiments into a single list of stim and resp data.

        Args:
        -----
            : _stimCode (dict or None): Dictionary of stim data. If None use self._stimCode.
            : _respCode (dict or None): Dictionary of resp data If None use self._respCode.

        Computes:
        --------
            : self._stimCodeFlat (list): Flattened list of stim data
            : self._respCodeFlat (list): Flattened list of resp data
        """
        if _stimCode is None:
            _stimCode = self._stimCode

        if _respCode is None:
            _respCode = self._respCode

        stimData = []
        respData = []
        for (_stimKey, _stimData), (_respKey, _respData) in zip(_stimCode.items(), _respCode.items()):
            stimData.extend(_stimData)
            respData.extend(_respData)

        self._stimCodeFlat = stimData
        self._respCodeFlat = respData
        return self


# Start :-> with 1 Rat, 1 Session Type
class query_metadata:

    @staticmethod
    def query(meta_data, key=None, rat_ID=None, SessionType=None, SessionNum=None):
        """
        Query metadata based on key: (rat_ID, SessionType, SessionNum).
        """
        if key is not None:
            rat_ID = key[0]
            SessionType = key[1]
            SessionNum = key[2]

        if rat_ID is not None:
            meta_data = meta_data[(meta_data.ratID == rat_ID)]

        if SessionType is not None:
            meta_data = meta_data[(meta_data.SessionType == SessionType)]

        if SessionNum is not None:
            meta_data = meta_data[(meta_data.SessionNum == SessionNum)]

        return meta_data

    @staticmethod
    def sort(meta_data, by):
        return meta_data.sort_values(by=by)


def extract_reward_action(response):
    """
    Extract reward and action vector from the response vector.

    Extraction:
    -----------

        : Reward: [0,1] (1=correct reject, 2=hit, 3=false alarm, 4=omission)
            reward=1 if response=[1,2]
            reward=0 if response=[3,4]

        : Action: [0,1] (1=nogo, 2=go, 3=go, 4=omissions)
            action=1 (go)  if response=[2,3]
            action=0 (nogo) if response=[1,4]

        Response vector encoding: (1=correct reject, 2=hit, 3=false alarm, 4=omission)
        Reward`: [0,1] (reward=1 if response=[1,2], else 0)
        Action`: [0,1] (action=1(go) if response=[1,3], else 0)

    Returns:
    --------

        : reward: list
        : action: list
    """

    _reward = [int(i == 1 or i == 2) for i in response] 
    _action = [int(i == 1 or i == 3) for i in response]

    return _reward, _action


def log_rat_metadata(meta_data):
    """Quick overview of the metadata structure."""
    print('--------------------------------------------------------------- \nmeta_data:')
    pprint.pprint(meta_data.info())
    print('\nExample:')
    pprint.pprint(meta_data.head(2))
    print('---------------------------------------------------------------')

    experiment_stats = f"""
    ---------------------------------------------------------------
    No. Rats (IDs):     {len(meta_data.ratID.unique())}
    ---------------------------------------------------------------
    """
    print(experiment_stats)

    exeriments_per_session = meta_data.groupby(['SessionType']).count()[['ratID']]
    exeriments_per_session.columns = ['No. Experiments per SessionType']
    print(exeriments_per_session)
