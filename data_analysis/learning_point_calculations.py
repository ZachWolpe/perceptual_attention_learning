# The learning point is a mean between the first time hdi crossed 0.5 line and the last time. 
# For most of the participants that would be the same point.
# 

import os
import pandas as pd
import numpy as np

# FUNCTIONS

def get_participant_id(file_name):
    return file_name[:2]

def get_all_participants_id(logs_folder):
    participants = []
    for file in os.listdir(logs_folder):
        if file[0] == '.':
            continue
        participants.append(get_participant_id(file))
    return sorted(list(set(participants)))

def merge_learning_points_with_difficulty_ams(learning_points_df, difficulty_ams_file):
    difficulty_df = pd.read_csv(difficulty_ams_file, index_col = 0)
    return pd.merge(learning_points_df, difficulty_df, on='participant', how='inner') 

def get_learning_point(logs_folder, difficulty_ams_file, skip_list):
    
    participants = get_all_participants_id(logs_folder)
    learning_points_df = pd.DataFrame(columns = ['participant','ids1','eds1','ids2','eds2'])
    stages = ['ids1', 'eds1', 'ids2', 'eds2']
    
    for participant in participants:
        #print('participant', participant)
        if participant in skip_list: 
            continue
        new_row = [int(participant)]
        for stage in stages:
            file_name = logs_folder + '/' + participant + '_' + stage + '.csv'
            df = pd.read_csv(file_name)
            column = stage + '_hdi_3'
            #df['mov_average'] = df[column].rolling(window=window,center=True).mean()
            over_threshold = df[column] > 0.5
            index = over_threshold.idxmax() if over_threshold.any() else float('nan')
            learning_point_first = df.trial_n[index]
            for i in range(len(over_threshold)):
                if over_threshold[i:].all(): 
                    learning_point_last = int(df.trial_n[i])
                    learning_point = (learning_point_first + learning_point_last)/2
                    break
                learning_point = np.nan
            #learning_point = (learning_point_first + learning_point_last)/2
            new_row.append(learning_point)
        learning_points_df.loc[len(learning_points_df)] = new_row

    learning_points_df = merge_learning_points_with_difficulty_ams(learning_points_df, difficulty_ams_file)
        
    return learning_points_df     


pilot = True # False - motivation

# Replace the file addresses
if pilot:
    # for pilot
    skip_list = ['03','08','09', '11', '21'] 
    difficult_ams_file = '/Users/natapost/Documents/data/perceptual_learning/pilot/processed/difficult_ams_pilot.csv'
    logs_folder = '/Users/natapost/Documents/data/perceptual_learning/pilot/processed/bayesian_pilot/outcomes_binomial_beta_prior_all_trials/tables'
else:
    # for motivation
    skip_list = ['05','12', '18'] # for motivation
    difficult_ams_file = '/Users/natapost/Documents/data/perceptual_learning/motivation/processed/difficult_ams_motivation.csv'
    logs_folder = '/Users/natapost/Documents/data/perceptual_learning/motivation/processed/bayesian_motivation/outcomes_binomial_beta_prior_all_trials_motivation/tables'

learning_points_df = get_learning_point(logs_folder, difficult_ams_file, skip_list)
print(learning_points_df)

if pilot:
    learning_points_df.to_csv('learning_points_pilot_average.csv')
else:
    learning_points_df.to_csv('learning_points_motivation_average.csv')