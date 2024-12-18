import pandas as pd
import os

# Update with the location of the logs
logs_folder_pilot = '/Users/natapost/Documents/data/perceptual_learning/pilot/processed/logs_to_csv_pilot'
logs_folder_motivation = '/Users/natapost/Documents/data/perceptual_learning/motivation/processed/logs_to_csv_motivation'

# Pilot
files = os.listdir(logs_folder_pilot)
stages = ['ams1', 'ams2']

participants = []
ams1s = []
ams2s = []
for file in files:
    #print('participant', file.split('.')[0])
    participants.append(int(file.split('.')[0]))
    df = pd.read_csv(logs_folder_pilot + '/' + file)

    ams1s.append(round(df[df.stage == 'ams1'].correct.mean(),2))
    ams2s.append(round(df[df.stage == 'ams2'].correct.mean(),2))
    
    #break
    
df = pd.DataFrame()
df['participant'] = participants
df['ams1'] = ams1s
df['ams2'] = ams2s

difficulty_df = pd.read_csv('/Users/natapost/Documents/data/perceptual_learning/pilot/processed/difficult_ams_pilot.csv', index_col = 0)
df = pd.merge(df, difficulty_df, on='participant')

df = df.sort_values(by = 'participant')

# Un-comment the line below to save the file
#df.to_csv('ams_performance_pilot.csv')
print(df)


# Motivation

files = os.listdir(logs_folder_motivation)
stages = ['ams1', 'ams2']

participants = []
ams1s = []
ams2s = []
for file in files:
    if file[0] == '.':
        continue
    #print(file)
    #print('participant', file.split('.')[0])
    participants.append(int(file.split('.')[0]))
    df = pd.read_csv(logs_folder_motivation + '/' + file)

    ams1s.append(round(df[df.stage == 'ams1'].correct.mean(),2))
    ams2s.append(round(df[df.stage == 'ams2'].correct.mean(),2))
    
    #break
    
df = pd.DataFrame()
df['participant'] = participants
df['ams1'] = ams1s
df['ams2'] = ams2s

difficulty_df = pd.read_csv('/Users/natapost/Documents/data/perceptual_learning/motivation/processed/difficult_ams_motivation.csv', index_col = 0)
df = pd.merge(df, difficulty_df, on='participant')

df = df.sort_values(by = 'participant')
#df.to_csv('ams_performance_motivation.csv')