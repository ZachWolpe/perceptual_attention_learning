# Creates a table with resulting nogo in both staircase sessions

import pandas as pd
import os

not_set = True
while not_set:
    answer = input('Are the calculations made on ultan server? (yes for ultan, no for local machine) ')
    if answer == 'yes':
        path_pilot = "/home/arkku/group/hipercog/dynocog/perceptual_learning/raw/"
        path_motivation = "/home/arkku/group/hipercog/dynocog/perceptual_learning/raw_motivation/"
        
        #path_for_data = '/home/arkku/group/hipercog/dynocog/perceptual_learning/processed/logs_to_csv/'
        not_set = False
    elif answer == 'no':
        path_pilot = "/Users/natapost/Documents/data/perceptual_learning/pilot/raw/"
        path_motivation = "/Users/natapost/Documents/data/perceptual_learning/motivation/raw/"
        #path_for_data = "/Users/natapost/Documents/perceptual_learning_drafts/processed_data/"
        not_set = False
    else:
        print('try again')
        break

skip_list_pilot = ['12']
skip_list_motivation = ['05']

def staircase_conversion_df_calculation(path, skip_list):
    file_names = []

    first_layer = True
    for root, dirs, files in os.walk(path):
        if first_layer:
            participants_folders = dirs
            first_layer = False

    print('participant_folders:', participants_folders)

    participants = []
    difficult_ams_list = []
    go_AUD_list = []
    go_VIS_list = []
    resulting_nogo_AUD_list = []
    resulting_nogo_VIS_list = []


    parts = ['part2', 'part3']
    for folder in participants_folders:
        #print(' ')
        participant = folder[-2:]
        #print('participant ', participant)
        if participant == '00' or participant == 'ss' or participant in skip_list:
            continue
        participants.append(participant)
        files = os.listdir(path + folder)
        data_parts = []
        for part in parts:
            #print('part ', part)
            resulting_nogo = -100
            for file in files:
                if 'log' in file and part in file:
                    log_file = path + folder + '/' + file
                    with open(log_file) as f:
                        #print(log_file)
                        lines = f.readlines()
                        for line in lines:
                            if 'relevant_modality:' in line:
                                #print(line)
                                #print(line.split(': ')[1][:3], '!')
                                relevant_modality = line.split(': ')[1][:3]
    
                            
                            if 'transition from staircase' in line:
                                #print(line)
                                #print('resulting_nogo =', resulting_nogo)
                                if relevant_modality == 'VIS':
                                    resulting_nogo_VIS_list.append(resulting_nogo)
                                    go_VIS_list.append(go)
                                else:
                                    resulting_nogo_AUD_list.append(resulting_nogo)
                                    go = int(go.split('_')[1].split('.')[0])
                                    go_AUD_list.append(go)
                                #print('---------------------------------')
                                break
                            if 'setting' in line and 'go' in line:
                                #print(line)
                                
                                #print(line.split('go=')[1].split(',')[0])
                                go = line.split('go=')[1].split(',')[0]
                                #print('resulting_nogo =', resulting_nogo)
                            if 'resulting_nogo' in line:
                                resulting_nogo = int(line.split(':')[1])

        #break

        
    df = pd.DataFrame()
    df['participant'] = participants
    df['go_AUD'] = go_AUD_list
    df['resulting_nogo_AUD'] = resulting_nogo_AUD_list
    df['go_VIS'] = go_VIS_list
    df['resulting_nogo_VIS'] = resulting_nogo_VIS_list
    df = df.sort_values(by='participant')
    return df
    
    
    
df = staircase_conversion_df_calculation(path_pilot, skip_list_pilot)
df.to_csv('staircase_conversion_pilot.csv')

df = staircase_conversion_df_calculation(path_motivation, skip_list_motivation)
df.to_csv('staircase_conversion_motivation.csv')
