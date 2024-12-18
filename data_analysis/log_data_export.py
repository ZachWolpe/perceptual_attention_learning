print('importing')
import pandas as pd
import os
#import csv
print('imported')

not_set = True
while not_set:
    answer = input('Are the calculations made on ultan server? (yes for ultan, no for local machine) ')
    if answer == 'yes':
        path = "/home/arkku/group/hipercog/dynocog/perceptual_learning/raw/"
        path_for_data = '/home/arkku/group/hipercog/dynocog/perceptual_learning/processed/logs_to_csv/'
        not_set = False
    elif answer == 'no':
        path = "/Users/natapost/Documents/ultan_files/pl/raw/"
        path_for_data = "/Users/natapost/Documents/ultan_files/pl/logs_to_csv/"
        not_set = False
    else:
        print('try again')
        break


#Functions

def performance_calc(correct_column, window):
    performance = [float('nan')]*(window-1)
    l = len(correct_column)
    for i in range(l - window+1):
        p = sum(correct_column[i:(i+window)])/window
        performance.append(p)
    
    return performance


def get_data_from_log(log_file):
    data = pd.DataFrame()
    trials = []
    go_list = []
    relevant_modality = []
    stages = []
    space_pressed_list = []
    correct_list = []
    times = []
    times_space = []
    timestamps = []
    timestamps_space = []
    reaction_times = []

    difficult_ams = ''

    stage = 'training'
    trial_on = False
    space_pressed = False

    blocks_info_not_read = True
    next_blocks_info = 0

    with open(log_file) as f:
        print(log_file)
        lines = f.readlines()

        for line in lines:

            if blocks_info_not_read:
                if next_blocks_info > 0:
                    
                    if ' 0' in line[28:]:
                        if line[0] == '2':
                            difficult_ams = 'ams1'
                        else:
                            difficult_ams = 'ams2'
                        blocks_info_not_read = False
                    next_blocks_info = next_blocks_info - 1


                if 'aud1' in line:
                    next_blocks_info = 6

            if 'utcTime:' in line:
                #print(line)
                t1 = float(line.split(' \t')[0])
                t2 = float(line.split('utcTime: ')[1])
                dt = t2-t1
                #print(t1,t2)
                #print('dt_________',dt)
            
            if 'break' in line and 'DATA' in line:
                #print(line)
                pass
            
            if 'relevant_modality:' in line:
                r_mod = line.split('relevant_modality: ')[1][:3]
                #print(line)
            if 'next_stimuli' in line:
                #print(line)
                trial_n = line.split('trial_number=')[1].split(',')[0]
                go = line.split('trial_go=')[1].split(',')[0]
            if 'transition to' in line:
                #print(line)
                stage = line.split('transition to ')[1][:-1]

            if 'trial' in line and 'DATA' in line and 'next' not in line and 'order' not in line:
                #print(line)
                trial_on = True
                space_pressed = False
                time_space = float('nan')
                timestamp_space = float('nan')
                reaction_time = float('nan')

            if 'grating_trial: autoDraw = True' in line:
                if trial_on:
                    time = float(line.split(' \t')[0])
                    #print(line)
                    timestamp = time + dt

            if 'Keypress: space' in line:
                #print(line)
                if trial_on:
                    space_pressed = True
                    time_space = float(line.split(' \t')[0])
                    timestamp_space = time_space + dt
                    reaction_time = time_space - time

            if 'response' in line:
                #print(line)

                correct = 0
                if go == str(space_pressed):
                    correct = 1

                trial_on = False
                
                times.append(time)
                times_space.append(time_space)
                timestamps.append(timestamp)
                timestamps_space.append(timestamp_space)
                reaction_times.append(reaction_time)
                trials.append(trial_n)
                go_list.append(go)
                relevant_modality.append(r_mod)
                stages.append(stage)
                space_pressed_list.append(space_pressed)
                correct_list.append(correct)
                  
    
    data['time_stimuli_start'] = times
    data['time_space'] = times_space
    data['timestamp_stimuli_start'] = timestamps
    data['timestamp_key_pressed'] = timestamps_space
    data['RT'] = reaction_times
    data['relevant_modality'] = relevant_modality
    data['stage'] = stages
    data['trial_n'] = trials
    data['go'] = go_list
    data['key_pressed'] = space_pressed_list
    data['correct'] = correct_list
    #data['performance_20wind'] = performance_calc(data.correct, 20)



    return data, difficult_ams




file_names = []

first_layer = True
for root, dirs, files in os.walk(path):
    if first_layer:
        participants_folders = dirs
        first_layer = False

print('participant_folders:', participants_folders)

participants = []
difficult_ams_list = []
parts = ['part1', 'part2', 'part3']
for folder in participants_folders:
    print(' ')
    participant = folder[-2:]
    print('participant ', participant)
    if participant == '00' or participant == 'ss':
        continue
    files = os.listdir(path + folder)
    data_parts = []
    for part in parts:
        print('part ', part)
        for file in files:
            if 'log' in file and part in file:
                print(file)
                data, difficult_ams = get_data_from_log(path + folder + '/' + file)
                data_parts.append(data)
    
    data_df = pd.concat(data_parts)
    filename = path_for_data + participant + '.csv'
    print('data is saved to: ', filename)
    data_df.to_csv(filename)

    participants.append(participant)
    difficult_ams_list.append(difficult_ams)

difficult_ams_df = pd.DataFrame()
difficult_ams_df['participant'] = participants
difficult_ams_df['difficult_ams'] = difficult_ams_list
difficult_ams_df.to_csv(path_for_data + '/difficult_ams.csv')
print('participant level data is saved to: difficult_ams.csv')

