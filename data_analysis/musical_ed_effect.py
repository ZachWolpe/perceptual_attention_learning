import pandas as pd
import os
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt




def shifcost_calc(learning_points): # IDS1 is used
    learning_points['shiftcost1'] = learning_points['eds1'] - learning_points['ids1']
    learning_points['shiftcost2'] = learning_points['eds2'] - learning_points['ids1']
    return learning_points



print('################### CALC ######################')


# Data merging

data_folder_pilot = '/Users/natapost/Documents/data/perceptual_learning/pilot/raw/'
data_folder_motivation = '/Users/natapost/Documents/data/perceptual_learning/motivation/raw/'

question_file_pilot = 'Perceptuallearning_DATA_2024-06-17_1419_background_only.csv'
question_file_motivation = 'Perceptuallearningmo_DATA_2024-06-17_1421_background_only.csv'

q_df_pilot = pd.read_csv(data_folder_pilot + question_file_pilot, index_col=0)
q_df_pilot = q_df_pilot.drop(q_df_pilot.index[23])
q_df_pilot = q_df_pilot.drop(q_df_pilot.index[0])

mus_vis_pilot = q_df_pilot.iloc[:, [2,7,8]]
mus_vis_pilot = mus_vis_pilot.rename(columns={'participant_id': 'participant'})
mus_vis_pilot['participant'] = mus_vis_pilot['participant'].apply(lambda x: int(x))

q_df_motivation = pd.read_csv(data_folder_motivation + question_file_motivation, index_col=0)
q_df_motivation.iloc[9,3] = 8

mus_vis_motivation = q_df_motivation.iloc[:, [2,7,8]].dropna()
mus_vis_motivation = mus_vis_motivation.rename(columns={'participant_id': 'participant'})
mus_vis_motivation['participant'] = mus_vis_motivation['participant'].apply(lambda x: int(x))


experiment = "pooled"

processed_data_folder_pilot = '/Users/natapost/Documents/data/perceptual_learning/pilot/processed'
modality_df_pilot = pd.read_csv(processed_data_folder_pilot + '/modalities_pilot.csv', index_col=0)
learning_points_file_pilot = '/Users/natapost/Documents/data/perceptual_learning/pilot/processed/learning_points_pilot_between.csv'
skip_list_pilot = [2,7,19,9,21]

processed_data_folder_motivation = '/Users/natapost/Documents/data/perceptual_learning/motivation/processed'
modality_df_motivation = pd.read_csv(processed_data_folder_motivation + '/modalities_motivation.csv', index_col=0)
learning_points_file_motivation = '/Users/natapost/Documents/data/perceptual_learning/motivation/processed/learning_points_motivation_between.csv'
skip_list_motivation = [17]

learning_points_p = pd.read_csv(learning_points_file_pilot, index_col=0)


learning_points_p = pd.merge(learning_points_p, modality_df_pilot, on='participant')
learning_points_p = pd.merge(learning_points_p, mus_vis_pilot, on='participant')
learning_points_p = learning_points_p[~learning_points_p['participant'].isin(skip_list_pilot)]
learning_points_p['participant'] = learning_points_p['participant'].apply(lambda x: str(int(x)) + '_' + 'p')


learning_points_m = pd.read_csv(learning_points_file_motivation, index_col=0)
learning_points_m = pd.merge(learning_points_m, modality_df_motivation, on='participant')
learning_points_m = pd.merge(learning_points_m, mus_vis_motivation, on='participant')
learning_points_m = learning_points_m[~learning_points_m['participant'].isin(skip_list_motivation)]
learning_points_m['participant'] = learning_points_m['participant'].apply(lambda x: str(int(x)) + '_' + 'm')

learning_points = pd.concat([learning_points_p, learning_points_m], ignore_index=True)
learning_points = shifcost_calc(learning_points)

# to long format
values = []
first_modalities = []
modalities = []
#stages_to_df = []
stages = ['ids1', 'eds1', 'ids2', 'eds2']
mus = []
vis = []
shifts = []
orders = []
modality_shifts = []

for index, row in learning_points.iterrows():
    first_modality = row.modality
    first_modalities.append(first_modality)
    first_modalities.append(first_modality)
    
    values.append(row.shiftcost1)
    orders.append('first')
    if first_modality == 'AUD':
        modality_shifts.append('AUD_VIS')
    else:
        modality_shifts.append('VIS_AUD')
    
    values.append(row.shiftcost2)
    orders.append('second')
    if first_modality == 'VIS':
        modality_shifts.append('AUD_VIS')
    else:
        modality_shifts.append('VIS_AUD')
    
    
    if row.musical > 0:
        mus.append('mus')
        mus.append('mus')
    else:
        mus.append('no_mus')
        mus.append('no_mus')
    if row.visual > 0:
        vis.append('vis')
        vis.append('vis')
    else:
        vis.append('no_vis')
        vis.append('no_vis')

df_long = pd.DataFrame()
#df_long['stage'] = stages_to_df
df_long['first_modality'] = first_modalities
df_long['musical_ed'] = mus
df_long['visual_ed'] = vis

df_long['order'] = orders
df_long['value'] = values
df_long['modality_shift'] = modality_shifts

model = sm.OLS.from_formula('value ~ musical_ed + modality_shift + first_modality + order', data=df_long).fit()
print(model.summary())

data = {'musical_ed': list(df_long['musical_ed']),
        'modality_shift': list(df_long['modality_shift']),
        'shiftcost': list(df_long['value'])
       }


df = pd.DataFrame(data)
sns.boxplot(x='musical_ed', y='shiftcost', hue='modality_shift', data=df)
#sns.scatterplot(x='x', y='shiftcost', hue='order', data=data, marker='o', s=100, zorder=3)

handles, labels = plt.gca().get_legend_handles_labels()

plt.legend(handles=handles[:-2], labels=labels[:-2])

plt.title('Musical education effect on shiftcost')
#plt.savefig('boxplot_bayesian_' + title + '.png')
plt.show()

print('done')
