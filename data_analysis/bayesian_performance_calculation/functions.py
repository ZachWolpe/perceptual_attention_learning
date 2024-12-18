import pymc as pm

import math
import matplotlib.pyplot as plt


def getStageIndex(df):
    stage_indeces = []
    stage = ''
    for index,row in df.iterrows():
        if row.stage != stage:
            stage = row.stage
            new_index = 1
        else:
            new_index = new_index + 1
        stage_indeces.append(new_index)
    df['stage_trial'] = stage_indeces
    return df


import math

def graph_plotting(trial_n, means, hdi_3_lower, hdi_97_higher, outcomes, participant, stage, go_trials, title, folder):
    
    point_mean_found = False
    for i in range(len(outcomes)):
        if means[i] >= 0.8:
            point_mean_x = trial_n[i]
            point_mean_y = means[i]
            point_mean_found = True
            break
    point_hdi_3_found = False
    for i in range(len(outcomes)):
        #print(i)
        #print(hdi_3_lower)
        if hdi_3_lower[i] > 0.5:
            point_hdi_3_x = trial_n[i]
            point_hdi_3_y = hdi_3_lower[i]
            point_hdi_3_found = True
            break
    point_hdi_97_found = False
    for i in range(len(outcomes)):
        if hdi_97_higher[i] < 0.5:
            point_hdi_97_x = trial_n[i]
            point_hdi_97_y = hdi_97_higher[i]
            point_hdi_97_found = True
            break
    #print('point mean', point_mean_x, point_mean_y)
    
    plt.figure()
    plt.plot(trial_n, means, marker='o', linestyle='-', color='blue', label='means')
    plt.plot(trial_n, hdi_3_lower, marker='.', linestyle='--', color='grey', label='hdi 3% lower bound')
    plt.plot(trial_n, hdi_97_higher, marker='.', linestyle='--', color='grey', label='hdi 97% higher bound')
    
    plt.plot(trial_n, [0.5]*len(trial_n), linestyle='--', color='red', label='threshold')
    plt.plot(trial_n, [0.8]*len(trial_n), linestyle='--', color='red', label='threshold')
    plt.scatter(trial_n, outcomes, marker='*', color='green', label='performance')
    plt.scatter(go_trials, [1.05]*len(go_trials), marker='*', color='red', label='go_trials')
    
    if point_mean_found:
        plt.scatter(point_mean_x, point_mean_y, marker='o', color='black')
        plt.annotate(f'trial {point_mean_x}', xy=(point_mean_x, point_mean_y), xytext=(point_mean_x + 0.1, point_mean_y + 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 )
    if point_hdi_3_found:
        plt.scatter(point_hdi_3_x, point_hdi_3_y, marker='o', color='black')
        plt.annotate(f'trial {point_hdi_3_x}', xy=(point_hdi_3_x, point_hdi_3_y), xytext=(point_hdi_3_x - 0.1, point_hdi_3_y - 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 )
    if point_hdi_97_found:
        plt.scatter(point_hdi_97_x, point_hdi_97_y, marker='o', color='black')
        plt.annotate(f'trial {point_hdi_97_x}', xy=(point_hdi_97_x, point_hdi_97_y), xytext=(point_hdi_97_x - 0.1, point_hdi_97_y - 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 )
    # Adding labels and title
    plt.xlabel('trial number')
    #plt.ylabel('l')
    plt.title(f'{participant} {stage} {title}')
    plt.xlim(0,max(trial_n)+1)
    plt.ylim(0, 1.1)
    ticks = [0]
    #ticks_n = 19
    #d = (max(trial_n))/ticks_n
    for i in range(math.ceil(max(trial_n)/60)*6):
        ticks.append((i+1)*10)
    #ticks.append(max(trial_n))
    plt.xticks(ticks)

    # Show plot
    plt.legend()
    plt.savefig(folder + '/plots/' + participant + '_' + stage + '_' + title + '.png')
    plt.show()
    


def distribution_update(observed_data):
    
    print(f'observed success: {sum(observed_data)}, fails: {len(observed_data)-sum(observed_data)}')
    with pm.Model() as model:
        # Prior distribution for guessing individuals
        prior = pm.Beta('performance', alpha=1, beta=1, testval=0.5)  # Prior belief: Probability of success is 0.5

        # Likelihood function for observed data
        likelihood_observed = pm.Bernoulli('likelihood_observed', p=prior, observed=observed_data)

        # Run MCMC sampling
        trace = pm.sample(2000, tune=1000)

    # Get summary statistics
    summary = pm.summary(trace, var_names=['performance'])
    print(summary)

    # Extract the mean
    mean = summary.loc['performance', 'mean']
    hdi_3 = summary.loc['performance', 'hdi_3%']
    hdi_97 = summary.loc['performance', 'hdi_97%']
    #print('hdi_3%', hdi_3)

    # Set the threshold
    threshold = 0.5
    threshold_for_hdi_3 = 0.5

    # Print the result
    '''
    if mean > threshold:
        print(f"The mean of the posterior ({mean:.2f}) is higher than the threshold ({threshold}).")
    else:
        print(f"The mean of the posterior ({mean:.2f}) is NOT higher than the threshold ({threshold}).")

    if hdi_3 > threshold_for_hdi_3:
        print(f"The lower bound of HDI of the posterior ({mean:.2f}) is higher than the threshold ({threshold_for_hdi_3}).")
    else:
        print(f"The lower bound of HDI of the posterior ({mean:.2f}) is NOT higher than the threshold ({threshold_for_hdi_3}).")
    '''
    return mean, hdi_3, hdi_97