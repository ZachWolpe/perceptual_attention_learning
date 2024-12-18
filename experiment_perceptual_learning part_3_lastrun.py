#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on toukokuu 07, 2023, at 17:22
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

import time
from psychopy import logging
import pandas as pd
import random

logging.console.setLevel(logging.DATA)

#timestamps logging
logging.data("utcTime: " + str(time.time()))

df_volume = pd.read_csv('sound_volume.csv')

#VARIABLES_1
df_stimuli = pd.read_csv('stimuli.csv')
noise_dur_list = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
intervals = pd.read_csv('intervals.csv')
block_n = 0
intervals_block = intervals.loc[block_n]
#intervals_block = intervals.iloc[block_n,:]
print(df_stimuli)
logging.data(df_stimuli)
df_trials = pd.DataFrame()

stages = ['previous', "staircase2", "ams2", "eds2"]
stage_number = 0
stage = stages[stage_number]
pass_stage = False
logging.data("stage " + str(stage))

#relevant_mod_only = True #for training only
relevant_mod_only = False

trials_in_block = 60 #has to be divisible by 4
mov_average = 20 #

index = 4 #index for the stimuli file
trial_number = 1 # trial order number
duration = 0.7 #feedback duration
correct_feedback = False
incorrect_feedback = False

go_correct = 0
nogo_correct = 0
performance_go = 0
performance_nogo = 0

#for progress bar
size = 0
x = 0.15-size/2
a = 0.3 #max length
progress = 0.1

#for staircase
correct_twice = False 
reversals = list()
converging = True
direction = "converging" # OR "diverging"

performance_record = list()
past_performance = 0
missed = 0

stat_text = "Useful info"

#FUNCTIONS
def get_volume(soundfile, df_volume = df_volume):
    sound_n = int(soundfile[13:16])
    volume = list(df_volume[df_volume['sound_n'] == sound_n].volume)[0]
    logging.data('got_volume:' + str(volume))
    return volume

def get_stair_sign(relevant_modality, df = df_stimuli):
    logging.data("get_stair_sign")
    
    if relevant_modality == "AUD":
        go = df.aud1[index]
        nogo = df.aud2[index]
    else:
        go = df.vis1[index]
        nogo = df.vis2[index]
    print("go", go)
    print("nogo", nogo)
    if go > nogo:
        stair_sign = 1
    else:
        stair_sign = -1
    logging.data("get_stair_sign: stair_sign=" + str(stair_sign) + ", go=" + str(go))
    return stair_sign, go

def get_soundfile_name(sound_number):
    logging.data("get_soundfile_name")
    
    # "sounds/sound_40.wav"
    soundfile_name = "sounds/sound_" + str(sound_number) + ".wav"
    
    return soundfile_name 

def trials_order(trials_in_block):
    print("trials_order")
    logging.data("trials_order")
    
    Go = list()
    Go.extend([True]*int(trials_in_block/2))
    Go.extend([False]*int(trials_in_block/2))
    irrelevant_mod = list()
    irrelevant_mod.extend([1]*int(trials_in_block/4))
    irrelevant_mod.extend([2]*int(trials_in_block/4))
    irrelevant_mod.extend([1]*int(trials_in_block/4))
    irrelevant_mod.extend([2]*int(trials_in_block/4))
    
    df = pd.DataFrame()
    df["Go"] = Go
    df["irrelevant_mod"] = irrelevant_mod
    
    df = df.sample(n = trials_in_block)
    df = df.reset_index(drop=True)
    
    return df

df_trials = trials_order(trials_in_block)
#print('df_trials')
#print(df_trials)
 
def stimuli_setting(relevant_modality, index, df = df_stimuli):
    print("stimuli_setting")
    logging.data("stimuli_setting")
    
    if relevant_modality == "AUD":
        go = get_soundfile_name(df.aud1[index])
        nogo = get_soundfile_name(df.aud2[index])
        irrel_mod1 = df.vis1[index]
        irrel_mod2 = df.vis2[index]
    else:
        irrel_mod1 = get_soundfile_name(df.aud1[index])
        irrel_mod2 = get_soundfile_name(df.aud2[index])
        go = df.vis1[index]
        nogo = df.vis2[index]
    
    logging.data("stimuli_setting: go=" +str(go) + ", nogo=" + str(nogo) + ", irrel_mod1=" + str(irrel_mod1) + ", irrel_mod2=" + str(irrel_mod2))
    return go, nogo, irrel_mod1, irrel_mod2
    
def next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df): #df = df_trials
    #logging.data("next_stimuli")
    n = trial_number - 1
    #print("n", n)
    if relevant_modality == "AUD":
        if df.irrelevant_mod[n] == 1:
            orien = irrel_mod1
        else:
            orien = irrel_mod2
        if df.Go[n]:
            trial_go = True
            sound_file1 = go
        else:
            trial_go = False
            sound_file1 = nogo
    else:
        if df.irrelevant_mod[n] == 1:
            sound_file1 = irrel_mod1
        else:
            sound_file1 = irrel_mod2
        if df.Go[n]:
            trial_go = True
            orien = go
        else:
            trial_go = False
            orien = nogo
    logging.data("next_stimuli: trial_number=" + str(trial_number) + ", trial_go=" + str(trial_go) + ", sound_file1=" + str(sound_file1) + ", orien=" + str(orien))
    
    return sound_file1, orien, trial_go


#VARIABLES_2

relevant_modality = df_stimuli.relevant_modality[index]
logging.data("relevant_modality: " + relevant_modality)

if relevant_modality == "AUD":
    sound_on = 1
    #vis_on = 0
    vis_on = 1
    logging.data("sound_on=" + str(sound_on) + ", vis_on=" + str(vis_on))
    
    #for staircase procedure:
    nogo_number = df_stimuli.aud2[index]
    angle = 1
else:
    #sound_on = 0
    sound_on = 1
    vis_on = 1
    logging.data("sound_on=" + str(sound_on) + ", vis_on=" + str(vis_on))
    
    #for staircase procedure:
    nogo_number = df_stimuli.vis2[index]
    #angle = 2
    angle = 1 

correct_opacity = 0
incorrect_opacity = 0
#noise_duration = round(random.uniform(0.5,1), 2)
noise_duration = noise_dur_list[intervals_block[trial_number-1]]

#STIMULI

go, nogo, irrel_mod1, irrel_mod2 = stimuli_setting(relevant_modality, index, df = df_stimuli)
sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)
volume = get_volume(sound_file1)


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'experiment_part3'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\labra\\Documents\\gitlab\\perceptual_attention_learning\\experiment_perceptual_learning part_3_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "intro"
introClock = core.Clock()
intro_text = visual.TextStim(win=win, name='intro_text',
    text='START part 3',
    font='Open Sans',
    pos=(0, 0.05), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
press_space_intro = visual.TextStim(win=win, name='press_space_intro',
    text='Press “space” when you are ready to begin.',
    font='Open Sans',
    pos=(0, -0.05), height=0.03, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
key_resp_intro = keyboard.Keyboard()

# Initialize components for Routine "progress_calc"
progress_calcClock = core.Clock()

# Initialize components for Routine "transition"
transitionClock = core.Clock()

# Initialize components for Routine "fixation_point"
fixation_pointClock = core.Clock()
noise = visual.NoiseStim(
    win=win, name='noise',
    noiseImage=None, mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.32, 0.32), sf=None,
    phase=0.0,
    color=[1,1,1], colorSpace='rgb',     opacity=None, blendmode='avg', contrast=0.7,
    texRes=128, filter=None,
    noiseType='Binary', noiseElementSize=[0.005], 
    noiseBaseSf=8.0, noiseBW=1.0,
    noiseBWO=30.0, noiseOri=0.0,
    noiseFractalPower=0.0,noiseFilterLower=1.0,
    noiseFilterUpper=8.0, noiseFilterOrder=0.0,
    noiseClip=3.0, imageComponent='Phase', interpolate=False, depth=-1.0)
noise.buildNoise()
fix_point1 = visual.Line(
    win=win, name='fix_point1',
    start=(-(0.05, 0.05)[0]/2.0, 0), end=(+(0.05, 0.05)[0]/2.0, 0),
    ori=0.0, pos=[0,0],
    lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)
fix_point2 = visual.Line(
    win=win, name='fix_point2',
    start=(-(0.05, 0.05)[0]/2.0, 0), end=(+(0.05, 0.05)[0]/2.0, 0),
    ori=90.0, pos=(0, 0),
    lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)

# Initialize components for Routine "trial"
trialClock = core.Clock()
sound_trial = sound.Sound('A', secs=1.01, stereo=True, hamming=True,
    name='sound_trial')
sound_trial.setVolume(1.0)
grating_trial = visual.GratingStim(
    win=win, name='grating_trial',
    tex='sin', mask='gauss',
    ori=1.0, pos=(0, 0), size=(0.3, 0.3), sf=5.0, phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=0.8, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
key_resp_trial = keyboard.Keyboard()

# Initialize components for Routine "response_feedback"
response_feedbackClock = core.Clock()
noise_3 = visual.NoiseStim(
    win=win, name='noise_3',
    noiseImage=None, mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.32, 0.32), sf=None,
    phase=0.0,
    color=[1,1,1], colorSpace='rgb',     opacity=None, blendmode='avg', contrast=0.7,
    texRes=128, filter=None,
    noiseType='Binary', noiseElementSize=[0.005], 
    noiseBaseSf=8.0, noiseBW=1.0,
    noiseBWO=30.0, noiseOri=0.0,
    noiseFractalPower=0.0,noiseFilterLower=1.0,
    noiseFilterUpper=8.0, noiseFilterOrder=0.0,
    noiseClip=3.0, imageComponent='Phase', interpolate=False, depth=-1.0)
noise_3.buildNoise()
correct_green_2 = visual.GratingStim(
    win=win, name='correct_green_2',
    tex='sin', mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.4, 0.4), sf=0.05, phase=0.0,
    color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
incorrect_red_2 = visual.GratingStim(
    win=win, name='incorrect_red_2',
    tex='sin', mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.4, 0.4), sf=0.1, phase=0.0,
    color=[0.7, -1.0000, -1.0000], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-3.0)

# Initialize components for Routine "progress_calc"
progress_calcClock = core.Clock()

# Initialize components for Routine "brk"
brkClock = core.Clock()
press_space_break = visual.TextStim(win=win, name='press_space_break',
    text='Press “space” when you are ready for the next round',
    font='Open Sans',
    pos=(0, -0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
key_resp_break = keyboard.Keyboard()
BREAK = visual.TextStim(win=win, name='BREAK',
    text='BREAK',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
polygon = visual.Rect(
    win=win, name='polygon',
    width=(0.3, 0.05)[0], height=(0.3, 0.05)[1],
    ori=0.0, pos=(0, -0.25),
    lineWidth=4.0,     colorSpace='rgb',  lineColor='white', fillColor='grey',
    opacity=None, depth=-4.0, interpolate=True)
polygon_2 = visual.Rect(
    win=win, name='polygon_2',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=4.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)

# Initialize components for Routine "transition"
transitionClock = core.Clock()

# Initialize components for Routine "fixation_point"
fixation_pointClock = core.Clock()
noise = visual.NoiseStim(
    win=win, name='noise',
    noiseImage=None, mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.32, 0.32), sf=None,
    phase=0.0,
    color=[1,1,1], colorSpace='rgb',     opacity=None, blendmode='avg', contrast=0.7,
    texRes=128, filter=None,
    noiseType='Binary', noiseElementSize=[0.005], 
    noiseBaseSf=8.0, noiseBW=1.0,
    noiseBWO=30.0, noiseOri=0.0,
    noiseFractalPower=0.0,noiseFilterLower=1.0,
    noiseFilterUpper=8.0, noiseFilterOrder=0.0,
    noiseClip=3.0, imageComponent='Phase', interpolate=False, depth=-1.0)
noise.buildNoise()
fix_point1 = visual.Line(
    win=win, name='fix_point1',
    start=(-(0.05, 0.05)[0]/2.0, 0), end=(+(0.05, 0.05)[0]/2.0, 0),
    ori=0.0, pos=[0,0],
    lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)
fix_point2 = visual.Line(
    win=win, name='fix_point2',
    start=(-(0.05, 0.05)[0]/2.0, 0), end=(+(0.05, 0.05)[0]/2.0, 0),
    ori=90.0, pos=(0, 0),
    lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)

# Initialize components for Routine "trial"
trialClock = core.Clock()
sound_trial = sound.Sound('A', secs=1.01, stereo=True, hamming=True,
    name='sound_trial')
sound_trial.setVolume(1.0)
grating_trial = visual.GratingStim(
    win=win, name='grating_trial',
    tex='sin', mask='gauss',
    ori=1.0, pos=(0, 0), size=(0.3, 0.3), sf=5.0, phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=0.8, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
key_resp_trial = keyboard.Keyboard()

# Initialize components for Routine "response_feedback"
response_feedbackClock = core.Clock()
noise_3 = visual.NoiseStim(
    win=win, name='noise_3',
    noiseImage=None, mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.32, 0.32), sf=None,
    phase=0.0,
    color=[1,1,1], colorSpace='rgb',     opacity=None, blendmode='avg', contrast=0.7,
    texRes=128, filter=None,
    noiseType='Binary', noiseElementSize=[0.005], 
    noiseBaseSf=8.0, noiseBW=1.0,
    noiseBWO=30.0, noiseOri=0.0,
    noiseFractalPower=0.0,noiseFilterLower=1.0,
    noiseFilterUpper=8.0, noiseFilterOrder=0.0,
    noiseClip=3.0, imageComponent='Phase', interpolate=False, depth=-1.0)
noise_3.buildNoise()
correct_green_2 = visual.GratingStim(
    win=win, name='correct_green_2',
    tex='sin', mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.4, 0.4), sf=0.05, phase=0.0,
    color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
incorrect_red_2 = visual.GratingStim(
    win=win, name='incorrect_red_2',
    tex='sin', mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.4, 0.4), sf=0.1, phase=0.0,
    color=[0.7, -1.0000, -1.0000], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-3.0)

# Initialize components for Routine "progress_calc"
progress_calcClock = core.Clock()

# Initialize components for Routine "brk"
brkClock = core.Clock()
press_space_break = visual.TextStim(win=win, name='press_space_break',
    text='Press “space” when you are ready for the next round',
    font='Open Sans',
    pos=(0, -0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
key_resp_break = keyboard.Keyboard()
BREAK = visual.TextStim(win=win, name='BREAK',
    text='BREAK',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
polygon = visual.Rect(
    win=win, name='polygon',
    width=(0.3, 0.05)[0], height=(0.3, 0.05)[1],
    ori=0.0, pos=(0, -0.25),
    lineWidth=4.0,     colorSpace='rgb',  lineColor='white', fillColor='grey',
    opacity=None, depth=-4.0, interpolate=True)
polygon_2 = visual.Rect(
    win=win, name='polygon_2',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=4.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)

# Initialize components for Routine "transition"
transitionClock = core.Clock()

# Initialize components for Routine "fixation_point"
fixation_pointClock = core.Clock()
noise = visual.NoiseStim(
    win=win, name='noise',
    noiseImage=None, mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.32, 0.32), sf=None,
    phase=0.0,
    color=[1,1,1], colorSpace='rgb',     opacity=None, blendmode='avg', contrast=0.7,
    texRes=128, filter=None,
    noiseType='Binary', noiseElementSize=[0.005], 
    noiseBaseSf=8.0, noiseBW=1.0,
    noiseBWO=30.0, noiseOri=0.0,
    noiseFractalPower=0.0,noiseFilterLower=1.0,
    noiseFilterUpper=8.0, noiseFilterOrder=0.0,
    noiseClip=3.0, imageComponent='Phase', interpolate=False, depth=-1.0)
noise.buildNoise()
fix_point1 = visual.Line(
    win=win, name='fix_point1',
    start=(-(0.05, 0.05)[0]/2.0, 0), end=(+(0.05, 0.05)[0]/2.0, 0),
    ori=0.0, pos=[0,0],
    lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)
fix_point2 = visual.Line(
    win=win, name='fix_point2',
    start=(-(0.05, 0.05)[0]/2.0, 0), end=(+(0.05, 0.05)[0]/2.0, 0),
    ori=90.0, pos=(0, 0),
    lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)

# Initialize components for Routine "trial"
trialClock = core.Clock()
sound_trial = sound.Sound('A', secs=1.01, stereo=True, hamming=True,
    name='sound_trial')
sound_trial.setVolume(1.0)
grating_trial = visual.GratingStim(
    win=win, name='grating_trial',
    tex='sin', mask='gauss',
    ori=1.0, pos=(0, 0), size=(0.3, 0.3), sf=5.0, phase=0.0,
    color=[1,1,1], colorSpace='rgb',
    opacity=1.0, contrast=0.8, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
key_resp_trial = keyboard.Keyboard()

# Initialize components for Routine "response_feedback"
response_feedbackClock = core.Clock()
noise_3 = visual.NoiseStim(
    win=win, name='noise_3',
    noiseImage=None, mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.32, 0.32), sf=None,
    phase=0.0,
    color=[1,1,1], colorSpace='rgb',     opacity=None, blendmode='avg', contrast=0.7,
    texRes=128, filter=None,
    noiseType='Binary', noiseElementSize=[0.005], 
    noiseBaseSf=8.0, noiseBW=1.0,
    noiseBWO=30.0, noiseOri=0.0,
    noiseFractalPower=0.0,noiseFilterLower=1.0,
    noiseFilterUpper=8.0, noiseFilterOrder=0.0,
    noiseClip=3.0, imageComponent='Phase', interpolate=False, depth=-1.0)
noise_3.buildNoise()
correct_green_2 = visual.GratingStim(
    win=win, name='correct_green_2',
    tex='sin', mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.4, 0.4), sf=0.05, phase=0.0,
    color=[-1.0000, 0.0039, -1.0000], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-2.0)
incorrect_red_2 = visual.GratingStim(
    win=win, name='incorrect_red_2',
    tex='sin', mask='gauss',
    ori=0.0, pos=(0, 0), size=(0.4, 0.4), sf=0.1, phase=0.0,
    color=[0.7, -1.0000, -1.0000], colorSpace='rgb',
    opacity=1.0, contrast=1.0, blendmode='avg',
    texRes=128.0, interpolate=True, depth=-3.0)

# Initialize components for Routine "progress_calc"
progress_calcClock = core.Clock()

# Initialize components for Routine "brk"
brkClock = core.Clock()
press_space_break = visual.TextStim(win=win, name='press_space_break',
    text='Press “space” when you are ready for the next round',
    font='Open Sans',
    pos=(0, -0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
key_resp_break = keyboard.Keyboard()
BREAK = visual.TextStim(win=win, name='BREAK',
    text='BREAK',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
polygon = visual.Rect(
    win=win, name='polygon',
    width=(0.3, 0.05)[0], height=(0.3, 0.05)[1],
    ori=0.0, pos=(0, -0.25),
    lineWidth=4.0,     colorSpace='rgb',  lineColor='white', fillColor='grey',
    opacity=None, depth=-4.0, interpolate=True)
polygon_2 = visual.Rect(
    win=win, name='polygon_2',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=4.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)

# Initialize components for Routine "end_of_experiment"
end_of_experimentClock = core.Clock()
text_4 = visual.TextStim(win=win, name='text_4',
    text='That’s it :) \nThank you!',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "intro"-------
continueRoutine = True
# update component parameters for each repeat
win.mouseVisible = False
logging.data("intro")
    
key_resp_intro.keys = []
key_resp_intro.rt = []
_key_resp_intro_allKeys = []
# keep track of which components have finished
introComponents = [intro_text, press_space_intro, key_resp_intro]
for thisComponent in introComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
introClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "intro"-------
while continueRoutine:
    # get current time
    t = introClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=introClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *intro_text* updates
    if intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        intro_text.frameNStart = frameN  # exact frame index
        intro_text.tStart = t  # local t and not account for scr refresh
        intro_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(intro_text, 'tStartRefresh')  # time at next scr refresh
        intro_text.setAutoDraw(True)
    
    # *press_space_intro* updates
    if press_space_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        press_space_intro.frameNStart = frameN  # exact frame index
        press_space_intro.tStart = t  # local t and not account for scr refresh
        press_space_intro.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(press_space_intro, 'tStartRefresh')  # time at next scr refresh
        press_space_intro.setAutoDraw(True)
    
    # *key_resp_intro* updates
    waitOnFlip = False
    if key_resp_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_intro.frameNStart = frameN  # exact frame index
        key_resp_intro.tStart = t  # local t and not account for scr refresh
        key_resp_intro.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_intro, 'tStartRefresh')  # time at next scr refresh
        key_resp_intro.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_intro.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_intro.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_intro.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_intro.getKeys(keyList=['space'], waitRelease=False)
        _key_resp_intro_allKeys.extend(theseKeys)
        if len(_key_resp_intro_allKeys):
            key_resp_intro.keys = _key_resp_intro_allKeys[-1].name  # just the last key pressed
            key_resp_intro.rt = _key_resp_intro_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in introComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "intro"-------
for thisComponent in introComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
logging.data("utcTime: " + str(time.time()))
event.clearEvents()
print("done intro")
thisExp.addData('intro_text.started', intro_text.tStartRefresh)
thisExp.addData('intro_text.stopped', intro_text.tStopRefresh)
thisExp.addData('press_space_intro.started', press_space_intro.tStartRefresh)
thisExp.addData('press_space_intro.stopped', press_space_intro.tStopRefresh)
# check responses
if key_resp_intro.keys in ['', [], None]:  # No response was made
    key_resp_intro.keys = None
thisExp.addData('key_resp_intro.keys',key_resp_intro.keys)
if key_resp_intro.keys != None:  # we had a response
    thisExp.addData('key_resp_intro.rt', key_resp_intro.rt)
thisExp.addData('key_resp_intro.started', key_resp_intro.tStartRefresh)
thisExp.addData('key_resp_intro.stopped', key_resp_intro.tStopRefresh)
thisExp.nextEntry()
# the Routine "intro" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "progress_calc"-------
continueRoutine = True
# update component parameters for each repeat
print('progress calc')


print('size',size)
print(x)
progress=progress+0.05

if stage == 'staircase2':
    if progress > 0.4:
        progress = 0.4
elif stage == 'ams2':
    progress=progress+0.05
elif stage == 'eds2':
    if progress > 0.75:
        progress = 0.75
else:
    print("STAGE IS NOT IDENTIFIED")

if progress >= 1:
    progress = 0.95
    
size = a * progress

x = -0.15+size/2
    
print('stage',stage,progress)
# keep track of which components have finished
progress_calcComponents = []
for thisComponent in progress_calcComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
progress_calcClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "progress_calc"-------
while continueRoutine:
    # get current time
    t = progress_calcClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=progress_calcClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in progress_calcComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "progress_calc"-------
for thisComponent in progress_calcComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "progress_calc" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "transition"-------
continueRoutine = True
# update component parameters for each repeat

logging.data("transition from " + str(stage))
print("transition from " + str(stage))

stage_number = stage_number + 1
stage = stages[stage_number]

logging.data("transition to " + str(stage))
print("transition to " + str(stage))

stimuli_change = False

if stage == "ids1":
    index = 1
    stimuli_change = True

elif stage == "staircase1":
    stair_sign, go_number = get_stair_sign(relevant_modality)
    if relevant_modality == "AUD":
        nogo_number = df_stimuli.aud2[index]
        angle = 1
    else:
        nogo_number = df_stimuli.vis2[index]
        angle = 1
    logging.data("nogo_number: " + str(nogo_number))
elif stage == "ams1":
    reversals = list()
    index = 2
    stimuli_change = True
elif stage == "eds1":
    index = 3
    stimuli_change = True
elif stage == "ids2":
    index = 4
    stimuli_change = True
elif stage == "staircase2":
    stair_sign, go_number = get_stair_sign(relevant_modality)
    if relevant_modality == "AUD":
        nogo_number = df_stimuli.aud2[index]
        angle = 1
    else:
        nogo_number = df_stimuli.vis2[index]
        angle = 1
    logging.data("nogo_number: " + str(nogo_number))
elif stage == "ams2":
    index = 5
    stimuli_change = True
elif stage == "eds2":
    index = 6
    stimuli_change = True


#STIMULI
if stimuli_change:
    relevant_modality = df_stimuli.relevant_modality[index]
    logging.data("relevant_modality: " + relevant_modality)

    go, nogo, irrel_mod1, irrel_mod2 = stimuli_setting(relevant_modality, index, df = df_stimuli)
    if stage == "ams1" or stage == "ams2":
        if relevant_modality == "AUD":
            if nogo == get_soundfile_name(0):
                nogo = get_soundfile_name(resulting_nogo)
        else:
            if nogo == 0:
                nogo = resulting_nogo

sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)

logging.data("go=" + str(go) + ", nogo=" + str(nogo) + ", irrel_mod1=" + str(irrel_mod1) + ", irrel_mod2=" + str(irrel_mod2))
# keep track of which components have finished
transitionComponents = []
for thisComponent in transitionComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
transitionClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "transition"-------
while continueRoutine:
    # get current time
    t = transitionClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=transitionClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in transitionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "transition"-------
for thisComponent in transitionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
event.clearEvents()
# the Routine "transition" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
staircase2 = data.TrialHandler(nReps=100.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='staircase2')
thisExp.addLoop(staircase2)  # add the loop to the experiment
thisStaircase2 = staircase2.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisStaircase2.rgb)
if thisStaircase2 != None:
    for paramName in thisStaircase2:
        exec('{} = thisStaircase2[paramName]'.format(paramName))

for thisStaircase2 in staircase2:
    currentLoop = staircase2
    # abbreviate parameter names if possible (e.g. rgb = thisStaircase2.rgb)
    if thisStaircase2 != None:
        for paramName in thisStaircase2:
            exec('{} = thisStaircase2[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    trials_staircase2 = data.TrialHandler(nReps=trials_in_block, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_staircase2')
    thisExp.addLoop(trials_staircase2)  # add the loop to the experiment
    thisTrials_staircase2 = trials_staircase2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_staircase2.rgb)
    if thisTrials_staircase2 != None:
        for paramName in thisTrials_staircase2:
            exec('{} = thisTrials_staircase2[paramName]'.format(paramName))
    
    for thisTrials_staircase2 in trials_staircase2:
        currentLoop = trials_staircase2
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_staircase2.rgb)
        if thisTrials_staircase2 != None:
            for paramName in thisTrials_staircase2:
                exec('{} = thisTrials_staircase2[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "fixation_point"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        logging.data("fix_point, trail_number=" + str(trial_number))
        
        print("fix point,", "trial_number:", trial_number,',', sound_file1,',visual degree:', orien, ",stage:", stage)
        # keep track of which components have finished
        fixation_pointComponents = [noise, fix_point1, fix_point2]
        for thisComponent in fixation_pointComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        fixation_pointClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "fixation_point"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = fixation_pointClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=fixation_pointClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *noise* updates
            if noise.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                noise.frameNStart = frameN  # exact frame index
                noise.tStart = t  # local t and not account for scr refresh
                noise.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noise, 'tStartRefresh')  # time at next scr refresh
                noise.setAutoDraw(True)
            if noise.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noise.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    noise.tStop = t  # not accounting for scr refresh
                    noise.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(noise, 'tStopRefresh')  # time at next scr refresh
                    noise.setAutoDraw(False)
            if noise.status == STARTED:
                if noise._needBuild:
                    noise.buildNoise()
                else:
                    if (frameN-noise.frameNStart) %             1==0:
                        noise.updateNoise()
            
            # *fix_point1* updates
            if fix_point1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_point1.frameNStart = frameN  # exact frame index
                fix_point1.tStart = t  # local t and not account for scr refresh
                fix_point1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_point1, 'tStartRefresh')  # time at next scr refresh
                fix_point1.setAutoDraw(True)
            if fix_point1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_point1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_point1.tStop = t  # not accounting for scr refresh
                    fix_point1.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_point1, 'tStopRefresh')  # time at next scr refresh
                    fix_point1.setAutoDraw(False)
            if fix_point1.status == STARTED:  # only update if drawing
                fix_point1.setPos((0, 0), log=False)
            
            # *fix_point2* updates
            if fix_point2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_point2.frameNStart = frameN  # exact frame index
                fix_point2.tStart = t  # local t and not account for scr refresh
                fix_point2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_point2, 'tStartRefresh')  # time at next scr refresh
                fix_point2.setAutoDraw(True)
            if fix_point2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_point2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_point2.tStop = t  # not accounting for scr refresh
                    fix_point2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_point2, 'tStopRefresh')  # time at next scr refresh
                    fix_point2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_pointComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "fixation_point"-------
        for thisComponent in fixation_pointComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        event.clearEvents()
        trials_staircase2.addData('noise.started', noise.tStartRefresh)
        trials_staircase2.addData('noise.stopped', noise.tStopRefresh)
        trials_staircase2.addData('fix_point1.started', fix_point1.tStartRefresh)
        trials_staircase2.addData('fix_point1.stopped', fix_point1.tStopRefresh)
        trials_staircase2.addData('fix_point2.started', fix_point2.tStartRefresh)
        trials_staircase2.addData('fix_point2.stopped', fix_point2.tStopRefresh)
        
        # ------Prepare to start Routine "trial"-------
        continueRoutine = True
        routineTimer.add(1.010000)
        # update component parameters for each repeat
        logging.data("trial")
        #print("trial:", trial_number)
        #print(trial_go)
        sound_trial.setSound(sound_file1, secs=1.01, hamming=True)
        sound_trial.setVolume(volume, log=False)
        grating_trial.setOpacity(vis_on)
        grating_trial.setOri(orien)
        key_resp_trial.keys = []
        key_resp_trial.rt = []
        _key_resp_trial_allKeys = []
        # keep track of which components have finished
        trialComponents = [sound_trial, grating_trial, key_resp_trial]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "trial"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = trialClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=trialClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # start/stop sound_trial
            if sound_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_trial.frameNStart = frameN  # exact frame index
                sound_trial.tStart = t  # local t and not account for scr refresh
                sound_trial.tStartRefresh = tThisFlipGlobal  # on global time
                sound_trial.play(when=win)  # sync with win flip
            if sound_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_trial.tStartRefresh + 1.01-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_trial.tStop = t  # not accounting for scr refresh
                    sound_trial.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_trial, 'tStopRefresh')  # time at next scr refresh
                    sound_trial.stop()
            
            # *grating_trial* updates
            if grating_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                grating_trial.frameNStart = frameN  # exact frame index
                grating_trial.tStart = t  # local t and not account for scr refresh
                grating_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating_trial, 'tStartRefresh')  # time at next scr refresh
                grating_trial.setAutoDraw(True)
            if grating_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating_trial.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    grating_trial.tStop = t  # not accounting for scr refresh
                    grating_trial.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(grating_trial, 'tStopRefresh')  # time at next scr refresh
                    grating_trial.setAutoDraw(False)
            
            # *key_resp_trial* updates
            waitOnFlip = False
            if key_resp_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_trial.frameNStart = frameN  # exact frame index
                key_resp_trial.tStart = t  # local t and not account for scr refresh
                key_resp_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_trial, 'tStartRefresh')  # time at next scr refresh
                key_resp_trial.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_trial.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_trial.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_trial.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_trial.tStop = t  # not accounting for scr refresh
                    key_resp_trial.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(key_resp_trial, 'tStopRefresh')  # time at next scr refresh
                    key_resp_trial.status = FINISHED
            if key_resp_trial.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_trial.getKeys(keyList=['space'], waitRelease=False)
                _key_resp_trial_allKeys.extend(theseKeys)
                if len(_key_resp_trial_allKeys):
                    key_resp_trial.keys = _key_resp_trial_allKeys[-1].name  # just the last key pressed
                    key_resp_trial.rt = _key_resp_trial_allKeys[-1].rt
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial"-------
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        keyPressed = event.getKeys()
        correct = False
        print('keyPressed:', keyPressed)
        if 'space' in keyPressed:
            if trial_go:
                correct_feedback = True
                correct = True
                go_correct = go_correct + 1
                #correct_response = correct_response + 1
            else: 
                incorrect_feedback = True
        else:
            if trial_go == False:
                correct = True
                nogo_correct = nogo_correct + 1
                #correct_response = correct_response + 1
            else:
                missed = missed + 1
        
        if correct:
            performance_record.append(1)
        else:
            performance_record.append(0)
            
        if correct_feedback:
            correct_opacity = 0.7
        if incorrect_feedback:
            incorrect_opacity = 0.7
            duration = 2
            noise_duration = 2
        
        logging.data("ITI duration: " + str(noise_duration))
        logging.data('correct: ' + str(correct))
        #print("noise_duration", noise_duration)
        event.clearEvents()
        
        sound_trial.stop()  # ensure sound has stopped at end of routine
        trials_staircase2.addData('sound_trial.started', sound_trial.tStartRefresh)
        trials_staircase2.addData('sound_trial.stopped', sound_trial.tStopRefresh)
        trials_staircase2.addData('grating_trial.started', grating_trial.tStartRefresh)
        trials_staircase2.addData('grating_trial.stopped', grating_trial.tStopRefresh)
        # check responses
        if key_resp_trial.keys in ['', [], None]:  # No response was made
            key_resp_trial.keys = None
        trials_staircase2.addData('key_resp_trial.keys',key_resp_trial.keys)
        if key_resp_trial.keys != None:  # we had a response
            trials_staircase2.addData('key_resp_trial.rt', key_resp_trial.rt)
        trials_staircase2.addData('key_resp_trial.started', key_resp_trial.tStartRefresh)
        trials_staircase2.addData('key_resp_trial.stopped', key_resp_trial.tStopRefresh)
        
        # ------Prepare to start Routine "response_feedback"-------
        continueRoutine = True
        # update component parameters for each repeat
        logging.data("response feedback: correct_opacity = " +str(correct_opacity) + ', incorrect_opacity = ' + str(incorrect_opacity))
        #print("response feedback")
        
        if stage == "staircase1" or stage == "staircase2":
            if correct:
                if correct_twice:
                    if converging:
                        pass
                    else:
                        reversals.append(nogo_number)
                        converging = True
                    nogo_number = nogo_number + stair_sign*angle
                    correct_twice = False 
                else:
                    correct_twice = True
            else:
                if converging:
                    reversals.append(nogo_number)
                    converging = False
                nogo_number = nogo_number - stair_sign*angle
            logging.data("nogo_number: " + str(nogo_number))
            
            if relevant_modality == "AUD":
                nogo = get_soundfile_name(nogo_number)
            else:
                nogo = nogo_number
            logging.data("nogo_number: " + str(nogo_number))
            logging.data("reversals: " + str(len(reversals)))
            
            print("reversals:", len(reversals))
            
            if len(reversals) >= 12:
                logging.data("reversals max")
                resulting_nogo = int(sum(reversals[-6:])/6)
                logging.data("resulting_nogo: " + str(resulting_nogo))
                if stage == "staircase1":
                    #trials_staircase1.finished = True
                    staircase1.finished = True
                if stage == "staircase2":
                    #trials_staircase2.finished = True
                    staircase2.finished = True
        correct_green_2.setOpacity(correct_opacity)
        incorrect_red_2.setOpacity(incorrect_opacity)
        # keep track of which components have finished
        response_feedbackComponents = [noise_3, correct_green_2, incorrect_red_2]
        for thisComponent in response_feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        response_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "response_feedback"-------
        while continueRoutine:
            # get current time
            t = response_feedbackClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=response_feedbackClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *noise_3* updates
            if noise_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                noise_3.frameNStart = frameN  # exact frame index
                noise_3.tStart = t  # local t and not account for scr refresh
                noise_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noise_3, 'tStartRefresh')  # time at next scr refresh
                noise_3.setAutoDraw(True)
            if noise_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noise_3.tStartRefresh + noise_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    noise_3.tStop = t  # not accounting for scr refresh
                    noise_3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(noise_3, 'tStopRefresh')  # time at next scr refresh
                    noise_3.setAutoDraw(False)
            if noise_3.status == STARTED:
                if noise_3._needBuild:
                    noise_3.buildNoise()
                else:
                    if (frameN-noise_3.frameNStart) %             1==0:
                        noise_3.updateNoise()
            
            # *correct_green_2* updates
            if correct_green_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                correct_green_2.frameNStart = frameN  # exact frame index
                correct_green_2.tStart = t  # local t and not account for scr refresh
                correct_green_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(correct_green_2, 'tStartRefresh')  # time at next scr refresh
                correct_green_2.setAutoDraw(True)
            if correct_green_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > correct_green_2.tStartRefresh + noise_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    correct_green_2.tStop = t  # not accounting for scr refresh
                    correct_green_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(correct_green_2, 'tStopRefresh')  # time at next scr refresh
                    correct_green_2.setAutoDraw(False)
            
            # *incorrect_red_2* updates
            if incorrect_red_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                incorrect_red_2.frameNStart = frameN  # exact frame index
                incorrect_red_2.tStart = t  # local t and not account for scr refresh
                incorrect_red_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(incorrect_red_2, 'tStartRefresh')  # time at next scr refresh
                incorrect_red_2.setAutoDraw(True)
            if incorrect_red_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > incorrect_red_2.tStartRefresh + noise_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    incorrect_red_2.tStop = t  # not accounting for scr refresh
                    incorrect_red_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(incorrect_red_2, 'tStopRefresh')  # time at next scr refresh
                    incorrect_red_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in response_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "response_feedback"-------
        for thisComponent in response_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        correct_opacity = 0
        incorrect_opacity = 0
        noise_duration = noise_dur_list[intervals_block[trial_number-1]]
        #noise_duration = round(random.uniform(0.5,1), 2)
        
        correct_feedback = False
        incorrect_feedback = False
        
        duration = 0.7
        
        trial_number = trial_number + 1
        if trial_number <= trials_in_block:
            sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)
            volume = get_volume(sound_file1)
            
        if stage != 'staircase2':
            if len(performance_record) >= mov_average:
                logging.data("mov average performance (sum): " +str(sum(performance_record[-mov_average:])))
                if sum(performance_record[-mov_average:])>=(mov_average*0.8):
                    pass_stage = True
                logging.data("pass_stage: " + str(pass_stage))
        performance = (sum(performance_record) - past_performance)/trials_in_block
        #performance_go = round(2*go_correct/trials_in_block, 2)
        #performance_nogo = round(2*nogo_correct/trials_in_block, 2)
        #stat_text = "Performance: " + str(performance)
        #stat_text = "Go: " + str(performance_go) + " NoGo: " + str(performance_nogo) + stage
        
        event.clearEvents()
        trials_staircase2.addData('noise_3.started', noise_3.tStartRefresh)
        trials_staircase2.addData('noise_3.stopped', noise_3.tStopRefresh)
        trials_staircase2.addData('correct_green_2.started', correct_green_2.tStartRefresh)
        trials_staircase2.addData('correct_green_2.stopped', correct_green_2.tStopRefresh)
        trials_staircase2.addData('incorrect_red_2.started', incorrect_red_2.tStartRefresh)
        trials_staircase2.addData('incorrect_red_2.stopped', incorrect_red_2.tStopRefresh)
        # the Routine "response_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed trials_in_block repeats of 'trials_staircase2'
    
    
    # ------Prepare to start Routine "progress_calc"-------
    continueRoutine = True
    # update component parameters for each repeat
    print('progress calc')
    
    
    print('size',size)
    print(x)
    progress=progress+0.05
    
    if stage == 'staircase2':
        if progress > 0.4:
            progress = 0.4
    elif stage == 'ams2':
        progress=progress+0.05
    elif stage == 'eds2':
        if progress > 0.75:
            progress = 0.75
    else:
        print("STAGE IS NOT IDENTIFIED")
    
    if progress >= 1:
        progress = 0.95
        
    size = a * progress
    
    x = -0.15+size/2
        
    print('stage',stage,progress)
    # keep track of which components have finished
    progress_calcComponents = []
    for thisComponent in progress_calcComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    progress_calcClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "progress_calc"-------
    while continueRoutine:
        # get current time
        t = progress_calcClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=progress_calcClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in progress_calcComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "progress_calc"-------
    for thisComponent in progress_calcComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "progress_calc" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "brk"-------
    continueRoutine = True
    # update component parameters for each repeat
    logging.data("break, stage: " + str(stage))
    logging.data("performance records: " + str(performance_record))
    logging.data("sum performance records: " + str(sum(performance_record)))
    logging.data("performance: " + str(performance))
    print("break")
    
    
    
    #block_n = block_n + 1
    
    if stage == "training":
        if pass_stage:
            block_n = 5
            training.finished=True
        else:
            if relevant_mod_only == False:
                training.finished=True
                print("TRAINING IS DONE")
                logging.data("training is done")
            relevant_mod_only = False
            logging.data("relevant_mode_only = False")
            
        if relevant_mod_only:
            pass
        else:
            sound_on = 1
            vis_on = 1
    else:
        if pass_stage:
            print('move to next stage!')
            if stage == "ids1":
                block_n = 10
                ids1.finished=True
            elif stage == "eds1":
                block_n = 25
                eds1.finished = True
            elif stage == "ids2":
                block_n = 30
                ids2.finished=True
            elif stage == "eds2":
                block_n = 45
                eds2.finished = True
        else:
            print(performance)
    
    trial_number = 1
    
    df_trials = trials_order(trials_in_block)
    
    sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)
    
    
    correct_response = 0
    missed = 0
    
    go_correct = 0
    nogo_correct = 0
    performance_go = 0
    performance_nogo = 0
    pass_stage = False
    key_resp_break.keys = []
    key_resp_break.rt = []
    _key_resp_break_allKeys = []
    polygon_2.setPos((x, -0.25))
    polygon_2.setSize((size, 0.05))
    # keep track of which components have finished
    brkComponents = [press_space_break, key_resp_break, BREAK, polygon, polygon_2]
    for thisComponent in brkComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    brkClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "brk"-------
    while continueRoutine:
        # get current time
        t = brkClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=brkClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *press_space_break* updates
        if press_space_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            press_space_break.frameNStart = frameN  # exact frame index
            press_space_break.tStart = t  # local t and not account for scr refresh
            press_space_break.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(press_space_break, 'tStartRefresh')  # time at next scr refresh
            press_space_break.setAutoDraw(True)
        
        # *key_resp_break* updates
        waitOnFlip = False
        if key_resp_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_break.frameNStart = frameN  # exact frame index
            key_resp_break.tStart = t  # local t and not account for scr refresh
            key_resp_break.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_break, 'tStartRefresh')  # time at next scr refresh
            key_resp_break.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_break.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_break.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_break.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_break.getKeys(keyList=['space'], waitRelease=False)
            _key_resp_break_allKeys.extend(theseKeys)
            if len(_key_resp_break_allKeys):
                key_resp_break.keys = _key_resp_break_allKeys[-1].name  # just the last key pressed
                key_resp_break.rt = _key_resp_break_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *BREAK* updates
        if BREAK.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            BREAK.frameNStart = frameN  # exact frame index
            BREAK.tStart = t  # local t and not account for scr refresh
            BREAK.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(BREAK, 'tStartRefresh')  # time at next scr refresh
            BREAK.setAutoDraw(True)
        
        # *polygon* updates
        if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon.frameNStart = frameN  # exact frame index
            polygon.tStart = t  # local t and not account for scr refresh
            polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
            polygon.setAutoDraw(True)
        
        # *polygon_2* updates
        if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_2.frameNStart = frameN  # exact frame index
            polygon_2.tStart = t  # local t and not account for scr refresh
            polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
            polygon_2.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in brkComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "brk"-------
    for thisComponent in brkComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    logging.data("utcTime: " + str(time.time()))
    event.clearEvents()
    staircase2.addData('press_space_break.started', press_space_break.tStartRefresh)
    staircase2.addData('press_space_break.stopped', press_space_break.tStopRefresh)
    # check responses
    if key_resp_break.keys in ['', [], None]:  # No response was made
        key_resp_break.keys = None
    staircase2.addData('key_resp_break.keys',key_resp_break.keys)
    if key_resp_break.keys != None:  # we had a response
        staircase2.addData('key_resp_break.rt', key_resp_break.rt)
    staircase2.addData('key_resp_break.started', key_resp_break.tStartRefresh)
    staircase2.addData('key_resp_break.stopped', key_resp_break.tStopRefresh)
    staircase2.addData('BREAK.started', BREAK.tStartRefresh)
    staircase2.addData('BREAK.stopped', BREAK.tStopRefresh)
    staircase2.addData('polygon.started', polygon.tStartRefresh)
    staircase2.addData('polygon.stopped', polygon.tStopRefresh)
    staircase2.addData('polygon_2.started', polygon_2.tStartRefresh)
    staircase2.addData('polygon_2.stopped', polygon_2.tStopRefresh)
    # the Routine "brk" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 100.0 repeats of 'staircase2'


# ------Prepare to start Routine "transition"-------
continueRoutine = True
# update component parameters for each repeat

logging.data("transition from " + str(stage))
print("transition from " + str(stage))

stage_number = stage_number + 1
stage = stages[stage_number]

logging.data("transition to " + str(stage))
print("transition to " + str(stage))

stimuli_change = False

if stage == "ids1":
    index = 1
    stimuli_change = True

elif stage == "staircase1":
    stair_sign, go_number = get_stair_sign(relevant_modality)
    if relevant_modality == "AUD":
        nogo_number = df_stimuli.aud2[index]
        angle = 1
    else:
        nogo_number = df_stimuli.vis2[index]
        angle = 1
    logging.data("nogo_number: " + str(nogo_number))
elif stage == "ams1":
    reversals = list()
    index = 2
    stimuli_change = True
elif stage == "eds1":
    index = 3
    stimuli_change = True
elif stage == "ids2":
    index = 4
    stimuli_change = True
elif stage == "staircase2":
    stair_sign, go_number = get_stair_sign(relevant_modality)
    if relevant_modality == "AUD":
        nogo_number = df_stimuli.aud2[index]
        angle = 1
    else:
        nogo_number = df_stimuli.vis2[index]
        angle = 1
    logging.data("nogo_number: " + str(nogo_number))
elif stage == "ams2":
    index = 5
    stimuli_change = True
elif stage == "eds2":
    index = 6
    stimuli_change = True


#STIMULI
if stimuli_change:
    relevant_modality = df_stimuli.relevant_modality[index]
    logging.data("relevant_modality: " + relevant_modality)

    go, nogo, irrel_mod1, irrel_mod2 = stimuli_setting(relevant_modality, index, df = df_stimuli)
    if stage == "ams1" or stage == "ams2":
        if relevant_modality == "AUD":
            if nogo == get_soundfile_name(0):
                nogo = get_soundfile_name(resulting_nogo)
        else:
            if nogo == 0:
                nogo = resulting_nogo

sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)

logging.data("go=" + str(go) + ", nogo=" + str(nogo) + ", irrel_mod1=" + str(irrel_mod1) + ", irrel_mod2=" + str(irrel_mod2))
# keep track of which components have finished
transitionComponents = []
for thisComponent in transitionComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
transitionClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "transition"-------
while continueRoutine:
    # get current time
    t = transitionClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=transitionClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in transitionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "transition"-------
for thisComponent in transitionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
event.clearEvents()
# the Routine "transition" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
ams2 = data.TrialHandler(nReps=2.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='ams2')
thisExp.addLoop(ams2)  # add the loop to the experiment
thisAms2 = ams2.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisAms2.rgb)
if thisAms2 != None:
    for paramName in thisAms2:
        exec('{} = thisAms2[paramName]'.format(paramName))

for thisAms2 in ams2:
    currentLoop = ams2
    # abbreviate parameter names if possible (e.g. rgb = thisAms2.rgb)
    if thisAms2 != None:
        for paramName in thisAms2:
            exec('{} = thisAms2[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    trials_ams2 = data.TrialHandler(nReps=trials_in_block, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_ams2')
    thisExp.addLoop(trials_ams2)  # add the loop to the experiment
    thisTrials_ams2 = trials_ams2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_ams2.rgb)
    if thisTrials_ams2 != None:
        for paramName in thisTrials_ams2:
            exec('{} = thisTrials_ams2[paramName]'.format(paramName))
    
    for thisTrials_ams2 in trials_ams2:
        currentLoop = trials_ams2
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_ams2.rgb)
        if thisTrials_ams2 != None:
            for paramName in thisTrials_ams2:
                exec('{} = thisTrials_ams2[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "fixation_point"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        logging.data("fix_point, trail_number=" + str(trial_number))
        
        print("fix point,", "trial_number:", trial_number,',', sound_file1,',visual degree:', orien, ",stage:", stage)
        # keep track of which components have finished
        fixation_pointComponents = [noise, fix_point1, fix_point2]
        for thisComponent in fixation_pointComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        fixation_pointClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "fixation_point"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = fixation_pointClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=fixation_pointClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *noise* updates
            if noise.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                noise.frameNStart = frameN  # exact frame index
                noise.tStart = t  # local t and not account for scr refresh
                noise.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noise, 'tStartRefresh')  # time at next scr refresh
                noise.setAutoDraw(True)
            if noise.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noise.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    noise.tStop = t  # not accounting for scr refresh
                    noise.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(noise, 'tStopRefresh')  # time at next scr refresh
                    noise.setAutoDraw(False)
            if noise.status == STARTED:
                if noise._needBuild:
                    noise.buildNoise()
                else:
                    if (frameN-noise.frameNStart) %             1==0:
                        noise.updateNoise()
            
            # *fix_point1* updates
            if fix_point1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_point1.frameNStart = frameN  # exact frame index
                fix_point1.tStart = t  # local t and not account for scr refresh
                fix_point1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_point1, 'tStartRefresh')  # time at next scr refresh
                fix_point1.setAutoDraw(True)
            if fix_point1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_point1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_point1.tStop = t  # not accounting for scr refresh
                    fix_point1.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_point1, 'tStopRefresh')  # time at next scr refresh
                    fix_point1.setAutoDraw(False)
            if fix_point1.status == STARTED:  # only update if drawing
                fix_point1.setPos((0, 0), log=False)
            
            # *fix_point2* updates
            if fix_point2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_point2.frameNStart = frameN  # exact frame index
                fix_point2.tStart = t  # local t and not account for scr refresh
                fix_point2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_point2, 'tStartRefresh')  # time at next scr refresh
                fix_point2.setAutoDraw(True)
            if fix_point2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_point2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_point2.tStop = t  # not accounting for scr refresh
                    fix_point2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_point2, 'tStopRefresh')  # time at next scr refresh
                    fix_point2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_pointComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "fixation_point"-------
        for thisComponent in fixation_pointComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        event.clearEvents()
        trials_ams2.addData('noise.started', noise.tStartRefresh)
        trials_ams2.addData('noise.stopped', noise.tStopRefresh)
        trials_ams2.addData('fix_point1.started', fix_point1.tStartRefresh)
        trials_ams2.addData('fix_point1.stopped', fix_point1.tStopRefresh)
        trials_ams2.addData('fix_point2.started', fix_point2.tStartRefresh)
        trials_ams2.addData('fix_point2.stopped', fix_point2.tStopRefresh)
        
        # ------Prepare to start Routine "trial"-------
        continueRoutine = True
        routineTimer.add(1.010000)
        # update component parameters for each repeat
        logging.data("trial")
        #print("trial:", trial_number)
        #print(trial_go)
        sound_trial.setSound(sound_file1, secs=1.01, hamming=True)
        sound_trial.setVolume(volume, log=False)
        grating_trial.setOpacity(vis_on)
        grating_trial.setOri(orien)
        key_resp_trial.keys = []
        key_resp_trial.rt = []
        _key_resp_trial_allKeys = []
        # keep track of which components have finished
        trialComponents = [sound_trial, grating_trial, key_resp_trial]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "trial"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = trialClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=trialClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # start/stop sound_trial
            if sound_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_trial.frameNStart = frameN  # exact frame index
                sound_trial.tStart = t  # local t and not account for scr refresh
                sound_trial.tStartRefresh = tThisFlipGlobal  # on global time
                sound_trial.play(when=win)  # sync with win flip
            if sound_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_trial.tStartRefresh + 1.01-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_trial.tStop = t  # not accounting for scr refresh
                    sound_trial.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_trial, 'tStopRefresh')  # time at next scr refresh
                    sound_trial.stop()
            
            # *grating_trial* updates
            if grating_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                grating_trial.frameNStart = frameN  # exact frame index
                grating_trial.tStart = t  # local t and not account for scr refresh
                grating_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating_trial, 'tStartRefresh')  # time at next scr refresh
                grating_trial.setAutoDraw(True)
            if grating_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating_trial.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    grating_trial.tStop = t  # not accounting for scr refresh
                    grating_trial.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(grating_trial, 'tStopRefresh')  # time at next scr refresh
                    grating_trial.setAutoDraw(False)
            
            # *key_resp_trial* updates
            waitOnFlip = False
            if key_resp_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_trial.frameNStart = frameN  # exact frame index
                key_resp_trial.tStart = t  # local t and not account for scr refresh
                key_resp_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_trial, 'tStartRefresh')  # time at next scr refresh
                key_resp_trial.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_trial.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_trial.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_trial.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_trial.tStop = t  # not accounting for scr refresh
                    key_resp_trial.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(key_resp_trial, 'tStopRefresh')  # time at next scr refresh
                    key_resp_trial.status = FINISHED
            if key_resp_trial.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_trial.getKeys(keyList=['space'], waitRelease=False)
                _key_resp_trial_allKeys.extend(theseKeys)
                if len(_key_resp_trial_allKeys):
                    key_resp_trial.keys = _key_resp_trial_allKeys[-1].name  # just the last key pressed
                    key_resp_trial.rt = _key_resp_trial_allKeys[-1].rt
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial"-------
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        keyPressed = event.getKeys()
        correct = False
        print('keyPressed:', keyPressed)
        if 'space' in keyPressed:
            if trial_go:
                correct_feedback = True
                correct = True
                go_correct = go_correct + 1
                #correct_response = correct_response + 1
            else: 
                incorrect_feedback = True
        else:
            if trial_go == False:
                correct = True
                nogo_correct = nogo_correct + 1
                #correct_response = correct_response + 1
            else:
                missed = missed + 1
        
        if correct:
            performance_record.append(1)
        else:
            performance_record.append(0)
            
        if correct_feedback:
            correct_opacity = 0.7
        if incorrect_feedback:
            incorrect_opacity = 0.7
            duration = 2
            noise_duration = 2
        
        logging.data("ITI duration: " + str(noise_duration))
        logging.data('correct: ' + str(correct))
        #print("noise_duration", noise_duration)
        event.clearEvents()
        
        sound_trial.stop()  # ensure sound has stopped at end of routine
        trials_ams2.addData('sound_trial.started', sound_trial.tStartRefresh)
        trials_ams2.addData('sound_trial.stopped', sound_trial.tStopRefresh)
        trials_ams2.addData('grating_trial.started', grating_trial.tStartRefresh)
        trials_ams2.addData('grating_trial.stopped', grating_trial.tStopRefresh)
        # check responses
        if key_resp_trial.keys in ['', [], None]:  # No response was made
            key_resp_trial.keys = None
        trials_ams2.addData('key_resp_trial.keys',key_resp_trial.keys)
        if key_resp_trial.keys != None:  # we had a response
            trials_ams2.addData('key_resp_trial.rt', key_resp_trial.rt)
        trials_ams2.addData('key_resp_trial.started', key_resp_trial.tStartRefresh)
        trials_ams2.addData('key_resp_trial.stopped', key_resp_trial.tStopRefresh)
        
        # ------Prepare to start Routine "response_feedback"-------
        continueRoutine = True
        # update component parameters for each repeat
        logging.data("response feedback: correct_opacity = " +str(correct_opacity) + ', incorrect_opacity = ' + str(incorrect_opacity))
        #print("response feedback")
        
        if stage == "staircase1" or stage == "staircase2":
            if correct:
                if correct_twice:
                    if converging:
                        pass
                    else:
                        reversals.append(nogo_number)
                        converging = True
                    nogo_number = nogo_number + stair_sign*angle
                    correct_twice = False 
                else:
                    correct_twice = True
            else:
                if converging:
                    reversals.append(nogo_number)
                    converging = False
                nogo_number = nogo_number - stair_sign*angle
            logging.data("nogo_number: " + str(nogo_number))
            
            if relevant_modality == "AUD":
                nogo = get_soundfile_name(nogo_number)
            else:
                nogo = nogo_number
            logging.data("nogo_number: " + str(nogo_number))
            logging.data("reversals: " + str(len(reversals)))
            
            print("reversals:", len(reversals))
            
            if len(reversals) >= 12:
                logging.data("reversals max")
                resulting_nogo = int(sum(reversals[-6:])/6)
                logging.data("resulting_nogo: " + str(resulting_nogo))
                if stage == "staircase1":
                    #trials_staircase1.finished = True
                    staircase1.finished = True
                if stage == "staircase2":
                    #trials_staircase2.finished = True
                    staircase2.finished = True
        correct_green_2.setOpacity(correct_opacity)
        incorrect_red_2.setOpacity(incorrect_opacity)
        # keep track of which components have finished
        response_feedbackComponents = [noise_3, correct_green_2, incorrect_red_2]
        for thisComponent in response_feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        response_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "response_feedback"-------
        while continueRoutine:
            # get current time
            t = response_feedbackClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=response_feedbackClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *noise_3* updates
            if noise_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                noise_3.frameNStart = frameN  # exact frame index
                noise_3.tStart = t  # local t and not account for scr refresh
                noise_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noise_3, 'tStartRefresh')  # time at next scr refresh
                noise_3.setAutoDraw(True)
            if noise_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noise_3.tStartRefresh + noise_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    noise_3.tStop = t  # not accounting for scr refresh
                    noise_3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(noise_3, 'tStopRefresh')  # time at next scr refresh
                    noise_3.setAutoDraw(False)
            if noise_3.status == STARTED:
                if noise_3._needBuild:
                    noise_3.buildNoise()
                else:
                    if (frameN-noise_3.frameNStart) %             1==0:
                        noise_3.updateNoise()
            
            # *correct_green_2* updates
            if correct_green_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                correct_green_2.frameNStart = frameN  # exact frame index
                correct_green_2.tStart = t  # local t and not account for scr refresh
                correct_green_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(correct_green_2, 'tStartRefresh')  # time at next scr refresh
                correct_green_2.setAutoDraw(True)
            if correct_green_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > correct_green_2.tStartRefresh + noise_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    correct_green_2.tStop = t  # not accounting for scr refresh
                    correct_green_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(correct_green_2, 'tStopRefresh')  # time at next scr refresh
                    correct_green_2.setAutoDraw(False)
            
            # *incorrect_red_2* updates
            if incorrect_red_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                incorrect_red_2.frameNStart = frameN  # exact frame index
                incorrect_red_2.tStart = t  # local t and not account for scr refresh
                incorrect_red_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(incorrect_red_2, 'tStartRefresh')  # time at next scr refresh
                incorrect_red_2.setAutoDraw(True)
            if incorrect_red_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > incorrect_red_2.tStartRefresh + noise_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    incorrect_red_2.tStop = t  # not accounting for scr refresh
                    incorrect_red_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(incorrect_red_2, 'tStopRefresh')  # time at next scr refresh
                    incorrect_red_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in response_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "response_feedback"-------
        for thisComponent in response_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        correct_opacity = 0
        incorrect_opacity = 0
        noise_duration = noise_dur_list[intervals_block[trial_number-1]]
        #noise_duration = round(random.uniform(0.5,1), 2)
        
        correct_feedback = False
        incorrect_feedback = False
        
        duration = 0.7
        
        trial_number = trial_number + 1
        if trial_number <= trials_in_block:
            sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)
            volume = get_volume(sound_file1)
            
        if stage != 'staircase2':
            if len(performance_record) >= mov_average:
                logging.data("mov average performance (sum): " +str(sum(performance_record[-mov_average:])))
                if sum(performance_record[-mov_average:])>=(mov_average*0.8):
                    pass_stage = True
                logging.data("pass_stage: " + str(pass_stage))
        performance = (sum(performance_record) - past_performance)/trials_in_block
        #performance_go = round(2*go_correct/trials_in_block, 2)
        #performance_nogo = round(2*nogo_correct/trials_in_block, 2)
        #stat_text = "Performance: " + str(performance)
        #stat_text = "Go: " + str(performance_go) + " NoGo: " + str(performance_nogo) + stage
        
        event.clearEvents()
        trials_ams2.addData('noise_3.started', noise_3.tStartRefresh)
        trials_ams2.addData('noise_3.stopped', noise_3.tStopRefresh)
        trials_ams2.addData('correct_green_2.started', correct_green_2.tStartRefresh)
        trials_ams2.addData('correct_green_2.stopped', correct_green_2.tStopRefresh)
        trials_ams2.addData('incorrect_red_2.started', incorrect_red_2.tStartRefresh)
        trials_ams2.addData('incorrect_red_2.stopped', incorrect_red_2.tStopRefresh)
        # the Routine "response_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed trials_in_block repeats of 'trials_ams2'
    
    
    # ------Prepare to start Routine "progress_calc"-------
    continueRoutine = True
    # update component parameters for each repeat
    print('progress calc')
    
    
    print('size',size)
    print(x)
    progress=progress+0.05
    
    if stage == 'staircase2':
        if progress > 0.4:
            progress = 0.4
    elif stage == 'ams2':
        progress=progress+0.05
    elif stage == 'eds2':
        if progress > 0.75:
            progress = 0.75
    else:
        print("STAGE IS NOT IDENTIFIED")
    
    if progress >= 1:
        progress = 0.95
        
    size = a * progress
    
    x = -0.15+size/2
        
    print('stage',stage,progress)
    # keep track of which components have finished
    progress_calcComponents = []
    for thisComponent in progress_calcComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    progress_calcClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "progress_calc"-------
    while continueRoutine:
        # get current time
        t = progress_calcClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=progress_calcClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in progress_calcComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "progress_calc"-------
    for thisComponent in progress_calcComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "progress_calc" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "brk"-------
    continueRoutine = True
    # update component parameters for each repeat
    logging.data("break, stage: " + str(stage))
    logging.data("performance records: " + str(performance_record))
    logging.data("sum performance records: " + str(sum(performance_record)))
    logging.data("performance: " + str(performance))
    print("break")
    
    
    
    #block_n = block_n + 1
    
    if stage == "training":
        if pass_stage:
            block_n = 5
            training.finished=True
        else:
            if relevant_mod_only == False:
                training.finished=True
                print("TRAINING IS DONE")
                logging.data("training is done")
            relevant_mod_only = False
            logging.data("relevant_mode_only = False")
            
        if relevant_mod_only:
            pass
        else:
            sound_on = 1
            vis_on = 1
    else:
        if pass_stage:
            print('move to next stage!')
            if stage == "ids1":
                block_n = 10
                ids1.finished=True
            elif stage == "eds1":
                block_n = 25
                eds1.finished = True
            elif stage == "ids2":
                block_n = 30
                ids2.finished=True
            elif stage == "eds2":
                block_n = 45
                eds2.finished = True
        else:
            print(performance)
    
    trial_number = 1
    
    df_trials = trials_order(trials_in_block)
    
    sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)
    
    
    correct_response = 0
    missed = 0
    
    go_correct = 0
    nogo_correct = 0
    performance_go = 0
    performance_nogo = 0
    pass_stage = False
    key_resp_break.keys = []
    key_resp_break.rt = []
    _key_resp_break_allKeys = []
    polygon_2.setPos((x, -0.25))
    polygon_2.setSize((size, 0.05))
    # keep track of which components have finished
    brkComponents = [press_space_break, key_resp_break, BREAK, polygon, polygon_2]
    for thisComponent in brkComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    brkClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "brk"-------
    while continueRoutine:
        # get current time
        t = brkClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=brkClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *press_space_break* updates
        if press_space_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            press_space_break.frameNStart = frameN  # exact frame index
            press_space_break.tStart = t  # local t and not account for scr refresh
            press_space_break.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(press_space_break, 'tStartRefresh')  # time at next scr refresh
            press_space_break.setAutoDraw(True)
        
        # *key_resp_break* updates
        waitOnFlip = False
        if key_resp_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_break.frameNStart = frameN  # exact frame index
            key_resp_break.tStart = t  # local t and not account for scr refresh
            key_resp_break.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_break, 'tStartRefresh')  # time at next scr refresh
            key_resp_break.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_break.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_break.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_break.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_break.getKeys(keyList=['space'], waitRelease=False)
            _key_resp_break_allKeys.extend(theseKeys)
            if len(_key_resp_break_allKeys):
                key_resp_break.keys = _key_resp_break_allKeys[-1].name  # just the last key pressed
                key_resp_break.rt = _key_resp_break_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *BREAK* updates
        if BREAK.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            BREAK.frameNStart = frameN  # exact frame index
            BREAK.tStart = t  # local t and not account for scr refresh
            BREAK.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(BREAK, 'tStartRefresh')  # time at next scr refresh
            BREAK.setAutoDraw(True)
        
        # *polygon* updates
        if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon.frameNStart = frameN  # exact frame index
            polygon.tStart = t  # local t and not account for scr refresh
            polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
            polygon.setAutoDraw(True)
        
        # *polygon_2* updates
        if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_2.frameNStart = frameN  # exact frame index
            polygon_2.tStart = t  # local t and not account for scr refresh
            polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
            polygon_2.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in brkComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "brk"-------
    for thisComponent in brkComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    logging.data("utcTime: " + str(time.time()))
    event.clearEvents()
    ams2.addData('press_space_break.started', press_space_break.tStartRefresh)
    ams2.addData('press_space_break.stopped', press_space_break.tStopRefresh)
    # check responses
    if key_resp_break.keys in ['', [], None]:  # No response was made
        key_resp_break.keys = None
    ams2.addData('key_resp_break.keys',key_resp_break.keys)
    if key_resp_break.keys != None:  # we had a response
        ams2.addData('key_resp_break.rt', key_resp_break.rt)
    ams2.addData('key_resp_break.started', key_resp_break.tStartRefresh)
    ams2.addData('key_resp_break.stopped', key_resp_break.tStopRefresh)
    ams2.addData('BREAK.started', BREAK.tStartRefresh)
    ams2.addData('BREAK.stopped', BREAK.tStopRefresh)
    ams2.addData('polygon.started', polygon.tStartRefresh)
    ams2.addData('polygon.stopped', polygon.tStopRefresh)
    ams2.addData('polygon_2.started', polygon_2.tStartRefresh)
    ams2.addData('polygon_2.stopped', polygon_2.tStopRefresh)
    # the Routine "brk" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 2.0 repeats of 'ams2'


# ------Prepare to start Routine "transition"-------
continueRoutine = True
# update component parameters for each repeat

logging.data("transition from " + str(stage))
print("transition from " + str(stage))

stage_number = stage_number + 1
stage = stages[stage_number]

logging.data("transition to " + str(stage))
print("transition to " + str(stage))

stimuli_change = False

if stage == "ids1":
    index = 1
    stimuli_change = True

elif stage == "staircase1":
    stair_sign, go_number = get_stair_sign(relevant_modality)
    if relevant_modality == "AUD":
        nogo_number = df_stimuli.aud2[index]
        angle = 1
    else:
        nogo_number = df_stimuli.vis2[index]
        angle = 1
    logging.data("nogo_number: " + str(nogo_number))
elif stage == "ams1":
    reversals = list()
    index = 2
    stimuli_change = True
elif stage == "eds1":
    index = 3
    stimuli_change = True
elif stage == "ids2":
    index = 4
    stimuli_change = True
elif stage == "staircase2":
    stair_sign, go_number = get_stair_sign(relevant_modality)
    if relevant_modality == "AUD":
        nogo_number = df_stimuli.aud2[index]
        angle = 1
    else:
        nogo_number = df_stimuli.vis2[index]
        angle = 1
    logging.data("nogo_number: " + str(nogo_number))
elif stage == "ams2":
    index = 5
    stimuli_change = True
elif stage == "eds2":
    index = 6
    stimuli_change = True


#STIMULI
if stimuli_change:
    relevant_modality = df_stimuli.relevant_modality[index]
    logging.data("relevant_modality: " + relevant_modality)

    go, nogo, irrel_mod1, irrel_mod2 = stimuli_setting(relevant_modality, index, df = df_stimuli)
    if stage == "ams1" or stage == "ams2":
        if relevant_modality == "AUD":
            if nogo == get_soundfile_name(0):
                nogo = get_soundfile_name(resulting_nogo)
        else:
            if nogo == 0:
                nogo = resulting_nogo

sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)

logging.data("go=" + str(go) + ", nogo=" + str(nogo) + ", irrel_mod1=" + str(irrel_mod1) + ", irrel_mod2=" + str(irrel_mod2))
# keep track of which components have finished
transitionComponents = []
for thisComponent in transitionComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
transitionClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "transition"-------
while continueRoutine:
    # get current time
    t = transitionClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=transitionClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in transitionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "transition"-------
for thisComponent in transitionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
event.clearEvents()
# the Routine "transition" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
eds2 = data.TrialHandler(nReps=100.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='eds2')
thisExp.addLoop(eds2)  # add the loop to the experiment
thisEds2 = eds2.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisEds2.rgb)
if thisEds2 != None:
    for paramName in thisEds2:
        exec('{} = thisEds2[paramName]'.format(paramName))

for thisEds2 in eds2:
    currentLoop = eds2
    # abbreviate parameter names if possible (e.g. rgb = thisEds2.rgb)
    if thisEds2 != None:
        for paramName in thisEds2:
            exec('{} = thisEds2[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    trials_eds2 = data.TrialHandler(nReps=trials_in_block, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_eds2')
    thisExp.addLoop(trials_eds2)  # add the loop to the experiment
    thisTrials_eds2 = trials_eds2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_eds2.rgb)
    if thisTrials_eds2 != None:
        for paramName in thisTrials_eds2:
            exec('{} = thisTrials_eds2[paramName]'.format(paramName))
    
    for thisTrials_eds2 in trials_eds2:
        currentLoop = trials_eds2
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_eds2.rgb)
        if thisTrials_eds2 != None:
            for paramName in thisTrials_eds2:
                exec('{} = thisTrials_eds2[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "fixation_point"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        logging.data("fix_point, trail_number=" + str(trial_number))
        
        print("fix point,", "trial_number:", trial_number,',', sound_file1,',visual degree:', orien, ",stage:", stage)
        # keep track of which components have finished
        fixation_pointComponents = [noise, fix_point1, fix_point2]
        for thisComponent in fixation_pointComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        fixation_pointClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "fixation_point"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = fixation_pointClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=fixation_pointClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *noise* updates
            if noise.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                noise.frameNStart = frameN  # exact frame index
                noise.tStart = t  # local t and not account for scr refresh
                noise.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noise, 'tStartRefresh')  # time at next scr refresh
                noise.setAutoDraw(True)
            if noise.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noise.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    noise.tStop = t  # not accounting for scr refresh
                    noise.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(noise, 'tStopRefresh')  # time at next scr refresh
                    noise.setAutoDraw(False)
            if noise.status == STARTED:
                if noise._needBuild:
                    noise.buildNoise()
                else:
                    if (frameN-noise.frameNStart) %             1==0:
                        noise.updateNoise()
            
            # *fix_point1* updates
            if fix_point1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_point1.frameNStart = frameN  # exact frame index
                fix_point1.tStart = t  # local t and not account for scr refresh
                fix_point1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_point1, 'tStartRefresh')  # time at next scr refresh
                fix_point1.setAutoDraw(True)
            if fix_point1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_point1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_point1.tStop = t  # not accounting for scr refresh
                    fix_point1.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_point1, 'tStopRefresh')  # time at next scr refresh
                    fix_point1.setAutoDraw(False)
            if fix_point1.status == STARTED:  # only update if drawing
                fix_point1.setPos((0, 0), log=False)
            
            # *fix_point2* updates
            if fix_point2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_point2.frameNStart = frameN  # exact frame index
                fix_point2.tStart = t  # local t and not account for scr refresh
                fix_point2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_point2, 'tStartRefresh')  # time at next scr refresh
                fix_point2.setAutoDraw(True)
            if fix_point2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_point2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_point2.tStop = t  # not accounting for scr refresh
                    fix_point2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_point2, 'tStopRefresh')  # time at next scr refresh
                    fix_point2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_pointComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "fixation_point"-------
        for thisComponent in fixation_pointComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        event.clearEvents()
        trials_eds2.addData('noise.started', noise.tStartRefresh)
        trials_eds2.addData('noise.stopped', noise.tStopRefresh)
        trials_eds2.addData('fix_point1.started', fix_point1.tStartRefresh)
        trials_eds2.addData('fix_point1.stopped', fix_point1.tStopRefresh)
        trials_eds2.addData('fix_point2.started', fix_point2.tStartRefresh)
        trials_eds2.addData('fix_point2.stopped', fix_point2.tStopRefresh)
        
        # ------Prepare to start Routine "trial"-------
        continueRoutine = True
        routineTimer.add(1.010000)
        # update component parameters for each repeat
        logging.data("trial")
        #print("trial:", trial_number)
        #print(trial_go)
        sound_trial.setSound(sound_file1, secs=1.01, hamming=True)
        sound_trial.setVolume(volume, log=False)
        grating_trial.setOpacity(vis_on)
        grating_trial.setOri(orien)
        key_resp_trial.keys = []
        key_resp_trial.rt = []
        _key_resp_trial_allKeys = []
        # keep track of which components have finished
        trialComponents = [sound_trial, grating_trial, key_resp_trial]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "trial"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = trialClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=trialClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # start/stop sound_trial
            if sound_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_trial.frameNStart = frameN  # exact frame index
                sound_trial.tStart = t  # local t and not account for scr refresh
                sound_trial.tStartRefresh = tThisFlipGlobal  # on global time
                sound_trial.play(when=win)  # sync with win flip
            if sound_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_trial.tStartRefresh + 1.01-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_trial.tStop = t  # not accounting for scr refresh
                    sound_trial.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_trial, 'tStopRefresh')  # time at next scr refresh
                    sound_trial.stop()
            
            # *grating_trial* updates
            if grating_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                grating_trial.frameNStart = frameN  # exact frame index
                grating_trial.tStart = t  # local t and not account for scr refresh
                grating_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating_trial, 'tStartRefresh')  # time at next scr refresh
                grating_trial.setAutoDraw(True)
            if grating_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating_trial.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    grating_trial.tStop = t  # not accounting for scr refresh
                    grating_trial.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(grating_trial, 'tStopRefresh')  # time at next scr refresh
                    grating_trial.setAutoDraw(False)
            
            # *key_resp_trial* updates
            waitOnFlip = False
            if key_resp_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_trial.frameNStart = frameN  # exact frame index
                key_resp_trial.tStart = t  # local t and not account for scr refresh
                key_resp_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_trial, 'tStartRefresh')  # time at next scr refresh
                key_resp_trial.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_trial.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_trial.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_trial.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_trial.tStop = t  # not accounting for scr refresh
                    key_resp_trial.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(key_resp_trial, 'tStopRefresh')  # time at next scr refresh
                    key_resp_trial.status = FINISHED
            if key_resp_trial.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_trial.getKeys(keyList=['space'], waitRelease=False)
                _key_resp_trial_allKeys.extend(theseKeys)
                if len(_key_resp_trial_allKeys):
                    key_resp_trial.keys = _key_resp_trial_allKeys[-1].name  # just the last key pressed
                    key_resp_trial.rt = _key_resp_trial_allKeys[-1].rt
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial"-------
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        keyPressed = event.getKeys()
        correct = False
        print('keyPressed:', keyPressed)
        if 'space' in keyPressed:
            if trial_go:
                correct_feedback = True
                correct = True
                go_correct = go_correct + 1
                #correct_response = correct_response + 1
            else: 
                incorrect_feedback = True
        else:
            if trial_go == False:
                correct = True
                nogo_correct = nogo_correct + 1
                #correct_response = correct_response + 1
            else:
                missed = missed + 1
        
        if correct:
            performance_record.append(1)
        else:
            performance_record.append(0)
            
        if correct_feedback:
            correct_opacity = 0.7
        if incorrect_feedback:
            incorrect_opacity = 0.7
            duration = 2
            noise_duration = 2
        
        logging.data("ITI duration: " + str(noise_duration))
        logging.data('correct: ' + str(correct))
        #print("noise_duration", noise_duration)
        event.clearEvents()
        
        sound_trial.stop()  # ensure sound has stopped at end of routine
        trials_eds2.addData('sound_trial.started', sound_trial.tStartRefresh)
        trials_eds2.addData('sound_trial.stopped', sound_trial.tStopRefresh)
        trials_eds2.addData('grating_trial.started', grating_trial.tStartRefresh)
        trials_eds2.addData('grating_trial.stopped', grating_trial.tStopRefresh)
        # check responses
        if key_resp_trial.keys in ['', [], None]:  # No response was made
            key_resp_trial.keys = None
        trials_eds2.addData('key_resp_trial.keys',key_resp_trial.keys)
        if key_resp_trial.keys != None:  # we had a response
            trials_eds2.addData('key_resp_trial.rt', key_resp_trial.rt)
        trials_eds2.addData('key_resp_trial.started', key_resp_trial.tStartRefresh)
        trials_eds2.addData('key_resp_trial.stopped', key_resp_trial.tStopRefresh)
        
        # ------Prepare to start Routine "response_feedback"-------
        continueRoutine = True
        # update component parameters for each repeat
        logging.data("response feedback: correct_opacity = " +str(correct_opacity) + ', incorrect_opacity = ' + str(incorrect_opacity))
        #print("response feedback")
        
        if stage == "staircase1" or stage == "staircase2":
            if correct:
                if correct_twice:
                    if converging:
                        pass
                    else:
                        reversals.append(nogo_number)
                        converging = True
                    nogo_number = nogo_number + stair_sign*angle
                    correct_twice = False 
                else:
                    correct_twice = True
            else:
                if converging:
                    reversals.append(nogo_number)
                    converging = False
                nogo_number = nogo_number - stair_sign*angle
            logging.data("nogo_number: " + str(nogo_number))
            
            if relevant_modality == "AUD":
                nogo = get_soundfile_name(nogo_number)
            else:
                nogo = nogo_number
            logging.data("nogo_number: " + str(nogo_number))
            logging.data("reversals: " + str(len(reversals)))
            
            print("reversals:", len(reversals))
            
            if len(reversals) >= 12:
                logging.data("reversals max")
                resulting_nogo = int(sum(reversals[-6:])/6)
                logging.data("resulting_nogo: " + str(resulting_nogo))
                if stage == "staircase1":
                    #trials_staircase1.finished = True
                    staircase1.finished = True
                if stage == "staircase2":
                    #trials_staircase2.finished = True
                    staircase2.finished = True
        correct_green_2.setOpacity(correct_opacity)
        incorrect_red_2.setOpacity(incorrect_opacity)
        # keep track of which components have finished
        response_feedbackComponents = [noise_3, correct_green_2, incorrect_red_2]
        for thisComponent in response_feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        response_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "response_feedback"-------
        while continueRoutine:
            # get current time
            t = response_feedbackClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=response_feedbackClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *noise_3* updates
            if noise_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                noise_3.frameNStart = frameN  # exact frame index
                noise_3.tStart = t  # local t and not account for scr refresh
                noise_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(noise_3, 'tStartRefresh')  # time at next scr refresh
                noise_3.setAutoDraw(True)
            if noise_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > noise_3.tStartRefresh + noise_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    noise_3.tStop = t  # not accounting for scr refresh
                    noise_3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(noise_3, 'tStopRefresh')  # time at next scr refresh
                    noise_3.setAutoDraw(False)
            if noise_3.status == STARTED:
                if noise_3._needBuild:
                    noise_3.buildNoise()
                else:
                    if (frameN-noise_3.frameNStart) %             1==0:
                        noise_3.updateNoise()
            
            # *correct_green_2* updates
            if correct_green_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                correct_green_2.frameNStart = frameN  # exact frame index
                correct_green_2.tStart = t  # local t and not account for scr refresh
                correct_green_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(correct_green_2, 'tStartRefresh')  # time at next scr refresh
                correct_green_2.setAutoDraw(True)
            if correct_green_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > correct_green_2.tStartRefresh + noise_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    correct_green_2.tStop = t  # not accounting for scr refresh
                    correct_green_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(correct_green_2, 'tStopRefresh')  # time at next scr refresh
                    correct_green_2.setAutoDraw(False)
            
            # *incorrect_red_2* updates
            if incorrect_red_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                incorrect_red_2.frameNStart = frameN  # exact frame index
                incorrect_red_2.tStart = t  # local t and not account for scr refresh
                incorrect_red_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(incorrect_red_2, 'tStartRefresh')  # time at next scr refresh
                incorrect_red_2.setAutoDraw(True)
            if incorrect_red_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > incorrect_red_2.tStartRefresh + noise_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    incorrect_red_2.tStop = t  # not accounting for scr refresh
                    incorrect_red_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(incorrect_red_2, 'tStopRefresh')  # time at next scr refresh
                    incorrect_red_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in response_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "response_feedback"-------
        for thisComponent in response_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        correct_opacity = 0
        incorrect_opacity = 0
        noise_duration = noise_dur_list[intervals_block[trial_number-1]]
        #noise_duration = round(random.uniform(0.5,1), 2)
        
        correct_feedback = False
        incorrect_feedback = False
        
        duration = 0.7
        
        trial_number = trial_number + 1
        if trial_number <= trials_in_block:
            sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)
            volume = get_volume(sound_file1)
            
        if stage != 'staircase2':
            if len(performance_record) >= mov_average:
                logging.data("mov average performance (sum): " +str(sum(performance_record[-mov_average:])))
                if sum(performance_record[-mov_average:])>=(mov_average*0.8):
                    pass_stage = True
                logging.data("pass_stage: " + str(pass_stage))
        performance = (sum(performance_record) - past_performance)/trials_in_block
        #performance_go = round(2*go_correct/trials_in_block, 2)
        #performance_nogo = round(2*nogo_correct/trials_in_block, 2)
        #stat_text = "Performance: " + str(performance)
        #stat_text = "Go: " + str(performance_go) + " NoGo: " + str(performance_nogo) + stage
        
        event.clearEvents()
        trials_eds2.addData('noise_3.started', noise_3.tStartRefresh)
        trials_eds2.addData('noise_3.stopped', noise_3.tStopRefresh)
        trials_eds2.addData('correct_green_2.started', correct_green_2.tStartRefresh)
        trials_eds2.addData('correct_green_2.stopped', correct_green_2.tStopRefresh)
        trials_eds2.addData('incorrect_red_2.started', incorrect_red_2.tStartRefresh)
        trials_eds2.addData('incorrect_red_2.stopped', incorrect_red_2.tStopRefresh)
        # the Routine "response_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed trials_in_block repeats of 'trials_eds2'
    
    
    # ------Prepare to start Routine "progress_calc"-------
    continueRoutine = True
    # update component parameters for each repeat
    print('progress calc')
    
    
    print('size',size)
    print(x)
    progress=progress+0.05
    
    if stage == 'staircase2':
        if progress > 0.4:
            progress = 0.4
    elif stage == 'ams2':
        progress=progress+0.05
    elif stage == 'eds2':
        if progress > 0.75:
            progress = 0.75
    else:
        print("STAGE IS NOT IDENTIFIED")
    
    if progress >= 1:
        progress = 0.95
        
    size = a * progress
    
    x = -0.15+size/2
        
    print('stage',stage,progress)
    # keep track of which components have finished
    progress_calcComponents = []
    for thisComponent in progress_calcComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    progress_calcClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "progress_calc"-------
    while continueRoutine:
        # get current time
        t = progress_calcClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=progress_calcClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in progress_calcComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "progress_calc"-------
    for thisComponent in progress_calcComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "progress_calc" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "brk"-------
    continueRoutine = True
    # update component parameters for each repeat
    logging.data("break, stage: " + str(stage))
    logging.data("performance records: " + str(performance_record))
    logging.data("sum performance records: " + str(sum(performance_record)))
    logging.data("performance: " + str(performance))
    print("break")
    
    
    
    #block_n = block_n + 1
    
    if stage == "training":
        if pass_stage:
            block_n = 5
            training.finished=True
        else:
            if relevant_mod_only == False:
                training.finished=True
                print("TRAINING IS DONE")
                logging.data("training is done")
            relevant_mod_only = False
            logging.data("relevant_mode_only = False")
            
        if relevant_mod_only:
            pass
        else:
            sound_on = 1
            vis_on = 1
    else:
        if pass_stage:
            print('move to next stage!')
            if stage == "ids1":
                block_n = 10
                ids1.finished=True
            elif stage == "eds1":
                block_n = 25
                eds1.finished = True
            elif stage == "ids2":
                block_n = 30
                ids2.finished=True
            elif stage == "eds2":
                block_n = 45
                eds2.finished = True
        else:
            print(performance)
    
    trial_number = 1
    
    df_trials = trials_order(trials_in_block)
    
    sound_file1, orien, trial_go = next_stimuli(relevant_modality, go, nogo, irrel_mod1, irrel_mod2, trial_number, df_trials)
    
    
    correct_response = 0
    missed = 0
    
    go_correct = 0
    nogo_correct = 0
    performance_go = 0
    performance_nogo = 0
    pass_stage = False
    key_resp_break.keys = []
    key_resp_break.rt = []
    _key_resp_break_allKeys = []
    polygon_2.setPos((x, -0.25))
    polygon_2.setSize((size, 0.05))
    # keep track of which components have finished
    brkComponents = [press_space_break, key_resp_break, BREAK, polygon, polygon_2]
    for thisComponent in brkComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    brkClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "brk"-------
    while continueRoutine:
        # get current time
        t = brkClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=brkClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *press_space_break* updates
        if press_space_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            press_space_break.frameNStart = frameN  # exact frame index
            press_space_break.tStart = t  # local t and not account for scr refresh
            press_space_break.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(press_space_break, 'tStartRefresh')  # time at next scr refresh
            press_space_break.setAutoDraw(True)
        
        # *key_resp_break* updates
        waitOnFlip = False
        if key_resp_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_break.frameNStart = frameN  # exact frame index
            key_resp_break.tStart = t  # local t and not account for scr refresh
            key_resp_break.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_break, 'tStartRefresh')  # time at next scr refresh
            key_resp_break.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_break.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_break.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_break.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_break.getKeys(keyList=['space'], waitRelease=False)
            _key_resp_break_allKeys.extend(theseKeys)
            if len(_key_resp_break_allKeys):
                key_resp_break.keys = _key_resp_break_allKeys[-1].name  # just the last key pressed
                key_resp_break.rt = _key_resp_break_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *BREAK* updates
        if BREAK.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            BREAK.frameNStart = frameN  # exact frame index
            BREAK.tStart = t  # local t and not account for scr refresh
            BREAK.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(BREAK, 'tStartRefresh')  # time at next scr refresh
            BREAK.setAutoDraw(True)
        
        # *polygon* updates
        if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon.frameNStart = frameN  # exact frame index
            polygon.tStart = t  # local t and not account for scr refresh
            polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
            polygon.setAutoDraw(True)
        
        # *polygon_2* updates
        if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_2.frameNStart = frameN  # exact frame index
            polygon_2.tStart = t  # local t and not account for scr refresh
            polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
            polygon_2.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in brkComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "brk"-------
    for thisComponent in brkComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    logging.data("utcTime: " + str(time.time()))
    event.clearEvents()
    eds2.addData('press_space_break.started', press_space_break.tStartRefresh)
    eds2.addData('press_space_break.stopped', press_space_break.tStopRefresh)
    # check responses
    if key_resp_break.keys in ['', [], None]:  # No response was made
        key_resp_break.keys = None
    eds2.addData('key_resp_break.keys',key_resp_break.keys)
    if key_resp_break.keys != None:  # we had a response
        eds2.addData('key_resp_break.rt', key_resp_break.rt)
    eds2.addData('key_resp_break.started', key_resp_break.tStartRefresh)
    eds2.addData('key_resp_break.stopped', key_resp_break.tStopRefresh)
    eds2.addData('BREAK.started', BREAK.tStartRefresh)
    eds2.addData('BREAK.stopped', BREAK.tStopRefresh)
    eds2.addData('polygon.started', polygon.tStartRefresh)
    eds2.addData('polygon.stopped', polygon.tStopRefresh)
    eds2.addData('polygon_2.started', polygon_2.tStartRefresh)
    eds2.addData('polygon_2.stopped', polygon_2.tStopRefresh)
    # the Routine "brk" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 100.0 repeats of 'eds2'


# ------Prepare to start Routine "end_of_experiment"-------
continueRoutine = True
routineTimer.add(5.000000)
# update component parameters for each repeat
logging.data("the end")
print("Done")
# keep track of which components have finished
end_of_experimentComponents = [text_4]
for thisComponent in end_of_experimentComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
end_of_experimentClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "end_of_experiment"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = end_of_experimentClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=end_of_experimentClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_4* updates
    if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_4.frameNStart = frameN  # exact frame index
        text_4.tStart = t  # local t and not account for scr refresh
        text_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
        text_4.setAutoDraw(True)
    if text_4.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_4.tStartRefresh + 5.0-frameTolerance:
            # keep track of stop time/frame for later
            text_4.tStop = t  # not accounting for scr refresh
            text_4.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_4, 'tStopRefresh')  # time at next scr refresh
            text_4.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in end_of_experimentComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end_of_experiment"-------
for thisComponent in end_of_experimentComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_4.started', text_4.tStartRefresh)
thisExp.addData('text_4.stopped', text_4.tStopRefresh)

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
