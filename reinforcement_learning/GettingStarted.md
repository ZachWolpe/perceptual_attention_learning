# Getting Started
## Rat & Human based Reinforcement Learning

#### Local scope

Navigate to the `reinforcement_learning` directory.

```bash
cd reinforcement_learning
```


#### Prepare the data

Convert the `.mat` data to python suitable `.csv` & `.pkl` files. `.mat` files are read using `scipy.io.loadmat` and split into:

    - `meta data`
    - `stimulus & response data`


To generate the files run (make sure the relative paths are correct in both scripts):

```bash
python process_data/extract_matlab_data_exe.py
```


##### Rat Data Encoding (matlab)

```
Data Schema

    % Make some docs
    info = {
    'SessionType',  {'EDS' 'EDS_BL_Easy' 'EDS_BL_Hard' 'IDS' 'IDS_BL_Easy' 'RE' 'Stair'}, {'Acronyms for each session type that occured'};
    'SessionNum',   [1 2 3 4 5 6 7 8 9 10 11 12 13], {'Consecutive number of each session within each session type'};
    'CondNum',      [1 2 3 4 5 6 7 8 9 10 11 12 13 14], {'Numeric ordering of occurence of conditions - same as numeric code for SessionType IFF these occured in same order for every rat.'};
    'ReleMode',     [0 1], {'Logical coding for relevant modality: 0=visual, 1=tone'};
    'StimCode',     [0 1], {'Logical coding for stimulus class: 0=nogo, 1=go'};
    'RespCode',     [1 2 3 4], {'Numeric coding for response type:
                        1=correct_rejection,
                        2=hit,
                        3=false_alarm,
                        4=omissions'}};
```


##### Rat Data Schema

This will produce two files:

- `action sequence data`: `./data/processsed_data/attention_behaviorals_actions.pkl`
- `meta data`: `./data/processsed_data/attention_behaviorals_metadata.csv`


**meta data**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 571 entries, 0 to 570
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   ratID        571 non-null    int64 
 1   SessionType  571 non-null    object
 2   SessionNum   571 non-null    int64 
 3   CondNum      571 non-null    int64 
 4   ReleMode     571 non-null    int64 
dtypes: int64(4), object(1)
memory usage: 22.4+ KB
None
```

Example:

```
   ratID  SessionType  SessionNum  CondNum  ReleMode
0   4151  IDS_BL_Easy           1        1         1
1   4151  IDS_BL_Easy           2        1         1
2   4151          IDS           1        2         1
3   4151          IDS           2        2         1
...
```

**action sequence data**

A pickle object, loaded as a nested python dictionary containing:

- (<rat ID>, <session Type>, <session Number>):
    - `stimCode`: The stimulus code for the given action sequence.
    - `respCode`: The response code for the given action sequence.

_*Primary key:*_

The primary key identifies each unique rat session. It is a tuple of the form `(<rat ID>, <session Type>, <session Number>)`.

_*Secondary key*_

Provide the data for a given rat session.

- `stimCode: array[int]`
- `respCode: array[int]`

Example:

```python
{
    (4151, 'IDS_BL_Easy', 1): {
        'stimCode': array([0,1,...,0]),
        'respCode': array([0,1,...,0])
    },
    (4151, 'IDS_BL_Easy', 2): {
        'stimCode': array([0,1,...,0]),
        'respCode': array([0,1,...,0])
    },
    ...
}
```

### Response Data: Infer Action/Reward Sequences.

Extracting action & reward.

I have extracted action and reward from the `response` (`RespCode`) vector:
The action space is `[go, no-go]`.

Response vector encoding:

```
(1=correct reject, 2=hit, 3=false alarm, 4=omission)
```

- `reward=1` if `response=[1,2]`, else 0
- `action=1` if `response=[1,3]`, else 0
UPDATED:
- `action=1` if `response=[2,3]`, else 0


- `Reward` in `[0,1]`
- `Action` in `[0,1]`



### Session Type

**Definitions:**

- *BL*: baseline.
- *IDS*: intra-dimensional shift. `VIS --> VIS` or `AUD --> AUD`.
- *EDS*: extra-dimensional shift. `VIS --> AUD` or `AUD --> VIS`.
- *RE*: re-establishment.
- *Stair*: staircase.


**Explanation:**

- *BL*: A baseline sessions where the rat performed a previously learned rule with known stimuli. 

- *IDS* or *EDS*: A new stimuli were introduced in when `session number = 1`. The trial at which the new stimuli were introduced in that session is coded there in the file (typically trial number 200, but sometimes trial number 100). Prior to that trial number, the rat received the stimuli from the preceding BL sessions. The additional session nums for IDS and EDS continue to use the new stimuli introduced at trial 200 (or 100) of session num = 1.  This was necessary because rats needed many sessions to learn. 

_NB: You can **concatenate** all `IDS` sessions (or `EDS` sessions) together into 1 long “problem solving session”**_

- *IDS vs EDS*:
    - Summary:
        - `IDS: VIS --> VIS`
        - `EDS: VIS --> AUD`
    - The task always starts with an IDS, It is followed by an EDS. that is followed by a 2nd IDS and a 2nd EDS. This means that if a rat learned that `auditory` (`AUD`) was the relevant modality, then the first IDS would have new AUD AND `visual` (`VIS`) stimuli, but the rat would need to learn that AUD was relevant and should discriminate the novel compound AUD-VIS stimuli in the AUD modality. The EDS would be new AUD-VIS stimuli again but they rat should learn that VIS is the relevant modailty and they should discriminate in that modality. After that, they would do another `IDS` (`VIS --> VIS`) and another `EDS` (`VIS --> AUD`).

- *AUD vs VIS*: Approx Half of the rats started with AUD and half started with VIS.

- *Task Difficulty*: The key manipulation was that, prior to each EDS, we wanted to expose half of the rats to 2 sessions of the stimuli that they had just learned in the prior IDS to the same stimuli but the NoGo stimulus should be difficult to discriminate from the Go stimulus. This means that the learned rewarded stimulus (the Go stimulus) remained the rewarded stimulus. The only difference from what the rat just learned, is that the NoGo stimulus was a bit harded to tell apart from the Go stimulus. The stimuli in the irrelevant modality remained unchanged (they were still irrelevant and were also easy to tell apart). The other half of the rats recieved 2 sessoins of the same old easy to discriminate stimuli that they had just learned in the prior IDS. In other words, for 2 sessions prior to the EDS, all rats got the same Go stimuli from the previously learned IDS, but half of them got a NoGo stimulus that was a bit harder to discriminate from the Go stimulus. We call these two sessions EDS_BL_Easy and EDS_BL_Hard.

- *Harder to discriminate*: How do we define “a bit harder to discriminate”? Well, just after the IDS, we run a “Stair” which is a psychophysical staircase for all rats. This was used to determine the NoGo stimulus adjustment needed (learned in prior IDS) to make them do 70% correct. We run this in all rats to make rats have similar exposure to task but we only use the resulting NoGo stmulus in half of the rats that will get the EDS_BL_Hard condition before the upcoming EDS.

- *RE*: We were required for ethical reasons to give rats a 2 day break after learning each shift. WE needed to make sure they still remember the stimuli from the last sessions, therefore, after the IDS there was a 2 day break. We gave an RE session to “re-establish” their performance and make sure they remember the IDS. We then ran the staircase (Stair). Then, they got the manipulaiton of perceptual attention (EDS_BL_Easy or EDs_BL_HarD) and then they got the EDS.
this would repeat for the 2nd IDS/EDS pair.





### StimCode

Logical coding for stimulus class:

- `0=nogo`,
- `1=go`

On 'go' trials, the stimulus displayed is one you should respond to (press the key). Vice versa for 'nogo'.


### CondNum

There's no unique ordering of sessions within rats, so we provide one: code a condition number field for unique orderings (with SessionNum).

### Deterministic Experiments

The experiments deterministic, I.e reward was given with probability=1.

### Omissions

`omission` refers to omission error, when the trial is 'go' and the subject doesn't respond.


----
# Human Experimental Data
----

**Human Stages**

The procedure for `rats` was slightly different, when they had to repeat some stages after the break. For `human` the stages are simpler:
- `ids1` or `ids2` - inter-dimensional shift (1st or 2nd)
- `staircase1` or `staircase2`
- `ams1` or `ams2` - attention manipulation sessions. In the rats data that would be EDS_BL_Easy and EDS_BL_Hard if I am not mistaken
- `eds1` and `eds2` - extra-dimensional shift
- `SessionNum` - the number of the session in one stage. For example, some of the participants had several sessions before they performed well enough to move to the next one. So you can have SessionType “training” and “SessionNum” 1,2,3 if this participant had 3 sessions before they could move to the next stage.
- `CondNum` - stage number:
    - training = 1
    - ids1 = 2
    - staircase1 = 3
    - ams1 = 4
    - eds1 = 5
    - ids2 = 6
    - staircase2 = 7
    - ams2 = 8
    - eds2 = 9




----
# Reinforcement Learning Background
----

[Wiki Q-Learning](https://en.wikipedia.org/wiki/Q-learning#Implementation)


----
# Rescorla-Wagner (RW) Model
----

Python implementation of the Rescorla-Wagner model.

This served as the basis of our implementation.

[Rescorla-Wagner Model](https://shawnrhoads.github.io/gu-psyc-347/module-03-01_Models-of-Learning.html)


----
# DQN: Deep Q-Learning
----

- Action Space: `[0,1]`, `Go=1), NoGo=0`
- Response Space: [0,1], `Correct=1`, `Incorrect=0`
- State Space: `n`-dimensional vector of `n` features. 

Initially we'll ignore the different types of response (Correct | Stimulus).

DQN is likely unneccessary for our purposes, but a good starting point for understanding the basics of reinforcement learning.

We will probably not need some a complex data encoding model.

[pyTorch implementation of DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)


