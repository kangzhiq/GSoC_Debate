#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:31:39 2020

@author: zxk93
"""
import pickle
from tqdm import tqdm
import numpy as np

pickle_dir = 'Combined/'
nb_candidate = 23
nb_emotion = 7
nb_keypoints = 12

debate_lst = ['Debate1Night1', 'Debate1Night2', 'Debate2Night1', 'Debate2Night2']
for i in range(3, 12):
    debate_lst.append('Debate{}'.format(i))

emotion_lst = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
can_lst =   ['John_Delaney',
             'Jay_Inslee',
             'Bill_de_Blasio',
             'Tim_Ryan',
             'Cory_Booker',
             'Amy_Klobuchar',
             "Beto_O'Rourke",
             'Elizabeth_Warren',
             'Tulsi_Gabbard',
             'Julian_Castro',
             'Michael_Bennet',
             'Joe_Biden',
             'Pete_Buttigieg',
             'Kirsten_Gillibrand',
             'Kamala_Harris',
             'John_Hickenlooper',
             'Bernie_Sanders',
             'Eric_Swalwell',
             'Marianne_Williamson',
             'Andrew_Yang',
             'Tom_Steyer',
             'Steve_Bullock',
             'Michael_Bloomberg']   

new_field = ['nose', 
             'neck', 
             'rshoulder', 
             'relbow', 
             'rwrist', 
             'lshoulder', 
             'leldow', 
             'lwrist', 
             'reye', 
             'leye', 
             'rear', 
             'lear', 
             'ref_dist']

valid_keypoints = ['nose', 
                 'rshoulder', 
                 'relbow', 
                 'rwrist', 
                 'lshoulder', 
                 'leldow', 
                 'lwrist', 
                 'reye', 
                 'leye', 
                 'rear', 
                 'lear']
pose_lst = []
for debate in debate_lst:
    debate_pickle_path = pickle_dir +  debate + '_per_emotion.pickle'
    with open(debate_pickle_path, 'rb') as handle:
        pose = pickle.load(handle)
    pose_lst.append(pose)

pose_avg = [[[0 for i in range(nb_keypoints)] for j in range(nb_emotion)] for k in range(nb_candidate)]    
count = [[[0 for i in range(nb_keypoints)] for j in range(nb_emotion)] for k in range(nb_candidate)]
std = [[[[] for i in range(nb_keypoints)] for j in range(nb_emotion)] for k in range(nb_candidate)]
for idx_video in range(len(pose_lst)):
    info = pose_lst[idx_video]
    for k in range(nb_candidate):
        for j in range(nb_emotion):
            for i in range(nb_keypoints):
                if info[k][j][i] != -1:
                    pose_avg[k][j][i] += info[k][j][i]
                    count[k][j][i] += 1
                    std[k][j][i].append(info[k][j][i])

for k in range(nb_candidate):
    for j in range(nb_emotion):
        for i in range(nb_keypoints):
            if pose_avg[k][j][i] != 0:
                pose_avg[k][j][i] /= count[k][j][i]
                std[k][j][i] = np.std(std[k][j][i])
            else:
                pose_avg[k][j][i] = 0
                std[k][j][i] = 0




############################
## Plot delta(x) w.r.t emotion
############################        
import matplotlib.pyplot as plt
import numpy as np

######### 4 keypoints
idx_can = 18
data = np.array(pose_avg[idx_can])
std_can = np.array(std[idx_can])

data_std = std_can[:,(0, 2, 3, 4)]/3   

    
    
length = len(data)
x_labels = emotion_lst

# Set plot parameters
fig, ax = plt.subplots()
width = 0.2 # width of bar
x = np.arange(length)

ax.bar(x, data[:,0], width, color='#000080', label='nose', yerr=data_std[:,0])
ax.bar(x + width, data[:,2], width, color='#0F52BA', label='rshoulder', yerr=data_std[:,1])
ax.bar(x + (2 * width), data[:,3], width, color='#6593F5', label='relbow', yerr=data_std[:,2])
ax.bar(x + (3 * width), data[:,4], width, color='#73C2FB', label='rwrist', yerr=data_std[:,3])
      
       
ax.set_ylabel('Delta_x')
ax.set_ylim(0, 0.3)
ax.set_xticks(x + width + width/2)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Emotion')
ax.set_title(can_lst[idx_can])
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=1, c='k', alpha=.3)

fig.tight_layout()
plt.show()


########### 9 keypointws

idx_can = 16
data = np.array(pose_avg[idx_can])
std_can = np.array(std[idx_can])

data_std = std_can[:,(0, 2, 3, 4, 5, 6, 7)]/3    

    
length = len(data)
x_labels = emotion_lst

# Set plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(10, 8,forward=True )
width = 0.2 # width of bar
x = np.arange(length)*1.5

ax.bar(x, data[:,0], width, color='#000080', label='nose', yerr=data_std[:,0])
ax.bar(x + width, data[:,2], width, color='#0F52BA', label='rshoulder', yerr=data_std[:,1])
ax.bar(x + (2 * width), data[:,3], width, color='#6593F5', label='relbow', yerr=data_std[:,2])
ax.bar(x + (3 * width), data[:,4], width, color='#73C2FB', label='rwrist', yerr=data_std[:,3])
ax.bar(x + (4 * width), data[:,5], width, color='#0F52BA', label='lshoulder', yerr=data_std[:,3])
ax.bar(x + (5 * width), data[:,6], width, color='#6593F5', label='lelbow', yerr=data_std[:,3])
ax.bar(x + (6 * width), data[:,7], width, color='#73C2FB', label='lwrist', yerr=data_std[:,3])

       
ax.set_ylabel('Delta_x')
ax.set_ylim(0, 0.3)
ax.set_xticks(x + width + width/2)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Emotion')
ax.set_title(can_lst[idx_can])
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=1, c='k', alpha=.3)

fig.tight_layout()
plt.show()
        

############################
## Avearage emotion
############################ 

pose_avg_aggre_emo = []
for i in range(nb_candidate):
    aggre_emo = []
    for j in range(nb_emotion):
        val = 0
        val_count = 0 
        for k in range(nb_keypoints):
            if pose_avg[i][j][k] != 0:
                val += pose_avg[i][j][k]
                val_count += 1
        if val_count != 0:
            val = val/val_count
        aggre_emo.append(val)
    pose_avg_aggre_emo.append(aggre_emo)

emo_avg = [0 for i in range(nb_emotion)]
for i in range(nb_emotion):
    count = 0
    for j in range(nb_candidate):
        val = pose_avg_aggre_emo[j][i]
        if val != 0:
            emo_avg[i] += val
            count += 1
    if count != 0:
        emo_avg[i] /= count



## Plot
x = np.arange(nb_emotion)

fig, ax = plt.subplots()
plt.bar(x, emo_avg)
plt.xticks(x, emotion_lst)
plt.show()
############################
## Variation over debates
############################ 

idx_can = 7
idx_emo = 1
delta_x_over_debate_lst = []
for idx_video in range(len(pose_lst)):
    info = pose_lst[idx_video]
    delta_x_over_debate_lst.append(info[idx_can][idx_emo])
    
delta_x_over_debate_lst = np.array(delta_x_over_debate_lst)
length = 5
x_labels = debate_lst

# Set plot parameters
fig, ax = plt.subplots()
width = 0.2 # width of bar
x = np.arange(length)

ax.bar(x, delta_x_over_debate_lst[5:10,0], width, color='#000080', label='nose')
ax.bar(x + width, delta_x_over_debate_lst[5:10,2], width, color='#0F52BA', label='rshoulder')
ax.bar(x + (2 * width), delta_x_over_debate_lst[5:10,3], width, color='#6593F5', label='relbow')
ax.bar(x + (3 * width), delta_x_over_debate_lst[5:10,4], width, color='#73C2FB', label='rwrist')
      
       
ax.set_ylabel('Delta_x')
ax.set_ylim(0, 0.5)
ax.set_xticks(x + width + width/2)
ax.set_xticklabels(debate_lst[5:10])
ax.set_xlabel('Debate')
ax.set_title(can_lst[idx_can])
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=1, c='k', alpha=.3)

fig.tight_layout()
plt.show()

############################
## Avearage candidate
############################ 

can_avg = [0 for i in range(nb_candidate)]
for i in range(nb_candidate):
    count = 0
    for j in range(nb_emotion):
        val = pose_avg_aggre_emo[i][j]
        if val != 0:
            can_avg[i] += val
            count += 1
    if count != 0:
        can_avg[i] /= count

can_aggre_avg = np.mean(can_avg)

## Plot
x = np.arange(nb_candidate)
money = can_avg

fig, ax = plt.subplots()
fig.set_size_inches(10, 8,forward=True )
plt.bar(x, money)
ax.set_ylim(0.02, 0.1)
plt.hlines(can_aggre_avg, xmin=0, xmax =nb_candidate-1 ,linestyles='dashed')
plt.xticks(x, can_lst)
ax.set_title('Average delta_x of each candidate')
fig.autofmt_xdate()
plt.show()


################
## Correlation
################
import seaborn as sns
import matplotlib.pyplot as plt

def CorrMtx(df, dropDuplicates = True):

    # Your dataset is already a correlation matrix.
    # If you have a dateset where you need to include the calculation
    # of a correlation matrix, just uncomment the line below:
    # df = df.corr()

    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:    
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sns.set_style(style = 'white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap, 
                    xticklabels=valid_keypoints + emotion_lst,
                    yticklabels=valid_keypoints + emotion_lst,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap, 
                    xticklabels=valid_keypoints + emotion_lst,
                    yticklabels=valid_keypoints + emotion_lst,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)


## Seelcting average is wrong
## We should concatenate row by row of each frame

corre_info_lst = []
for debate in debate_lst:
    debate_pickle_path = pickle_dir +  debate + '_corr_info.pickle'
    with open(debate_pickle_path, 'rb') as handle:
        corre = pickle.load(handle)
    corre_info_lst.append(corre)

corre_lst = []
idx_select = np.zeros(nb_emotion+nb_keypoints)
idx_select[1] = 1
idx_select = idx_select.astype(int)
for idx_debate in range(len(corre_info_lst)):
    for idx_can in range(nb_candidate):
        if corre_info_lst[idx_debate][idx_can][0] != -1:
            info = np.array(corre_info_lst[idx_debate][idx_can])
            if not np.any(info == -1):
                val = info[info != 0]
                corre_lst.append(val)

corre_lst = np.array(corre_lst)
corr = np.corrcoef(corre_lst, rowvar=False)
CorrMtx(corr, dropDuplicates = True)


#correlation on each complete samples, not on average
valid_corre_info_lst = []
for debate in debate_lst:
    debate_pickle_path = pickle_dir +  debate + '_valid_corr_info.pickle'
    with open(debate_pickle_path, 'rb') as handle:
        corre = pickle.load(handle)
    if len(corre) != 0:
        for i in range(len(corre)):
            valid_corre_info_lst.append(corre[i])

valid_corre_info_lst = np.array(valid_corre_info_lst)
corr = np.corrcoef(valid_corre_info_lst, rowvar=False)
CorrMtx(corr, dropDuplicates = True)


################################
## Speaking or not + emotion
################################
speaking_info_lst = []
for debate in debate_lst:
    debate_pickle_path = pickle_dir +  debate + '_speaking_emotion.pickle'
    with open(debate_pickle_path, 'rb') as handle:
        data = pickle.load(handle)
    speaking_info_lst.append(data)

idx_can = 7

speaking_lst = np.array([[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]).astype(float)
count_speaking_lst = np.array([[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]).astype(float)
no_speaking_lst = np.array([[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]).astype(float)
count_no_speaking_lst = np.array([[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]).astype(float)

for i in range(len(speaking_info_lst)):
    speaking_lst += np.array(speaking_info_lst[i][idx_can][0][0])
    count_speaking_lst += np.array(speaking_info_lst[i][idx_can][0][1])
    no_speaking_lst += np.array(speaking_info_lst[i][idx_can][1][0])
    count_no_speaking_lst += np.array(speaking_info_lst[i][idx_can][1][1])
    

for j in range(nb_emotion):
    for i in range(nb_keypoints):
        if speaking_lst[j][i] != 0:
            speaking_lst[j][i] /= count_speaking_lst[j][i]
        else:
            speaking_lst[j][i] = 0
        if no_speaking_lst[j][i] != 0:
            no_speaking_lst[j][i] /= count_no_speaking_lst[j][i]
        else:
            no_speaking_lst[j][i] = 0      

## Plot            
######### 4 keypoints

data = np.array(no_speaking_lst) 
    
length = len(data)
x_labels = emotion_lst

# Set plot parameters
fig, ax = plt.subplots()
width = 0.2 # width of bar
x = np.arange(length)

ax.bar(x, data[:,0], width, color='#000080', label='nose')
ax.bar(x + width, data[:,2], width, color='#0F52BA', label='rshoulder')
ax.bar(x + (2 * width), data[:,3], width, color='#6593F5', label='relbow')
ax.bar(x + (3 * width), data[:,4], width, color='#73C2FB', label='rwrist')
      
       
ax.set_ylabel('Delta_x')
ax.set_ylim(0, 0.3)
ax.set_xticks(x + width + width/2)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Emotion')
ax.set_title(can_lst[idx_can]+': Not speaking')
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=1, c='k', alpha=.3)

fig.tight_layout()
plt.show()           


## Use this data to calculate the overall average
emo_avg = [0 for i in range(nb_emotion)]
count = [0 for i in range(nb_emotion)]

for k in range(len(speaking_info_lst)):
    for j in range(nb_candidate):
        for i in range(nb_emotion):
            emo_avg[i] += np.sum(speaking_info_lst[k][j][0][0][i])
            count += np.sum(speaking_info_lst[k][j][0][1][i])
            emo_avg[i] += np.sum(speaking_info_lst[k][j][1][0][i])
            count += np.sum(speaking_info_lst[k][j][1][1][i])

for i in range(nb_emotion):
    emo_avg[i] /= count[i]
    
x = np.arange(nb_emotion)

fig, ax = plt.subplots()
plt.bar(x, emo_avg)
plt.xticks(x, emotion_lst)
plt.show()