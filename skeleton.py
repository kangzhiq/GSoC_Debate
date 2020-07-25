#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:47:48 2020

@author: zxk93
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv

## Load file and extract keypoints for a candidate
debate = 'Debate6'
pickle_dir = 'Combined/'

csv_path = 'tables_speaking/{}.csv'.format(debate)
fps=15
nb_half = fps/2
nb_candidate = 23
# number of feld for each candidate in csv
nb_field = 15
nb_col_head = 5
nb_keypoints = 12

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
nb_new_field = len(new_field)    
# Total number of field of each candidate
nb_total_field = nb_new_field + nb_field



print 'Debate: ', debate

# Read csv
table = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        table.append(row)

# table 0 is the header
header = table[0]
info = table[1:]
nb_info = len(info)

can_lst = []
for i in range(nb_candidate):
    can_lst.append(header[nb_col_head+i*nb_field][13:])


## get keypoints of the canidate
idx_can =11

def str2array(coordinate):
    coor = coordinate.split(',')
    x = float(coor[0][1:])
    y = float(coor[1][1:-1])
    return np.array([x, y])

debate_lst = ['Debate1Night1', 'Debate1Night2', 'Debate2Night1', 'Debate2Night2']
for i in range(3, 12):
    debate_lst.append('Debate{}'.format(i))



keypoints_lst= []
emo_val_lst = []
for debate in debate_lst:
    debate_pickle_path = pickle_dir +  debate + '_combined.pickle'
    with open(debate_pickle_path, 'rb') as handle:
        combined = pickle.load(handle)

    nb_frame_fps15 = len(combined)
    
    for idx_frame in range(nb_frame_fps15):
        frame = combined[idx_frame]
        on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
        if on_screen:
            info = frame[nb_col_head+idx_can*nb_total_field:nb_col_head+(1+idx_can)*nb_total_field] 
            info = np.array(info)
            if info[nb_field+4] != '-1' or info[nb_field+7] != '-1':
                keypoints = []
                for i in range(nb_field, nb_field+nb_keypoints):
                    if info[i] != '-1':
                        kp = str2array(info[i])
                        if i == 1:
                            kp = np.array([640, 400])
                        else:                
                            kp += np.array([640, 400])
                    else:
                        kp = np.array([0, 0])
                    keypoints.append(kp)
                keypoints = np.array(keypoints)
                keypoints_lst.append(keypoints)
                emo_val_lst.append(info[6:13])


with open('skeleton/{}_keypoints_emo_scores.pickle'.format(can_lst[idx_can]), 'rb') as handle:
    (keypoints_lst, emo_val_lst) = pickle.load(handle)


with open('skeleton/{}_keypoints.pickle'.format(can_lst[idx_can]), 'rb') as handle:
    keypoints_lst = pickle.load(handle)


    
temp = []
emo_vals = [0 for i in range(nb_emotion)]
for i in range(len(keypoints_lst)):
    kps = keypoints_lst[i]
    if not np.all(kps[4] == 0) and not np.all(kps[7] == 0):
        #if (kps[5][0] - kps[2][0]) > 230:
        temp.append(kps)
        for j in range(nb_emotion):
            emo_vals[j] += float(emo_val_lst[i][j])
            
for i in range(nb_emotion):
    emo_vals[i] /= len(temp) 
               
temp = np.array(temp)
keypoints_lst_temp = temp.copy()
## Plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

save_path = 'skeleton/{}/'.format(can_lst[idx_can])

bone_list = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [0, 8], [0, 9], [8, 10], [9, 11]]

number_of_postures = len(keypoints_lst_temp)



for i in range(0, 1600):
    fig, ax = plt.subplots(1, figsize=(8, 6))
    width, height = fig.get_size_inches() * fig.get_dpi()
    canvaswidth = int(width)
    height = int(height)
#     = FigureCanvas(fig)
    
    plt.title(can_lst[idx_can])
    plt.xlim(0, 1280)
    plt.ylim(-720, 0)

    skeleton = keypoints_lst_temp[i]

    x = skeleton[:, 0]
    y = -skeleton[:, 1]

    sc = ax.scatter(x, y, s=100, c=[(0, 0, 0) for k in range(len(x))])
    for bone in bone_list:
        if not np.all(skeleton[bone[0]] == 0) and not np.all(skeleton[bone[1]] == 0):
            ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], linewidth=4, color=(0, 0, 0))
#    canvas.draw()
#    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    fig.savefig(save_path+'%06d.png'%i, bbox_inches='tight')
    plt.close(fig)
    if i % 100 == 0:
        print i

### Read plot and regroup final image
import cv2
import os

#nb_img = len(os.listdir(save_path))
nb_img = 1600
res = cv2.imread(save_path+'%06d.png'%0).astype(int)
for i in range(nb_img):
    im = cv2.imread(save_path+'%06d.png'%i).astype(int)
    
    res += im

res = res / nb_img

temp = res.copy().astype(float)
temp /= float(255)

temp2 = 1-temp
temp2 = temp2**(0.4)

temp3 = 1 - temp2
plt.imshow(temp3)


fig, ax = plt.subplots(1, figsize=(8, 6))
ax.imshow(temp3)
fig.savefig('skeleton/{}{}.png'.format(can_lst[idx_can], nb_img), bbox_inches='tight')



###### 
## Skeleton with emotion
######

ske_emo_lst = [[] for i in range(7)]

emo_ske = np.zeros((nb_emotion, 12, 2))
emo_count = [[0 for i in range(12)] for j in range(nb_emotion)]
for idx_can in [7, 11, 16]:
    with open('skeleton/{}_keypoints_emo.pickle'.format(can_lst[idx_can]), 'rb') as handle:
        ske_emo = pickle.load(handle)

    for i in range(len(ske_emo)):
        ske, emo_idx = ske_emo[i]
        if not np.all(ske[4] == 0) and not np.all(ske[7] == 0):
            if (ske[5][0] - ske[2][0]) > 180:
                ske_emo_lst[emo_idx].append(ske)
                emo_ske[emo_idx] += ske
            
                for j, kp in enumerate(ske):
                    if not np.all(kp == 0):
                        emo_count[emo_idx][j] += 1

for i in range(nb_emotion):
    for j in range(12):
        if emo_count[i][j] != 0:
            emo_ske[i, j, :] /= emo_count[i][j]


## Plot
fig, ax = plt.subplots(1, figsize=(8, 6))
plt.title(can_lst[idx_can])
plt.xlim(0, 1280)
plt.ylim(-720, 0)


color_lst = ['b','g', 'r', 'c', 'm','y', 'k']
for i in range(len(ske_emo_lst[0])):

    skeleton = ske_emo_lst[0][i]

    x = skeleton[:, 0]
    y = -skeleton[:, 1]

    sc = ax.scatter(x, y, s=100, c=[color_lst[i] for k in range(len(x))])
    for bone in bone_list:
        if not np.all(skeleton[bone[0]] == 0) and not np.all(skeleton[bone[1]] == 0):
            ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], linewidth=3, color=color_lst[i])
            



## Plot emotion simple
x = range(nb_emotion)
fig, ax = plt.subplots()
fig.set_size_inches(5.5, 2,forward=True )
plt.bar(x, emo_vals, color='#6593F5')
plt.xticks(x, emotion_lst)
ax.set_ylabel('Emotion scores')
#fig.autofmt_xdate()
plt.show()





## plot emotion
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# set the style of the axes and the text color
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams['text.color']='#333F4B'

# create some fake data
percentages = pd.Series(emo_vals, 
                        index=emotion_lst)
df = pd.DataFrame({'percentage' : percentages})
df = df.sort_values(by='percentage')

# we first need a numeric placeholder for the y axis
my_range=list(range(1,len(df.index)+1))

fig, ax = plt.subplots(figsize=(6,2.5))

# create for each expense type an horizontal line that starts at x = 0 with the length 
# represented by the specific expense percentage value.
plt.hlines(y=my_range, xmin=0, xmax=df['percentage'], color='#007ACC', alpha=0.4, linewidth=15)

# create for each expense type a dot at the level of the expense percentage value
plt.plot(df['percentage'], my_range, "o", markersize=10, color='#007ACC', alpha=0.9)

# set labels
ax.set_xlabel('Emotion scores', fontsize=14, fontweight='black', color = '#333F4B')
ax.set_ylabel('')

# set axis
ax.tick_params(axis='both', which='major', labelsize=12)
plt.yticks(my_range, df.index)

# add an horizonal label for the y axis 
fig.text(-0.23, 0.96, 'Transaction Type', fontsize=15, fontweight='black', color = '#333F4B')

# change the style of the axis spines
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)

# set the spines position
ax.spines['bottom'].set_position(('axes', -0.04))
ax.spines['left'].set_position(('axes', 0.015))

plt.savefig('hist2.png', dpi=300, bbox_inches='tight')