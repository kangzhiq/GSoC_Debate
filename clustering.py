#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:37:31 2020

@author: zxk93
"""

import csv
import os
import numpy as np
import pickle


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


# Load data
clustering_lst = []
idx_c_lst = []
emo_lst = []
idx_frame_lst = []
for debate in debate_lst:
    debate_pickle_path = pickle_dir +  debate + '_clutering_info.pickle'
    with open(debate_pickle_path, 'rb') as handle:
        data = pickle.load(handle)
    for i in range(len(data)):
        idx, idx_c, emo, sample = data[i]
        if not np.all(sample == 0):
            clustering_lst.append(sample)
            emo_lst.append(emo)
            idx_c_lst.append(idx_c)
            idx_frame = idx / 15
            idx_supple = idx % 15
            idx_frame_lst.append(debate+'_Frame'+str(idx_frame)+'_'+str(idx_supple)+'_'+can_lst[idx_c])


clustering_lst = np.array(clustering_lst)
idx_c_lst = np.array(idx_c_lst)
emo_lst = np.array(emo_lst)
idx_frame_lst = np.array(idx_frame_lst)
emo_label_lst = np.array([np.argmax(emo_lst[i]) for i in range(len(emo_lst))])

## Discard the first 30%
sum_lst = clustering_lst.sum(axis = 1)
sort_idx_lst = np.argsort(sum_lst)
nb_discard = int(len(sort_idx_lst)/10 * 3)
kpt_idx = sort_idx_lst[nb_discard:]

kpt_cluster_lst = clustering_lst[kpt_idx]
kpt_idx_c_lst = idx_c_lst[kpt_idx]
kpt_emo_lst = emo_lst[kpt_idx]
kpt_emo_label_lst = emo_label_lst[kpt_idx]
kpt_idx_frame_lst = idx_frame_lst[kpt_idx]


# Recover the class of each frame

from sklearn import svm
from sklearn.cluster import KMeans
from time import time

tt = time()
kmeans = KMeans(n_clusters=100, random_state=0).fit(kpt_cluster_lst)
print("Time: {}".format(time() - tt))

kmeans.labels_[-20:]

## Addign cluster to initial list
res_cluster = np.zeros(len(clustering_lst))
res_cluster -= 1
for i in range(len(kpt_idx)):
    idx =int( kpt_idx[i])
    res_cluster[idx] = kmeans.labels_[i]

idx_emo = 0
idx_emo_lst = emo_label_lst == idx_emo
a = res_cluster[idx_emo_lst]

unique, counts = np.unique(a, return_counts=True)
print np.asarray((unique[1:], counts[1:])).T

kpt_idx_frame_lst[kmeans.labels_==63][-30:]

for i in range(100):
    idx_cluster = i
    b = emo_label_lst[res_cluster == idx_cluster]
    unique, counts = np.unique(b, return_counts=True)
    print np.argmax((emo_lst[res_cluster == idx_cluster].mean(axis=0))[:-1])
    if np.argmax(emo_lst[res_cluster == idx_cluster].mean(axis=0)[:-1]) == 3:
        break
i
print np.asarray((unique, counts)).T

emo_lst[res_cluster == idx_cluster].mean(axis=0)


kpt_idx_frame_lst[kmeans.labels_ == 2][-20:]

np.where(idx_frame_lst == 'Debate6_Frame1286_0_Elizabeth_Warren')
idx_frame_lst[110299-5:110299+5]
kmeans.labels_[110299-5:110299+5]
kpt_idx_frame_lst[kmeans.labels_ == 27][-30:]

### We can also check the candidate in first quater and last quater

kpt_idx_c_lst[-100:]

###################################
## Kmeans with cross correlation
###################################

#### Classical K-means
center_1 = np.array([1,1])
center_2 = np.array([5,5])
center_3 = np.array([8,1])

# Generate random data and center it to the three centers
data_1 = np.random.randn(200, 2) + center_1
data_2 = np.random.randn(200,2) + center_2
data_3 = np.random.randn(200,2) + center_3

data = np.concatenate((data_1, data_2, data_3), axis = 0)


#number of samples:
n, c = data.shape[:2]
#number of clusters:
k = 3

centers_init = np.random.randn(k, c)

centers_old = np.zeros(centers_init.shape) # to store old centers
centers_new = centers_init.copy() # Store new centers

clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
count_ite = 0
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        distances[:,i] = np.linalg.norm(data - centers_new[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = centers_new.copy()
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
    count_ite += 1
centers_new[0]

#### Cross correlation
a = [0, 0, 0, 1, 3, 1, 5, 7, 1]
b = [0, 0, 0, 0, 0, 1, 3, 1, 5]
c = [0, 0, 0, 2, 8, 1, 1, 4, 2]

a = (a - np.mean(a)) / (np.std(a) * len(a))
b = (b - np.mean(b)) / (np.std(b))
c = (c - np.mean(c)) / (np.std(c))

print max(np.correlate(a, b, 'full'))
print max(np.correlate(a, c, 'full'))

#### Cross correlation Kmeans:
data = clustering_lst.copy()
n, c = data.shape[:2]
#number of clusters:
k = 3
centers_init = np.random.randn(k, c)

centers_old = np.zeros(centers_init.shape) # to store old centers
centers_new = centers_init.copy() # Store new centers

clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

count_ite = 0
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        for j in range(n):
            a = data[j]
            b = centers_new[i]
            a = (a - np.mean(a)) / (np.std(a) * len(a))
            b = (b - np.mean(b)) / (np.std(b))
            distances[j,i] = max(np.correlate(a, b, 'full'))
    # Assign all training data to closest center
    clusters = np.argmax(distances, axis = 1)
    
    centers_old = centers_new.copy()
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
    count_ite += 1
    print count_ite
centers_new[0]

