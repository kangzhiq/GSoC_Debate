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
emo_lst = []
idx_frame_lst = []
for debate in debate_lst:
    debate_pickle_path = pickle_dir +  debate + '_clutering_info.pickle'
    with open(debate_pickle_path, 'rb') as handle:
        data = pickle.load(handle)
    for i in range(len(data)):
        idx, emo, sample = data[i]
        if not np.all(sample == 0):
            clustering_lst.append(sample)
            emo_lst.append(emo)
            idx_frame_lst.append(debate+'_'+str(idx))

clustering_lst = np.array(clustering_lst)

from sklearn import svm
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(clustering_lst)
kmeans.labels_[:10]




