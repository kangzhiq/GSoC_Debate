#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:37:31 2020

@author: zxk93
"""

import csv
import os
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


debate = 'Debate3'
json_pickle_path = 'json_fps15/{}.pickle'.format(debate)
ref_dist_path = 'candidate/can_ref_dist.pickle'
csv_path = 'Combined/{}_pose.csv'.format(debate)
is_speaking_path = 'tables_speaking/{}.csv'.format(debate)
save_path = 'Combined/' 
fps=15
nb_half = fps/2

# Read csv
table = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        table.append(row)

# table 0 is the header
header = table[0]
final_table = table[1:]



