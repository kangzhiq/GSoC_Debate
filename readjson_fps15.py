#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 07:01:20 2020

@author: zxk93
"""

import json
import csv
import os
import numpy as np
import json

import pickle
from tqdm import tqdm


def extract_keypoints(data):
	keypoints = []
	people = data['people']
	if len(people) == 0:
		return keypoints

	for i in range(len(people)):
		person = people[i]
		kp_all = person["pose_keypoints_2d"]
		assert len(kp_all) == 75
		
		kp_all = np.array(kp_all)
		kp_all = kp_all.reshape((-1, 3))
		# we are only interested in the upper body
		ind = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
		keypoints.append(kp_all[ind])		

	return keypoints


start = 9
end = 11
#for idx_video in range(start, end+1):
debate = 'Debate1Night2'
json_dir = 'json_fps15/{}/'.format(debate)
csv_path = 'tables/{}.csv'.format(debate.lower())
save_path = 'json_fps15/' 
fps = 15

nb_file = len(os.listdir(json_dir))
nb_frame_from_video = nb_file / fps


keypoints_lst = []

for i in tqdm(range(nb_file)):
    if i % fps == 0:    
        filename = json_dir+'frame%06d_keypoints.json'%(i / fps)
    else:
        idx_f = i / fps
        idx_i = i % fps
        filename = json_dir+'frame%06d_%02d_keypoints.json'%(idx_f, idx_i)
    with open(filename) as f:
        data = json.load(f)
    keypoints_lst.append(extract_keypoints(data))
	

# Save data
with open(save_path+'{}.pickle'.format(debate), 'wb') as handle:
	pickle.dump(keypoints_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
