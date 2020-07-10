#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:56:51 2020

@author: zxk93
"""

import json
import csv
import os
import numpy as np
import json

import pickle
from tqdm import tqdm

from plot_keypoints import *


def get_points_dist(pt1, pt2):
    return np.sqrt(((pt1 - pt2)**2).sum())

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


can_json_dir = 'candidate/'
nb_candidate = 23
can_keypoints_lst = []

for i in tqdm(range(nb_candidate)):   
    filename = can_json_dir+'can_%02d_keypoints.json'%(i )
    with open(filename) as f:
        data = json.load(f)
    can_keypoints_lst.append(extract_keypoints(data))
	




##########
## Verification
##########
import matplotlib.pyplot as plt

idx_can = 1
im = cv2.imread("{}can_{:02d}.jpg".format(can_json_dir, idx_can))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#plt.imshow(im)
plt.imshow(plot_keypoint_on_image(im, keypoint_set=can_keypoints_lst[idx_can], color=[0, 255, 0]))

## template distance:
ref_kp_dist = ['nose2neck', 'neck2rshoulder', 'neck2lshoulder', 'lshoudler2rshoulder']
nb_ref_kp_dist = len(ref_kp_dist)

ref_kp_dist_lst = []
for i in range(nb_candidate):
    # we know that there is only one person in each image
    keypoints = can_keypoints_lst[i][0]
    idx_nose = 0
    idx_neck = 1
    idx_rshoulder = 2
    idx_lshoulder = 5
    temp = []
    temp.append(get_points_dist(keypoints[idx_nose, :2], keypoints[idx_neck, :2]))
    temp.append(get_points_dist(keypoints[idx_rshoulder, :2], keypoints[idx_neck, :2]))
    temp.append(get_points_dist(keypoints[idx_lshoulder, :2], keypoints[idx_neck, :2])) 
    temp.append(get_points_dist(keypoints[idx_lshoulder, :2], keypoints[idx_rshoulder, :2]))
    ref_kp_dist_lst.append(temp)

# Save data
with open(can_json_dir+'can_ref_dist.pickle', 'wb') as handle:
	pickle.dump(ref_kp_dist_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
   