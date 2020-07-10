#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:12:06 2020

@author: zxk93


This file is used to further process the data, including assigning the keypoints
to a person, performing interpolation for missing value, calculating the posture
movement and normalizing the distance.

"""

# TODO: Identify the same person
# TODO: interpolation
# TODO: normalization of the values
# TODO: defined delta x? idea: delta horizaontal and delta vertical, 


import json
import csv
import os
import numpy as np
import json

from plot_keypoints import *

import pickle
from tqdm import tqdm

def num2time(num):
    seconds = num % 60
    minutes = num / 60
    hours = minutes / 60 
    minutes = minutes % 60
    
    return "%02d:%02d:%02d"%(hours, minutes, seconds)

def time2num(time):
    nums = time.split(':')
    if int(nums[1]) >= 60 or int(nums[1]) < 0:
        print "Invalid minute: ", time
        return
    if int(nums[2]) >= 60 or int(nums[2]) < 0:
        print "Invalid second: ", time
        return 
    return int(nums[0])*3600 + int(nums[1])*60 + int(nums[2])        

def get_points_dist(pt1, pt2):
    return np.sqrt(((pt1 - pt2)**2).sum())

def get_ref_dist(kps):
    if np.all(kps[0] != 0) and np.all(kps[1] != 0):
        return get_points_dist(kps[0,:2], kps[1, :2])
    return -1

def get_corres(kp, candidates):
    if np.all(kp[0] == 0):
        return -1
    ref_dist = get_ref_dist(kp)
    for i in range(len(candidates)):
        kp_can = candidates[i]
        if np.all(kp_can[0] ==0):
            continue
        if ref_dist == -1: ref_dist = get_ref_dist(kp_can)
        if ref_dist == -1: continue
        head_dist = get_points_dist(kp[0, :2], kp_can[0, :2])
        # The same person
        if head_dist/ref_dist < 0.8:
            return i
    return -1


def interpolate(kp, pre, post):
    # The interpolation is highly dependent on the 
    # We assume that the nose and neck are available.
    idx_pre = get_corres(kp, pre)
    idx_post = get_corres(kp, post)
    has_run = 0
    if idx_pre != -1 and idx_post != -1:
        # Do interpolation
        kp_pre = pre[idx_pre]
        kp_post = post[idx_post]
        for j in np.arange(2, len(kp)):
            if np.all(kp[j] == 0):
                # If we can find value for interpolation
                if np.all(kp_pre[j]!=0) and np.all(kp_post[j]!=0):
                    kp[j] = (kp_pre[j] + kp_post[j])/2
                    has_run = 1
    return has_run
                    
                
####################
## Loading data
####################

debate = 'Debate1Night1'
json_pickle_path = 'json/{}.pickle'.format(debate)
ref_dist_path = 'candidate/can_ref_dist.pickle'
csv_path = 'tables/{}.csv'.format(debate.lower())
save_path = 'Combined/' 

with open(json_pickle_path, 'rb') as handle:
    keypoints_lst = pickle.load(handle)

with open(ref_dist_path, 'rb') as handle:
    ref_kp_dist_lst = pickle.load(handle)

nb_frame = len(keypoints_lst)

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
# Check the correspondence
assert nb_frame == nb_info


####################
## Intepolation:
####################
count_interpolation = 0
count_valid_frame = 0
for i in np.arange(1, nb_frame-1):
    if info[i][1] == '-1':
        continue    
    count_valid_frame += 1
    if len(keypoints_lst[i]) == 0:
        print 'csv > 0 but keypoint = 0: ', i
    for idx_kp in range(len(keypoints_lst[i])):
        kp = keypoints_lst[i][idx_kp]
        if np.any(np.all(kp[:, :2] == 0, axis = 1)):
            if info[i-1][1]=='-1' or info[i+1][1]=='-1':
                break
            res = interpolate(kp, keypoints_lst[i-1], keypoints_lst[i+1])
            if res == 1:
                count_interpolation += 1
#            if res == 1:
#                load_path_pre = "Frames_fps1/{}/frame{:06d}.jpg".format(debate, i-1)
#                im_pre = cv2.imread(load_path_pre)
#                plot_keypoint_on_image(im_pre, keypoints[i-1], "test/frames{:06d}.jpg".format(i-1))
#        
#                load_path = "Frames_fps1/{}/frame{:06d}.jpg".format(debate, i)
#                im = cv2.imread(load_path)
#                plot_keypoint_on_image(im, keypoints[i], "test/frames{:06d}.jpg".format(i))
#
#                load_path_post = "Frames_fps1/{}/frame{:06d}.jpg".format(debate, i+1)
#                im_post = cv2.imread(load_path_post)
#                plot_keypoint_on_image(im_post, keypoints[i+1], "test/frames{:06d}.jpg".format(i+1))


print 'Total valid frame: ', count_valid_frame
print 'Interpolation: ', count_interpolation            
    

####################
## Combine info:
####################
# number of feld for each candidate in csv
nb_candidate = 23
nb_field = 15

new_field = ['nose', 'neck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'leldow', 'lwrist', 'reye', 'leye', 'rear', 'lear', 'ref_dist']
nb_keypoints = 12
nb_new_field = len(new_field)    
# Total number of field of each candidate
nb_total_field = nb_new_field + nb_field

#combined = - np.ones((nb_frame, 2+nb_candidate*(nb_field + nb_new_field)))
combined = [['-1' for i in range(2+nb_candidate*(nb_field + nb_new_field))] for j in range(nb_frame)]
#for idx_frame in tqdm(range(nb_frame)):
count_combined = 0
count_on_screen = 0
for idx_frame in range(0, nb_frame):
    combined[idx_frame][:2] = info[idx_frame][:2]
    # no candidate is on screen
    if info[idx_frame][1] == '-1':
        continue
    keypoints_frame = keypoints_lst[idx_frame]
    for idx_can in range(nb_candidate):
        on_screen = info[idx_frame][2+idx_can*nb_field] != '-1'
        # Load information to combined table for on screen candidate
        if on_screen:
            count_on_screen += 1
            combined[idx_frame][2+idx_can*nb_total_field:17+idx_can*nb_total_field] = info[idx_frame][2+idx_can*nb_field:17+idx_can*nb_field]
            x = int(info[idx_frame][4+idx_can*nb_field])
            y = int(info[idx_frame][5+idx_can*nb_field])
            h = int(info[idx_frame][6+idx_can*nb_field])
            w = int(info[idx_frame][7+idx_can*nb_field])

            no_found = True
            for keypoints in keypoints_frame:
                # Find corresponding pose info    
                nose = keypoints[0]
                if y <= nose[1] <= y+h and x <= nose[0] <= x+w:
                    no_found = False

                    # We center the distance with respect to neck
                    neck = keypoints[1]
                    if not np.all(neck == 0):                                        
                        count_combined += 1
                        for i in range(nb_keypoints):
                            # We only consider non zero value
                            kp = keypoints[i]
                            if not np.all(kp == 0):
                                kp[:2] -= neck[:2]
                                # coordinate is at (x, y)
                                coor = '('+ str(kp[0]) + ', ' + str(kp[1]) + ')'
                                combined[idx_frame][2+idx_can*nb_total_field+nb_field+i] = coor
                                
                        # Append ref distance
                        # Nose and neck != 0 already satisfied
                        dist = get_points_dist(keypoints[0][:2], keypoints[1][:2])
                        dist = str(dist)

                        combined[idx_frame][2+idx_can*nb_total_field+nb_field+nb_keypoints] = dist
                    
            if no_found:
                print 'idx_frame: ', idx_frame
                print 'idx_can: ', idx_can

#                break
#    if no_found:
#        break
            
####################
## Verification combine
####################                

#idx_frame = 669
#idx_can = 3
        
#idx_frame=824
#idx_can = 2

idx_frame = 800
while info[idx_frame][1] == '-1':
    idx_frame += 1  
       
x = int(info[idx_frame][4]) 
idx_can = 0 
while int(x) == -1:
    idx_can += 1
    x = int(info[idx_frame][4+idx_can*nb_field])
    y = int(info[idx_frame][5+idx_can*nb_field])
    h = int(info[idx_frame][6+idx_can*nb_field])
    w = int(info[idx_frame][7+idx_can*nb_field])
    
idx_frame = 669
import matplotlib.pyplot as plt
im = cv2.imread("Frames_fps1/{}/frame{:06d}.jpg".format(debate, idx_frame))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.imshow(plot_bounding_box_on_image(im, x, y, h, w, [255, 0, 0]))
temp = plot_bounding_box_on_image(im, x, y, h, w, [255, 0, 0])
plt.imshow(plot_keypoint_on_image(temp, keypoint_set=keypoints_lst[idx_frame], color=[0, 255, 0]))
plt.imshow(im[y:y+h, x:x+w, :])

im = cv2.imread("Frames_fps1/{}/frame{:06d}.jpg".format(debate, idx_frame))
plot_keypoint_on_image(im, keypoint_set=keypoints_lst[idx_frame], save_path='test/frame{:06d}.jpg'.format(idx_frame))

print combined[idx_frame][17+idx_can*nb_total_field:2+(idx_can+1)*nb_total_field]


####################
## Delta x
####################

def str2array(coordinate):
    coor = coordinate.split(',')
    x = float(coor[0][1:])
    y = float(coor[1][1:-1])
    return np.array([x, y])

# For each record, relace the keypoint coor by the delta_x.
for idx_frame in range(nb_frame - 1):
    frame_current = combined[idx_frame]
    frame_post = combined[idx_frame + 1]
    
    for idx_can in range(nb_candidate):
        on_screen_current = frame_current[2+idx_can*nb_total_field] != '-1'
        on_screen_post = frame_post[2+idx_can*nb_total_field] != '-1'
        # If possible to calculate delta_x
        if on_screen_post and on_screen_current:
            # 17 = 2 + nb_field
            keypoints_current = frame_current[17+idx_can*nb_total_field:17+idx_can*nb_total_field+nb_keypoints]
            keypoints_post = frame_post[17+idx_can*nb_total_field:17+idx_can*nb_total_field+nb_keypoints]
                            
            for i in range(nb_keypoints):

                kp_current = keypoints_current[i]
                kp_post = keypoints_post[i]
                
                if kp_current == '-1' or kp_post == '-1':
                    dist = '-1'
                    
                else:                
                    kp_current = str2array(kp_current)
                    kp_post = str2array(kp_post)
                    # zero here means the missing information
                    if np.all(kp_post == 0) or np.all(kp_current == 0):
                        dist = '-1'
                    else:
                        dist = str(get_points_dist(kp_current, kp_post))

                frame_current[17+idx_can*nb_total_field+i] = dist
        # If only one is on screen, meaning last frame or missing frame we, we cannot
        # calculate the delta_x
        else:
            for i in range(nb_new_field):
                frame_current[2+idx_can*nb_total_field+nb_field+i] = '-1'
            
            

####################
## Verification delta_x
####################                

#idx_frame = 669
#idx_can = 3
        
#idx_frame=824
#idx_can = 2

idx_frame = 500
while combined[idx_frame][1] == '-1':
    idx_frame += 1  
       
x = int(combined[idx_frame][4]) 
idx_can = 0     
while int(x) == -1:
    idx_can += 1
    x = int(combined[idx_frame][4+idx_can*nb_total_field])
    y = int(combined[idx_frame][5+idx_can*nb_total_field])
    h = int(combined[idx_frame][6+idx_can*nb_total_field])
    w = int(combined[idx_frame][7+idx_can*nb_total_field])

print combined[idx_frame+2][17+idx_can*nb_total_field:2+(idx_can+1)*nb_total_field]

for i in range(nb_frame):
    if combined[i][17] != '-1':
        print(combined[i][17])


####################
## Nomalization
####################       


count_no_ref_dist = 0     
count_final = 0    
for idx_frame in range(nb_frame - 1):
    frame = combined[idx_frame]
    
    for idx_can in range(nb_candidate):
        on_screen = frame[2+idx_can*nb_total_field] != '-1'
        if on_screen:
            count_final += 1
            # 17 = 2 + nb_field
            keypoints = frame[17+idx_can*nb_total_field:17+idx_can*nb_total_field+nb_keypoints]      
            ref_dist = float(frame[17+idx_can*nb_total_field+nb_keypoints]) 
            if ref_dist == -1:
                # We assume that the ref_dist is avaible for all frames
                # TODO: deal with the case where ref_dist = -1
                ref_dist = 1
                # print 'No ref dist available at frame ', idx_frame, ', candidate:', idx_can
                count_no_ref_dist += 1
            for i in range(nb_keypoints):
                if keypoints[i] != '-1':
                    kp = float(keypoints[i]) / ref_dist 
                    if kp > 5:
                        kp = -1
                    frame[17+idx_can*nb_total_field+i] = str(kp)
                    
                    
                
# Fianl result
# in total 5904 frames
# 774 no ref dist
                
print combined[idx_frame+1][2+idx_can*nb_total_field:2+(idx_can+1)*nb_total_field]

####################
## Analysis
####################  
delta_x_lst = []
#delta_x_lst.append(new_field[:nb_keypoints])
for idx_can in range(nb_candidate):
    value = [0 for i in range(nb_keypoints)]
    count = [0 for i in range(nb_keypoints)]
    for idx_frame in range(nb_frame):
        frame = combined[idx_frame]
        on_screen = frame[2+idx_can*nb_total_field] != '-1'
        if on_screen:
            keypoints = frame[17+idx_can*nb_total_field:17+idx_can*nb_total_field+nb_keypoints]  
            #print(keypoints)
            for idx_kp in range(nb_keypoints):
                if keypoints[idx_kp] != '-1':
                    value[idx_kp] += float(keypoints[idx_kp])
                    count[idx_kp] += 1
    for i in range(nb_keypoints):
        if count[i] > 0:
            value[i] = value[i]/count[i]
        else:
            value[i] = -1
    print(count)
    delta_x_lst.append(value)

res = np.array(delta_x_lst)


np.savetxt('Debate1Night1_with_coordinate.csv', combined, delimiter=';', fmt= '%s')