#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:00:41 2020

@author: zxk93
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


def get_normalization_ref(keypoints, idx_can, ref_kp_dist_lst):
    keypoints_dist_ref = ref_kp_dist_lst[idx_can]
    # Nose and neck
    confidence_lst = []
    dist = []
    # nose2neck
    if np.all(keypoints[0, :2] != 0):
        dist.append(get_points_dist(keypoints[0, :2], keypoints[1, :2]))
        confidence_lst.append(keypoints[0, 2] * keypoints[1, 2])
    else:
        dist.append(0)
        confidence_lst.append(0)
        
    # neck2rshoulder
    if np.all(keypoints[2, :2] != 0):
        dist.append(get_points_dist(keypoints[2, :2], keypoints[1, :2]))
        confidence_lst.append(keypoints[2, 2] * keypoints[1, 2])
    else:
        dist.append(0)
        confidence_lst.append(0)

    # neck2lshoulder
    if np.all(keypoints[5, :2] != 0):
        dist.append(get_points_dist(keypoints[5, :2], keypoints[1, :2]))
        confidence_lst.append(keypoints[5, 2] * keypoints[1, 2])
    else:
        dist.append(0)
        confidence_lst.append(0)

        
    conf = 0
    for i in range(len(keypoints_dist_ref)-1):
        conf += confidence_lst[i]
    if conf == 0:
        return -1
    
    temp = 0
    for i in range(len(keypoints_dist_ref)-1):
        temp += dist[i] / keypoints_dist_ref[i] * confidence_lst[i]
    scale = temp/conf
    return keypoints_dist_ref[0] * scale
                          
                
####################
## Loading data
####################
debate_lst = ['Debate1Night1', 'Debate1Night2', 'Debate2Night1', 'Debate2Night2']
for i in range(3, 12):
    debate_lst.append('Debate{}'.format(i))

for debate in debate_lst:
    debate = 'Debate6'
    json_pickle_path = 'json_fps15/{}.pickle'.format(debate)
    ref_dist_path = 'candidate/can_ref_dist.pickle'
    csv_path = 'tables_speaking/{}.csv'.format(debate)
    is_speaking_path = 'tables_speaking/{}.csv'.format(debate)
    save_path = 'Combined/' 
    fps=15
    nb_half = fps/2
    
    
    print 'Debate: ', debate
    
    with open(json_pickle_path, 'rb') as handle:
        keypoints_lst = pickle.load(handle)
    
    with open(ref_dist_path, 'rb') as handle:
        ref_kp_dist_lst = pickle.load(handle)
    
    nb_frame_fps15 = len(keypoints_lst)
    nb_frame = nb_frame_fps15 / 15
    
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
    if nb_frame != nb_info:
        nb_frame = min(nb_frame, nb_info)
    
    
    ####################
    ## Intepolation:
    ####################
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
    nb_keypoints = 12
    
    
    count_interpolation = 0
    count_valid_frame = 0
    for i in tqdm(np.arange(1, nb_frame-1)):
        if info[i][1] == '-1':
            continue    
        count_valid_frame += 1
    
        # Assume the 7 frame before and 7 frame after is related to this frame
        for idx_frame_t in range(i*fps-nb_half, i*fps+nb_half+1):
            for idx_kp in range(len(keypoints_lst[idx_frame_t])):
                kp = keypoints_lst[idx_frame_t][idx_kp]
                if np.any(np.all(kp[:, :2] == 0, axis = 1)):
                    if info[i-1][1]=='-1' or info[i+1][1]=='-1':
                        break
                    res = interpolate(kp, keypoints_lst[idx_frame_t-1], keypoints_lst[idx_frame_t+1])
                    if res == 1:
                        count_interpolation += 1
    
    
    print 'Total valid frame: ', count_valid_frame
    print 'Interpolation: ', count_interpolation            
        
    
    ####################
    ## Combine info:
    ####################
    nb_candidate = 23
    # number of feld for each candidate in csv
    nb_field = 15
    nb_col_head = 5
    
    can_lst = []
    for i in range(nb_candidate):
        can_lst.append(header[nb_col_head+i*nb_field][13:])
    
    nb_new_field = len(new_field)    
    # Total number of field of each candidate
    nb_total_field = nb_new_field + nb_field
    
    combined = [['-1' for i in range(nb_col_head+nb_candidate*nb_total_field)] for j in range(nb_frame_fps15)]
    #for idx_frame in tqdm(range(nb_frame)):
    count_combined = 0
    count_on_screen = 0
    # Combined data from csv and openpose
    for idx_frame in range(0, nb_frame):
        # 15 frame for each second(except time 0)
        range_left = max(0, idx_frame*fps-nb_half)
        range_right = min(nb_frame_fps15, idx_frame*fps+nb_half+1)
        for idx_frame_t in range(range_left, range_right):
            combined[idx_frame_t][:nb_col_head] = info[idx_frame][:nb_col_head]
            # no candidate is on screen
            if info[idx_frame][3] == '0':
                continue
            keypoints_frame = keypoints_lst[idx_frame_t]
            nb_on_screen = int(info[idx_frame][3])
            if len(keypoints_frame) != nb_on_screen:
                continue
            for idx_can in range(nb_candidate):
                on_screen = info[idx_frame][nb_col_head+idx_can*nb_field] != '-1'
                # Load information to combined table for on screen candidate
                if on_screen:
                    count_on_screen += 1
                    combined[idx_frame_t][nb_col_head+idx_can*nb_total_field:nb_col_head+nb_field+idx_can*nb_total_field] = info[idx_frame][nb_col_head+idx_can*nb_field:nb_col_head+nb_field+idx_can*nb_field]
                    x = int(info[idx_frame][nb_col_head+2+idx_can*nb_field])
                    y = int(info[idx_frame][nb_col_head+3+idx_can*nb_field])
                    h = int(info[idx_frame][nb_col_head+4+idx_can*nb_field])
                    w = int(info[idx_frame][nb_col_head+5+idx_can*nb_field])
        
                    no_found = True
                    for keypoints in keypoints_frame:
                        # Find corresponding pose info    
                        nose = keypoints[0]
                        if y <= nose[1] <= y+h and x <= nose[0] <= x+w:
                            no_found = False
        
                            # We center the distance with respect to neck
                            # TO eliminate the camera moving
                            neck = keypoints[1].copy()
                            if not np.all(neck == 0):                                        
                                count_combined += 1
                                for i in range(nb_keypoints):
                                    # We only consider non zero value
                                    kp = keypoints[i].copy()
                                    if not np.all(kp == 0):
                                        kp[:2] -= neck[:2]
                                        # coordinate is at (x, y)
                                        coor = '('+ str(kp[0]) + ', ' + str(kp[1]) + ')'
                                        combined[idx_frame_t][nb_col_head+idx_can*nb_total_field+nb_field+i] = coor
                                        
                                # Append ref distance
                                # Nose and neck != 0 already satisfied
                                dist = get_normalization_ref(keypoints, idx_can, ref_kp_dist_lst)
                                dist = str(dist)
        
                                combined[idx_frame_t][nb_col_head+idx_can*nb_total_field+nb_field+nb_keypoints] = dist
    print 'total number of candidate on screen: ', count_on_screen
    print 'number of opse info assigned: ', count_combined                       
    
    
                
    ####################
    ## Verification combine
    ####################                
    
    #idx_frame = 669
    #idx_can = 3
            
    
    #idx_frame=824
    #idx_can = 2
    #
    #idx_frame = 1588
    #while combined[idx_frame][2] == '0':
    #    idx_frame += 1  
    #       
    #x = int(info[idx_frame][2+nb_col_head]) 
    #y = int(info[idx_frame][3+nb_col_head])
    #h = int(info[idx_frame][4+nb_col_head])
    #w = int(info[idx_frame][5+nb_col_head])
    #
    #idx_can =0
    #while int(x) == -1:
    #    idx_can += 1
    #    x = int(info[idx_frame][2+nb_col_head+idx_can*nb_field])
    #    y = int(info[idx_frame][3+nb_col_head+idx_can*nb_field])
    #    h = int(info[idx_frame][4+nb_col_head+idx_can*nb_field])
    #    w = int(info[idx_frame][5+nb_col_head+idx_can*nb_field])
    #    
    #idx_frame += 1
    #import matplotlib.pyplot as plt
    #im = cv2.imread("Frames_fps1/{}/frame{:06d}.jpg".format(debate, idx_frame))
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #plt.imshow(im)
    ##plt.imshow(plot_bounding_box_on_image(im, x, y, h, w, [255, 0, 0]))
    #temp = plot_bounding_box_on_image(im, x, y, h, w, [255, 0, 0])
    #plt.imshow(plot_keypoint_on_image(temp, keypoint_set=keypoints_lst[idx_frame*fps], color=[0, 255, 0]))
    #plt.imshow(plot_keypoint_on_image(temp, keypoint_set=[keypoints], color=[0, 255, 0]))
    
    #plt.imshow(im[y:y+h, x:x+w, :])
    #
    #im = cv2.imread("Frames_fps1/{}/frame{:06d}.jpg".format(debate, idx_frame))
    #plot_keypoint_on_image(im, keypoint_set=keypoints_lst[idx_frame], save_path='test/frame{:06d}.jpg'.format(idx_frame))
    #
    #print combined[idx_frame*fps+6][nb_col_head+idx_can*nb_total_field:nb_col_head+(idx_can+1)*nb_total_field]
    
    
    ####################
    ## Delta x
    ####################
    
    def str2array(coordinate):
        coor = coordinate.split(',')
        x = float(coor[0][1:])
        y = float(coor[1][1:-1])
        return np.array([x, y])
    
    
    def get_average_delta_x(combined_lst):
        res = [i for i in combined_lst[nb_half-1]]
        keypoints = [0 for i in range(nb_candidate*nb_keypoints)]
        count = [0 for i in range(nb_candidate*nb_keypoints)]
        for i in range(len(combined_lst)):
            for idx_can in range(nb_candidate):
                for idx_kp in range(nb_keypoints):
                    temp = combined_lst[i][nb_col_head+idx_can*nb_total_field+nb_field+idx_kp]
                    if temp != '-1':
                        val = float(temp)
                        keypoints[idx_can*nb_keypoints+idx_kp] += val
                        count[idx_can*nb_keypoints+idx_kp] += 1
        for idx_can in range(nb_candidate):
            for idx_kp in range(nb_keypoints):
                idx = idx_can*nb_keypoints+idx_kp
                if count[idx] != 0:
                    res[nb_col_head+idx_can*nb_total_field+nb_field+idx_kp] = str(keypoints[idx]/count[idx])
                else:
                    res[nb_col_head+idx_can*nb_total_field+nb_field+idx_kp] = '-1'
        return res
                    
                        
    
    # For each record, relace the keypoint coor by the delta_x.
    # we average over 15 frames
    print 'Delta_x .......'
    
    final_table = []
    for idx_frame in range(nb_frame):
        delta_x_lst = []
        range_left = max(0, idx_frame*fps-nb_half)
        range_right = min(nb_frame_fps15, idx_frame*fps+nb_half)
        for idx_frame_t in range(range_left, range_right):
            # no candidate is on screen
            if info[idx_frame][3] == '0':
                continue
    
            # Calculate the delta_x based on current frame and the frame after
            frame_current = combined[idx_frame_t]
            frame_post = combined[idx_frame_t + 1]
            
            for idx_can in range(nb_candidate):
                on_screen_current = frame_current[nb_col_head+idx_can*nb_total_field] != '-1'
                on_screen_post = frame_post[nb_col_head+idx_can*nb_total_field] != '-1'
                # If possible to calculate delta_x
                if on_screen_post and on_screen_current:
                    # 17 = 2 + nb_field
                    keypoints_current = frame_current[nb_col_head+nb_field+idx_can*nb_total_field:nb_col_head+nb_field+idx_can*nb_total_field+nb_keypoints]
                    keypoints_post = frame_post[nb_col_head+nb_field+idx_can*nb_total_field:nb_col_head+nb_field+idx_can*nb_total_field+nb_keypoints]
                                    
                    for i in range(nb_keypoints):
        
                        kp_current = keypoints_current[i]
                        kp_post = keypoints_post[i]
                        
                        if kp_current == '-1' or kp_post == '-1':
                            dist = '-1'
                            
                        else:                
                            kp_current = str2array(kp_current)
                            kp_post = str2array(kp_post)
                            dist = str(get_points_dist(kp_current, kp_post))
        
                        frame_current[nb_col_head+nb_field+idx_can*nb_total_field+i] = dist
                # If only one is on screen, meaning last frame or missing frame we, we cannot
                # calculate the delta_x
                else:
                    for i in range(nb_new_field):
                        frame_current[nb_col_head+idx_can*nb_total_field+nb_field+i] = '-1'
        final_table.append(get_average_delta_x(combined[range_left: range_right]))
                
    ####################
    ## Verification delta_x
    ####################                
    
    #idx_frame = 669
    #idx_can = 3
            
    #idx_frame=824
    #idx_can = 2
    #
    #idx_frame = 81
    #while final_table[idx_frame][3] == '0':
    #    idx_frame += 1  
    #       
    #x = int(final_table[idx_frame][2+nb_col_head]) 
    #idx_can = 0     
    #while int(x) == -1:
    #    idx_can += 1
    #    x = int(final_table[idx_frame][2+nb_col_head+idx_can*nb_total_field])
    #    y = int(final_table[idx_frame][3+nb_col_head+idx_can*nb_total_field])
    #    h = int(final_table[idx_frame][4+nb_col_head+idx_can*nb_total_field])
    #    w = int(final_table[idx_frame][5+nb_col_head+idx_can*nb_total_field])
    #
    #print final_table[idx_frame][17+idx_can*nb_total_field:2+(idx_can+1)*nb_total_field]
    
    #for i in range(nb_frame):
    #    if combined[i][17] != '-1':
    #        print(combined[i][17])
    
    
    ####################
    ## Nomalization
    ####################     
    count_no_ref_dist = 0     
    count_final = 0   
    count_wrong_value = 0 
    for idx_frame in range(nb_frame - 1):
        frame = final_table[idx_frame]
        
        for idx_can in range(nb_candidate):
            on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
            if on_screen:
                count_final += 1
                # 17 = 2 + nb_field
                keypoints = frame[nb_col_head+nb_field+idx_can*nb_total_field:nb_col_head+nb_field+idx_can*nb_total_field+nb_keypoints]      
                ref_dist = float(frame[nb_col_head+nb_field+idx_can*nb_total_field+nb_keypoints]) 
                if ref_dist == -1:
                    # We assume that the ref_dist is avaible for all frames
                    # TODO: deal with the case where ref_dist = -1
                    for i in range(nb_keypoints):
                        frame[nb_col_head+nb_field+idx_can*nb_total_field+i] = '-1'
                    # print 'No ref dist available at frame ', idx_frame, ', candidate:', idx_can
                    count_no_ref_dist += 1
                    continue
    
                for i in range(nb_keypoints):
                    if keypoints[i] != '-1':
                        dist = float(keypoints[i]) / ref_dist 
                        # A manual check shows that normalized dist>0.5 is often due to the changes
                        # of scene(from one person to two or from one angle to another)
                        # TODO: Integrate the camera angle estimate to eliminate this defect
                        if dist > 0.5:
                            count_wrong_value += 1
                            dist = -1
                        frame[nb_col_head+nb_field+idx_can*nb_total_field+i] = str(dist)
    
    print 'Valid candidate: ', count_final               
    print 'Bad estiamtion: ', count_wrong_value
                    
    # Fianl result
    # in total 5904 frames
    # 774 no ref dist
    
    
    
    ####################
    ## Analysis
    ####################  
    
    # Average delta_x on each candidate of each keypoint
    # Result:
    # nb_candidate * nb_keypoints
    delta_x_lst = []
    #delta_x_lst.append(new_field[:nb_keypoints])
    for idx_can in range(nb_candidate):
        value = [0 for i in range(nb_keypoints)]
        count = [0 for i in range(nb_keypoints)]
        for idx_frame in range(nb_frame-1):
            frame = final_table[idx_frame]
            on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
            if on_screen:
                keypoints = frame[nb_col_head+nb_field+idx_can*nb_total_field:nb_col_head+nb_field+idx_can*nb_total_field+nb_keypoints]  
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
        #print(count)
        delta_x_lst.append(value)
    
    res = np.array(delta_x_lst)
    
    with open(save_path+'{}_aggregated.pickle'.format(debate), 'wb') as handle:
        	pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Average delta_x of each candidate of each keypoint w.r.t. each emotion
    # Result:
    # nb_candidate * nb_emotion * nb_keypoints
    nb_emotion = 7
    emo_lst = []
    for i in range(8, 15):
        head = header[i].split('_')
        emo_lst.append(head[1])
        
    emo_delta_x_lst = []
    for idx_can in range(nb_candidate):
        value = [[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]
        count = [[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]
        for idx_frame in range(nb_frame-1):
            frame = final_table[idx_frame]
            on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
            if on_screen:
                emo_lst = frame[nb_col_head+6+idx_can*nb_total_field:nb_col_head+13+idx_can*nb_total_field] 
                idx_emo = np.argmax(emo_lst)
                keypoints = frame[nb_col_head+nb_field+idx_can*nb_total_field:nb_col_head+nb_field+idx_can*nb_total_field+nb_keypoints]  
                #print(keypoints)
                for idx_kp in range(nb_keypoints):
                    if keypoints[idx_kp] != '-1':
                        value[idx_emo][idx_kp] += float(keypoints[idx_kp])
                        count[idx_emo][idx_kp] += 1
                        
        for idx_emo in range(nb_emotion):
            for i in range(nb_keypoints):
                if count[idx_emo][i] > 0:
                    value[idx_emo][i] = value[idx_emo][i]/count[idx_emo][i]
                else:
                    value[idx_emo][i] = -1
        #print(count)
        emo_delta_x_lst.append(value)
    
    with open(save_path+'{}_per_emotion.pickle'.format(debate), 'wb') as handle:
        	pickle.dump(emo_delta_x_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    ####################
    ## Average emo and delta_X for corelation
    ####################
    # Result:
    # nb_candidate * (nb_keypoints + nb_emotion)
    corr_info_lst = []
    for idx_can in range(nb_candidate):
        value = [ 0 for i in range(nb_keypoints+nb_emotion)] 
        count = [ 0 for i in range(nb_keypoints+nb_emotion)] 
        for idx_frame in range(nb_frame-1):
            frame = final_table[idx_frame]
            on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
            if on_screen:
                emo_lst = frame[nb_col_head+6+idx_can*nb_total_field:nb_col_head+13+idx_can*nb_total_field] 
                idx_emo = np.argmax(emo_lst)
                keypoints = frame[nb_col_head+nb_field+idx_can*nb_total_field:nb_col_head+nb_field+idx_can*nb_total_field+nb_keypoints]  
                #print(keypoints)
                for idx_kp in range(nb_keypoints):
                    if keypoints[idx_kp] != '-1':
                        value[idx_kp] += float(keypoints[idx_kp])
                        count[idx_kp] += 1
                for idx_emo in range(nb_emotion):
                    if emo_lst[idx_emo] != '-1':
                        value[nb_keypoints+idx_emo] += float(emo_lst[idx_emo])
                        count[nb_keypoints+idx_emo] += 1
                        
        for idx in range(nb_keypoints+nb_emotion):
            if count[idx] > 0:
                value[idx] = value[idx]/count[idx]
            else:
                value[idx] = -1
        #print(count)
        corr_info_lst.append(value)
    
    with open(save_path+'{}_corr_info.pickle'.format(debate), 'wb') as handle:
        	pickle.dump(corr_info_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ####################
    ##Correlation for all candidates
    ####################
    # Result:
    # nb_valid_Frame* (nb_keypoints + nb_emotion)       
    def average_keypoints(keypoints):
        # neck already removed
        val = np.zeros(15)
        val[0] = keypoints[0]
        val[-11:] = keypoints[-11:]
        mask = np.array(keypoints[1:7]) != '-1'
        if (mask[0]+mask[3]) != 0:
            val[1] = str((mask[0]*float(keypoints[1]) + mask[3]*float(keypoints[4])) / (mask[0]+mask[3]))
        else:
            val[1] = '-1'
            
        if (mask[1]+mask[4]) != 0:
            val[2] = str((mask[1]*float(keypoints[2]) + mask[4]*float(keypoints[5])) / (mask[1]+mask[4]))
        else:
            val[2] = '-1'    
            
        if (mask[2]+mask[5]) != 0:
            val[3] = str((mask[2]*float(keypoints[3]) + mask[5]*float(keypoints[6])) / (mask[2]+mask[5]))
        else:
            val[3] = '-1' 
            
        return val
    
    idx_can = 7
    idx_select = np.array([False for i in range(nb_total_field)])
    idx_select[6:13] = True
    idx_select[15:15+nb_keypoints] = True
    # Exclude neck
    idx_select[16] = False
    
    valid_corr_info_lst = []
    for idx_frame in range(nb_frame):
        for idx_can in range(nb_candidate):
            frame = final_table[idx_frame]
            on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
            if on_screen:
                info = frame[nb_col_head+idx_can*nb_total_field:nb_col_head+(1+idx_can)*nb_total_field] 
                info = np.array(info)
                temp = info[idx_select]
                val = np.zeros_like(temp)
                val[-nb_emotion:] = temp[:nb_emotion]
                val[:-nb_emotion] = temp[nb_emotion:]
                val = average_keypoints(val)
                if not np.any(val == -1):
                    valid_corr_info_lst.append(val)
            
    valid_corr_info_lst = np.array(valid_corr_info_lst).astype(float)   
    
    with open(save_path+'{}_valid_corr_info.pickle'.format(debate), 'wb') as handle:
        	pickle.dump(valid_corr_info_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    ###############
    ## Correlation on each shot
    ##############
    # Here we use combined table, after replacing coordinate by delta_x
    corr_lst = []
    nb_emotion = 7
    idx_start = 0
    idx_end = 1
    
    while idx_end < nb_frame_fps15:
        while combined[idx_start][3] != '1' and idx_start < nb_frame_fps15-2:
            idx_start += 1
        if idx_start >= nb_frame_fps15-2:
            break
        
        idx_end = idx_start + 1
        candidates = combined[idx_start][4]
        while combined[idx_end][4] == candidates and idx_end < nb_frame_fps15-1:
            idx_end += 1
        
        if idx_end - idx_start < 30:
            idx_start = idx_end
            continue
        print '['+str(idx_start)+', '+str(idx_end)+']'
        for idx_can in range(nb_candidate):
            on_screen = combined[idx_start][nb_col_head+idx_can*nb_total_field] != '-1'
            if on_screen:
                break
        val_all = np.zeros(18)
        count_all = np.zeros(18)
        count_all[-7:] = idx_end - idx_start
        for idx_frame_t in range(idx_start, idx_end):
            ref_dist = float(combined[idx_frame_t][nb_col_head+(idx_can+1)*nb_total_field-1])
            if ref_dist == -1:
                continue
            info = combined[idx_frame_t][nb_col_head+idx_can*nb_total_field:nb_col_head+(1+idx_can)*nb_total_field] 
            if info[nb_field].startswith('('):
                continue
            info = np.array(info)
            temp = info[idx_select].astype(float)
            val = np.zeros_like(temp)
            count = np.zeros_like(temp)
            val[-nb_emotion:] = temp[:nb_emotion]
            val[:-nb_emotion] = temp[nb_emotion:]
            for i in range(11):
                if val[i] == -1:
                    val[i] = 0
                else:
                    val[i] = val[i]/ref_dist
                    count[i] = 1
            val_all += val
            count_all += count
                
        for i in range(18):
            if count_all[i] != 0:
                val_all[i] = val_all[i] / count_all[i]
            else:
                val_all[i] = 0
        
        if not np.any(val_all == 0):
            corr_lst.append(val_all)
        idx_start = idx_end+1
    
    corr_lst = np.array(corr_lst)
    
    with open(save_path+'{}_shot_corr_info.pickle'.format(debate), 'wb') as handle:
        	pickle.dump(corr_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    #corr_lst = np.array(corr_lst)
    #corr = np.corrcoef(corr_lst, rowvar=False)
    #corr[0, 0] = -1
    #CorrMtx(corr, dropDuplicates = True)        
    
    
    ##############
    ## Clustering
    ##############
    clustering_lst = []
    nb_frame_window = 30
    
    idx_select = np.array([False for i in range(nb_total_field)])
    idx_select[nb_field:nb_field+8] = True
    # Exclude neck
    idx_select[16] = False
    nb_select = idx_select.sum()
    
    for idx_frame_t in range(0, nb_frame_fps15, 10):
        if combined[idx_frame_t][3] != '1':
            continue
        
        candidates = combined[idx_frame_t][4]
        same_shot = True
        for i in range(idx_frame_t, idx_frame_t+nb_frame_window):
            if combined[i][4] != candidates:         
                same_shot = False
                break
        if not same_shot:
            continue
    #    idx_frame = (idx_frame_t + 7) / 15
        frame = combined[idx_frame_t]
    
        for idx_can in range(7, 8):
            on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
            if on_screen:
                sample = np.zeros(nb_frame_window * nb_select)
                emo_lst = np.zeros(nb_emotion)
                for i, idx in enumerate(range(idx_frame_t, idx_frame_t+nb_frame_window)):
                    info = combined[idx][nb_col_head+idx_can*nb_total_field:nb_col_head+(1+idx_can)*nb_total_field] 
                    info = np.array(info)
                    val = info[idx_select].copy()
                    if val[0].startswith('('):
                        continue
                    val = val.astype(float)
                    for j in range(len(val)):
                        if val[j] == -1:
                            val[j] = 0
                    sample[i*nb_select:(i+1)*nb_select] = val
                    emo_lst += info[6:13].astype(float)
                
                emo = np.argmax(emo_lst)
                clustering_lst.append((idx_frame_t, emo, sample))
    
#    clustering_lst = np.array(clustering_lst)
    
    with open(save_path+'{}_clutering_info.pickle'.format(debate), 'wb') as handle:
        	pickle.dump(clustering_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ################
    ## Classification each every frames
    ################
    #cluster_lst = []
    #emotion_label_lst = []
    #
    #for idx_frame in range(nb_frame):
    #    delta_x_lst = []
    #    range_left = max(0, idx_frame*fps-nb_half)
    #    range_right = min(nb_frame_fps15, idx_frame*fps+nb_half)
    #    frame = final_table[idx_frame]
    #    for idx_frame_t in range(range_left, range_right):
    #        # no candidate is on screen
    #        if info[idx_frame][3] == '0':
    #            continue    
    #        for idx_can in range(7, 8):
    #            on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
    #            has_delta_x = frame[nb_col_head+nb_field+idx_can*nb_total_field] != '-1'
    #            ref_dist = float(frame[nb_col_head+(idx_can+1)*nb_total_field-1])
    #            if ref_dist == -1:
    #                continue
    #            if on_screen and has_delta_x:
    #                sample= []
    #                for idx_frame_t in range(range_left, range_right):
    #                    # Append nose, shoulder, elbow, wrist
    #                    for i in [0, 2, 3, 4, 5, 6, 7]:
    #                        val = combined[idx_frame_t][nb_col_head+nb_field+idx_can*nb_total_field+i]
    #                        if val == '-1':
    #                            sample.append(0)
    #                        else:
    #                            val = str2array(val)
    #                            dist = get_points_dist(val, np.zeros(2))
    #                            sample.append(dist/ref_dist)
    #                cluster_lst.append(sample)
    #                emo_lst = frame[nb_col_head+idx_can*nb_total_field+6:nb_col_head+idx_can*nb_total_field+13]
    #                emotion_label_lst.append(np.argmax(emo_lst))
    #                    
    #cluster_lst = np.array(cluster_lst) 
    #emotion_label_lst = np.array(emotion_label_lst)
    #
    #from sklearn import svm
    #from sklearn.cluster import KMeans
    #from sklearn.model_selection import train_test_split
    #
    #x_train, x_test, y_train, y_test = train_test_split(cluster_lst, emotion_label_lst, test_size=0.3)
    #kmeans = KMeans(n_clusters=7, random_state=0).fit(x_train)
    #kmeans.labels_[:50]
    #
    ### SVM
    #clf = svm.SVC()
    #clf.fit(x_train, y_train)
    #
    #pred = clf.predict(x_test)
    
    ## Interpolation for missing value
    #mean_val = np.zeros(nb_keypoints-1)
    #for i in range(nb_keypoints-1):
    #    col = valid_corr_info_lst[:, i]
    #    mean_val[i] = col[col != -1].mean()
    ## replace -1 by mean value
    #for i in range(nb_keypoints-1):
    #    col = valid_corr_info_lst[:, i]
    #    col[col == -1] = mean_val[i]
    
    
    #
    #corr = np.corrcoef(valid_corr_info_lst, rowvar=False)
    #CorrMtx(corr, dropDuplicates = True)
    #        
        
         
    
    
    
    ################
    ## is speaking or not
    ################     
    speaking_info_lst = []
    count_is_speaking = 0
    count_not_speaking = 0
    for idx_can in range(nb_candidate):
        value_sp = [[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]
        count_sp = [[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]
        value_not_sp = [[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]
        count_not_sp = [[0 for i in range(nb_keypoints)] for j in range(nb_emotion)]
    
        for idx_frame in range(nb_frame-1):
            frame = final_table[idx_frame]
            on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
            if on_screen:
                emo_lst = frame[nb_col_head+6+idx_can*nb_total_field:nb_col_head+13+idx_can*nb_total_field] 
                idx_emo = np.argmax(emo_lst)
                keypoints = frame[nb_col_head+nb_field+idx_can*nb_total_field:nb_col_head+nb_field+idx_can*nb_total_field+nb_keypoints]  
    
                is_speaking = frame[nb_col_head+idx_can*nb_total_field+1] != '-1'
                if is_speaking:
                    count_is_speaking += 1
                    for idx_kp in range(nb_keypoints):
                        if keypoints[idx_kp] != '-1':
                            value_sp[idx_emo][idx_kp] += float(keypoints[idx_kp])
                            count_sp[idx_emo][idx_kp] += 1
                else:
                    count_not_speaking += 1
                    for idx_kp in range(nb_keypoints):
                        if keypoints[idx_kp] != '-1':
                            value_not_sp[idx_emo][idx_kp] += float(keypoints[idx_kp])
                            count_not_sp[idx_emo][idx_kp] += 1
    #    print 'Is speaking: ', count_is_speaking
    #    print 'Not speaking: ', 
    #    print value_sp
        
        speaking_info_lst.append(((value_sp, count_sp), (value_not_sp, count_not_sp)))
    
    with open(save_path+'{}_speaking_emotion.pickle'.format(debate), 'wb') as handle:
        	pickle.dump(speaking_info_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ## Check average
    #emo_avg = [0 for i in range(nb_emotion)]
    #count = [0 for i in range(nb_emotion)]
    #
    #for j in range(nb_candidate):
    #    for i in range(nb_emotion):
    #        emo_avg[i] += np.sum(speaking_info_lst[j][0][0][i])
    #        count[i] += np.sum(speaking_info_lst[j][0][1][i])
    #        emo_avg[i] += np.sum(speaking_info_lst[j][1][0][i])
    #        count[i] += np.sum(speaking_info_lst[j][1][1][i])
    #
    #for i in range(nb_emotion):
    #    emo_avg[i] /= count[i]
    #    
    #x = np.arange(nb_emotion)
    #
    #import matlotlib.pyplot as plt
    #emotion_lst = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    #
    #fig, ax = plt.subplots()
    #plt.bar(x, emo_avg)
    #ax.set_ylim(0.02, 0.06)
    #plt.xticks(x, emotion_lst)
    #plt.show()
    #
    
    
    ###### Verification of ref distance
    ## Need the table Combined
    #ref_dist_lst = []
    #for idx_can in tqdm(range(nb_candidate)):
    #    value = []
    #    for idx_frame_t in range(0, nb_frame_fps15):
    #        frame = combined[idx_frame_t]
    #        on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
    #        if on_screen:
    #            nose_str = frame[nb_col_head+idx_can*nb_total_field+nb_field]
    #            neck_str = frame[nb_col_head+idx_can*nb_total_field+nb_field+1]
    #            if nose_str != '-1' and neck_str != '-1':
    #                nose = str2array(nose_str)
    #                neck = str2array(neck_str)
    #                dist = get_points_dist(nose, neck)
    #                ref_dist = float(frame[nb_col_head+idx_can*nb_total_field+nb_field+nb_keypoints])
    #                value.append(dist/ref_dist)
    ##                if dist/ref_dist>1.2:
    ##                    break
    #    ref_dist_lst.append(value)
    #
    #
    ################
    ### Sum of delta_x along time
    ################
    #    
    #idx_can = 7
    #delta_x_sum_lst = []
    #last = [0 for i in range(nb_keypoints+1)]
    #for idx_frame in range(nb_frame):
    #    frame = final_table[idx_frame]
    #    on_screen = frame[nb_col_head+idx_can*nb_total_field] != '-1'
    #    if on_screen:
    #        # Sum of delta_x
    #        for i in range(nb_keypoints):
    #            val = float(frame[nb_col_head+idx_can*nb_total_field+nb_field+i])
    #            if val != -1:
    #                last[i] += float(val)
    #        # Append loudness
    #        loud = frame[nb_col_head+idx_can*nb_total_field+nb_field-1]
    #        loud = float(loud)
    #        if loud != -1:
    #            last[-1] = loud
    #        else:
    #            last[-1] = 0
    #        # Update last
    #        delta_x_sum_lst.append([i for i in last])
    #    else:
    #        last[-1] = 0
    #        delta_x_sum_lst.append([i for i in last])
    #
    #
    ### Plot
    #delta_x_sum_lst = np.array(delta_x_sum_lst)
    #fig = plt.figure()
    #fig.set_size_inches(10, 8,forward=True )
    #fig.show()
    #ax = fig.add_subplot(111)
    #for i in range(nb_keypoints):
    #    ax.plot(range(len(delta_x_sum_lst)), delta_x_sum_lst[:, i], label=new_field[i])
    #ax.plot(range(len(delta_x_sum_lst)), delta_x_sum_lst[:, -1], label='loudness')
    #plt.legend(loc=2)
    #ax.set_title('Cumulative sum of delta(x): '+can_lst[idx_can])
    #ax.set_xlabel('Active frame index')
    #plt.draw()
    #
    #####################
    ### Verification time series
    #####################                
    #
    ##idx_frame = 669
    ##idx_can = 3
    #        
    #
    ##idx_frame=824
    ##idx_can = 2
    ##
    ##idx_frame = 1588
    ##while info[idx_frame][2] == '0':
    ##    idx_frame += 1  
    ##       
    #x = int(info[idx_frame][6]) 
    #y = int(info[idx_frame][7])
    #h = int(info[idx_frame][8])
    #w = int(info[idx_frame][9])
    #
    #idx_can =0
    #while int(x) == -1:
    #    idx_can += 1
    #    x = int(info[idx_frame][6+idx_can*nb_field])
    #    y = int(info[idx_frame][7+idx_can*nb_field])
    #    h = int(info[idx_frame][8+idx_can*nb_field])
    #    w = int(info[idx_frame][9+idx_can*nb_field])
    ##    
    #idx_frame +=1
    #import matplotlib.pyplot as plt
    #im = cv2.imread("Frames_fps1/{}/frame{:06d}.jpg".format(debate, idx_frame))
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #plt.imshow(im)
    ##plt.imshow(plot_bounding_box_on_image(im, x, y, h, w, [255, 0, 0]))
    #temp = plot_bounding_box_on_image(im, x, y, h, w, [255, 0, 0])
    #plt.imshow(plot_keypoint_on_image(temp, keypoint_set=keypoints_lst[idx_frame*fps], color=[0, 255, 0]))
    #plt.imshow(plot_keypoint_on_image(temp, keypoint_set=[keypoints], color=[0, 255, 0]))
    #
    ##plt.imshow(im[y:y+h, x:x+w, :])
    ##
    ##im = cv2.imread("Frames_fps1/{}/frame{:06d}.jpg".format(debate, idx_frame))
    ##plot_keypoint_on_image(im, keypoint_set=keypoints_lst[idx_frame], save_path='test/frame{:06d}.jpg'.format(idx_frame))
    ##
    #print combined[idx_frame*fps][nb_col_head+idx_can*nb_total_field:nb_col_head+(idx_can+1)*nb_total_field]
    
    
    ##############
    ## Save csv files
    #############
    final_header = [0 for i in range(nb_col_head+nb_candidate*nb_total_field)]
    final_header[:nb_col_head] = header[:nb_col_head]
    for i in range(nb_candidate):
        final_header[nb_col_head+i*nb_total_field: nb_col_head+i*nb_total_field+nb_field] = header[nb_col_head+i*nb_field: nb_col_head+i*nb_field+nb_field]
        for j in range(nb_new_field):
            final_header[nb_col_head+i*nb_total_field+nb_field+j] = new_field[j] + '_' + can_lst[i]
    
    res = []
    res.append(final_header)
    for i in range(nb_frame-1):
        res.append(final_table[i])
    
    np.savetxt(save_path+'{}_pose.csv'.format(debate), res, delimiter=';', fmt='%s')
