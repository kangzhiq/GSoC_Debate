#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 21:24:57 2020

@author: zxk93
"""

import cv2


save_dir = 'test/'

def plot_keypoint_on_image(image, keypoint_set, save_path=None, color = None):
    im = image.copy()
    if color is not None:
        cl = color
    else:
        cl = [0, 0, 255]
    for keypoint in keypoint_set:
        for lm in keypoint:
            im[int(lm[1])-3:int(lm[1]+4), int(lm[0])-3:int(lm[0])+4, :] = cl
        
    if save_path:
        cv2.imwrite(save_path, im)
        
    return im

def plot_bounding_box_on_image(image, x, y, h, w, color = None):
    im = image.copy()
    if color is not None:
        cl = color
    else:
        cl = [0, 0, 255]
    
    im[y-3:y+3, x:x+w, :] = cl
    im[y+h-3:y+h+3, x:x+w, :] = cl
    im[y:y+h, x-3:x+3, :] = cl
    im[y:y+h, x+w-3:x+w+3, :] = cl
    return im
    