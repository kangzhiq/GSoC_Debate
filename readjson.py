'''
This file will only focus on reading json file from Openpose and save the data
to a pickle object.

Further processing such as intepolation of value will be performed after having 
the csv file.

'''
import json
import csv
import os
import numpy as np
import json

import pickle
from tqdm import tqdm


debate = 'Debate1Night1'
json_dir = 'json/{}/'.format(debate)
csv_path = 'tables/{}.csv'.format(debate.lower())
save_path = 'json/' 

nb_frame = len(os.listdir(json_dir))


keypoints_lst = []

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


for i in tqdm(range(nb_frame)):
	filename = json_dir+'frame%06d_keypoints.json'%i
	with open(filename) as f:
		data = json.load(f)
	keypoints_lst.append(extract_keypoints(data))
	




# Save data
with open(save_path+'{}.pickle'.format(debate), 'wb') as handle:
	pickle.dump(keypoints_lst, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''


'''
	
