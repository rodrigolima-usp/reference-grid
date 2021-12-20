from torch.utils.data import Dataset, DataLoader
import torch

from pathlib import Path
import numpy as np
import json
import cv2
import os
import h5py
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
  
def get_data_info_from_csv(data_info_path):
  info = pd.read_csv(data_info_path, sep=" ", header=None)
  return info
  
def get_data_from_mat(path, user_to_days_list):
    for user_to_days in user_to_days_list: 
        for day in user_to_days['days']:
            mat_file = path + user_to_days['user'] + "/" + day
            data = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['data']
            left = list(zip(data.left.image, data.left.gaze, data.left.pose))
            right = list(zip(data.right.image, data.right.gaze, data.right.pose))
            
            sample_pairs = [SamplePair(l, r) for l,r in zip(left, right)]
            
            #ler aquivo na pasta original e buscar rotulo e adiciona ao sample
            path_to_ground_truth = path + user_to_days['user'] + '/annotation.txt'
            ground_truth = pd.read_csv(path_to_ground_truth, sep=" ", header=None)

            for idx, sample_pair in enumerate(sample_pairs):
                sample_pair.setLabel(ground_truth.iloc[idx][24:26])
                
    return sample_pairs

class Sample():
    def __init__(self, image, gaze, pose):
        self.image = image
        self.gaze = gaze
        self.pose = pose       
        
class SamplePair():
    def __init__(self, left, right):
        image, gaze, pose = left
        self.left = Sample(image, gaze, pose)
        
        image, gaze, pose = right
        self.right = Sample(image, gaze, pose)
        
    def setLabel(self, label):
        self.label = label

class Loader(Dataset):
    def __init__(self, path, user_to_days_list):
        self.sample_pairs = get_data_from_mat(path, user_to_days_list)
        
    def __len__(self):
        return len(self.sample_pairs)
	
    def __getitem__(self, index):
    
        left_eye = self.sample_pairs[index].left
        right_eye = self.sample_pairs[index].right
        #print(self.sample_pairs[index].label)
        return {
            "leftEye": torch.from_numpy(np.array(left_eye.image)).type(torch.FloatTensor).unsqueeze_(0).repeat(3, 1, 1),
            "rightEye": torch.from_numpy(np.array(right_eye.image)).type(torch.FloatTensor).unsqueeze_(0).repeat(3, 1, 1),
            "headPose": torch.from_numpy(np.array([0,0,0])).type(torch.FloatTensor),
            "label": torch.from_numpy(np.array(self.sample_pairs[index].label)).type(torch.FloatTensor)
        }
            
def load_data():
    path = ""

    user_to_days_list = [{"user":"p00","days":["day01.mat"]}]
    
    loader = Loader(path, user_to_days_list)
    return loader
            
if __name__ == '__main__':
    path = "/home/rodrigo/Downloads/MPIIGaze/Data/Normalized"
    user_to_days_list = [{"user":"p00","days":["day01.mat"]}]
    
    loader = Loader(path, user_to_days_list)
    
    print(loader.__getitem__(0)['leftEye'].shape)
    print(loader.__getitem__(0)['rightEye'].shape)
    
