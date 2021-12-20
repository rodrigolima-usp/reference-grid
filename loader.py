from torch.utils.data import Dataset, DataLoader
import torch

from pathlib import Path
import numpy as np
import json
import cv2
import os

from scipy import io
import matplotlib.pyplot as plt
  
def get_data_info_from_csv(data_info_path):
  info = pd.read_csv(data_info_path, sep=" ", header=None)
  return info
  
def get_data_from_mat(path, user_to_days_list):
  sample_pairs = []
  for user_to_days in user_to_days_list:
        for day in user_to_days['days']:
            mat_file = path + "/" + user_to_days['user'] + "/" + day
            data = io.loadmat(mat_file)['data']
            for i in range(len(data[0][0][0][0][0][0])):
                
                left = [
                    data[0]['left'][0]['image'][0][0][i],
                    data[0]['left'][0]['gaze'][0][0][i],
                    data[0]['left'][0]['pose'][0][0][i]
                ]
                
                right = [
                   data[0]['right'][0]['image'][0][0][i],
                   data[0]['right'][0]['gaze'][0][0][i],
                   data[0]['right'][0]['pose'][0][0][i]
                ]
                
                sample_pair = SamplePair(left, right)
                sample_pairs.append(sample_pair)

  return sample_pairs

class Sample():
    def __init__(self, image, gaze, pose):
        print('Sample-image',image.shape)
        self.image = image
        self.gaze = gaze
        self.pose = pose       
        
class SamplePair():
    def __init__(self, left, right):
        image, gaze, pose = left
        self.left = Sample(image, gaze, pose)
        
        image, gaze, pose = right
        self.right = Sample(image, gaze, pose)

class Loader(Dataset):
    def __init__(self, path, user_to_days_list):
        self.sample_pairs = get_data_from_mat(path, user_to_days_list)
        
    def __len__(self):
        return len(self.sample_pairs)
	
    def __getitem__(self, index):
    
        left_eye = self.sample_pairs[index].left
        right_eye = self.sample_pairs[index].right
        
        print('Loader-left.image',left_eye.image.shape)

        return {
            "leftEye": torch.from_numpy(np.array(left_eye.image.reshape((3, 36, 60)))).type(torch.FloatTensor),
            "rightEye": torch.from_numpy(np.array(right_eye.image.reshape((3, 36, 60)))).type(torch.FloatTensor),
            "headPose": torch.from_numpy(np.array([0,0,0])).type(torch.FloatTensor)
        }
            
def load_data():
    path = "/home/rodrigo/Downloads/MPIIGaze/Data/Normalized"
    user_to_days_list = [{"user":"p00","days":["day01.mat"]}]
    
    loader = Loader(path, user_to_days_list)
    return loader
            
if __name__ == '__main__':
    path = "/home/rodrigo/Downloads/MPIIGaze/Data/Normalized"
    user_to_days_list = [{"user":"p00","days":["day01.mat"]}]
    
    loader = Loader(path, user_to_days_list)
    
    print(loader.__getitem__(0))
    
