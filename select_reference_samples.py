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
                
                left = {
                   'image':data[0]['left'][0]['image'][0][0][i],
                   'gaze':data[0]['left'][0]['gaze'][0][0][i],
                   'pose':data[0]['left'][0]['pose'][0][0][i]
                }
                
                right = {
                   'image':data[0]['right'][0]['image'][0][0][i],
                   'gaze':data[0]['right'][0]['gaze'][0][0][i],
                   'pose':data[0]['right'][0]['pose'][0][0][i]
                }
                
                sample_pair = SamplePair(left, right)
                sample_pairs.append(sample_pair)
  return sample_pairs

class Sample():
    def __init__(self, image, gaze, pose):
        self.image = image
        self.gaze = gaze
        self.pose = pose           
        
class SamplePair():
    def __init__(self, left, right):
        image, gaze, pose = left
        self.left = Sample(gaze, image, pose)
        
        image, gaze, pose = right
        self.right = Sample(gaze, image, pose)

class Loader():
    def __init__(self, path, user_to_days_list):
        self.sample_pairs = get_data_from_mat(path, user_to_days_list)
	
    def __getitem__(self, index):
    
        left_eye = self.sample_pairs[index].left
        right_eye = self.sample_pairs[index].right

        return {
            "leftEye": left_eye, #torch.from_numpy(left_eye.image).type(torch.FloatTensor),
            "rightEye": right_eye, #torch.from_numpy(right_eye.image).type(torch.FloatTensor),
            "headPose": 0         #torch.from_numpy(0).type(torch.FloatTensor)
            }
            
if __name__ == '__main__':
    path = "/home/rodrigo/Downloads/MPIIGaze/Data/Normalized"
    user_to_days_list = [{"user":"p00","days":["day01.mat"]}]
    
    loader = Loader(path, user_to_days_list)
    
    # criar cnn
    
    # \seleciona 500 amostras
    # para cada amostra
        #calcula na rede 
        # calcula erro e salva
        
    # calcula media dos erros
    
    # seleciona as amostras boas
    # calcula (x_min,y_min) (x_max,y_max)
    # calcula total eixo x, eixo_y
    # define o grid ->  espaçamento será total_eixo_x/4 e t_eiixo_y/3 -> começa x com (espaçameno_x / 2) e o mesmo para y
    # para cada amostra
        # calcula distancia entre ela e cada um dos pontos, se a distância for a menor para algum dos potnos, salva ela como a amostra mais próxima
    print(loader.__getitem__(0))
    
