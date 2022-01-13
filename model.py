import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy.linalg as la
import numpy as np

class DEANet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.regularDataBranch = BranchLayer()
        self.referenceDataBranch = BranchLayer()
        
        self.conv = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)            
        )
       
        self.output = nn.Linear(256, 2)
        self.loss = 0

    def forward(self, leftEye, rightEye, headPose, referenceLeftEye, referenceRightEye, referenceHeadPose):
        feature_regular = self.regularDataBranch(leftEye, rightEye, headPose)
        feature_reference = self.referenceDataBranch(referenceLeftEye, referenceRightEye, referenceHeadPose)
        
        feature = torch.cat((feature_regular, feature_reference), 1)
        feature = self.conv(feature)
        gaze = self.output(feature)

        return gaze
        
    def training_step(self, data, reference_data, device):
        output = self(data['leftEye'].to(device), 
                     data['rightEye'].to(device),
                     data['headPose'].to(device),
                     reference_data['leftEye'].to(device), 
                     reference_data['rightEye'].to(device),
                     reference_data['headPose'].to(device))
        loss = 0
        k = len(data)
        for i in range(0,k):
                loss = loss + la.norm(data['label'][i] - reference_data['label'][i], ord = 2) / k

        return loss
        
        #loss = 0
        #print(data)
        #for input_k in data:
        #    output = self(input_k)
        #    loss = loss + la.norm(input_k['rotulo'], output)

        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)
                
class BranchLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.eyesLayer = EyesLayer()
        
        self.conv = nn.Sequential(
            nn.BatchNorm1d(515),
            nn.ReLU(inplace=True),
            nn.Linear(515, 256),          
        )
        
        self.output = nn.ReLU(inplace=True)
        
    def forward(self, leftEye, rightEye, headPose):  
        featureEyes = self.eyesLayer(leftEye, rightEye)
        
        feature = torch.cat((headPose, featureEyes), 1)
        feature = self.conv(feature)
        feature = self.output(feature)
        
        return feature
        
     
class EyesLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.leftEyeLayer = EyeLayer()
        self.rightEyeLayer = EyeLayer()
        
        self.output = nn.Linear(2048, 512)
        
    def forward(self, leftEye, rightEye):

        feature_left_eye = self.leftEyeLayer(leftEye)
        feature_right_eye = self.rightEyeLayer(rightEye)
        
        feature = torch.cat((feature_left_eye, feature_right_eye), 1)
        feature = self.output(feature)
        
        return feature
        
class EyeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        vgg16 = torchvision.models.vgg16(pretrained=True)
        
        self.conv = nn.Sequential(
            vgg16,
            nn.Linear(1000, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)            
        )
        
    def forward(self, eyeFeature):
        feature = self.conv(eyeFeature)
        return feature
                   
if __name__ == '__main__':
    m = DEANet()
    '''feature = {"face":torch.zeros(10, 3, 224, 224).cuda(),
                "left":torch.zeros(10,1, 36,60).cuda(),
                "right":torch.zeros(10,1, 36,60).cuda()
              }'''
    feature = {"head_pose": torch.zeros(10, 2),
               "left_eye": torch.zeros(10, 3, 36, 60),
               "right_eye": torch.zeros(10, 3, 36, 60)
               }
    a = m(feature)
    print(m)
