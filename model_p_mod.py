import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import complex_new as complex
import math


class ManifoldNetComplex(nn.Module):
    def __init__(self):
        super(ManifoldNetComplex, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 20, (5, 5), (2, 2))
        self.complex_conv2 = complex.ComplexConv2Deffgroup(20, 20, (5, 5), (2, 2))
        
        #self.proj1 = complex.manifoldReLUv2angle(10) #complex.ReLU4Dsp(20)
        self.proj2 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(40)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear_1 = complex.ComplexLinearangle2Dmw_outfield(20*22*22)
        self.conv_1 = nn.Conv2d(20, 30, (5,5))
        self.mp_1 = nn.MaxPool2d((2,2))
        self.conv_2 = nn.Conv2d(30, 40, (5,5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(40)
        self.bn_3 = nn.BatchNorm2d(50)
        self.conv_3 = nn.Conv2d(40, 50, (2, 2))
        self.linear_2 = nn.Linear(50, 30)
        self.linear_4 = nn.Linear(30, 10)
        
    def forward(self, x):
        x = self.complex_conv1(x)
        x = self.proj2(x)
        x = self.complex_conv2(x)
        x = self.proj2(x)
        #x = self.dropout(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
#         print(x.shape)
#         x = self.mp_2(x)
        #print(x.shape)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        #print(x.shape)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        #x = self.dropout(x)
#         x = self.linear_3(x)
#         x = self.relu(x)
        x = self.linear_4(x)
         
        
#         x = self.complex_conv3(x)
#         x = self.proj3(x)
#         #x = self.complex_bn(x)
#         #print(x.shape)
#         x = self.complex_conv4(x)
#         x = self.proj4(x)
#         x = self.complex_conv5(x)
#         x = self.proj5(x)
#         x = self.complex_conv6(x)
#         x = self.proj6(x)
#         x = self.complex_fc(x)
        return x

def make_complex_layer(complex_c, complex_kern, complex_stride):
    complex_parts = []
    prev_c = 1
    prev_relu = None
    for (c, kern, stride) in zip(complex_c, complex_kern, complex_stride):
        complex_parts.append(complex.ComplexConv2Deffgroup(prev_c, c, (kern, kern), (stride, stride)))
        if prev_c == c and prev_relu is not None:
            complex_parts.append(prev_relu)
        else:
            prev_relu = complex.manifoldReLUv2angle(c)
            complex_parts.append(prev_relu)
        prev_c = c             
    return complex_parts, prev_c
    
def make_regular_layer(prev_channel, regular_c, regular_kern, regular_stride, max_pool):
    regular_parts = []
    prev_c = prev_channel
    relu = nn.ReLU()
    for (c, kern, stride, pool) in zip(regular_c, regular_kern, regular_stride, max_pool):
        regular_parts.append(nn.Conv2d(prev_c, c, (kern, kern), (stride, stride)))
        regular_parts.append(nn.BatchNorm2d(c))
        regular_parts.append(relu)
        if pool != 1:
            regular_parts.append(nn.MaxPool2d((pool, pool)))
        prev_c = c
    return regular_parts, prev_c
    

class CustomManifold(nn.Module):
    def __init__(self, params):
        super(CustomManifold, self).__init__()
        
        complex_parts, channel = make_complex_layer(params["complex_c"], params["complex_kern"], params["complex_stride"])
        
        self.complex_parts = nn.Sequential(*complex_parts)
        self.linear_1 = complex.ComplexLinearangle2Dmw_outfield(channel*params["middle"])
        
        regular_parts, channel = make_regular_layer(channel, params["regular_c"], params["regular_kern"], params["regular_stride"], params["max_pool"])
        
        self.regular_parts = nn.Sequential(*regular_parts)
        self.linear_2 = nn.Linear(channel, int(channel/2))
        self.relu = nn.ReLU()
        self.linear_3 = nn.Linear(int(channel/2), 11)
        
        
    
        
    def forward(self, x):
        
        x = self.complex_parts(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.regular_parts(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        
        return x
        
class ManifoldNetAll(nn.Module):
    def __init__(self):
        super(ManifoldNetAll, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 5, (5, 5), (2, 2))
        self.complex_conv2 = complex.ComplexConv2Deffgroup(5, 10, (5, 5), (2, 2))
        self.complex_conv3 = complex.ComplexConv2Deffgroup(10, 20, (5, 5), (2, 2))
        self.complex_conv4 = complex.ComplexConv2Deffgroup(20, 40, (5, 5), (2, 2))
        self.complex_conv5 = complex.ComplexConv2Deffgroup(40, 80, (3, 3), (1, 1))
        self.complex_bn = complex.ComplexBN()
        self.proj1 = complex.manifoldReLUv2angle(5) #complex.ReLU4Dsp(20)
        self.proj2 = complex.manifoldReLUv2angle(10) #complex.ReLU4Dsp(40)
        self.proj3 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(80)
        self.proj4 = complex.manifoldReLUv2angle(40) #complex.ReLU4Dsp(80)
        self.proj5 = complex.manifoldReLUv2angle(80) #complex.ReLU4Dsp(80)
        self.complex_fc = complex.ComplexLinearangle(80, 1, 30, 11) #10
    def forward(self, x):
        x = self.complex_conv1(x)
        x = self.proj1(x)
        
        x = self.complex_conv2(x)
        x = self.proj2(x)
        
        x = self.complex_conv3(x)
        x = self.proj3(x)
        
        x = self.complex_conv4(x)
        x = self.proj4(x)
        x = self.complex_conv5(x)
        x = self.proj5(x)
        
        x = self.complex_fc(x)
        return x

    
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 20, (5, 5), (2, 2))
        self.complex_conv2 = complex.ComplexConv2Deffgroup(20, 20, (5, 5), (2, 2))
        self.proj2 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(40)
#         self.proj3 = complex.manifoldReLUv2angle(30)
        self.relu = nn.ReLU()
        self.linear_1 = complex.ComplexLinearangle2Dmw_outfield(20*22*22)
        self.conv_1 = nn.Conv2d(20, 30, (5, 5))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(30, 40, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(40)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(40, 50, (2, 2))
        self.bn_3 = nn.BatchNorm2d(50)
        self.linear_2 = nn.Linear(50, 30)
        self.linear_4 = nn.Linear(30, 11)
        
        
    def forward(self, x):
        
        x1 = self.complex_conv1(x)
        
        x = self.proj2(x1)
        x2 = self.complex_conv2(x)
        x = self.proj2(x2)
#         print(x.shape)
        x = self.linear_1(x)
        x3 = self.relu(x)
        x = self.conv_1(x3)
        
        x = self.relu(x)
        x4 = self.mp_1(x)
#         print(x.shape)
        x = self.conv_2(x4)
        
        x5 = self.relu(x)
#         print(x.shape)
        x = self.conv_3(x5)
       
        x6 = self.relu(x)
        x = x6.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x, x1, x2, x3, x4, x5, x6

    
    

class ManifoldNetComplex_11(nn.Module):
    def __init__(self):
        super(ManifoldNetComplex_11, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 20, (5, 5), (2, 2))
        self.complex_conv2 = complex.ComplexConv2Deffgroup(20, 20, (5, 5), (2, 2))
        self.proj2 = complex.manifoldReLUv2angle(20) #complex.ReLU4Dsp(40)
#         self.proj3 = complex.manifoldReLUv2angle(30)
        self.relu = nn.ReLU()
        self.linear_1 = complex.ComplexLinearangle2Dmw_outfield(20*22*22)
        self.conv_1 = nn.Conv2d(20, 30, (5, 5))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(30, 40, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(40)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(40, 50, (2, 2))
        self.bn_3 = nn.BatchNorm2d(50)
        self.linear_2 = nn.Linear(50, 30)
        self.linear_4 = nn.Linear(30, 11)
        
        
    def forward(self, x):
        
        x = self.complex_conv1(x)
        x = self.proj2(x)
        x = self.complex_conv2(x)
        x = self.proj2(x)
#         print(x.shape)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.mp_1(x)
#         print(x.shape)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
#         print(x.shape)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.tensor(np.load('../data_polar/' + ID).astype(np.float)).float().unsqueeze(1)
        X[0,...] = torch.acos(X[0,0,...])
        X[1,...] = X[4,...]
        #X = X/torch.norm(X,dim=0).unsqueeze(0) 
       	#X = X.unsqueeze(0)
        #print(X.shape)
        y = self.labels[ID]

        return X[:2,...], y

