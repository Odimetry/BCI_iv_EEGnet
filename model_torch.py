# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:01:52 2024

@author: User
"""

import torch.nn as nn

'''
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
    ised to do some model searching to get optimal performance on your
    ticular dataset.
    
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 
    
    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    
'''
#%% version 2    
class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, depth_multiplier, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * depth_multiplier, 
                                   kernel_size=kernel_size, padding=padding, groups=in_channels)
    
    def forward(self, x):
        return self.depthwise(x)

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = DepthwiseConv2D(in_channels, depth_multiplier=1, kernel_size=kernel_size, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class EEGNet_v2(nn.Module):
    def __init__(self, nb_classes, Chans=15, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, dropoutType='Dropout'):
        super().__init__()
        self.block1_conv2d = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.block1_batchnorm1 = nn.BatchNorm2d(F1)
        self.block1_depthwise = DepthwiseConv2D(F1, D, (Chans, 1), padding=0)
        self.block1_batchnorm2 = nn.BatchNorm2d(D*F1)
        self.block1_activation = nn.ELU()
        self.block1_avgpool = nn.AvgPool2d((1, 4))
        self.block1_dropout = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)
        
        self.block2_sepconv2d = SeparableConv2D(F1*D, F2, (1, 16), padding='same')
        self.block2_batchnorm = nn.BatchNorm2d(F2)
        self.block2_activation = nn.ELU()
        self.block2_avgpool = nn.AvgPool2d((1, 8))
        self.block2_dropout = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // 32), nb_classes)  # Adjust the size according to the model's output
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Shape: (Batch, 1, C, T)
        x = self.block1_conv2d(x)
        # Shape: (Batch, F1, C, T)
        x = self.block1_batchnorm1(x)
        x = self.block1_depthwise(x)
        # Shape: (Batch, F1*D, 1, T)
        x = self.block1_batchnorm2(x)
        x = self.block1_activation(x)
        x = self.block1_avgpool(x)
        # Shape: (Batch, F1*D, 1, T//4)
        x = self.block1_dropout(x)
        
        x = self.block2_sepconv2d(x)
        # Shape: (Batch, F2, 1, T//4)
        x = self.block2_batchnorm(x)
        x = self.block2_activation(x)
        x = self.block2_avgpool(x)
        # Shape: (Batch, F2, 1, T//32)
        x = self.block2_dropout(x)
        
        x = self.flatten(x)
        # Shape: (Batch, F2 * T//32)
        x = self.dense(x)
        # Shape: (Batch, nb_classes)
        x = self.softmax(x)
        return x


if "__main__" == __name__:
    # 모델 초기화
    model = EEGNet_v2(num_classes=2, in_channels=1, samples=128, dropout_rate=0.25, F1=8, D=2, F2=16)
    
    # 모델 정보 출력
    print(model)
