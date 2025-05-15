import os
import sys
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader, random_split


import sys
print(sys.argv[0])


class my_model(torch.nn.Module):
    def __init__(self, verbose = False, use_dropout = False, grid_size = [150,150], channels = 4, seed = 0):

        super().__init__()

        self.verbose = verbose
        self.use_dropout = use_dropout
        self.grid_size = grid_size
        self.seed = seed

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        C_in = channels
        H_in = grid_size[0]
        W_in = grid_size[1]
        in_shape = [C_in, H_in, W_in]
        
        self.conv2d_1 = torch.nn.Conv2d(in_channels=C_in, out_channels=16, kernel_size=(5,5), stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv2d_2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(33,33), stride=1, padding=16, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv2d_3 = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(9,9), stride=1, padding=4, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv2d_4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.conv2d_5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5), stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=(4,4), stride=4, padding=2, dilation=1, return_indices=True, ceil_mode=False)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1, dilation=1, return_indices=True, ceil_mode=False)


        self.conv2d_14 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5,5), stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv2d_13 = torch.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(9,9), stride=1, padding=4, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv2d_12 = torch.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(33,33), stride=1, padding=16, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv2d_11 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(5,5), stride=1, padding=2, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.max_unpool_2 = torch.nn.MaxUnpool2d(kernel_size=(2,2), stride=2, padding=1)
        self.max_unpool_1 = torch.nn.MaxUnpool2d(kernel_size=(4,4), stride=4, padding=2)
        

        self.batch_norm_1 = torch.nn.BatchNorm2d(16, momentum=0.1)
        self.batch_norm_2 = torch.nn.BatchNorm2d(16, momentum=0.1)
        self.batch_norm_3 = torch.nn.BatchNorm2d(64, momentum=0.1)
        self.batch_norm_4 = torch.nn.BatchNorm2d(128, momentum=0.1)
        self.batch_norm_5 = torch.nn.BatchNorm2d(128, momentum=0.1)

        self.batch_norm_14 = torch.nn.BatchNorm2d(64, momentum=0.1)
        self.batch_norm_13 = torch.nn.BatchNorm2d(16, momentum=0.1)
        self.batch_norm_12 = torch.nn.BatchNorm2d(4, momentum=0.1)
        self.batch_norm_11 = torch.nn.BatchNorm2d(1, momentum=0.1)


        self.identity = torch.nn.Identity()

        self.act_relu = torch.nn.ReLU()
        self.act_leaky_relu = torch.nn.LeakyReLU()
        self.act_elu = torch.nn.ELU()

        self.no_neg = torch.nn.ReLU()

    def forward(self,x):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        #Encoder
        x=x.unsqueeze(0)
        x=self.conv2d_1(x)
        x=self.act_elu(x)
        x=self.batch_norm_1(x)
        x=self.conv2d_2(x)
        x=self.act_elu(x)
        x=self.batch_norm_2(x)
        before_pool1_shape = x.shape[2:]
        before_pool1_clone = x.clone()
        x, ind_1=self.max_pool_1(x)

        x=self.conv2d_3(x)
        x=self.act_elu(x)
        x=self.batch_norm_3(x)
        x=self.conv2d_4(x)
        x=self.act_elu(x)
        x=self.batch_norm_4(x)
        before_pool2_shape=x.shape[2:]
        before_pool2_clone=x.clone()
        x, ind_2=self.max_pool_2(x)

        x=self.conv2d_5(x)
        x=self.act_elu(x)
        x=self.batch_norm_5(x)
        if self.verbose: print("Before max_unpool_2: x.shape = ",x.shape)


        #Decoder
        x=self.max_unpool_2(x, ind_2, output_size=before_pool2_shape)
        x = x + before_pool2_clone
        x=self.conv2d_14(x)
        x=self.act_elu(x)
        x=self.batch_norm_14(x)
        x=self.conv2d_13(x)
        x=self.act_elu(x)
        x=self.batch_norm_13(x)

        x=self.max_unpool_1(x, ind_1, output_size=before_pool1_shape)
        x = x + before_pool1_clone
        x=self.conv2d_12(x)
        x=self.act_elu(x)
        x=self.batch_norm_12(x)
        x=self.conv2d_11(x)
        x=self.batch_norm_11(x)
        x=x.squeeze(0)
        x=self.identity(x)
        x=self.no_neg(x)
        return x
    
    
    def loss(self, pred, label):
        return torch.nn.MSELoss()(pred, label)

