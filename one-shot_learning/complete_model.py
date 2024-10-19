
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from one_shot_learning_block import ConvBlock

import torch
import torch.nn as nn
import torch.nn.functional as F  # Importing PyTorch libraries for neural networks
from torchsummary import summary  # For summarizing the model architecture
from torchvision import transforms  # For performing transformations on images
from torch import utils  # Utilities for PyTorch like DataLoader and Tensor manipulation

import cv2  # OpenCV for image processing tasks
import numpy as np  # NumPy for numerical operations
import os  # For file and directory operations
import random  # For generating random values

from tqdm.auto import trange  # TQDM for progress bars in loops



class OneShot(nn.Module):
    def __init__(self, n_blocks: int = 7, n_high_refine: int = 3, n_conv_high_refine: int = 3, 
                 n_conv_end: int = 2, filters: int = 64, start_kernel: int = 5, kernel: int = 3, 
                 growth_factor: float = 2.0, alpha: float = 0.07, moment: float = 0.7, dense: int = 512, 
                 final: int = 100, drop: float = 0.2, stride: bool = True):
        super(OneShot, self).__init__()

        START_PADDING = (start_kernel - 1) // 2  # Padding for the starting convolution
        PADDING = (kernel - 1) // 2  # Padding for regular convolution layers

        # Initial convolutional layer with stride
        self.conv = nn.ModuleList([nn.Conv2d(3, filters, start_kernel, stride=2, padding=START_PADDING)])
        
        # Adding convolutional blocks
        for c in range(n_blocks):
            filters = int(filters * growth_factor)
            if c <= n_high_refine:
                # Add a high refinement ConvBlock
                self.conv.append(ConvBlock(n_conv_high_refine, filters, kernel, growth_factor, moment, stride, alpha))
            else:
                # Add an end refinement ConvBlock
                self.conv.append(ConvBlock(n_conv_end, filters, kernel, growth_factor, moment, stride, alpha))
        
        # Adaptive average pooling to reduce feature map size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Flatten the output of the pooling layer
        self.flat = nn.Flatten()
        
        # Fully connected layers for classification
        self.lden = nn.Linear(int(filters * growth_factor), dense)
        self.lvec = nn.Linear(dense, final)
        
        # Activation and dropout
        self.nlin = nn.LeakyReLU(alpha)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Pass input through each convolutional block
        for conv in self.conv:
            x = conv(x)

        # Pool, flatten, and apply the fully connected layers with dropout
        x = self.flat(self.pool(x))
        x = self.drop(self.nlin(self.lden))
        return self.lvec(x)


# Training loop

