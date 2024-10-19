from utility_functions import ConvBlock
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, n_layers: int, filters: int, kernel: int = 3, growth_factor: float = 2.0, 
                 moment: float = 0.7, stride: bool = True, alpha: float = 0.03):
        super(ConvBlock, self).__init__()
        PADDING = (kernel - 1) // 2  # Padding based on the kernel size
        self.stride = stride

        # Batch normalization layers
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_features=filters, momentum=moment) for _ in range(n_layers)])
        
        # Convolutional layers
        self.conv = nn.ModuleList([nn.Conv2d(filters, filters, kernel, padding=PADDING) for _ in range(n_layers - 1)])
        
        # Non-linear activation function
        self.nlin = nn.LeakyReLU(alpha)

        # Add final convolutional layer with stride or max pooling if applicable
        if stride:
            self.conv.append(nn.Conv2d(int(filters // growth_factor), filters, kernel, stride=2, padding=PADDING))
        else:
            self.conv.append(nn.Conv2d(int(filters // growth_factor), filters, kernel, padding=PADDING))
            self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer

        self.conv.reverse()  # Reverse the convolution layers for the forward pass

    def forward(self, x):
        if not self.stride:
            x = self.pool(x)
        
        # Pass through convolutional and normalization layers with activation
        for conv, norm in zip(self.conv, self.norm):
            x = self.nlin(norm(conv(x)))
        return x

