import torch  # PyTorch for building and training models
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimizer modules
import numpy as np  # NumPy for numerical operations
import torchvision.transforms as transforms  # Image transformations
from torch.utils.data import DataLoader, Dataset  # DataLoader and Dataset utilities
from torchsummary import summary  # Summary of the model architecture

# Define the ConvBlock class, inheriting from nn.Module
class ConvBlock(nn.Module):
    def __init__(self, n_layers: int, filters: int, kernel: int = 3, growth_factor: float = 2.0, 
                 moment: float = 0.7, stride: bool = True, alpha: float = 0.03):
        """
        Initialize the convolutional block with specified parameters.

        Args:
            n_layers (int): Number of layers in the block.
            filters (int): Number of filters for the convolutional layers.
            kernel (int): Kernel size for the convolutional layers.
            growth_factor (float): Growth factor for the filters.
            moment (float): Momentum for batch normalization.
            stride (bool): If True, apply stride to reduce the size of the feature map.
            alpha (float): Negative slope coefficient for the LeakyReLU activation function.
        """
        super(ConvBlock, self).__init__()
        PADDING = (kernel - 1) // 2  # Padding is calculated based on kernel size to preserve input dimensions
        self.stride = stride  # Whether to use stride for downsampling

        # Initialize batch normalization layers for each convolutional layer
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_features=filters, momentum=moment) for _ in range(n_layers)])
        
        # Initialize convolutional layers
        self.conv = nn.ModuleList([nn.Conv2d(filters, filters, kernel, padding=PADDING) for _ in range(n_layers - 1)])
        
        # Non-linear activation function (LeakyReLU)
        self.nlin = nn.LeakyReLU(alpha)

        # Final convolutional layer, with optional stride or max pooling
        if stride:
            # Add convolutional layer with stride for downsampling
            self.conv.append(nn.Conv2d(int(filters // growth_factor), filters, kernel, stride=2, padding=PADDING))
        else:
            # Add convolutional layer without stride
            self.conv.append(nn.Conv2d(int(filters // growth_factor), filters, kernel, padding=PADDING))
            # Add a max pooling layer if stride is not used
            self.pool = nn.MaxPool2d(2, 2)

        # Reverse the convolutional layers for the forward pass order
        self.conv.reverse()

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Args:
            x: Input tensor (batch_size, channels, height, width).
        
        Returns:
            Output tensor after passing through the convolutional layers, normalization, and activation.
        """
        if not self.stride:
            # Apply max pooling if stride is not used
            x = self.pool(x)
        
        # Iterate over the convolutional and normalization layers
        for conv, norm in zip(self.conv, self.norm):
            x = self.nlin(norm(conv(x)))  # Apply convolution, batch normalization, and activation
        return x
