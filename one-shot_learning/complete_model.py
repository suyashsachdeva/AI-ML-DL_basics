import torch
import torch.nn as nn  # Importing PyTorch's neural network modules
import torch.optim as optim  # Optimizer module for model optimization
import numpy as np  # Importing NumPy for numerical computations
import torchvision.transforms as transforms  # For applying transformations to input data
from torch.utils.data import DataLoader, Dataset  # For creating custom datasets and loading data

from one_shot_learning_block import ConvBlock  # Importing a custom convolutional block module

import torch.nn.functional as F  # Importing additional functions like activation functions
from torchsummary import summary  # For summarizing the model architecture (showing layer information)
from torchvision import transforms  # For transforming images before feeding them to the model
from torch import utils  # Various utilities from PyTorch, such as data loaders

import cv2  # OpenCV library for image processing tasks like reading and modifying images
import os  # OS library for file and directory management
import random  # Random library for generating random numbers, used in data augmentations
from tqdm.auto import trange  # TQDM for creating progress bars during loops


class OneShot(nn.Module):
    """
    The OneShot class defines a deep convolutional neural network for one-shot learning tasks.
    It uses multiple convolutional blocks followed by fully connected layers for classification.
    """
    def __init__(self, n_blocks: int = 7, n_high_refine: int = 3, n_conv_high_refine: int = 3, 
                 n_conv_end: int = 2, filters: int = 64, start_kernel: int = 5, kernel: int = 3, 
                 growth_factor: float = 2.0, alpha: float = 0.07, moment: float = 0.7, dense: int = 512, 
                 final: int = 100, drop: float = 0.2, stride: bool = True):
        """
        Initializes the OneShot neural network model.
        
        Args:
            n_blocks: Number of convolutional blocks in the network.
            n_high_refine: Number of blocks that use high refinement (deep convolutional layers).
            n_conv_high_refine: Number of convolutional layers in high refinement blocks.
            n_conv_end: Number of convolutional layers in the later blocks.
            filters: Number of filters in the first convolutional layer.
            start_kernel: Kernel size for the first convolutional layer.
            kernel: Kernel size for subsequent convolutional layers.
            growth_factor: Multiplier for increasing the number of filters after each block.
            alpha: Negative slope value for Leaky ReLU activation.
            moment: Momentum for batch normalization.
            dense: Number of neurons in the fully connected layer.
            final: Number of output neurons (for classification).
            drop: Dropout rate for regularization.
            stride: Whether to apply stride (downsampling) in the convolutional layers.
        """
        super(OneShot, self).__init__()

        START_PADDING = (start_kernel - 1) // 2  # Calculate padding for the starting convolutional layer
        PADDING = (kernel - 1) // 2  # Padding for subsequent convolutional layers

        # Initial convolutional layer: 3 input channels (RGB), 'filters' output channels, stride of 2 to downsample
        self.conv = nn.ModuleList([nn.Conv2d(3, filters, start_kernel, stride=2, padding=START_PADDING)])
        
        # Adding a series of convolutional blocks to the network
        for c in range(n_blocks):
            filters = int(filters * growth_factor)  # Increase the number of filters based on the growth factor
            if c <= n_high_refine:
                # Add a high refinement ConvBlock (deep convolutional layers) for early stages
                self.conv.append(ConvBlock(n_conv_high_refine, filters, kernel, growth_factor, moment, stride, alpha))
            else:
                # Add an end refinement ConvBlock for later stages (shallower layers)
                self.conv.append(ConvBlock(n_conv_end, filters, kernel, growth_factor, moment, stride, alpha))
        
        # Adaptive average pooling to reduce the spatial dimensions to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Flatten the output of the pooling layer to prepare for fully connected layers
        self.flat = nn.Flatten()
        
        # Fully connected layer 1: connects to a dense layer with 'dense' number of neurons
        self.lden = nn.Linear(int(filters * growth_factor), dense)
        
        # Fully connected layer 2 (output layer): connects 'dense' neurons to the final output (e.g., for classification)
        self.lvec = nn.Linear(dense, final)
        
        # Activation function: Leaky ReLU is used to introduce non-linearity with a small slope for negative values
        self.nlin = nn.LeakyReLU(alpha)
        
        # Dropout for regularization, reducing overfitting
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Defines the forward pass of the model, describing how data flows through the network.
        
        Args:
            x: Input image batch.
        
        Returns:
            Output of the network after passing through all layers (classification output).
        """
        # Pass the input through each convolutional block
        for conv in self.conv:
            x = conv(x)

        # Apply adaptive pooling, flatten the output, and pass through fully connected layers
        x = self.flat(self.pool(x))  # Pooling reduces spatial dimensions to 1x1, then flatten to a vector
        x = self.drop(self.nlin(self.lden))  # Fully connected layer followed by dropout and activation
        return self.lvec(x)  # Final output layer for classification


# Example Training Loop (for illustration, not complete)

# Define hyperparameters and other training settings
batch_size = 32  # Number of samples in a batch
epochs = 10  # Number of epochs to train the model

# Sample code for loading data using PyTorch's DataLoader
# Assuming dataset is already prepared and wrapped in a Dataset object

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# model = OneShot()  # Instantiate the model
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer for updating weights
# criterion = nn.CrossEntropyLoss()  # Loss function for classification tasks

# for epoch in range(epochs):
#     model.train()  # Set model to training mode
#     running_loss = 0.0

#     # Loop over batches of data
#     for i, data in enumerate(train_loader):
#         inputs, labels = data  # Get the inputs and labels from the data loader
#         optimizer.zero_grad()  # Zero out gradients before backpropagation
#         outputs = model(inputs)  # Forward pass through the model
#         loss = criterion(outputs, labels)  # Calculate loss
#         loss.backward()  # Backpropagation
#         optimizer.step()  # Update model weights
#         running_loss += loss.item()

#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")
